# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RNN chunking actor: recurrent alternative to the transformer actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.modules.base import BaseActor, BaseActorConfig
from isaac_rl_posttraining.modules.base.distributions import (
    CategoricalDistribution,
    SquashedDiagGaussianDistribution,
)
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["RNNActorConfig", "RNNActor"]

_RNN_TYPES = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}


@dataclass
class RNNActorConfig(BaseActorConfig):
    """Configuration for :class:`RNNActor`.

    Attributes:
        rnn_type: ``LSTM``, ``GRU``, or ``RNN``.
        rnn_hidden_size: Recurrent hidden width.
        rnn_num_layers: Stacked recurrent layers.
        rnn_dropout: Inter-layer recurrent dropout.
        rnn_bidirectional: Use a bidirectional RNN.
        feature_hidden_dims: MLP before the RNN (empty = none).
        output_hidden_dims: MLP after the RNN (empty = none).
        activation: Activation for the MLP parts.
        use_layer_norm: LayerNorm inside the MLP parts.
        dropout_rate: Dropout inside the MLP parts.
        log_std_init: Initial bias of the ``log_std`` head.
        log_std_min: Lower clamp for predicted ``log_std``.
        log_std_max: Upper clamp for predicted ``log_std``.
        chunk_size: Number of action steps predicted per observation.
        featurizer: Per-key featurizer spec for dict observations.
        image_encoder_type: ``"impala"`` enables IMPALA encoders for image keys.
        impala_nn_scale: IMPALA channel-width multiplier.
        impala_num_blocks_per_stack: IMPALA residual blocks per stack.
        impala_use_smaller: Use the reduced-depth IMPALA variant.
        impala_output_dim: IMPALA embedding projection width.
    """

    rnn_type: str = "LSTM"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_dropout: float = 0.0
    rnn_bidirectional: bool = False

    feature_hidden_dims: Optional[list[int]] = None
    output_hidden_dims: Optional[list[int]] = None

    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    log_std_init: float = -0.5
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    chunk_size: int = 30

    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[list[Any]] = None

    image_encoder_type: Optional[str] = None
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: Optional[int] = None

    def __post_init__(self) -> None:
        """Fill list defaults and validate RNN options."""
        if self.feature_hidden_dims is None:
            self.feature_hidden_dims = []
        if self.output_hidden_dims is None:
            self.output_hidden_dims = []
        if self.rnn_type not in _RNN_TYPES:
            raise ValueError(f"rnn_type must be one of {sorted(_RNN_TYPES)}, got {self.rnn_type}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    @property
    def actor_class(self) -> type:
        return RNNActor


def _flat_2d(v: torch.Tensor) -> torch.Tensor:
    """View a batched tensor as ``[B, D]``; promote a 1-D tensor to ``[1, D]``."""
    return v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)


def _first_batch_size(obs: dict) -> int:
    """Batch size of the first present entry (1 for unbatched or empty)."""
    for other in obs.values():
        if other is not None:
            return (other.size(0) if other.dim() > 1 else 1) or 1
    return 1


class _RNNFeaturizerMixin:
    """Shared featurizer construction and observation flattening for RNN nets."""

    config: Any
    featurizer_cfg: dict
    featurizers: Optional[nn.ModuleDict]
    _flatten_keys: Optional[list[str]]

    def _init_featurizers(self) -> int:
        """Build per-key featurizers and return the flattened feature width."""
        config = self.config
        self.featurizer_cfg = dict(config.featurizer) if config.featurizer is not None else {}

        if config.image_encoder_type == "impala":
            self._add_impala_featurizer_cfg()

        if not self.featurizer_cfg:
            self.featurizers = None
            return self._plain_obs_dim()
        self.featurizers = nn.ModuleDict()
        self._build_featurizer_modules()
        self._flatten_keys = None
        return sum(
            int(value.output_dim) if hasattr(value, "output_dim") else int(sum(value))
            for value in self.featurizer_cfg.values()
        )

    def _plain_obs_dim(self) -> int:
        """Record the keys to flatten (dict spaces) and return the flat width."""
        space = self.config.observation_space
        if isinstance(space, gym.spaces.Dict):
            self._flatten_keys = [k for k, sub in space.spaces.items() if getattr(sub, "shape", None) is not None]
            return sum(int(np.prod(space.spaces[k].shape)) for k in self._flatten_keys)
        self._flatten_keys = None
        return int(np.prod(space.shape))

    def _add_impala_featurizer_cfg(self) -> None:
        """IMPALA encoders for image keys, ``[256]`` MLPs for any other unset key."""
        from isaac_rl_posttraining.utils.featurizers import (
            build_impala_featurizers,
            identify_image_keys,
        )

        config = self.config
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError("IMPALA encoder requires a Dict observation space")
        impala_featurizers = build_impala_featurizers(
            observation_space=config.observation_space,
            keys=None,
            nn_scale=config.impala_nn_scale,
            num_blocks_per_stack=config.impala_num_blocks_per_stack,
            use_smaller=config.impala_use_smaller,
            requires_grad=True,
            output_dim=config.impala_output_dim,
        )
        image_keys = identify_image_keys(list(config.observation_space.spaces.keys()))
        for key in image_keys:
            self.featurizer_cfg.setdefault(key, impala_featurizers[key])
        for key in config.observation_space.spaces:
            self.featurizer_cfg.setdefault(key, [256])

    def _build_featurizer_modules(self) -> None:
        """Register one featurizer module per configured key."""
        config = self.config
        for key, value in self.featurizer_cfg.items():
            if isinstance(value, (list, tuple)):
                obs_dim = int(np.prod(config.observation_space.spaces[key].shape))
                self.featurizers[key] = build_mlp(
                    obs_dim, value, config.activation, config.use_layer_norm, config.dropout_rate
                )
            elif isinstance(value, nn.Module):
                self.featurizers[key] = value
            else:
                raise ValueError(f"Featurizer for key {key!r} must be list/tuple or nn.Module")

    def _flatten_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Featurize/flatten an observation batch to ``[B, D]``."""
        if isinstance(obs, dict):
            if self.featurizer_cfg:
                return self._featurize_dict_obs(obs)
            if self._flatten_keys:
                return self._flatten_keyed_dict_obs(obs)
            feats = [_flat_2d(v) for v in obs.values() if v is not None]
            return torch.cat(feats, dim=-1) if feats else torch.empty(0)

        if obs.dim() > 2:
            return obs.view(obs.size(0), -1)
        if obs.dim() == 1:
            return obs.unsqueeze(0)
        return obs

    def _featurize_dict_obs(self, obs: dict) -> torch.Tensor:
        """Run each key through its featurizer (pass-through when it has none)."""
        feats = []
        for k, v in obs.items():
            v_flat = _flat_2d(v)
            if k in self.featurizers:
                feats.append(self.featurizers[k](v_flat))
            else:
                feats.append(v_flat)
        if not feats:
            raise ValueError(f"No featurizable keys present in observation dict. Keys: {list(obs.keys())}")
        return torch.cat(feats, dim=-1)

    def _flatten_keyed_dict_obs(self, obs: dict) -> torch.Tensor:
        """Concatenate ``_flatten_keys`` in order, zero-filling missing keys."""
        feats = []
        batch_size: Optional[int] = None
        for k in self._flatten_keys:
            v = obs.get(k)
            flat_dim = int(np.prod(self.config.observation_space.spaces[k].shape))
            if v is None:
                if batch_size is None:
                    batch_size = _first_batch_size(obs)
                feats.append(torch.zeros(batch_size, flat_dim, device=next(self.parameters()).device))
            else:
                feats.append(_flat_2d(v))
        return torch.cat(feats, dim=-1) if feats else torch.empty(0)

    def _build_rnn(self, input_size: int) -> nn.Module:
        """Instantiate the configured recurrent core."""
        rnn_cls = _RNN_TYPES[self.config.rnn_type]
        return rnn_cls(
            input_size=input_size,
            hidden_size=self.config.rnn_hidden_size,
            num_layers=self.config.rnn_num_layers,
            dropout=self.config.rnn_dropout if self.config.rnn_num_layers > 1 else 0,
            bidirectional=self.config.rnn_bidirectional,
            batch_first=True,
        )


class RNNActor(_RNNFeaturizerMixin, BaseActor):
    """Recurrent chunking policy predicting ``[B, chunk_size, A]`` actions.

    The observation is featurized once and unrolled through the recurrent
    core to produce one distribution per chunk step.
    """

    def __init__(self, config: RNNActorConfig) -> None:
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform

        obs_dim = self._init_featurizers()

        if config.feature_hidden_dims:
            self.feature_mlp: Optional[nn.Module] = build_mlp(
                obs_dim, config.feature_hidden_dims, config.activation, config.use_layer_norm, config.dropout_rate
            )
            rnn_input_size = config.feature_hidden_dims[-1]
        else:
            self.feature_mlp = None
            rnn_input_size = obs_dim

        self.rnn_input_size = rnn_input_size
        self.rnn = self._build_rnn(rnn_input_size)

        rnn_output_size = config.rnn_hidden_size * (2 if config.rnn_bidirectional else 1)
        if config.output_hidden_dims:
            self.output_mlp: Optional[nn.Module] = build_mlp(
                rnn_output_size,
                config.output_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            final_hidden_dim = config.output_hidden_dims[-1]
        else:
            self.output_mlp = None
            final_hidden_dim = rnn_output_size

        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))
            self.mean_layer = nn.Linear(final_hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(final_hidden_dim, action_dim)
            self.log_std_layer.weight.data.fill_(0.0)
            self.log_std_layer.bias.data.fill_(config.log_std_init)
            self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
        else:
            action_dim = config.action_space.n
            self.logits_layer = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    def _process_observations(self, obs: Any) -> torch.Tensor:
        """Preprocess, flatten, and (optionally) MLP-encode observations."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        obs_flat = self._flatten_obs(obs)
        if self.feature_mlp is not None:
            obs_flat = self.feature_mlp(obs_flat)
        return obs_flat

    def _unroll_chunk(self, obs_processed: torch.Tensor, hidden_state: Any = None) -> torch.Tensor:
        """Run the recurrent core once per chunk step and stack the outputs."""
        rnn_outputs = []
        for _ in range(self.config.chunk_size):
            rnn_output, _ = self.rnn(obs_processed, hidden_state) if hidden_state is not None else self.rnn(
                obs_processed
            )
            rnn_outputs.append(rnn_output)
        stacked = torch.stack(rnn_outputs, dim=1)  # [B, K, hidden]
        if self.output_mlp is not None:
            stacked = self.output_mlp(stacked.reshape(-1, stacked.size(-1)))
        return stacked

    def _forward(
        self, obs: Any, actor_state: Any = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-step distribution parameters for the whole chunk."""
        if isinstance(obs, dict):
            obs = {k: v.unsqueeze(1) if v.dim() == 2 else v for k, v in obs.items()}
            batch_size = next(iter(obs.values())).size(0)
        else:
            obs = obs.unsqueeze(1)
            batch_size = obs.size(0)

        obs_processed = self._process_observations(obs)
        rnn_outputs = self._unroll_chunk(obs_processed)

        if self.is_continuous:
            mean = self.mean_layer(rnn_outputs)
            log_std = torch.clamp(self.log_std_layer(rnn_outputs), self.config.log_std_min, self.config.log_std_max)
            return (
                mean.view(batch_size, self.config.chunk_size, -1),
                log_std.view(batch_size, self.config.chunk_size, -1),
            )
        logits = self.logits_layer(rnn_outputs)
        return logits.view(batch_size, self.config.chunk_size, -1)

    def _act(self, obs: Any, deterministic: bool = False, actor_state: Any = None) -> tuple[torch.Tensor, Any]:
        """Sample a full action chunk for a single observation."""
        obs_processed = self._process_observations(obs).unsqueeze(1)  # [B, 1, feature]

        if actor_state is not None and "hidden_latent" in actor_state and actor_state["hidden_latent"] is not None:
            hidden_state = actor_state["hidden_latent"]
        else:
            hidden_state = self._get_initial_hidden_state(obs_processed.size(0))

        rnn_outputs = self._unroll_chunk(obs_processed, hidden_state)

        batch_size = obs_processed.size(0)
        if self.is_continuous:
            mean = self.mean_layer(rnn_outputs)
            log_std = torch.clamp(self.log_std_layer(rnn_outputs), self.config.log_std_min, self.config.log_std_max)
            action = self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic)
        else:
            logits = self.logits_layer(rnn_outputs)
            action = self.action_dist.actions_from_params(logits, deterministic=deterministic)

        action = action.view(batch_size, self.config.chunk_size, -1)
        return action, {"hidden_latent": hidden_state}

    def get_action_dist_params(
        self, obs: Any, hidden: Any = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Return ``(mean, log_std, extra_kwargs)`` from a recurrent pass."""
        obs_processed = self._process_observations(obs)
        if hidden is None:
            rnn_output, _ = self.rnn(obs_processed)
        else:
            rnn_output, _ = self.rnn(obs_processed.unsqueeze(1), hidden)

        if self.is_continuous:
            mean = self.mean_layer(rnn_output)
            log_std = torch.clamp(self.log_std_layer(rnn_output), self.config.log_std_min, self.config.log_std_max)
            return mean, log_std, {}
        return self.logits_layer(rnn_output), None, {}

    def get_initial_state(self) -> dict[str, Any]:
        """Initial recurrent state placeholder, filled on first use."""
        return {"hidden_latent": None}

    def _get_initial_hidden_state(self, batch_size: int) -> Any:
        """Zero recurrent state matching the configured RNN geometry."""
        device = next(self.parameters()).device
        num_directions = 2 if self.config.rnn_bidirectional else 1
        shape = (self.config.rnn_num_layers * num_directions, batch_size, self.config.rnn_hidden_size)
        h_0 = torch.zeros(*shape, device=device)
        if self.config.rnn_type == "LSTM":
            return (h_0, torch.zeros(*shape, device=device))
        return h_0
