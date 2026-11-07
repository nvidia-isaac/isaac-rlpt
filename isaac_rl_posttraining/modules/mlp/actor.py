# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP actor for single-step (non-chunked) action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.modules.base import BaseActor, BaseActorConfig
from isaac_rl_posttraining.modules.base.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from isaac_rl_posttraining.utils.featurizers import ObservationFeaturizer
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["MLPActorConfig", "MLPActor"]


def build_obs_featurizer(config: Any) -> Optional[ObservationFeaturizer]:
    """Build the per-key featurizer when the config asks for one, else ``None``."""
    if config.featurizer is None and config.image_encoder_type not in ("small", "impala"):
        return None
    return ObservationFeaturizer(
        observation_space=config.observation_space,
        featurizer_cfg=config.featurizer,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        dropout_rate=config.dropout_rate,
        image_encoder_type=config.image_encoder_type,
        impala_nn_scale=config.impala_nn_scale,
        impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
        impala_use_smaller=config.impala_use_smaller,
        impala_bounded_head=config.impala_bounded_head,
        small_bottleneck_dim=config.small_bottleneck_dim,
        impala_output_dim=config.impala_output_dim,
    )


def flat_obs_layout(observation_space: gym.Space) -> tuple[Optional[list[str]], int]:
    """Return ``(flatten_keys, obs_dim)`` for un-featurized observations."""
    if isinstance(observation_space, gym.spaces.Dict):
        flatten_keys: Optional[list[str]] = [
            k for k, space in observation_space.spaces.items() if getattr(space, "shape", None) is not None
        ]
        obs_dim = sum(int(np.prod(observation_space.spaces[k].shape)) for k in flatten_keys)
    else:
        flatten_keys = None
        obs_dim = int(np.prod(observation_space.shape))
    if obs_dim <= 0:
        raise ValueError(f"Invalid observation dimension from observation_space: {obs_dim}")
    return flatten_keys, obs_dim


def _dict_batch_size(obs: dict, keys: list[str]) -> int:
    """Batch size from the first clearly batched (>=2-D) tensor, else 1."""
    for k in keys:
        v = obs.get(k)
        if torch.is_tensor(v) and v.dim() >= 2:
            return int(v.size(0))
    return 1


def _flatten_dict_value(v: Any, batch_size: int) -> Any:
    """Orient one dict entry as ``[B, D]`` (1-D per-batch scalars become ``(B, 1)``)."""
    if torch.is_tensor(v):
        if v.dim() == 0:
            v = v.view(1, 1)
        elif v.dim() == 1:
            v = v.unsqueeze(-1) if int(v.size(0)) == batch_size else v.unsqueeze(0)
    return v.reshape(v.size(0), -1) if torch.is_tensor(v) and v.dim() > 1 else v


def flatten_plain_obs(
    obs: Union[dict, torch.Tensor], flatten_keys: Optional[list[str]], device: torch.device
) -> torch.Tensor:
    """Flatten an un-featurized observation batch to ``[B, D]`` on ``device``."""
    if isinstance(obs, dict):
        batch_size = _dict_batch_size(obs, flatten_keys)
        feats = [
            _flatten_dict_value(obs[k], batch_size).to(device) for k in flatten_keys if obs.get(k) is not None
        ]
        return torch.cat(feats, dim=-1) if feats else torch.empty(0, device=device)

    if obs.dim() > 2:
        return obs.reshape(obs.size(0), -1).to(device)
    if obs.dim() == 1:
        return obs.unsqueeze(0).to(device)
    return obs.to(device)


@dataclass
class MLPActorConfig(BaseActorConfig):
    """Configuration for :class:`MLPActor`.

    Attributes:
        hidden_dims: Trunk layer widths.
        activation: Trunk activation name.
        use_layer_norm: LayerNorm after each trunk layer.
        dropout_rate: Dropout after each trunk activation.
        use_tanh_output: Squash continuous actions through ``tanh``.
        log_std_init: Initial bias of the ``log_std`` head.
        log_std_min: Lower clamp for predicted ``log_std``.
        log_std_max: Upper clamp for predicted ``log_std``.
        deterministic: Build a deterministic (non-distributional) policy.
        featurizer: Per-key featurizer spec for dict observations — a
            list/tuple of hidden dims builds an MLP for that key, an
            ``nn.Module`` is used directly.
        image_encoder_type: ``"impala"`` enables IMPALA encoders for image keys.
        impala_nn_scale: IMPALA channel-width multiplier.
        impala_num_blocks_per_stack: IMPALA residual blocks per stack.
        impala_use_smaller: Use the reduced-depth IMPALA variant.
        impala_output_dim: IMPALA embedding projection width.
    """

    hidden_dims: tuple = (256, 256)
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    use_tanh_output: bool = False
    log_std_init: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    deterministic: bool = False

    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[Callable] = None

    image_encoder_type: Optional[str] = None
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_bounded_head: bool = False
    small_bottleneck_dim: int = 50
    official_policy_init: bool = False
    impala_output_dim: Optional[int] = None

    @property
    def actor_class(self) -> type:
        return MLPActor


class MLPActor(BaseActor):
    """Feed-forward Gaussian (or categorical) policy over flattened observations.

    Dict observations are flattened key-by-key, optionally through per-key
    featurizers (see :class:`~isaac_rl_posttraining.utils.featurizers.ObservationFeaturizer`).
    """

    def __init__(self, config: MLPActorConfig) -> None:
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform

        self.obs_featurizer: Optional[ObservationFeaturizer] = build_obs_featurizer(config)
        self._flatten_keys: Optional[list[str]]
        if self.obs_featurizer is not None:
            obs_dim = self.obs_featurizer.output_dim
            if obs_dim <= 0:
                raise ValueError("ObservationFeaturizer produced an invalid output dimension for the actor.")
            self._flatten_keys = None
        else:
            self._flatten_keys, obs_dim = flat_obs_layout(config.observation_space)

        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))
        else:
            action_dim = config.action_space.n

        self.mlp = build_mlp(
            obs_dim,
            config.hidden_dims,
            config.activation,
            config.use_layer_norm,
            config.dropout_rate,
        )
        final_hidden_dim = int(config.hidden_dims[-1])
        self.hidden_dim = final_hidden_dim

        if self.is_continuous:
            self._build_gaussian_heads(final_hidden_dim, action_dim)
        else:
            self.logits_layer = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    def _build_gaussian_heads(self, final_hidden_dim: int, action_dim: int) -> None:
        """Build the mean/log-std heads and the continuous action distribution."""
        config = self.config
        self.mean_layer = nn.Linear(final_hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(final_hidden_dim, action_dim)
        self.log_std_layer.weight.data.fill_(0.0)
        self.log_std_layer.bias.data.fill_(config.log_std_init)
        if config.official_policy_init:
            self._apply_official_policy_init()
        if config.use_tanh_output:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
        else:
            self.action_dist = DiagGaussianDistribution(action_dim=action_dim)

    def _apply_official_policy_init(self) -> None:
        """Re-initialize so the steering prior is observation-independent N(0, sigma^2)."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.zeros_(self.log_std_layer.weight)
        self.log_std_layer.bias.data.fill_(self.config.log_std_init)

    def _flatten_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Flatten (and featurize) an observation batch to ``[B, D]``."""
        device = next(self.parameters()).device
        if self.obs_featurizer is not None:
            return self.obs_featurizer.flatten_obs(obs, device=device)
        return flatten_plain_obs(obs, self._flatten_keys, device)

    def _hidden(self, obs: Any) -> torch.Tensor:
        """Run the trunk on flattened observations."""
        return self.mlp(self._flatten_obs(obs).float())

    def _act(self, obs: Any, deterministic: bool = False) -> torch.Tensor:
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        hidden = self._hidden(obs)
        mean_actions, log_std, _ = self.get_action_dist_params(obs, hidden=hidden)
        if log_std is not None:
            return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        return self.action_dist.actions_from_params(mean_actions, deterministic=deterministic)

    def log_prob(self, obs: Any, action: torch.Tensor) -> torch.Tensor:
        """Log-probability of ``action`` under the current policy."""
        if self.config.deterministic:
            raise ValueError("Cannot compute log_prob for a deterministic policy")
        hidden = self._hidden(obs)
        mean_actions, log_std, _ = self.get_action_dist_params(obs, hidden=hidden)
        if self.is_continuous:
            self.action_dist.proba_distribution(mean_actions, log_std)
        else:
            self.action_dist.proba_distribution(mean_actions)
        return self.action_dist.log_prob(action)

    def _forward(self, obs: Any, actor_state: Any = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden = self._hidden(obs)
        if self.is_continuous:
            mean = self.mean_layer(hidden)
            log_std = torch.clamp(self.log_std_layer(hidden), self.config.log_std_min, self.config.log_std_max)
            return mean, log_std
        return self.logits_layer(hidden)

    def get_hidden_state(self, obs: Any) -> torch.Tensor:
        """Expose the trunk output for feature-extractor reuse."""
        return self._hidden(obs)

    def get_action_dist_params(
        self, obs: Any, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Return ``(mean, log_std, extra_kwargs)``; logits carry ``log_std=None``."""
        if hidden is None:
            hidden = self._hidden(obs)
        if self.is_continuous:
            mean = self.mean_layer(hidden)
            log_std = torch.clamp(self.log_std_layer(hidden), self.config.log_std_min, self.config.log_std_max)
            return mean, log_std, {}
        return self.logits_layer(hidden), None, {}
