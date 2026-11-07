# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RNN critic scoring an observation (sequence) plus an action chunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.modules.base import BaseCritic, BaseCriticConfig
from isaac_rl_posttraining.modules.rnn.actor import _RNNFeaturizerMixin
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["RNNCriticConfig", "RNNCritic"]


@dataclass
class RNNCriticConfig(BaseCriticConfig):
    """Configuration for :class:`RNNCritic`.

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
        chunk_size: Expected action-chunk length for Q-functions.
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
        if self.rnn_type not in ("LSTM", "GRU", "RNN"):
            raise ValueError(f"rnn_type must be one of ['LSTM', 'GRU', 'RNN'], got {self.rnn_type}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    @property
    def critic_class(self) -> type:
        return RNNCritic


def _as_sequences(
    obs: Any, action: Optional[torch.Tensor]
) -> tuple[Any, Optional[torch.Tensor]]:
    """Promote single observations/actions to length-1 sequences."""
    first_value = next(iter(obs.values())) if isinstance(obs, dict) else obs
    if first_value.dim() == 3:
        return obs, action
    if isinstance(obs, dict):
        obs = {k: v.unsqueeze(1) if v.dim() == 2 else v for k, v in obs.items()}
    else:
        obs = obs.unsqueeze(1)
    if action is not None:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        elif action.dim() == 1:
            action = action.unsqueeze(0).unsqueeze(1)
    return obs, action


class RNNCritic(_RNNFeaturizerMixin, BaseCritic):
    """Recurrent value network over observation and action-chunk sequences.

    Q-functions embed the per-step actions with a small MLP and append them
    to the observation sequence before the recurrent pass; the last hidden
    state feeds a scalar value head. ``use_action=False`` yields a
    V-function over the observation sequence alone.
    """

    def __init__(self, config: RNNCriticConfig) -> None:
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform
        self.use_action = config.use_action

        obs_dim = self._init_featurizers()

        if obs_dim > 0 and config.feature_hidden_dims:
            self.feature_mlp: Optional[nn.Module] = build_mlp(
                obs_dim, config.feature_hidden_dims, config.activation, config.use_layer_norm, config.dropout_rate
            )
            rnn_input_size = config.feature_hidden_dims[-1]
        else:
            self.feature_mlp = None
            rnn_input_size = obs_dim

        if self.use_action:
            if isinstance(config.action_space, gym.spaces.Box):
                action_dim = int(np.prod(config.action_space.shape))
            elif isinstance(config.action_space, gym.spaces.Discrete):
                action_dim = config.action_space.n
            else:
                raise ValueError(f"Unsupported action space: {config.action_space}")
            # Actions are projected to the RNN input width so they can be
            # appended to the observation sequence as extra timesteps.
            self.action_mlp = build_mlp(
                action_dim, [rnn_input_size], config.activation, config.use_layer_norm, config.dropout_rate
            )

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

        self.value_head = nn.Linear(final_hidden_dim, 1)

    def _process_observations(self, obs: Any) -> torch.Tensor:
        """Preprocess, flatten, and (optionally) MLP-encode one timestep."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        obs_flat = self._flatten_obs(obs)
        if self.feature_mlp is not None:
            obs_flat = self.feature_mlp(obs_flat)
        return obs_flat

    def forward(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return values of shape ``[B, 1]``."""
        return self.value_head(self.compute_pooled(obs, action))

    def compute_pooled(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the last recurrent output feeding the value head.

        Mirrors the transformer critic's ``compute_pooled`` so ensembles can
        share the trunk and apply per-member value heads.
        """
        obs, action = _as_sequences(obs, action)
        if isinstance(obs, dict):
            batch_size, seq_len = next(iter(obs.values())).shape[:2]
        else:
            batch_size, seq_len = obs.shape[:2]

        # Featurize each observation timestep independently.
        obs_steps = []
        for i in range(seq_len):
            obs_t = {k: v[:, i] for k, v in obs.items()} if isinstance(obs, dict) else obs[:, i]
            obs_steps.append(self._process_observations(obs_t))
        obs_processed = torch.stack(obs_steps, dim=1)  # [B, T, feature]

        if self.use_action and action is not None:
            action_steps = self._embed_action_chunk(action, batch_size, seq_len)
            rnn_input = torch.cat([obs_processed, action_steps], dim=1)
        else:
            rnn_input = obs_processed

        rnn_output, _ = self.rnn(rnn_input)  # [B, T(+K), hidden]
        last_hidden = rnn_output[:, -1]
        if self.output_mlp is not None:
            last_hidden = self.output_mlp(last_hidden)
        return last_hidden

    def _embed_action_chunk(
        self, action: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Project each action step to the RNN input width: ``[B, K, rnn_input]``."""
        device = next(self.parameters()).device
        action = action.to(device)
        if isinstance(self.config.action_space, gym.spaces.Discrete):
            action = (
                action.float() if action.dim() == 3 else action.unsqueeze(-1).float()
            )
        if action.dim() == 2:
            action = action.unsqueeze(1).expand(-1, seq_len, -1)

        chunk_len = action.size(1)
        action = self.action_mlp(action.reshape(batch_size * chunk_len, -1))
        return action.reshape(batch_size, chunk_len, -1)
