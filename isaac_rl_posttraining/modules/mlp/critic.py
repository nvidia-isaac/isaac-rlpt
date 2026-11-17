# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP critic (Q- or V-function) for flattened observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.modules.base import BaseCritic, BaseCriticConfig
from isaac_rl_posttraining.modules.mlp.actor import (
    build_obs_featurizer,
    flat_obs_layout,
    flatten_plain_obs,
)
from isaac_rl_posttraining.utils.featurizers import ObservationFeaturizer
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["MLPCriticConfig", "MLPCritic"]


@dataclass
class MLPCriticConfig(BaseCriticConfig):
    """Configuration for :class:`MLPCritic`.

    Attributes:
        hidden_dims: Trunk layer widths.
        activation: Trunk activation name.
        use_layer_norm: LayerNorm after each trunk layer.
        dropout_rate: Dropout after each trunk activation.
        featurizer: Per-key featurizer spec for dict observations.
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

    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[Callable] = None

    image_encoder_type: Optional[str] = None
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_bounded_head: bool = False
    small_bottleneck_dim: int = 50
    impala_output_dim: Optional[int] = None

    @property
    def critic_class(self) -> type:
        return MLPCritic


class MLPCritic(BaseCritic):
    """Feed-forward value network over flattened observations.

    With ``config.use_action=True`` the action is concatenated to the
    flattened observation (a Q-function); otherwise the critic scores the
    observation alone (a V-function).
    """

    def __init__(self, config: MLPCriticConfig) -> None:
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform
        # Kept for interface parity with the transformer critic's pooled path.
        self.output_mlp: Optional[nn.Module] = None

        self.obs_featurizer: Optional[ObservationFeaturizer] = build_obs_featurizer(config)
        self._flatten_keys: Optional[list[str]]
        if self.obs_featurizer is not None:
            obs_dim = self.obs_featurizer.output_dim
            if obs_dim <= 0:
                raise ValueError("ObservationFeaturizer produced an invalid output dimension for the critic.")
            self._flatten_keys = None
        else:
            self._flatten_keys, obs_dim = flat_obs_layout(config.observation_space)

        if isinstance(config.action_space, gym.spaces.Discrete):
            action_dim = 1
        else:
            action_dim = int(np.prod(config.action_space.shape))

        input_dim = obs_dim + action_dim if config.use_action else obs_dim
        self.critic_trunk = build_mlp(
            int(input_dim),
            config.hidden_dims,
            config.activation,
            config.use_layer_norm,
            config.dropout_rate,
        )
        self.value_head = nn.Linear(int(config.hidden_dims[-1]), 1)

    def _flatten_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Flatten (and featurize) an observation batch to ``[B, D]``."""
        device = next(self.parameters()).device
        if self.obs_featurizer is not None:
            return self.obs_featurizer.flatten_obs(obs, device=device)

        return flatten_plain_obs(obs, self._flatten_keys, device)

    def _combine_obs_action(self, obs: Any, action: Optional[torch.Tensor]) -> torch.Tensor:
        """Preprocess, flatten, and (for Q-functions) concatenate the action."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        obs = self._remove_obs_keys(obs)
        obs = self._flatten_obs(obs)
        device = next(self.parameters()).device

        if not self.config.use_action:
            return obs
        if action is None:
            raise ValueError("Action must be provided when use_action=True")

        action = self._shape_action(action)

        # Broadcast singleton batches so obs/action pairs line up.
        if obs.size(0) != action.size(0):
            if obs.size(0) == 1:
                obs = obs.expand(action.size(0), -1)
            elif action.size(0) == 1:
                action = action.expand(obs.size(0), -1)

        return torch.cat([obs, action.to(device)], dim=1)

    def _shape_action(self, action: torch.Tensor) -> torch.Tensor:
        """Reshape an action batch to ``[B, A]`` (discrete indices become float ``[B, 1]``)."""
        if isinstance(self.config.action_space, gym.spaces.Discrete):
            if action.dim() == 1:
                return action.unsqueeze(-1).float()
            if action.dim() == 0:
                return action.unsqueeze(0).unsqueeze(-1).float()
            return action.float()
        if action.dim() > 2:
            return action.reshape(action.size(0), -1)
        if action.dim() == 1:
            return action.unsqueeze(0)
        return action

    def forward(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return values of shape ``[B, 1]``."""
        return self.value_head(self.compute_pooled(obs, action))

    def compute_pooled(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the trunk output feeding the value head (``[B, hidden]``).

        Mirrors ``TransformerCritic.compute_pooled`` so ensembles can share a
        trunk and apply per-member value heads.
        """
        combined = self._combine_obs_action(obs, action)
        pooled = self.critic_trunk(combined)
        if self.output_mlp is not None:
            pooled = self.output_mlp(pooled)
        return pooled
