# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base critic interface and shared configuration fields."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import torch
from torch import nn

__all__ = ["BaseCriticConfig", "BaseCritic"]


@dataclass
class BaseCriticConfig(abc.ABC):
    """Shared configuration fields for all critic networks.

    Attributes:
        observation_space: Space of the (possibly dict) observations.
        action_space: Space of the actions scored by a Q-function.
        feature_extractor: Optional externally-built feature module.
        preprocess_obs_transform: Optional callables applied to observations.
        remove_obs_keys: Observation keys dropped before scoring.
        use_action: ``True`` builds a Q-function ``Q(s, a)``; ``False`` builds
            a state-value function ``V(s)``.
    """

    observation_space: Optional[gym.Space] = None
    action_space: Optional[gym.Space] = None
    feature_extractor: Optional[nn.Module] = None
    preprocess_obs_transform: Optional[Callable] = None
    remove_obs_keys: Optional[list[str]] = None
    use_action: bool = True

    @property
    def critic_class(self) -> type:
        """Concrete critic class this config instantiates."""
        return BaseCritic

    def create(self) -> "BaseCritic":
        """Instantiate the critic described by this config."""
        critic_class = self.critic_class
        if critic_class is None:
            raise ValueError(f"critic_class not defined for {self.__class__.__name__}")
        return critic_class(self)


class BaseCritic(nn.Module):
    """Common scaffolding for value networks (Q- and V-functions)."""

    def __init__(self, config: BaseCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.remove_obs_keys = config.remove_obs_keys
        self.preprocess_obs_transform: Optional[Callable] = None

    def forward(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score a batch of observations (and actions, for Q-functions)."""
        raise NotImplementedError("Subclasses must implement forward")

    def save(self, path: str) -> None:
        """Save the critic's ``state_dict`` to ``path``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load a ``state_dict`` saved by :meth:`save`."""
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def _remove_obs_keys(self, obs: Any) -> Any:
        """Return the observation without the configured keys.

        A filtered copy, not an in-place drop: the SAC core evaluates the
        critic on the actor's observation and then reads that same dict for
        reference-action regularization, so mutating it here would corrupt the
        downstream read.
        """
        if self.remove_obs_keys is None or not isinstance(obs, dict):
            return obs
        return {key: value for key, value in obs.items() if key not in self.remove_obs_keys}
