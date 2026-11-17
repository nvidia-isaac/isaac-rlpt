# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base actor interface: action normalization, act/forward split, config."""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

__all__ = ["BaseActorConfig", "BaseActor"]


@dataclass
class BaseActorConfig(abc.ABC):
    """Shared configuration fields for all actor networks.

    Attributes:
        observation_space: Space of the (possibly dict) observations.
        action_space: Space the actor emits actions in.
        feature_extractor: Optional externally-built feature module.
        preprocess_obs_transform: Optional callables applied to observations
            before the network sees them.
        min_action: Override for the lower action bound (with ``max_action``,
            enables normalization regardless of the space bounds).
        max_action: Override for the upper action bound.
        remove_obs_keys: Observation keys dropped before inference.
    """

    observation_space: Optional[gym.Space] = None
    action_space: Optional[gym.Space] = None
    feature_extractor: Optional[nn.Module] = None
    preprocess_obs_transform: Optional[Callable] = None
    min_action: Union[float, torch.Tensor, None] = None
    max_action: Union[float, torch.Tensor, None] = None
    remove_obs_keys: Optional[list[str]] = None

    @property
    def actor_class(self) -> type:
        """Concrete actor class this config instantiates."""
        return BaseActor

    def create(self) -> "BaseActor":
        """Instantiate the actor described by this config."""
        actor_class = self.actor_class
        if actor_class is None:
            raise ValueError(f"actor_class not defined for {self.__class__.__name__}")
        return actor_class(self)


class BaseActor(nn.Module):
    """Common behavior for policy networks (template-method pattern).

    Subclasses implement ``_forward`` (distribution parameters) and ``_act``
    (action selection); this base class owns the environment-facing
    :meth:`act` wrapper, action (un)normalization between the network's
    ``[-1, 1]`` range and the environment's bounds, and observation-key
    filtering.
    """

    def __init__(self, config: BaseActorConfig) -> None:
        """Set up action-space bookkeeping from the config.

        Args:
            config: Actor configuration; ``action_space`` must be ``Box`` or
                ``Discrete``.

        Raises:
            ValueError: For unsupported action-space types.
        """
        super().__init__()
        self.config = config
        self.action_space = config.action_space
        self.remove_obs_keys = config.remove_obs_keys
        self.preprocess_obs_transform: Optional[Callable] = None

        self.is_continuous = isinstance(self.action_space, gym.spaces.Box)
        self.is_discrete = isinstance(self.action_space, gym.spaces.Discrete)
        if not (self.is_continuous or self.is_discrete):
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")

        # Continuous actions with finite bounds are normalized to [-1, 1]
        # internally and mapped back to the environment range in act().
        self.normalize_actions = False
        if isinstance(self.action_space, gym.spaces.Box) and not (
            np.isinf(self.action_space.low).any() or np.isinf(self.action_space.high).any()
        ):
            self.action_low = torch.tensor(self.action_space.low, dtype=torch.float32)
            self.action_high = torch.tensor(self.action_space.high, dtype=torch.float32)
            self.normalize_actions = True

        if self.config.min_action is not None and self.config.max_action is not None:
            self.action_low = torch.tensor(self.config.min_action, dtype=torch.float32)
            self.action_high = torch.tensor(self.config.max_action, dtype=torch.float32)
            self.normalize_actions = True

        # Set by subclasses to a Distribution instance.
        self.action_dist: Any = None

    def _move_action_tensors_to_device(self, device: torch.device) -> None:
        """Keep the bound tensors co-located with the actions they scale."""
        if self.normalize_actions:
            self.action_low = self.action_low.to(device)
            self.action_high = self.action_high.to(device)

    def normalize_action(
        self, action: torch.Tensor, *, clamp: bool = True
    ) -> torch.Tensor:
        """Map an environment-range action to learner coordinates.

        ``clamp=False`` preserves exploratory actions outside the actor's
        reachable bounds, so unnormalizing a replayed value reconstructs the
        latent that actually executed.
        """
        if not self.normalize_actions:
            return action
        self._move_action_tensors_to_device(action.device)
        normalized = (action - self.action_low) / (self.action_high - self.action_low)
        normalized = normalized * 2.0 - 1.0
        if clamp:
            return torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Map a ``[-1, 1]`` action back into the environment range."""
        if not self.normalize_actions:
            return normalized_action
        self._move_action_tensors_to_device(normalized_action.device)
        return (normalized_action + 1) / 2 * (self.action_high - self.action_low) + self.action_low

    def forward(self, obs: Any, actor_state: Any = None) -> Any:
        """Delegate to the subclass ``_forward``, passing state only if accepted."""
        if "actor_state" not in inspect.signature(self._forward).parameters:
            return self._forward(obs)
        return self._forward(obs, actor_state=actor_state)

    def _forward(self, obs: Any, actor_state: Any = None) -> Any:
        """Compute action-distribution parameters (subclass hook)."""
        raise NotImplementedError("Subclasses must implement _forward")

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = False, actor_state: Any = None) -> tuple[torch.Tensor, Any]:
        """Produce environment-range actions for rollout/evaluation.

        Runs in eval mode and without gradients; the module's training mode
        is restored afterwards.

        Args:
            obs: Observation batch (dict or tensor).
            deterministic: Take the distribution mode instead of sampling.
            actor_state: Recurrent state for stateful actors (``None`` for
                stateless ones).

        Returns:
            Tuple of ``(actions, new_actor_state)``.
        """
        was_training = self.training
        if was_training:
            self.eval()
        obs = self._remove_obs_keys(obs)

        if "actor_state" not in inspect.signature(self._act).parameters:
            action = self._act(obs, deterministic=deterministic)
            actor_state = None
        else:
            action, actor_state = self._act(obs, deterministic=deterministic, actor_state=actor_state)

        if self.normalize_actions:
            action = self.unnormalize_action(action)

        if was_training:
            self.train()
        return action, actor_state

    def get_action_dist_params(self, obs: Any) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Return ``(mean, log_std, extra_kwargs)`` for the action distribution."""
        raise NotImplementedError("Subclasses must implement get_action_dist_params")

    def action_log_prob(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with log-probabilities (for actor-critic updates).

        Actions are returned in the network's internal ``[-1, 1]`` range.

        Returns:
            Tuple ``(actions, log_prob)``.
        """
        if self.action_dist is None:
            raise ValueError("action_dist not initialized by subclass")

        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        if log_std is not None:
            return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        return self.action_dist.log_prob_from_params(mean_actions, **kwargs)

    def get_initial_state(self) -> Any:
        """Return the initial recurrent state (``None`` for stateless actors)."""
        return None

    def save(self, path: str) -> None:
        """Save the actor's ``state_dict`` to ``path``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load a ``state_dict`` saved by :meth:`save`."""
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def _remove_obs_keys(self, obs: Any) -> Any:
        """Return the observation without the configured keys.

        A filtered copy, not an in-place drop: the caller's dict is reused
        after the forward (e.g. the SAC core evaluates the critic on the same
        observation it later reads for reference-action regularization), so
        mutating it here would corrupt those downstream reads.
        """
        if self.remove_obs_keys is None or not isinstance(obs, dict):
            return obs
        return {key: value for key, value in obs.items() if key not in self.remove_obs_keys}
