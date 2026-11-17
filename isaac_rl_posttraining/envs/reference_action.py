# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-conditioned direct-action environment wrapper."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch


__all__ = [
    "ReferenceActionEnv",
    "environment_actions_from_learner",
    "reference_observation_entries",
]

FEATURE_KEY = "vlm_features"
REFERENCE_KEY = "reference_action"
ACTION_MASK_KEY = "action_mask"


def environment_actions_from_learner(
    learner_action: np.ndarray,
    *,
    action_indices: tuple[int, ...],
    environment_action_defaults: np.ndarray,
) -> np.ndarray:
    """Place a learner chunk ``[..., A]`` into full environment commands ``[..., A_env]``.

    Dimensions the learner does not control keep their configured defaults.
    """
    learner_action = np.asarray(learner_action, dtype=np.float32)
    if learner_action.shape[-1] != len(action_indices):
        raise ValueError(
            f"expected learner action width {len(action_indices)}, got {learner_action.shape}"
        )
    environment_action = np.broadcast_to(
        environment_action_defaults,
        (*learner_action.shape[:-1], environment_action_defaults.shape[0]),
    ).copy()
    environment_action[..., action_indices] = learner_action
    return environment_action


def reference_observation_entries(
    features: torch.Tensor,
    reference: np.ndarray,
    state: np.ndarray,
    *,
    action_indices: tuple[int, ...],
    action_horizon: int,
    reference_action_transform: str,
    relative_action_scale: float,
) -> dict[str, np.ndarray]:
    """The learner-facing entries of one frozen decode for a batch-one observation.

    Returns the pooled backbone features, the reference chunk ``[1, H, A]`` in the
    learner's coordinates (the controlled dimensions of the raw chunk, made
    relative to the previous target when configured) and an all-ones action mask.
    """
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape[-1] <= max(action_indices):
        raise ValueError(
            "frozen policy reference width does not cover action_indices: "
            f"shape={reference.shape}, indices={action_indices}"
        )
    reference = reference[..., action_indices]
    if reference_action_transform == "absolute_to_relative":
        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] <= max(action_indices):
            raise ValueError(
                "state width does not cover action_indices: "
                f"shape={state.shape}, indices={action_indices}"
            )
        current = state[..., action_indices]
        previous = np.concatenate([current[:, None, :], reference[:, :-1, :]], axis=1)
        reference = np.clip((reference - previous) / relative_action_scale, -1.0, 1.0)
    expected = (1, action_horizon, len(action_indices))
    if reference.shape != expected:
        raise ValueError(
            f"frozen policy reference shape {reference.shape} does not match {expected}"
        )
    return {
        FEATURE_KEY: features.detach().cpu().numpy().astype(np.float32),
        REFERENCE_KEY: reference,
        ACTION_MASK_KEY: np.ones((1, action_horizon, 1), dtype=np.float32),
    }


def _controlled_action_indices(
    action_indices: tuple[int, ...] | list[int] | None,
    action_dim: int,
    environment_action_dim: int,
) -> tuple[int, ...]:
    """The environment action dimensions the learner controls, validated."""
    if action_indices is None:
        indices = tuple(range(action_dim))
    else:
        indices = tuple(int(index) for index in action_indices)
    if len(indices) != action_dim:
        raise ValueError(
            "action_indices length must equal action_dim, got "
            f"{len(indices)} and {action_dim}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("action_indices must be unique")
    if any(index < 0 or index >= environment_action_dim for index in indices):
        raise ValueError(
            "action_indices must index the environment action width "
            f"{environment_action_dim}, got {indices}"
        )
    return indices


def _environment_action_defaults(
    defaults: tuple[float, ...] | list[float] | None,
    action_dim: int,
    environment_action_dim: int,
) -> np.ndarray:
    """The values of the environment action dimensions the learner does not control."""
    if defaults is None:
        if environment_action_dim != action_dim:
            raise ValueError(
                "environment_action_defaults is required when the learner controls "
                "only a subset of the environment action"
            )
        return np.zeros(environment_action_dim, dtype=np.float32)
    array = np.asarray(defaults, dtype=np.float32)
    if array.shape != (environment_action_dim,):
        raise ValueError(
            "environment_action_defaults must match the environment action "
            f"width {environment_action_dim}, got "
            f"{array.shape}"
        )
    return array


class ReferenceActionEnv(gym.vector.VectorWrapper):
    """Attach frozen-policy features and proposals to a vector environment.

    The wrapped learner acts directly in bounded environment-command chunks.
    Raw observations remain present for video capture, but callers should use
    :attr:`learner_remove_obs_keys` to keep them out of actor and critic input.
    """

    FEATURE_KEY = FEATURE_KEY
    REFERENCE_KEY = REFERENCE_KEY
    ACTION_MASK_KEY = ACTION_MASK_KEY

    def __init__(
        self,
        env: gym.Env,
        base_model: Any,
        *,
        action_horizon: int,
        action_dim: int,
        action_bound: float = 1.0,
        action_indices: tuple[int, ...] | list[int] | None = None,
        environment_action_defaults: tuple[float, ...] | list[float] | None = None,
        include_sim_state: bool = False,
        state_key: str = "state",
        reference_action_transform: str = "identity",
        relative_action_scale: float = 0.025,
        reference_image_transform_mode: str = "eval",
    ) -> None:
        super().__init__(env)
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if action_bound <= 0 or not np.isfinite(action_bound):
            raise ValueError("action_bound must be finite and positive")
        self.base_model = base_model
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        base_action_space = getattr(env, "single_action_space", env.action_space)
        if not isinstance(base_action_space, gym.spaces.Box):
            raise TypeError("ReferenceActionEnv requires a Box environment action space")
        if base_action_space.shape is None or len(base_action_space.shape) != 1:
            raise ValueError("ReferenceActionEnv requires a one-dimensional environment action")
        self.environment_action_dim = int(base_action_space.shape[0])
        self.action_indices = _controlled_action_indices(
            action_indices, self.action_dim, self.environment_action_dim
        )
        self.environment_action_defaults = _environment_action_defaults(
            environment_action_defaults, self.action_dim, self.environment_action_dim
        )
        self.include_sim_state = bool(include_sim_state)
        self.state_key = str(state_key)
        if reference_action_transform not in {"identity", "absolute_to_relative"}:
            raise ValueError(
                "reference_action_transform must be 'identity' or 'absolute_to_relative'"
            )
        self.reference_action_transform = reference_action_transform
        self.relative_action_scale = float(relative_action_scale)
        if self.relative_action_scale <= 0.0:
            raise ValueError("relative_action_scale must be positive")
        if reference_image_transform_mode not in {"train", "eval"}:
            raise ValueError("reference_image_transform_mode must be train or eval")
        self.reference_image_transform_mode = reference_image_transform_mode

        single_obs = copy.deepcopy(getattr(env, "single_observation_space", env.observation_space))
        if not isinstance(single_obs, gym.spaces.Dict):
            raise TypeError("ReferenceActionEnv requires a Dict observation space")
        feature_dim = self._probe_feature_dim(single_obs)
        single_obs.spaces[self.FEATURE_KEY] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32,
        )
        single_obs.spaces[self.REFERENCE_KEY] = gym.spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(self.action_horizon, self.action_dim),
            dtype=np.float32,
        )
        single_obs.spaces[self.ACTION_MASK_KEY] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.action_horizon, 1),
            dtype=np.float32,
        )
        self.single_observation_space = single_obs
        num_envs = int(getattr(env, "num_envs", 1))
        self.observation_space = gym.vector.utils.batch_space(single_obs, num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(self.action_horizon, self.action_dim),
            dtype=np.float32,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

        image_keys = list(getattr(self._base_env(), "steering_remove_obs_keys", []))
        self.learner_remove_obs_keys = image_keys
        if not self.include_sim_state:
            self.learner_remove_obs_keys.append(self.state_key)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        environment_action = environment_actions_from_learner(
            action,
            action_indices=self.action_indices,
            environment_action_defaults=self.environment_action_defaults,
        )
        obs, reward, terminated, truncated, info = self.env.step(environment_action)
        return self._augment(obs), reward, terminated, truncated, info

    def sample_action(self) -> np.ndarray:
        """Return the native GR00T proposal cached in the current observation."""
        if not hasattr(self, "_latest_reference"):
            raise RuntimeError("reset the environment before requesting a reference action")
        return self._latest_reference.copy()

    def _augment(self, obs: dict[str, Any]) -> dict[str, Any]:
        features, reference = self.base_model.infer_reference(
            obs,
            image_transform_mode=self.reference_image_transform_mode,
        )
        entries = reference_observation_entries(
            features,
            reference,
            obs[self.state_key],
            action_indices=self.action_indices,
            action_horizon=self.action_horizon,
            reference_action_transform=self.reference_action_transform,
            relative_action_scale=self.relative_action_scale,
        )
        self._latest_reference = entries[self.REFERENCE_KEY]
        return {**obs, **entries}

    def _probe_feature_dim(self, observation_space: gym.spaces.Dict) -> int:
        del observation_space
        policy = getattr(self.base_model, "policy", None)
        model = getattr(policy, "model", None)
        feature_dim = getattr(model, "feat_dim", None)
        if feature_dim is None:
            raise ValueError("the frozen policy must expose model.feat_dim for pooled VLM features")
        return 2 * int(feature_dim)

    def _base_env(self) -> Any:
        current = self.env
        for _ in range(12):
            if hasattr(current, "envs") and current.envs:
                return current.envs[0]
            next_env = getattr(current, "env", None)
            if next_env is None:
                return current
            current = next_env
        return current
