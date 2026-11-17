# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for reference-conditioned direct-action SACEdit."""

from __future__ import annotations

from dataclasses import dataclass

from isaac_rl_posttraining.algorithms.sac import SACConfig


__all__ = ["SACEditConfig"]


@dataclass
class SACEditConfig(SACConfig):
    """SAC hyperparameters plus GR00T-reference conditioning for direct actions."""

    reference_action_key: str = "reference_action"
    reference_action_dropout: float = 0.0
    reference_action_l2_coef: float = 0.1
    normalize_actor_q_loss: bool = True
    alpha_weight_decay: float = 0.01
    learner_update_anchors: list[list[int]] | None = None
    learner_update_batch_size: int = 1
    use_adaptive_reward_scaler: bool = False
    # Symmetric value-support bound: the adaptive reward scaler keeps the
    # discounted return within [q_value_min, q_value_max] via its
    # max(|q_value_min|, |q_value_max|) support (only read when the scaler is on).
    q_value_min: float = -5.0
    q_value_max: float = 0.0
    reward_scaling_g_max: float | None = None
    reward_scaling_epsilon: float = 1.0e-8
    normalize_input: bool = False
    input_normalizer_key: str = "state"
    input_normalizer_dim: int = 0
    input_normalizer_clip: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.q_value_min >= self.q_value_max:
            raise ValueError(
                f"q_value_min must be less than q_value_max, got "
                f"{self.q_value_min} >= {self.q_value_max}"
            )
        if not 0.0 <= self.reference_action_dropout <= 1.0:
            raise ValueError("reference_action_dropout must be in [0, 1]")
        if self.reference_action_l2_coef < 0.0:
            raise ValueError("reference_action_l2_coef must be non-negative")
        if self.alpha_weight_decay < 0.0:
            raise ValueError("alpha_weight_decay must be non-negative")
        if self.learner_update_batch_size <= 0:
            raise ValueError("learner_update_batch_size must be positive")
        if self.reward_scaling_g_max is not None and self.reward_scaling_g_max <= 0.0:
            raise ValueError("reward_scaling_g_max must be positive when set")
        if self.reward_scaling_epsilon <= 0.0:
            raise ValueError("reward_scaling_epsilon must be positive")
        if self.normalize_input and self.input_normalizer_dim <= 0:
            raise ValueError("input_normalizer_dim must be positive when normalize_input=true")
        if self.input_normalizer_clip <= 0.0:
            raise ValueError("input_normalizer_clip must be positive")
        if self.learner_update_anchors is not None:
            _validate_learner_update_anchors(self.learner_update_anchors)

    @property
    def algorithm_class(self) -> type:
        from isaac_rl_posttraining.algorithms.sac_edit.sac_edit import SACEdit

        return SACEdit


def _validate_learner_update_anchors(anchors: list[list[int]]) -> None:
    """Anchors are ``[env_steps, updates]`` pairs, strictly increasing in env steps."""
    if len(anchors) < 2:
        raise ValueError("learner_update_anchors requires at least two points")
    previous_steps = -1
    previous_updates = -1
    for anchor in anchors:
        if len(anchor) != 2:
            raise ValueError(
                "each learner update anchor must contain [env_steps, updates]"
            )
        env_steps, updates = (int(value) for value in anchor)
        if env_steps <= previous_steps or updates < previous_updates:
            raise ValueError(
                "learner update anchors must increase by env step and "
                "must not decrease by update count"
            )
        previous_steps = env_steps
        previous_updates = updates
