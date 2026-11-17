# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-conditioned direct-action SACEdit."""

from __future__ import annotations

from typing import Any

from loguru import logger
import torch

from isaac_rl_posttraining.algorithms.constants import INPUT_NORMALIZER, REWARD_SCALER
from isaac_rl_posttraining.algorithms.sac_edit.config import SACEditConfig
from isaac_rl_posttraining.algorithms.sac_edit.input_normalizer import RunningMeanStdNormalizer
from isaac_rl_posttraining.algorithms.sac_edit.reward_scaler import AdaptiveRewardScaler
from isaac_rl_posttraining.algorithms.sac import SAC


__all__ = ["SACEdit"]


class SACEdit(SAC):
    """Direct chunk-action SAC regularized toward frozen GR00T proposals."""

    def __init__(self, config: SACEditConfig) -> None:
        edit_scale = getattr(getattr(config.actor, "config", None), "edit_scale", None)
        reference_centered = bool(
            getattr(
                getattr(config.actor, "config", None),
                "reference_centered_full_support",
                False,
            )
        )
        if (edit_scale is not None or reference_centered) and float(config.reference_action_dropout) > 0.0:
            # The reference is both the edit center and the actor's conditioning input.
            raise ValueError(
                "reference_action_dropout must be 0 in the bounded-edit and reference-centered "
                f"modes, got {config.reference_action_dropout}"
            )
        super().__init__(config)
        self.config = config
        if edit_scale is not None:
            logger.info(
                "SAC-edit bounded reference-edit mode: "
                f"edit_scale={float(edit_scale)}, "
                f"reference_action_l2_coef={float(config.reference_action_l2_coef)}; "
                "the L2 term is optional because edit geometry already bounds deviation"
            )
        elif reference_centered:
            logger.info(
                "SAC-edit full-support reference-centered mode: the actor predicts "
                "pre-tanh edits around the frozen action"
            )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            betas=tuple(config.actor_optimizer_betas),
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_optimizer_lr,
            betas=tuple(config.critic_optimizer_betas),
            eps=config.critic_optimizer_eps,
            weight_decay=config.critic_optimizer_weight_decay,
        )
        if self.log_ent_coef is not None:
            self.ent_coef_optimizer = torch.optim.AdamW(
                [self.log_ent_coef],
                lr=config.ent_coef_lr,
                weight_decay=config.alpha_weight_decay,
            )
        if config.reward_scaling_g_max is not None:
            support_g_max = float(config.reward_scaling_g_max)
        else:
            support_g_max = max(abs(config.q_value_min), abs(config.q_value_max))
        self.reward_scaler = (
            AdaptiveRewardScaler(
                gamma=config.gamma,
                support_g_max=support_g_max,
                epsilon=config.reward_scaling_epsilon,
                device=self.device,
            )
            if config.use_adaptive_reward_scaler
            else None
        )
        if self.reward_scaler is not None:
            self.component_names.append(REWARD_SCALER)
        self.input_normalizer = (
            RunningMeanStdNormalizer(
                config.input_normalizer_dim,
                clip=config.input_normalizer_clip,
                device=self.device,
            )
            if config.normalize_input
            else None
        )
        if self.input_normalizer is not None:
            self.component_names.append(INPUT_NORMALIZER)
            self._attach_input_normalizer()

    def _attach_input_normalizer(self) -> None:
        for module in [
            self.actor,
            *([self.actor_behavior_anchor] if hasattr(self, "actor_behavior_anchor") else []),
            *self.critic.critics,
            *self.critic_target.critics,
        ]:
            module.input_normalizer = self.input_normalizer
            module.input_normalizer_key = self.config.input_normalizer_key

    def _update_input_normalizer(self, observations: Any, next_observations: Any) -> None:
        if self.input_normalizer is None:
            return
        key = self.config.input_normalizer_key
        if not isinstance(observations, dict) or key not in observations:
            raise KeyError(f"SAC-edit observation is missing {key!r}")
        if not isinstance(next_observations, dict) or key not in next_observations:
            raise KeyError(f"SAC-edit next observation is missing {key!r}")
        self.input_normalizer.update(torch.cat([observations[key], next_observations[key]], dim=0))

    def observe_rewards(self, rewards: Any, dones: Any, truncateds: Any, episode_ids: Any) -> None:
        if self.reward_scaler is None:
            return
        self.reward_scaler.update(rewards, dones, truncateds, episode_ids)

    def _shape_training_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_scaler is None:
            return rewards
        return self.reward_scaler.normalize(rewards.detach()).to(dtype=rewards.dtype)

    def _reduce_action_log_prob(
        self,
        log_prob: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim == 3:
            return log_prob.sum(dim=1, keepdim=True)
        return log_prob.reshape(-1, 1)

    def _actor_q_scale(self, q_values: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_actor_q_loss:
            return super()._actor_q_scale(q_values)
        return torch.reciprocal(q_values.detach().abs().mean().clamp_min(1.0e-6))

    def target_update_count(self, env_steps: int) -> int | None:
        anchors = self.config.learner_update_anchors
        if anchors is None:
            return None
        steps = int(env_steps)
        lower = anchors[0]
        if steps <= int(lower[0]):
            return int(lower[1])
        for candidate in anchors[1:]:
            upper = candidate
            if steps <= int(candidate[0]):
                break
            lower = candidate
        else:
            lower = anchors[-2]
            upper = anchors[-1]
        span = int(upper[0]) - int(lower[0])
        fraction = (steps - int(lower[0])) / span
        interpolated = int(lower[1]) + fraction * (int(upper[1]) - int(lower[1]))
        batch = int(self.config.learner_update_batch_size)
        return max(0, int(interpolated) // batch * batch)

    def _actor_policy_observations(self, observations: Any) -> Any:
        if not isinstance(observations, dict):
            raise TypeError("SAC-edit requires Dict observations")
        key = self.config.reference_action_key
        if key not in observations:
            raise KeyError(f"SAC-edit observation is missing {key!r}")
        probability = float(self.config.reference_action_dropout)
        if probability <= 0.0:
            return observations
        actor_observations = dict(observations)
        reference = observations[key]
        if probability >= 1.0:
            actor_observations[key] = torch.zeros_like(reference)
            return actor_observations
        keep = (
            torch.rand(
                (reference.shape[0],) + (1,) * (reference.ndim - 1),
                device=reference.device,
            )
            >= probability
        )
        actor_observations[key] = reference * keep.to(reference.dtype)
        return actor_observations

    def _actor_regularization_loss(
        self,
        observations: Any,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reference = observations[self.config.reference_action_key]
        if reference.shape != actions.shape:
            reference = reference.reshape_as(actions)
        action_l2 = torch.square(actions - reference).mean()
        loss = float(self.config.reference_action_l2_coef) * action_l2
        return loss, {
            "reference_action_l2": float(action_l2.detach().item()),
            "reference_action_l2_loss": float(loss.detach().item()),
        }
