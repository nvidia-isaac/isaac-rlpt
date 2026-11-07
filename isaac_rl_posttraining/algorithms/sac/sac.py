# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REDQ-style soft actor-critic core.

Implements SAC with the ensemble tricks that make it stable for
latent-steering post-training: a randomized critic subset for TD targets, a
shared critic backbone with per-member value heads, entropy auto-tuning, and
optional prior/snapshot KL anchors on the actor. The algorithm is agnostic
to what the action *is* — for DSRL (:mod:`isaac_rl_posttraining.algorithms.dsrl`)
the "action" is a steering latent decoded by a base model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
import copy
import math
import time
from typing import Any, NamedTuple, Optional, Union

from loguru import logger
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from isaac_rl_posttraining.algorithms.base import BaseAlgorithm, LearnerModule, LearnerOptimizer
from isaac_rl_posttraining.algorithms.constants import (
    ACTOR,
    ACTOR_BEHAVIOR_ANCHOR,
    ACTOR_BEHAVIOR_ANCHOR_CAPTURED,
    ACTOR_OPTIMIZER,
    CRITIC,
    CRITIC_OPTIMIZER,
    CRITIC_TARGET,
    ENT_COEF_OPTIMIZER,
    LOG_ENT_COEF,
)
from isaac_rl_posttraining.algorithms.replay_learner import pack_batches
from isaac_rl_posttraining.algorithms.sac.config import SACConfig
from isaac_rl_posttraining.algorithms.sac.diagnostics import (
    diagonal_gaussian_prior_kl,
    gaussian_prior_diagnostics,
    record_actor_diagnostics,
    record_latent_diagnostics,
    summarize_learning_diagnostics,
)
from isaac_rl_posttraining.algorithms.sac.train_stats import TrainStepStats, latent_noise_metrics
from isaac_rl_posttraining.buffers import Batch, Sampler, UniformSampler, to_tensors
from isaac_rl_posttraining.modules.augmentations import PixelAugmentation
from isaac_rl_posttraining.modules.base.distributions import kl_divergence
from isaac_rl_posttraining.runtimes.learner import LearnerBatch
from isaac_rl_posttraining.utils.networks import CriticEnsemble, TargetUpdater, frozen_parameters, polyak_update
from isaac_rl_posttraining.utils.torch_utils import apply_torch_perf_knobs, resolve_amp_dtype


__all__ = ["SAC"]


class _ActorForward(NamedTuple):
    """Actor distribution parameters and one sample on the actor slice."""

    mean_actions: torch.Tensor
    log_std: torch.Tensor
    dist_kwargs: dict[str, Any]
    actions_pi: torch.Tensor
    log_prob: torch.Tensor


class SAC(BaseAlgorithm):
    """Soft actor-critic with a randomized Q-ensemble.

    Each :meth:`train_step` draws one oversized batch, splits it into
    ``num_critic_updates_per_actor_update`` slices for critic updates, and
    performs one actor (plus temperature) update on the first slice.
    """

    training_state_tensors = (LOG_ENT_COEF,)
    controller_attributes = (*BaseAlgorithm.controller_attributes, "expert_sampler", "success_sampler")

    def __init__(self, config: SACConfig) -> None:
        """Build networks, target networks, and optimizers from the config.

        Args:
            config: Populated :class:`SACConfig` with ``actor``, ``critic``,
                ``buffer`` and ``sampler`` wired in.
        """
        super().__init__(config)
        apply_torch_perf_knobs()
        self.config = config
        self.actor = config.actor
        self.device = next(self.actor.parameters()).device
        self.amp_dtype = resolve_amp_dtype() if config.use_amp else None
        if torch.cuda.is_available():
            logger.info(f"SAC autocast dtype: {self.amp_dtype or torch.float32}")

        if self.config.train_actor_with_kl_divergence:
            self.update_old_actor(self.actor)

        self.component_names = [
            ACTOR,
            CRITIC,
            CRITIC_TARGET,
            ACTOR_OPTIMIZER,
            CRITIC_OPTIMIZER,
            ENT_COEF_OPTIMIZER,
            LOG_ENT_COEF,
        ]
        self._setup_behavior_anchor(config)
        self.pooled_critic_features = config.pooled_critic_features
        if config.critic_target_reduction not in ("mean", "min"):
            raise ValueError(
                f"critic_target_reduction must be 'mean' or 'min', got {config.critic_target_reduction!r}"
            )

        self.critic = self._create_critic_ensemble(
            config.critic, config.num_critics, share_backbone=self.pooled_critic_features
        )
        self.buffer = config.buffer
        self.sampler: Sampler | None = config.sampler
        self._setup_behavior_cloning(config)
        self.latent_prior_std = float(config.latent_prior_std)
        if not math.isfinite(self.latent_prior_std) or self.latent_prior_std <= 0.0:
            raise ValueError("latent_prior_std must be finite and positive")

        self.batch_size = config.batch_size
        self.pixel_augmentation = (
            PixelAugmentation(color_jitter=config.augment_color_jitter) if config.augment_pixels else None
        )
        self.gamma = config.gamma
        self.compute_chunked_gamma = config.compute_chunked_gamma
        self.tau = config.tau
        self.target_update_interval = config.target_update_interval
        self.learning_starts = config.learning_starts
        self.actor_learning_starts = (
            config.learning_starts
            if config.actor_learning_starts is None
            else config.actor_learning_starts
        )
        self.actor_q_loss_starts = (
            self.actor_learning_starts
            if config.actor_q_loss_starts is None
            else config.actor_q_loss_starts
        )
        self.actor_learning_stops = config.actor_learning_stops
        self.num_critics = config.num_critics
        self.critic_reduction = config.critic_reduction

        self._setup_entropy_coefficient(config)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            betas=tuple(config.actor_optimizer_betas),
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        critic_params = list(self.critic.parameters())
        logger.info(f"Critic ensemble has {len(critic_params)} parameters (deduplicated)")
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=config.critic_optimizer_lr,
            betas=config.critic_optimizer_betas,
            eps=config.critic_optimizer_eps,
            weight_decay=config.critic_optimizer_weight_decay,
        )

        if self.log_ent_coef is not None:
            self.ent_coef_optimizer: Optional[torch.optim.Adam] = torch.optim.Adam(
                [self.log_ent_coef], lr=config.ent_coef_lr, weight_decay=0.0
            )
        else:
            self.ent_coef_optimizer = None

        # Independent target ensemble, hard-copied from the online ensemble.
        self.critic_target = self._create_critic_ensemble(
            copy.deepcopy(config.critic),
            config.num_critics,
            share_backbone=self.pooled_critic_features,
        )
        for param in self.critic_target.parameters():
            param.requires_grad = False
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), tau=1.0)
        self.critic_target.eval()
        self._target_updater = TargetUpdater()

        self._n_updates = 0

    def _setup_behavior_anchor(self, config: SACConfig) -> None:
        """Freeze a copy of the actor as the behavior-KL anchor when that term is on."""
        self.actor_behavior_kl_coef = float(config.actor_behavior_kl_coef)
        self.actor_behavior_anchor_step = config.actor_behavior_anchor_step
        self.actor_behavior_kl_starts = (
            config.actor_behavior_anchor_step
            if config.actor_behavior_kl_starts is None
            else config.actor_behavior_kl_starts
        )
        if self.actor_behavior_kl_coef > 0.0:
            self.actor_behavior_anchor = copy.deepcopy(self.actor).to(self.device)
            self.actor_behavior_anchor.eval()
            for parameter in self.actor_behavior_anchor.parameters():
                parameter.requires_grad = False
            self.actor_behavior_anchor_captured = torch.tensor(False, device=self.device)
            self.component_names.extend([ACTOR_BEHAVIOR_ANCHOR, ACTOR_BEHAVIOR_ANCHOR_CAPTURED])

    def _setup_behavior_cloning(self, config: SACConfig) -> None:
        """Read and check the expert and online-success behavior-cloning settings."""
        self.expert_bc_coef = float(config.expert_bc_coef)
        if self.expert_bc_coef < 0.0:
            raise ValueError("expert_bc_coef must be non-negative")
        self.expert_bc_loss_type = str(config.expert_bc_loss_type)
        if self.expert_bc_loss_type not in ("nll", "mse"):
            raise ValueError("expert_bc_loss_type must be 'nll' or 'mse'")
        # The behavior-cloning terms draw from demonstrations only and from
        # the successful episodes of the robot's own replay.
        self.expert_sampler = None if config.demonstrations is None else UniformSampler(config.demonstrations)
        if self.expert_bc_coef > 0.0 and self.expert_sampler is None:
            raise ValueError("expert_bc_coef requires demonstrations (buffer.demonstrations)")
        self.online_success_bc_coef = float(config.online_success_bc_coef)
        if self.online_success_bc_coef < 0.0:
            raise ValueError("online_success_bc_coef must be non-negative")
        self.online_success_bc_loss_type = str(config.online_success_bc_loss_type)
        if self.online_success_bc_loss_type not in ("nll", "mse"):
            raise ValueError("online_success_bc_loss_type must be 'nll' or 'mse'")
        self.success_sampler = None if self.buffer is None else UniformSampler(self.buffer, outcome="success")
        if self.online_success_bc_coef > 0.0 and self.success_sampler is None:
            raise ValueError("online_success_bc_coef requires the replay buffer of the robot's own episodes")

    @staticmethod
    def _create_critic_ensemble(
        base_critic: nn.Module, num_critics: int, share_backbone: bool = False
    ) -> CriticEnsemble:
        """Clone ``base_critic`` into an ensemble with re-initialized heads.

        Args:
            base_critic: Prototype critic.
            num_critics: Ensemble size.
            share_backbone: When set, heavy feature modules (transformer
                encoder / featurizers / trunk) are shared by reference across
                members, so each member differs only in its value head. The
                deduplicating :class:`CriticEnsemble` keeps the shared
                parameters from receiving multiple gradient updates.

        Returns:
            The assembled :class:`CriticEnsemble`.
        """

        def reset_parameters(module: nn.Module) -> None:
            if hasattr(module, "reinitialize_parameters"):
                # Modules with a bespoke init scheme (e.g. the conv encoders'
                # orthogonal/xavier weights) re-run it; the child-by-child
                # walk would silently replace it with PyTorch defaults.
                module.reinitialize_parameters()
                return
            for child in module.children():
                reset_parameters(child)
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        critics = []
        for _ in range(num_critics):
            critic_copy = copy.deepcopy(base_critic)
            reset_parameters(critic_copy)
            if share_backbone:
                # Transformer critics share the sequence trunk.
                if hasattr(base_critic, "obs_feature_extractor"):
                    critic_copy.obs_feature_extractor = base_critic.obs_feature_extractor
                    critic_copy.position_embedding = base_critic.position_embedding
                    critic_copy.transformer_encoder = base_critic.transformer_encoder
                    critic_copy.action_embedding = base_critic.action_embedding
                    critic_copy.action_projection = getattr(base_critic, "action_projection", None)
                    critic_copy.obs_projection = getattr(base_critic, "obs_projection", None)
                    critic_copy.input_norm = getattr(base_critic, "input_norm", None)
                # MLP critics share the featurizer and trunk.
                if hasattr(base_critic, "obs_featurizer"):
                    critic_copy.obs_featurizer = base_critic.obs_featurizer
                    critic_copy.output_mlp = base_critic.output_mlp
            critics.append(critic_copy)
        return CriticEnsemble(critics)

    @staticmethod
    def _slice_batch_data(
        data: Union[dict[str, torch.Tensor], torch.Tensor], indices: Union[np.ndarray, torch.Tensor]
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        """Index a (possibly dict) batch with numpy or tensor indices."""

        def _index(value: torch.Tensor) -> torch.Tensor:
            if isinstance(indices, np.ndarray):
                idx = torch.from_numpy(indices).long().to(value.device)
            else:
                idx = indices.to(value.device).long()
            return value[idx]

        if isinstance(data, dict):
            return {key: _index(value) for key, value in data.items()}
        return _index(data)

    @staticmethod
    def _behavior_cloning_action_mask(
        action_mask: Any,
        actions: torch.Tensor,
    ) -> torch.Tensor | None:
        """Broadcast a replay chunk mask to one weight per action scalar."""
        if action_mask is None:
            return None
        mask = torch.as_tensor(action_mask, device=actions.device)
        while mask.ndim < actions.ndim:
            mask = mask.unsqueeze(-1)
        try:
            mask = torch.broadcast_to(mask, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                "behavior cloning action mask cannot broadcast to actions: "
                f"{tuple(mask.shape)} versus {tuple(actions.shape)}"
            ) from exc
        if not torch.isfinite(mask.float()).all():
            raise ValueError("behavior cloning action mask must be finite")
        return mask > 0

    @staticmethod
    def _record_reward_model_diagnostics(batch: dict[str, Any], stats: TrainStepStats) -> None:
        """Collect reward-model relabeling diagnostics, when present in infos."""
        batch_info = batch.get("info")
        if isinstance(batch_info, (list, tuple)):
            for info in batch_info:
                if isinstance(info, dict):
                    if info.get("relabeled_reward") is not None:
                        stats.progress_rewards.append(float(info["relabeled_reward"]))
                    if info.get("success_prob") is not None:
                        stats.success_probs.append(float(info["success_prob"]))

    def _actor_action_dim(self) -> int:
        """Dimension of the actor's action distribution (the width the entropy target is set from)."""
        dist = getattr(self.actor, "action_dist", None)
        dim = getattr(dist, "action_dim", None)
        if dim is None:
            raise ValueError(
                "target_entropy='auto' needs an actor whose action_dist exposes action_dim; "
                "set algorithm.target_entropy explicitly for this actor"
            )
        return int(dim)

    def _setup_entropy_coefficient(self, config: SACConfig) -> None:
        """Initialize the (fixed or learned) entropy temperature."""
        if config.ent_coef_parameterization not in ("log", "softplus"):
            raise ValueError(
                f"ent_coef_parameterization must be 'log' or 'softplus', got "
                f"{config.ent_coef_parameterization!r}"
            )
        if (
            config.ent_coef_parameterization == "softplus"
            and config.temperature_loss_type != "alpha"
        ):
            raise ValueError(
                "ent_coef_parameterization='softplus' pairs with the linear-in-alpha "
                "temperature loss; set temperature_loss_type='alpha' "
                f"(got {config.temperature_loss_type!r})."
            )
        if config.ent_coef == "auto":
            # Both parameterizations start at alpha = 1: log-form raw is
            # log(1) = 0, softplus-form raw is softplus^-1(1) = log(e - 1).
            # ``log_ent_coef`` holds the RAW parameter; its mapping to alpha
            # is _ent_coef_from_raw (the name keeps checkpoint compatibility).
            if config.ent_coef_parameterization == "softplus":
                raw_init = float(np.log(np.expm1(1.0)))
            else:
                raw_init = 0.0
            self.log_ent_coef: Optional[torch.Tensor] = torch.tensor(
                raw_init, device=self.device, requires_grad=True
            )
            if config.target_entropy == "auto" or config.target_entropy is None:
                # -dim(action) of the actor's own distribution, so the target does not
                # depend on an environment object being attached to the config.
                self.target_entropy = -float(self._actor_action_dim())
            else:
                self.target_entropy = float(config.target_entropy)
            logger.info(
                f"SAC: auto-tuned entropy coefficient, target entropy = {self.target_entropy}"
            )
        else:
            self.log_ent_coef = None
            self.target_entropy = None
            self.ent_coef_tensor = torch.tensor(float(config.ent_coef), device=self.device)
            logger.info(f"SAC: fixed entropy coefficient = {self.ent_coef_tensor.item()}")

    def _ent_coef_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Map the raw temperature parameter to alpha per the parameterization."""
        if self.config.ent_coef_parameterization == "softplus":
            return torch.nn.functional.softplus(raw)
        return torch.exp(raw)

    def _autocast(self) -> Any:
        """Autocast context (or a no-op when AMP is off)."""
        if self.amp_dtype:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _clip_gradients(self, module: torch.nn.Module, per_network_clip: Optional[float]) -> None:
        """Clip a module's gradients by its per-network norm, else clip_grad_norm."""
        clip = per_network_clip if per_network_clip is not None else self.config.clip_grad_norm
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=clip)

    def _critic_outputs(
        self,
        ensemble: Any,
        observations: Any,
        actions: torch.Tensor,
        *,
        critic_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate critic heads while preserving their output dimension."""
        critics = ensemble.critics
        if critic_indices is not None:
            critics = [critics[int(index)] for index in critic_indices]
        if self.pooled_critic_features:
            pooled = ensemble.critics[0].compute_pooled(observations, actions)
            outputs = [critic.value_head(pooled) for critic in critics]
        else:
            outputs = ensemble(
                observations,
                actions,
                critic_indices=critic_indices,
            )
        return torch.stack(outputs, dim=1)

    def _critic_values_from_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert scalar critic outputs to one value per ensemble member."""
        if outputs.shape[-1] != 1:
            raise ValueError(f"Scalar SAC critics must output one value, got {outputs.shape[-1]}")
        return outputs.squeeze(-1)

    def _critic_td_target(
        self,
        *,
        next_outputs: torch.Tensor,
        next_log_prob: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float,
        ent_coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build scalar SAC targets and values used for metrics."""
        next_values = self._critic_values_from_outputs(next_outputs)
        if self.config.critic_target_reduction == "min":
            reduced_next_values = next_values.min(dim=1, keepdim=True).values
        elif self.config.critic_target_reduction == "mean":
            reduced_next_values = next_values.mean(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Invalid critic target reduction: {self.config.critic_target_reduction}"
            )
        if self.config.train_critic_with_entropy:
            reduced_next_values = reduced_next_values - ent_coef * next_log_prob
        target_values = rewards + (1.0 - terminals) * gamma * reduced_next_values
        if self.config.td_target_clip is not None:
            low, high = self.config.td_target_clip
            target_values = target_values.clamp(min=low, max=high)
        return target_values, reduced_next_values

    def _critic_training_loss(
        self,
        current_outputs: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scalar critic loss and current values for metrics."""
        current_values = self._critic_values_from_outputs(current_outputs).float()
        expanded_targets = target_values.float().expand_as(current_values)
        return F.mse_loss(current_values, expanded_targets), current_values

    def _after_critic_optimizer_step(self) -> None:
        """Allow algorithm variants to enforce critic parameter constraints."""

    def _should_update_actor(self, gradient_update_step: int) -> bool:
        """Return whether this gradient step includes actor and temperature updates."""
        del gradient_update_step
        return True

    def _actor_policy_observations(self, observations: Any) -> Any:
        """Return the actor-conditioning view used for an optimization step."""
        return observations

    def _actor_regularization_loss(
        self,
        observations: Any,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return an optional additive actor loss and its diagnostics."""
        del observations
        return actions.new_zeros(()), {}

    def _actor_q_scale(self, q_values: torch.Tensor) -> torch.Tensor:
        """Return the detached multiplier applied only to the actor's Q term."""
        return q_values.new_ones(())

    def _effective_actor_q_loss_coef(self, rollout_step: Optional[int]) -> float:
        """Return the configured actor Q weight at this environment step."""
        boundary = self.config.actor_q_loss_coef_after_step
        if boundary is not None and rollout_step is not None and rollout_step > boundary:
            value = self.config.actor_q_loss_coef_after_step_value
            assert value is not None
            return float(value)
        return float(self.config.actor_q_loss_coef)

    def _shape_training_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply an optional algorithm-specific reward transform."""
        return rewards

    def _capture_actor_behavior_anchor(self, rollout_step: Optional[int]) -> None:
        """Freeze the actor at the configured BC-to-RL handoff."""
        if self.actor_behavior_kl_coef <= 0.0 or rollout_step is None:
            return
        assert self.actor_behavior_anchor_step is not None
        if rollout_step < self.actor_behavior_anchor_step:
            return
        if bool(self.actor_behavior_anchor_captured.item()):
            return
        self.actor_behavior_anchor.load_state_dict(self.actor.state_dict())
        self.actor_behavior_anchor.eval()
        for parameter in self.actor_behavior_anchor.parameters():
            parameter.requires_grad = False
        self.actor_behavior_anchor_captured.fill_(True)
        logger.info(
            "Captured fixed actor behavior anchor at rollout step "
            f"{rollout_step} (configured step {self.actor_behavior_anchor_step})"
        )

    def begin_actor_rl_phase(
        self,
        rollout_step: int,
        *,
        actor_q_loss_starts: int | None = None,
        actor_behavior_kl_starts: int | None = None,
    ) -> None:
        """Freeze the current BC actor and enable its RL objective."""
        if rollout_step < 0:
            raise ValueError("rollout_step must be non-negative")
        if not hasattr(self, "actor_behavior_anchor"):
            self.actor_behavior_anchor = copy.deepcopy(self.actor).to(self.device)
            self.actor_behavior_anchor_captured = torch.tensor(
                False, device=self.device
            )
            self.component_names.extend([ACTOR_BEHAVIOR_ANCHOR, ACTOR_BEHAVIOR_ANCHOR_CAPTURED])
        self.actor_behavior_anchor.load_state_dict(self.actor.state_dict())
        self.actor_behavior_anchor.eval()
        for parameter in self.actor_behavior_anchor.parameters():
            parameter.requires_grad_(False)
        self.actor_behavior_anchor_captured.fill_(True)
        # Collection ends on episode boundaries, so it can pass a configured step
        # boundary by one trajectory. Keep both loss schedules causal at handoff.
        self.actor_q_loss_starts = max(
            rollout_step,
            rollout_step if actor_q_loss_starts is None else int(actor_q_loss_starts),
        )
        self.actor_behavior_kl_starts = max(
            rollout_step,
            rollout_step if actor_behavior_kl_starts is None else int(actor_behavior_kl_starts),
        )
        if self.config.reset_actor_optimizer_at_rl_handoff:
            self.actor_optimizer.state.clear()
            logger.info("Cleared actor optimizer state at BC-to-RL handoff")
        logger.info(f"Started anchored actor RL phase at rollout step {rollout_step}")

    def _update_input_normalizer(self, observations: Any, next_observations: Any) -> None:
        """Update optional algorithm-specific observation statistics."""
        del observations, next_observations

    def _reduce_action_log_prob(
        self,
        log_prob: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce structured-action log probability to one value per sample."""
        if actions.ndim == 3:
            return log_prob.mean(dim=1, keepdim=True)
        return log_prob.reshape(-1, 1)

    def _draw(self, sampler: Sampler | None, batch_size: int, source: str) -> Batch:
        if sampler is None:
            raise RuntimeError(f"SAC has no {source!r} sampler to draw from")
        batch = sampler.sample(batch_size)
        if not batch:
            raise RuntimeError(f"the {source!r} sampler produced no batch for {batch_size} samples")
        return batch

    def learner_modules(self) -> dict[str, LearnerModule]:
        """The actor (published), the critic ensemble and its target."""
        return {
            ACTOR: LearnerModule(self.actor, publish=True),
            CRITIC: LearnerModule(self.critic),
            CRITIC_TARGET: LearnerModule(self.critic_target),
        }

    def learner_optimizers(self) -> dict[str, LearnerOptimizer]:
        return {
            ACTOR_OPTIMIZER: LearnerOptimizer(self.actor_optimizer, ACTOR),
            CRITIC_OPTIMIZER: LearnerOptimizer(self.critic_optimizer, CRITIC),
        }

    @property
    def respects_step_schedules(self) -> bool:
        return bool(self.config.respect_step_schedules)

    def learner_batch(self) -> LearnerBatch:
        """One oversized replay draw per gradient step, plus the behavior-cloning draws in use.

        The success draw is left out while replay holds no successful episode,
        which is how the learner knows to skip that term for the step.
        """
        steps: list[dict[str, Batch]] = []
        for _ in range(self.config.num_updates_per_train_step):
            draws = {
                "replay": self._draw(
                    self.sampler, self.batch_size * self.config.num_critic_updates_per_actor_update, "replay"
                )
            }
            if self.expert_bc_coef > 0.0:
                draws["expert"] = self._draw(self.expert_sampler, self.batch_size, "expert")
            if self.online_success_bc_coef > 0.0 and self.success_sampler.can_sample(self.batch_size):
                draws["online_success"] = self.success_sampler.sample(self.batch_size)
            steps.append(draws)
        return pack_batches(steps)

    def train_step(
        self,
        rollout_step: int = None,
        batches: Sequence[Mapping[str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Run ``num_updates_per_train_step`` gradient steps.

        Each gradient step draws one oversized batch, runs one critic update
        per slice, then (while the actor window is open) one temperature and
        one actor update on the first slice.

        Args:
            rollout_step: Current environment step; below
                ``learning_starts`` the call is a no-op.
            batches: Pre-drawn batches, one mapping per gradient step keyed by
                draw (``"replay"`` plus ``"expert"`` / ``"online_success"``
                when those terms are on), as :meth:`learner_batch` packs
                them. When given, replay is never sampled here.

        Returns:
            Averaged training metrics (empty when no update ran).
        """
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}
        gradient_steps = self.config.num_updates_per_train_step
        if batches is not None and len(batches) != gradient_steps:
            raise ValueError(
                f"train_step expects {gradient_steps} pre-drawn batches, got {len(batches)}"
            )

        self._capture_actor_behavior_anchor(rollout_step)

        t_total_start = time.perf_counter()
        stats = TrainStepStats()
        num_chunks = self.config.num_critic_updates_per_actor_update
        # Stable global counter so target-update and KL cadences survive
        # across train_step calls.
        base_update_step = int(self._n_updates)

        for gradient_step in range(gradient_steps):
            ent_coef = self._current_ent_coef()
            supplied = None if batches is None else batches[gradient_step]
            batch = self._draw_update_batch(supplied)
            if batch is None:
                return {}

            # One oversized draw, split into per-critic-update slices.
            shuffled_indices = torch.as_tensor(
                np.random.permutation(batch["reward"].shape[0]),  # NOSONAR - draws from the globally seeded NumPy stream
                device=batch["reward"].device,
            )
            per_chunk = max(1, int(batch["reward"].shape[0] // max(1, num_chunks)))

            # Actor forward once, on the first slice (reused for the actor
            # and temperature losses after the critic loop).
            actor_observations = self._slice_batch_data(batch["obs"], shuffled_indices[:per_chunk])
            actor_replay_actions = self._slice_batch_data(batch["action"], shuffled_indices[:per_chunk])
            actor_policy_observations = self._actor_policy_observations(actor_observations)
            forward = self._actor_forward(actor_policy_observations)

            update_step = self._critic_updates(
                batch,
                shuffled_indices,
                per_chunk=per_chunk,
                first_update_step=base_update_step + gradient_step * num_chunks,
                ent_coef=ent_coef,
                stats=stats,
            )

            if not self._actor_window_open(rollout_step) or not self._should_update_actor(
                base_update_step + gradient_step
            ):
                continue

            log_prob = self._reduce_action_log_prob(forward.log_prob, forward.actions_pi).float()
            ent_coef = self._temperature_step(log_prob, stats)
            self._record_reward_model_diagnostics(batch, stats)
            # The actor loss needs dQ/da only; the critic's own weight gradients would be discarded.
            with frozen_parameters(self.critic):
                self._actor_update(
                    forward,
                    log_prob,
                    ent_coef=ent_coef,
                    actor_observations=actor_observations,
                    actor_policy_observations=actor_policy_observations,
                    rollout_step=rollout_step,
                    update_step=update_step,
                    stats=stats,
                    supplied_batches=supplied,
                    actor_replay_actions=actor_replay_actions,
                    actor_update_index=base_update_step + gradient_step,  # update_step counts critic slices
                )

        self._n_updates += gradient_steps
        self.step_counter = self._n_updates

        metrics = self._assemble_metrics(stats)
        metrics["train_step_total_time_s"] = float(time.perf_counter() - t_total_start)
        return metrics

    def _critic_updates(
        self,
        batch: dict[str, Any],
        shuffled_indices: torch.Tensor,
        *,
        per_chunk: int,
        first_update_step: int,
        ent_coef: torch.Tensor,
        stats: TrainStepStats,
    ) -> int:
        """One critic update per slice of the oversized batch; return the last update step."""
        num_chunks = self.config.num_critic_updates_per_actor_update
        for critic_update in range(num_chunks):
            update_step = first_update_step + critic_update
            start = critic_update * per_chunk
            end = min(start + per_chunk, shuffled_indices.shape[0])
            if start >= end:
                start = max(0, shuffled_indices.shape[0] - per_chunk)
                end = shuffled_indices.shape[0]
            critic_loss, current_q_values, next_q_values = self._critic_update(
                batch,
                shuffled_indices[start:end],
                ent_coef=ent_coef,
                update_step=update_step,
                stats=stats,
            )
            stats.critic_losses.append(critic_loss)

        stats.reward_values.append(batch["reward"].mean())
        stats.q_values.append(current_q_values.detach().mean())
        stats.q_next_values.append(next_q_values.detach().mean())
        return update_step

    def _draw_update_batch(self, supplied: Mapping[str, dict[str, Any]] | None) -> dict[str, Any] | None:
        """This gradient step's replay batch: the pre-drawn one if supplied, else a fresh sample."""
        if supplied is None:
            return self._sample_update_batch()
        return self._prepare_update_batch(supplied["replay"])

    def _temperature_step(self, log_prob: torch.Tensor, stats: TrainStepStats) -> torch.Tensor:
        """Update the entropy temperature and record it; return the coefficient for the actor loss."""
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(log_prob)
        if ent_coef_loss is not None:
            stats.ent_coef_losses.append(ent_coef_loss)
        stats.ent_coefs.append(ent_coef.item())
        return ent_coef

    def _actor_window_open(self, rollout_step: int | None) -> bool:
        """Whether ``rollout_step`` lies inside the actor's learning window."""
        return rollout_step is None or (
            rollout_step >= self.actor_learning_starts
            and (self.actor_learning_stops is None or rollout_step < self.actor_learning_stops)
        )

    def _current_ent_coef(self) -> torch.Tensor:
        """Detached temperature entering this gradient step's critic targets."""
        if self.log_ent_coef is not None:
            return self._ent_coef_from_raw(self.log_ent_coef.detach())
        return self.ent_coef_tensor

    def _sample_update_batch(self) -> dict[str, Any] | None:
        """Draw one oversized replay batch, augmented and ready to slice.

        Returns ``None`` (after logging why) when replay cannot supply a
        batch, in which case the caller skips the training step.
        """
        if self.sampler is None:
            raise RuntimeError("SAC has no replay sampler; inject config.sampler or pass pre-drawn batches")
        batch = self.sampler.sample(self.batch_size * self.config.num_critic_updates_per_actor_update)
        if not batch:
            logger.info("Replay produced no batch; skipping this training step")
            return None
        return self._prepare_update_batch(to_tensors(batch, self.device))

    def _prepare_update_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """Default the bootstrap steps, augment pixels, and update input statistics."""
        actions = batch["action"]
        if batch.get("bootstrap_steps") is None:
            default_steps = actions.shape[1] if actions.ndim == 3 else 1
            batch["bootstrap_steps"] = torch.full(
                (actions.shape[0],),
                default_steps,
                dtype=torch.long,
                device=actions.device,
            )

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        # Pixel augmentation is shared by every critic slice and the actor
        # batch of this gradient step (the reference augments both obs
        # and next_obs).
        if self.pixel_augmentation is not None:
            self.pixel_augmentation.apply_(obs, next_obs)
        if len(obs) == 0:
            logger.info("Buffer produced an empty batch; skipping this training step")
            return None
        self._update_input_normalizer(obs, next_obs)
        return batch

    def _actor_forward(self, actor_policy_observations: Any) -> _ActorForward:
        """Sample actions and their log-probabilities from the actor on its slice."""
        with self._autocast():
            mean_actions, log_std, dist_kwargs = self.actor.get_action_dist_params(
                actor_policy_observations
            )
            actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(
                mean_actions, log_std, **dist_kwargs
            )
        return _ActorForward(mean_actions, log_std, dist_kwargs, actions_pi, log_prob)

    def _critic_update(
        self,
        batch: dict[str, Any],
        idx: torch.Tensor,
        *,
        ent_coef: torch.Tensor,
        update_step: int,
        stats: TrainStepStats,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One critic gradient step on the batch slice ``idx``.

        Returns the detached critic loss plus the slice's current and target Q
        values (for the metrics; read back once per ``train_step``).
        """
        critic_observations = self._slice_batch_data(batch["obs"], idx)
        critic_actions = self._slice_batch_data(batch["action"], idx)
        critic_rewards = self._shape_training_rewards(batch["reward"][idx])
        critic_next_obs = self._slice_batch_data(batch["next_obs"], idx)
        critic_dones = batch["done"][idx]
        critic_truncateds = batch["truncated"][idx]
        critic_bootstrap_steps = batch["bootstrap_steps"][idx]

        with torch.no_grad():
            critic_next_actions, critic_next_log_prob = self.actor.action_log_prob(
                critic_next_obs
            )

            critic_next_log_prob = self._reduce_action_log_prob(
                critic_next_log_prob, critic_next_actions
            )

            # TD target from a random subset of the ensemble; the
            # reduction over members is configurable ('min' = REDQ
            # pessimism, 'mean' = the legacy default with no
            # overestimation control).
            critic_indices = torch.randperm(self.num_critics)[
                : self.config.n_critics_to_sample
            ]
            next_outputs = self._critic_outputs(
                self.critic_target,
                critic_next_obs,
                critic_next_actions,
                critic_indices=critic_indices,
            )

            # K-step action chunks discount by gamma^K unless the
            # caller already folded that into gamma.
            if critic_next_actions.ndim == 3 and self.compute_chunked_gamma:
                gamma_power = torch.pow(
                    torch.full_like(
                        critic_bootstrap_steps.reshape(-1, 1),
                        self.gamma,
                        dtype=torch.float32,
                    ),
                    critic_bootstrap_steps.reshape(-1, 1),
                )
            else:
                gamma_power = self.gamma

            # TD terminal mask. Legacy default treats truncation like
            # termination (finite-horizon: "succeed within the cap");
            # bootstrap_at_truncation applies the standard time-limit
            # convention instead (bootstrap through the cut, so equal
            # states are not aliased by their distance to the cap).
            critic_terminal = critic_dones.reshape(-1, 1).bool()
            if not self.config.bootstrap_at_truncation:
                critic_terminal = critic_terminal | critic_truncateds.reshape(-1, 1).bool()
            critic_terminal = critic_terminal.float()
            target_q_values, next_q_values = self._critic_td_target(
                next_outputs=next_outputs,
                next_log_prob=critic_next_log_prob.reshape(-1, 1),
                rewards=critic_rewards.reshape(-1, 1),
                terminals=critic_terminal,
                gamma=gamma_power,
                ent_coef=ent_coef,
            )
            stats.td_targets.append(target_q_values.detach())
            stats.td_terminals.append(critic_terminal.detach())

        current_outputs = self._critic_outputs(
            self.critic,
            critic_observations,
            critic_actions,
        )
        critic_loss, current_q_values = self._critic_training_loss(
            current_outputs,
            target_q_values,
        )
        critic_loss_value = critic_loss.detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._clip_gradients(self.critic, self.config.critic_clip_grad_norm)
        self.critic_optimizer.step()
        self._after_critic_optimizer_step()

        if self.target_update_interval is not None and int(self.target_update_interval) > 0:
            if update_step % int(self.target_update_interval) == 0:
                self._target_updater.soft_update(self.critic, self.critic_target, self.tau)
                stats.target_network_updates += 1
        return critic_loss_value, current_q_values, next_q_values

    def _update_entropy_coefficient(
        self, log_prob: torch.Tensor
    ) -> tuple[torch.Tensor, float | None]:
        """Tune the temperature toward the target entropy.

        Returns the detached temperature entering the actor loss and the
        temperature loss value (``None`` when the temperature is not tuned).
        """
        # Only tune the temperature when entropy actually enters a loss.
        should_tune_entropy = bool(
            self.config.train_actor_with_entropy or self.config.train_critic_with_entropy
        )
        if (
            should_tune_entropy
            and self.ent_coef_optimizer is not None
            and self.log_ent_coef is not None
        ):
            ent_coef = self._ent_coef_from_raw(self.log_ent_coef.detach())
            entropy_gap = (log_prob + float(self.target_entropy)).detach()
            if self.config.temperature_loss_type == "alpha":
                # Reference-DSRL parameterization: the gradient w.r.t.
                # log-alpha scales with alpha itself, so a converged
                # (near-zero) temperature re-inflates only slowly
                # instead of snapping back and re-injecting exploration
                # into a converged policy.
                ent_coef_loss = -(self._ent_coef_from_raw(self.log_ent_coef) * entropy_gap).mean()
            else:
                ent_coef_loss = -(self.log_ent_coef * entropy_gap).mean()
            ent_coef_loss_value = ent_coef_loss.item()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            return ent_coef, ent_coef_loss_value
        if self.log_ent_coef is not None:
            # ``auto`` still supplies the configured fixed coefficient for
            # KL/prior anchoring when entropy losses are disabled.
            return self._ent_coef_from_raw(self.log_ent_coef.detach()), None
        return self.ent_coef_tensor, None

    def _actor_update(
        self,
        forward: _ActorForward,
        log_prob: torch.Tensor,
        *,
        ent_coef: torch.Tensor,
        actor_observations: Any,
        actor_policy_observations: Any,
        rollout_step: Optional[int],
        update_step: int,
        stats: TrainStepStats,
        supplied_batches: Mapping[str, dict[str, Any]] | None = None,
        actor_replay_actions: torch.Tensor | None = None,
        actor_update_index: int = 0,
    ) -> None:
        """One actor gradient step on the actor slice with the configured anchors and BC terms."""
        with self._autocast():
            self.critic.eval()
            actor_outputs = self._critic_outputs(
                self.critic,
                actor_observations,
                forward.actions_pi,
            )
            q_values_pi = self._critic_values_from_outputs(actor_outputs)
            self.critic.train()

        if self.critic_reduction == "min":
            min_qf_pi = torch.min(q_values_pi, dim=1, keepdim=True)[0].float()
        elif self.critic_reduction == "mean":
            min_qf_pi = q_values_pi.mean(dim=1, keepdim=True).float()
        else:
            raise ValueError(f"Invalid critic reduction: {self.critic_reduction}")

        current_actor_distribution = self.actor.action_dist.proba_distribution(
            forward.mean_actions, forward.log_std, **forward.dist_kwargs
        )
        kl_div = self._snapshot_kl(
            current_actor_distribution, actor_policy_observations, update_step, stats
        )
        behavior_kl_div = self._behavior_anchor_kl(
            current_actor_distribution, actor_policy_observations, rollout_step, stats
        )
        pk, pk_ps = self._prior_kl(forward.mean_actions, forward.log_std, stats)

        actor_q_loss_active = rollout_step is None or rollout_step >= self.actor_q_loss_starts
        actor_q_weight = float(actor_q_loss_active)
        effective_actor_q_loss_coef = self._effective_actor_q_loss_coef(rollout_step)
        actor_q_objective = (
            actor_q_weight
            * effective_actor_q_loss_coef
            * self._actor_q_scale(min_qf_pi)
            * min_qf_pi
        )
        stats.actor_q_loss_active_values.append(actor_q_weight)
        stats.actor_q_loss_coef_values.append(effective_actor_q_loss_coef)

        if self.config.train_actor_with_prior_kl:
            # Replace max-entropy with a prior pull: the actor stays at
            # N(0,1) until -Q outweighs the anchor.
            actor_loss = (ent_coef.float() * pk_ps - actor_q_objective).mean()
        elif self.config.train_actor_with_kl_divergence and (kl_div is not None):
            actor_loss = (ent_coef.float() * kl_div - actor_q_objective).mean()
        elif self.config.train_actor_with_entropy:
            actor_loss = (ent_coef.float() * log_prob - actor_q_objective).mean()
        else:
            actor_loss = (-actor_q_objective).mean()

        # Additive fixed-weight prior anchor; skipped when the prior-KL
        # objective above is on (it already carries the anchor).
        prior_kl_coef = float(self.config.latent_prior_kl_coef or 0.0)
        if prior_kl_coef > 0.0 and not self.config.train_actor_with_prior_kl:
            actor_loss = actor_loss + prior_kl_coef * pk

        if behavior_kl_div is not None:
            actor_loss = actor_loss + self.actor_behavior_kl_coef * behavior_kl_div.mean()

        actor_regularization, stats.actor_regularization_metrics = self._actor_regularization_loss(
            actor_observations, forward.actions_pi
        )
        actor_loss = actor_loss + actor_regularization
        actor_loss = self._add_behavior_cloning_terms(actor_loss, stats, supplied=supplied_batches)

        stats.min_qf_pis.append(min_qf_pi.mean().item())
        stats.actor_losses.append(actor_loss.item())
        record_latent_diagnostics(self, forward, stats)
        if self.config.diagnostics_interval and actor_update_index % self.config.diagnostics_interval == 0:
            record_actor_diagnostics(
                self, forward, log_prob, min_qf_pi, ent_coef=ent_coef, actor_observations=actor_observations,
                replay_actions=actor_replay_actions, stats=stats,
            )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_gradients(self.actor, self.config.actor_clip_grad_norm)
        self.actor_optimizer.step()

        stats.actor_log_pis.append(log_prob.mean().item())


    def _snapshot_kl(
        self,
        current_actor_distribution: Any,
        actor_policy_observations: Any,
        update_step: int,
        stats: TrainStepStats,
    ) -> torch.Tensor | None:
        """Per-sample KL(actor || frozen snapshot) at the KL cadence, else ``None``."""
        kl_update_interval = int(self.config.kl_update_interval)
        compute_kl_this_step = self.config.train_actor_with_kl_divergence and (
            kl_update_interval <= 1 or (update_step % kl_update_interval == 0)
        )
        if not compute_kl_this_step:
            return None
        with torch.no_grad():
            old_mean, old_log_std, old_dist_kwargs = self.old_actor.get_action_dist_params(
                actor_policy_observations
            )
            old_actor_distribution = self.old_actor.action_dist.proba_distribution(
                old_mean,
                old_log_std,
                **old_dist_kwargs,
            )
        kl_div = kl_divergence(current_actor_distribution, old_actor_distribution).float()
        kl_div = kl_div.sum(dim=-1, keepdim=True)
        # Chunked distributions are [B, K, A]; collapse the chunk dim
        # to match log_prob / min_qf_pi at [B, 1].
        if kl_div.ndim == 3:
            kl_div = kl_div.mean(dim=1)
        stats.kl_divs.append(kl_div.mean().item())
        return kl_div

    def _behavior_anchor_kl(
        self,
        current_actor_distribution: Any,
        actor_policy_observations: Any,
        rollout_step: Optional[int],
        stats: TrainStepStats,
    ) -> torch.Tensor | None:
        """Per-sample KL(actor || fixed BC-handoff actor) once the anchor is active, else ``None``."""
        behavior_kl_active = bool(
            self.actor_behavior_kl_coef > 0.0
            and bool(self.actor_behavior_anchor_captured.item())
            and (
                rollout_step is None
                or self.actor_behavior_kl_starts is None
                or rollout_step >= self.actor_behavior_kl_starts
            )
        )
        if not behavior_kl_active:
            return None
        with torch.no_grad():
            anchor_mean, anchor_log_std, anchor_dist_kwargs = (
                self.actor_behavior_anchor.get_action_dist_params(actor_policy_observations)
            )
            anchor_distribution = self.actor_behavior_anchor.action_dist.proba_distribution(
                anchor_mean,
                anchor_log_std,
                **anchor_dist_kwargs,
            )
        behavior_kl_div = kl_divergence(
            current_actor_distribution,
            anchor_distribution,
        ).float()
        behavior_kl_div = behavior_kl_div.sum(dim=-1, keepdim=True)
        if behavior_kl_div.ndim == 3:
            behavior_kl_div = behavior_kl_div.mean(dim=1)
        stats.behavior_kl_divs.append(float(behavior_kl_div.mean().item()))
        return behavior_kl_div

    def _prior_kl(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        stats: TrainStepStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL(N(mean, std) || configured zero-mean prior), as a scalar and per sample.

        The scalar (summed over chunk*A, batch-mean) is always logged as a
        diagnostic; the per-sample ``[B, 1]`` form feeds the loss when
        ``train_actor_with_prior_kl`` is on.
        """
        pk_dim = diagonal_gaussian_prior_kl(
            mean_actions,
            log_std,
            self.latent_prior_std,
        )
        pk = pk_dim.reshape(pk_dim.shape[0], -1).sum(dim=1).mean()
        stats.prior_kls.append(float(pk.item()))
        for name, value in gaussian_prior_diagnostics(
            mean_actions, log_std, self.latent_prior_std
        ).items():
            stats.prior_diagnostics.setdefault(name, []).append(value)
        pk_ps = pk_dim.sum(dim=-1, keepdim=True)
        if pk_ps.ndim == 3:
            pk_ps = pk_ps.mean(dim=1)
        return pk, pk_ps

    def _behavior_cloning_batches(
        self, supplied: Mapping[str, dict[str, Any]] | None
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The expert and successful-online batches of this step: pre-drawn or sampled now."""
        expert_batch = None
        online_success_batch = None
        if supplied is not None:
            if self.expert_bc_coef > 0.0:
                expert_batch = supplied["expert"]
            if self.online_success_bc_coef > 0.0 and "online_success" in supplied:
                online_success_batch = supplied["online_success"]
            return expert_batch, online_success_batch
        if self.expert_bc_coef > 0.0:
            expert_batch = to_tensors(self._draw(self.expert_sampler, self.batch_size, "expert"), self.device)
        if self.online_success_bc_coef > 0.0 and self.success_sampler.can_sample(self.batch_size):
            online_success_batch = to_tensors(self.success_sampler.sample(self.batch_size), self.device)
        return expert_batch, online_success_batch

    def _add_behavior_cloning_terms(
        self,
        actor_loss: torch.Tensor,
        stats: TrainStepStats,
        *,
        supplied: Mapping[str, dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Add the weighted expert and successful-online-episode BC losses to ``actor_loss``."""
        expert_batch, online_success_batch = self._behavior_cloning_batches(supplied)
        if self.expert_bc_coef > 0.0:
            if not expert_batch:
                raise RuntimeError(
                    "expert_bc_coef is enabled but the expert replay pool produced no batch"
                )
            expert_bc_loss, expert_bc_metrics = self._behavior_cloning_loss(
                expert_batch,
                loss_type=self.expert_bc_loss_type,
            )
            actor_loss = actor_loss + self.expert_bc_coef * expert_bc_loss
            stats.expert_bc_losses.append(float(expert_bc_loss.detach().item()))
            for key, value in expert_bc_metrics.items():
                stats.expert_bc_diagnostics.setdefault(key, []).append(value)

        if online_success_batch:
            online_success_bc_loss, online_success_bc_metrics = self._behavior_cloning_loss(
                online_success_batch,
                loss_type=self.online_success_bc_loss_type,
            )
            actor_loss = actor_loss + self.online_success_bc_coef * online_success_bc_loss
            stats.online_success_bc_losses.append(float(online_success_bc_loss.detach().item()))
            for key, value in online_success_bc_metrics.items():
                stats.online_success_bc_diagnostics.setdefault(key, []).append(value)
        return actor_loss

    def _assemble_metrics(self, stats: TrainStepStats) -> dict[str, Any]:
        """Average the accumulated diagnostics into the train-step metrics."""
        # One host read-back per list instead of a sync per critic step.
        critic_losses, q_values, q_next_values, rewards = (
            torch.stack(values).tolist() if values else []
            for values in (stats.critic_losses, stats.q_values, stats.q_next_values, stats.reward_values)
        )
        metrics_dict: dict[str, Any] = {
            "critic_loss": np.mean(critic_losses),
            "q_values_mean": np.mean(q_values),
            "target_q_mean": np.mean(q_next_values),
            "reward_mean": np.mean(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "actor_optimizer_updates": len(stats.actor_losses),
            "critic_optimizer_updates": len(stats.critic_losses),
            "temperature_optimizer_updates": len(stats.ent_coef_losses),
            "target_network_updates": stats.target_network_updates,
        }
        metrics_dict.update(summarize_learning_diagnostics(stats))
        if stats.actor_losses:
            metrics_dict.update(
                {
                    "ent_coef": np.mean(stats.ent_coefs),
                    "actor_loss": np.mean(stats.actor_losses),
                    "actor_q_loss_active": np.mean(stats.actor_q_loss_active_values),
                    "actor_q_loss_coef": np.mean(stats.actor_q_loss_coef_values),
                    "actor_q_loss_coef_base": self.config.actor_q_loss_coef,
                    "prior_kl": (np.mean(stats.prior_kls) if stats.prior_kls else 0.0),
                    "actor_log_pis_mean": np.mean(stats.actor_log_pis),
                    "actor_chosen_q_mean": np.mean(stats.min_qf_pis),
                }
            )
            metrics_dict.update(stats.actor_regularization_metrics)
        if stats.w_means:
            metrics_dict.update(latent_noise_metrics(stats))
        metrics_dict.update({f"diag/{k}": float(np.mean(v)) for k, v in stats.diagnostics.items() if v})
        if stats.ent_coef_losses:
            metrics_dict["ent_coef_loss"] = np.mean(stats.ent_coef_losses)
        if stats.kl_divs:
            metrics_dict["kl_div_mean"] = np.mean(stats.kl_divs)
        metrics_dict.update(self._actor_regularization_metrics(stats))
        if stats.progress_rewards:
            metrics_dict["progress_reward_mean"] = np.mean(stats.progress_rewards)
            metrics_dict["progress_reward_min"] = np.min(stats.progress_rewards)
            metrics_dict["progress_reward_max"] = np.max(stats.progress_rewards)
        if stats.success_probs:
            metrics_dict["success_prob_mean"] = np.mean(stats.success_probs)

        distribution_diagnostics = getattr(self.actor.action_dist, "diagnostics", None)
        if callable(distribution_diagnostics):
            metrics_dict.update(distribution_diagnostics())
        return metrics_dict

    def _actor_regularization_metrics(self, stats: TrainStepStats) -> dict[str, Any]:
        """Metrics of the behavior-KL anchor and the behavior-cloning terms that are on."""
        metrics: dict[str, Any] = {}
        if self.actor_behavior_kl_coef > 0.0:
            metrics["actor_behavior_anchor_captured"] = float(
                self.actor_behavior_anchor_captured.item()
            )
            metrics["actor_behavior_kl_active"] = float(bool(stats.behavior_kl_divs))
            metrics["actor_behavior_kl_coef"] = self.actor_behavior_kl_coef
            metrics["actor_behavior_kl_mean"] = (
                float(np.mean(stats.behavior_kl_divs)) if stats.behavior_kl_divs else 0.0
            )
        if stats.expert_bc_losses:
            metrics["expert_bc_loss"] = np.mean(stats.expert_bc_losses)
            metrics["expert_bc_coef"] = self.expert_bc_coef
            metrics.update(
                {
                    f"expert_bc_{key}": float(np.mean(values))
                    for key, values in stats.expert_bc_diagnostics.items()
                }
            )
        if self.online_success_bc_coef > 0.0:
            metrics["online_success_bc_coef"] = self.online_success_bc_coef
            metrics["online_success_bc_loss"] = (
                np.mean(stats.online_success_bc_losses) if stats.online_success_bc_losses else 0.0
            )
            metrics.update(
                {
                    f"online_success_bc_{key}": float(np.mean(values))
                    for key, values in stats.online_success_bc_diagnostics.items()
                }
            )
        return metrics

    def _behavior_cloning_loss(
        self,
        batch: dict[str, Any],
        *,
        loss_type: str = "nll",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Supervise the actor from replay actions with NLL or deterministic MSE."""
        observations = batch["obs"]
        actions = batch["action"]
        action_mask = None
        if isinstance(observations, dict):
            action_mask = observations.get("action_mask")
        diagnostics: dict[str, float] = {}
        with self._autocast():
            mean, log_std, distribution_kwargs = self.actor.get_action_dist_params(observations)
            if loss_type == "mse":
                predicted_actions = self.actor.action_dist.actions_from_params(
                    mean,
                    log_std,
                    deterministic=True,
                    **distribution_kwargs,
                )
                targets = actions.to(
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                )
                projector = getattr(
                    self.actor.action_dist,
                    "project_actions_to_support",
                    None,
                )
                clipped = torch.zeros_like(targets, dtype=torch.bool)
                if callable(projector):
                    targets, clipped = projector(targets, **distribution_kwargs)

                scalar_mask = self._behavior_cloning_action_mask(
                    action_mask,
                    targets,
                )
                squared_error = (predicted_actions.float() - targets.float()).square()
                if scalar_mask is None:
                    loss = squared_error.mean()
                    valid = torch.ones_like(squared_error)
                else:
                    valid = scalar_mask.float()
                    denominator = valid.sum()
                    if denominator.item() <= 0:
                        raise ValueError("behavior cloning action mask contains no valid scalars")
                    loss = (squared_error * valid).sum() / denominator

                projection = targets.float() - actions.to(
                    device=targets.device,
                    dtype=torch.float32,
                )
                valid_count = valid.sum().clamp_min(1.0)
                diagnostics.update(
                    {
                        "valid_action_fraction": float(valid.mean().detach().item()),
                        "target_projection_l2": float(
                            torch.sqrt((projection.square() * valid).sum() / valid_count)
                            .detach()
                            .item()
                        ),
                        "target_clipped_fraction": float(
                            (clipped.float() * valid).sum().div(valid_count).detach().item()
                        ),
                    }
                )
                return loss, diagnostics
            if loss_type != "nll":
                raise ValueError(f"Unknown behavior cloning loss type: {loss_type}")
            distribution = self.actor.action_dist.proba_distribution(
                mean,
                log_std,
                **distribution_kwargs,
            )
            log_prob = distribution.log_prob(actions)
        if action_mask is None:
            return -log_prob.float().reshape(log_prob.shape[0], -1).mean(), diagnostics
        scalar_mask = self._behavior_cloning_action_mask(action_mask, actions)
        assert scalar_mask is not None
        timestep_mask = scalar_mask[..., 0]
        if timestep_mask.shape != log_prob.shape:
            raise ValueError(
                "behavior cloning action mask does not match NLL shape: "
                f"{tuple(timestep_mask.shape)} versus {tuple(log_prob.shape)}"
            )
        denominator = timestep_mask.sum()
        if denominator.item() <= 0:
            raise ValueError("behavior cloning action mask contains no valid timesteps")
        diagnostics["valid_action_fraction"] = float(scalar_mask.float().mean().item())
        masked_log_prob = torch.where(
            timestep_mask.bool(),
            log_prob.float(),
            torch.zeros_like(log_prob, dtype=torch.float32),
        )
        return -masked_log_prob.sum() / denominator, diagnostics

    def pretrain_expert_action_step(
        self,
        logging_prefix: str = "offline_action_bc_pretrain",
    ) -> dict[str, float]:
        """Fit the actor to inverted expert latents before online collection."""
        if self.expert_sampler is None:
            raise TypeError("Action behavior cloning requires demonstrations (buffer.demonstrations)")
        batch = to_tensors(self._draw(self.expert_sampler, self.batch_size, "expert"), self.device)

        loss, bc_diagnostics = self._behavior_cloning_loss(
            batch,
            loss_type=self.expert_bc_loss_type,
        )
        self.actor_optimizer.zero_grad()
        loss.backward()
        self._clip_gradients(self.actor, self.config.actor_clip_grad_norm)
        self.actor_optimizer.step()
        self._n_updates += 1
        self.step_counter = self._n_updates

        metrics = {
            "expert_bc_loss": float(loss.detach().item()),
            "expert_bc_coef": self.expert_bc_coef,
            **{f"expert_bc_{key}": value for key, value in bc_diagnostics.items()},
        }
        self.log_metrics(metrics, step=self._n_updates, prefix=logging_prefix)
        return metrics

    def load(self, load_dir: str, *, load_optimizer_state: bool = True) -> None:
        """Restore a per-component checkpoint and anchor KL regularization to its actor."""
        super().load(load_dir, load_optimizer_state=load_optimizer_state)
        self._n_updates = int(self.step_counter)
        if self.config.train_actor_with_kl_divergence:
            self.update_old_actor(self.actor)

    def load_learner_state(
        self, state: Mapping[str, np.ndarray], *, managed: Collection[str]
    ) -> None:
        """Restore the runtime-checkpoint state and re-anchor the KL snapshot."""
        super().load_learner_state(state, managed=managed)
        self._n_updates = int(self.step_counter)
        if self.config.train_actor_with_kl_divergence:
            self.update_old_actor(self.actor)

    def update_old_actor(self, other_actor: Any) -> None:
        """Refresh the frozen actor snapshot used by the KL anchor."""
        self.old_actor = type(other_actor)(other_actor.config)
        if hasattr(other_actor, "feature_extractor") and hasattr(
            self.old_actor, "feature_extractor"
        ):
            self.old_actor.feature_extractor = copy.deepcopy(other_actor.feature_extractor)
        self.old_actor.load_state_dict(other_actor.state_dict())
        self.old_actor.to(self.device)
        self.old_actor.eval()
