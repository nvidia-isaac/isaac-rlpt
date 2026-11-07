# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the REDQ-style SAC core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Union

import gymnasium as gym

from isaac_rl_posttraining.algorithms.base import RUNTIME_INJECTED, BaseAlgorithmConfig


__all__ = ["STEP_SCHEDULED_FIELDS", "SACConfig"]

#: Settings keyed on the environment step; they act only with ``respect_step_schedules``.
STEP_SCHEDULED_FIELDS: dict[str, object] = {
    "actor_learning_starts": None,
    "actor_learning_stops": None,
    "actor_q_loss_starts": None,
    "actor_q_loss_coef_after_step": None,
    "actor_q_loss_coef_after_step_value": None,
    "actor_behavior_kl_coef": 0.0,
    "actor_behavior_anchor_step": None,
    "actor_behavior_kl_starts": None,
}


@dataclass
class SACConfig(BaseAlgorithmConfig):
    """Hyperparameters for :class:`~isaac_rl_posttraining.algorithms.sac.SAC`.

    Attributes:
        action_space: Action space the actor emits in (injected at runtime
            with the other wiring slots).
        learning_starts: Environment steps collected before updates begin.
        actor_learning_starts: Optional environment step at which actor and
            temperature updates begin while critic learning continues.
        actor_q_loss_starts: Optional environment step at which the critic's
            action-value objective begins contributing to actor updates.
            This can be later than ``actor_learning_starts`` to allow a
            behavior-cloning-only predeployment stage.
        actor_q_loss_coef: Non-negative weight on the critic's action-value
            objective in the actor loss.
        actor_q_loss_coef_after_step: Optional environment-step boundary after
            which ``actor_q_loss_coef_after_step_value`` replaces
            ``actor_q_loss_coef``. The base coefficient remains effective at
            the boundary itself.
        actor_q_loss_coef_after_step_value: Non-negative actor Q-loss weight
            used after ``actor_q_loss_coef_after_step``.
        actor_learning_stops: Optional exclusive environment-step boundary
            after which actor and temperature updates stop.
        ent_coef: Entropy temperature — ``"auto"`` learns it, a float fixes it.
        target_entropy: Target for the temperature loss; ``"auto"`` uses
            ``-|A|``, a number sets it directly (e.g. ``0`` for latent spaces).
        target_update_interval: Critic-updates between target-net updates.
        gamma: Per-step discount factor.
        compute_chunked_gamma: Discount TD targets by ``gamma**K`` when the
            sampled actions are ``K``-step chunks. Disable when rewards are
            already stored per executed chunk and the discount is folded into
            ``gamma`` by the caller.
        tau: Polyak coefficient for target-network updates.
        pooled_critic_features: Share the critic backbone across ensemble
            members (per-member value heads only) and evaluate it once per
            batch.
        batch_size: Per-update minibatch size.
        n_critics_to_sample: Ensemble members sampled per TD target.
        num_critics: Ensemble size.
        critic_reduction: ``"min"`` or ``"mean"`` over member Q-values in the
            actor objective.
        critic_target_reduction: ``"mean"`` (legacy default) or ``"min"``
            (REDQ subset pessimism) over the sampled members in the TD
            target. ``"mean"`` provides no overestimation control; ``"min"``
            is the published REDQ semantics.
        num_critic_updates_per_actor_update: Critic updates per actor update;
            one oversized batch is drawn and split into this many slices.
        num_updates_per_train_step: Gradient steps per ``train_step`` call.
        bootstrap_at_truncation: Bootstrap the TD target through time-limit
            truncations (the standard time-limit convention; avoids aliasing
            states by their distance to the cap). ``False`` (legacy) treats a
            truncation as a terminal, making the objective "succeed within
            the cap".
        augment_pixels: Apply random-shift augmentation to image observation
            keys (name contains ``"image"``) in each update batch -- the
            DrQ-style regularizer pixel critics need at high update ratios.
        augment_color_jitter: Additionally jitter brightness/contrast/
            saturation/hue of image keys (reference-DSRL augmentation);
            requires ``augment_pixels``.
        temperature_loss_type: ``"log_alpha"`` (torch convention: constant
            log-space gradient) or ``"alpha"`` (reference DSRL: gradient
            scales with alpha, so a converged temperature re-inflates
            slowly).
        use_amp: Train under BF16 autocast on devices that support it. Disable for
            full-FP32 updates (the reference JAX implementation's precision).
        train_critic_with_entropy: Include the entropy term in TD targets.
        train_actor_with_entropy: Include the entropy term in the actor loss.
        actor_optimizer_lr: Actor Adam learning rate.
        actor_optimizer_eps: Actor Adam epsilon.
        actor_optimizer_weight_decay: Actor Adam weight decay.
        critic_optimizer_lr: Critic Adam learning rate.
        critic_optimizer_betas: Critic Adam betas.
        critic_optimizer_eps: Critic Adam epsilon.
        critic_optimizer_weight_decay: Critic Adam weight decay.
        ent_coef_lr: Temperature Adam learning rate.
        train_actor_with_kl_divergence: Replace the actor entropy term with a
            KL anchor to a frozen snapshot of the actor (see
            ``update_old_actor``).
        kl_update_interval: Update-steps between KL-anchor recomputations.
        actor_behavior_kl_coef: Weight of an additive KL penalty from the
            current actor to a fixed behavior-policy snapshot. ``0`` disables
            the constraint.
        actor_behavior_anchor_step: Environment step at which the behavior
            snapshot is captured, before that step's actor update. This is
            typically the end of a behavior-cloning phase.
        actor_behavior_kl_starts: Optional environment step at which the fixed
            behavior KL begins contributing to the actor loss. Defaults to the
            snapshot step and can be delayed until the RL objective starts.
        reset_actor_optimizer_at_rl_handoff: Clear actor optimizer momentum at
            an explicit BC-to-RL handoff. This prevents a long BC phase from
            carrying stale momentum into the anchored RL objective.
        latent_prior_kl_coef: Weight of an additive Gaussian-prior KL
            penalty on the actor loss. For latent-steering policies this
            keeps the emitted latent near the base policy's sampling prior so
            decoded actions stay on-distribution. ``0`` disables it.
        latent_prior_std: Standard deviation of the zero-mean Gaussian prior
            in the actor's normalized action coordinates.
        train_actor_with_prior_kl: Replace the actor entropy term with
            ``ent_coef * KL(actor || N(0,1))`` so the actor leaves the prior
            only where the Q-landscape justifies it. Mutually exclusive with
            ``latent_prior_kl_coef`` (the additive term is skipped when on).
        expert_bc_coef: Weight of an auxiliary behavior-cloning loss on noise
            actions sampled from the demonstrations buffer
            (``buffer.demonstrations``). ``0`` disables it.
        expert_bc_loss_type: ``"nll"`` clones the full stochastic
            distribution; ``"mse"`` clones only its deterministic action.
        online_success_bc_coef: Weight of an auxiliary negative-log-likelihood
            loss on actions from successful online episodes.
            ``0`` disables it.
        online_success_bc_loss_type: ``"nll"`` clones the full stochastic
            distribution; ``"mse"`` clones only its deterministic action.
        clip_grad_norm: Optional gradient-norm clip for actor and critic.
        respect_step_schedules: Pass the collected environment steps to every
            update so the step-scheduled settings above (actor learning window,
            actor Q-loss start and switch, behavior anchor) take effect. Off by
            default, which keeps the update independent of the step count
            (today's behaviour); setting any of those fields then fails at
            construction instead of being ignored.
        diagnostics_interval: Every this many actor updates, record the
            ``diag/*`` metrics (gradient scales of the Q and entropy terms,
            critic responses to the actor's, the replayed, a prior and a
            uniform latent, Q spread over latents versus observations).
            ``0`` (default) records nothing; the update itself is unchanged.
    """

    action_space: gym.Space | None = field(default=None, metadata=RUNTIME_INJECTED)

    learning_starts: int = 1000
    respect_step_schedules: bool = False
    actor_learning_starts: Optional[int] = None
    actor_q_loss_starts: Optional[int] = None
    actor_q_loss_coef: float = 1.0
    actor_q_loss_coef_after_step: Optional[int] = None
    actor_q_loss_coef_after_step_value: Optional[float] = None
    actor_learning_stops: Optional[int] = None
    ent_coef: Union[str, float] = "auto"
    # How the learned temperature is parameterized: "log" (alpha = exp(raw),
    # the classic SAC form) or "softplus" (alpha = softplus(raw)). Both start
    # at alpha = 1. Softplus decays alpha more gently from large values —
    # the reference-validated DSRL schedule — and requires
    # temperature_loss_type="alpha" (the linear-in-alpha loss).
    ent_coef_parameterization: str = "log"
    target_entropy: Union[str, float, int] = "auto"
    target_update_interval: int = 1
    gamma: float = 0.99
    compute_chunked_gamma: bool = True
    tau: float = 0.005
    pooled_critic_features: bool = True

    batch_size: int = 128
    n_critics_to_sample: int = 2
    num_critics: int = 2
    critic_reduction: str = "min"
    critic_target_reduction: str = "mean"
    # Optional bounds on the reward-added, terminal-masked regression target.
    td_target_clip: tuple[float, float] | None = None
    num_critic_updates_per_actor_update: int = 4
    num_updates_per_train_step: int = 1
    bootstrap_at_truncation: bool = False
    augment_pixels: bool = False
    augment_color_jitter: bool = False
    use_amp: bool = True
    temperature_loss_type: str = "log_alpha"
    train_critic_with_entropy: bool = False
    train_actor_with_entropy: bool = True

    actor_optimizer_lr: float = 3e-4
    actor_optimizer_betas: tuple = (0.9, 0.999)
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 1e-6

    critic_optimizer_lr: float = 3e-4
    critic_optimizer_betas: tuple = (0.95, 0.999)
    critic_optimizer_eps: float = 1e-8
    critic_optimizer_weight_decay: float = 1e-6

    ent_coef_lr: float = 3e-4

    train_actor_with_kl_divergence: bool = True
    kl_update_interval: int = 1

    actor_behavior_kl_coef: float = 0.0
    actor_behavior_anchor_step: Optional[int] = None
    actor_behavior_kl_starts: Optional[int] = None
    reset_actor_optimizer_at_rl_handoff: bool = False

    latent_prior_kl_coef: float = 0.0
    latent_prior_std: float = 1.0
    train_actor_with_prior_kl: bool = False
    expert_bc_coef: float = 0.0
    expert_bc_loss_type: str = "nll"
    online_success_bc_coef: float = 0.0
    online_success_bc_loss_type: str = "nll"

    clip_grad_norm: Optional[float] = None
    diagnostics_interval: int = 0
    # Per-network clips; each falls back to clip_grad_norm when unset.
    actor_clip_grad_norm: Optional[float] = None
    critic_clip_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        self._require_step_schedules_respected()
        self._validate_td_target_clip()
        effective_actor_start = self._validate_actor_starts()
        self._validate_actor_q_loss_coef()
        self._validate_behavior_kl()
        if (
            self.actor_learning_stops is not None
            and self.actor_learning_stops <= effective_actor_start
        ):
            raise ValueError(
                "actor_learning_stops must be greater than the effective actor "
                f"start, got {effective_actor_start} -> {self.actor_learning_stops}"
            )

    def _require_step_schedules_respected(self) -> None:
        if self.respect_step_schedules:
            return
        configured = sorted(
            name
            for name, default in STEP_SCHEDULED_FIELDS.items()
            if getattr(self, name) != default
        )
        if configured:
            raise ValueError(
                f"algorithm.{', algorithm.'.join(configured)} "
                f"{'is' if len(configured) == 1 else 'are'} scheduled by environment step "
                "and only act with algorithm.respect_step_schedules=true (the learner "
                "then receives the step count with every update); set it, or leave "
                "these fields at their defaults"
            )

    def _validate_td_target_clip(self) -> None:
        if self.td_target_clip is not None:
            low, high = (float(bound) for bound in self.td_target_clip)
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(
                    "td_target_clip must have finite bounds with low < high, "
                    f"got {self.td_target_clip!r}"
                )

    def _validate_actor_starts(self) -> int:
        """Check the actor start steps; return the effective actor learning start."""
        effective_actor_start = (
            self.learning_starts
            if self.actor_learning_starts is None
            else self.actor_learning_starts
        )
        effective_actor_q_start = (
            effective_actor_start if self.actor_q_loss_starts is None else self.actor_q_loss_starts
        )
        if self.learning_starts < 0 or effective_actor_start < 0:
            raise ValueError("learning starts must be non-negative")
        if effective_actor_q_start < effective_actor_start:
            raise ValueError(
                "actor_q_loss_starts must not precede the effective actor learning "
                f"start, got {effective_actor_q_start} < {effective_actor_start}"
            )
        return effective_actor_start

    def _validate_actor_q_loss_coef(self) -> None:
        if not math.isfinite(self.actor_q_loss_coef) or self.actor_q_loss_coef < 0.0:
            raise ValueError(
                f"actor_q_loss_coef must be finite and non-negative, got {self.actor_q_loss_coef}"
            )
        scheduled_fields = (
            self.actor_q_loss_coef_after_step,
            self.actor_q_loss_coef_after_step_value,
        )
        if (scheduled_fields[0] is None) != (scheduled_fields[1] is None):
            raise ValueError(
                "actor_q_loss_coef_after_step and "
                "actor_q_loss_coef_after_step_value must be configured together"
            )
        if self.actor_q_loss_coef_after_step is not None:
            if self.actor_q_loss_coef_after_step < 0:
                raise ValueError("actor_q_loss_coef_after_step must be non-negative")
            assert self.actor_q_loss_coef_after_step_value is not None
            if (
                not math.isfinite(self.actor_q_loss_coef_after_step_value)
                or self.actor_q_loss_coef_after_step_value < 0.0
            ):
                raise ValueError(
                    "actor_q_loss_coef_after_step_value must be finite and non-negative, "
                    f"got {self.actor_q_loss_coef_after_step_value}"
                )

    def _validate_behavior_kl(self) -> None:
        if not math.isfinite(self.actor_behavior_kl_coef) or self.actor_behavior_kl_coef < 0.0:
            raise ValueError(
                "actor_behavior_kl_coef must be finite and non-negative, "
                f"got {self.actor_behavior_kl_coef}"
            )
        if self.actor_behavior_anchor_step is not None and self.actor_behavior_anchor_step < 0:
            raise ValueError("actor_behavior_anchor_step must be non-negative")
        if self.actor_behavior_kl_starts is not None and self.actor_behavior_kl_starts < 0:
            raise ValueError("actor_behavior_kl_starts must be non-negative")
        if self.actor_behavior_kl_coef > 0.0 and self.actor_behavior_anchor_step is None:
            raise ValueError("positive actor_behavior_kl_coef requires actor_behavior_anchor_step")
        if (
            self.actor_behavior_anchor_step is not None
            and self.actor_behavior_kl_starts is not None
            and self.actor_behavior_kl_starts < self.actor_behavior_anchor_step
        ):
            raise ValueError("actor_behavior_kl_starts must not precede actor_behavior_anchor_step")

    @property
    def algorithm_class(self) -> type:
        from isaac_rl_posttraining.algorithms.sac.sac import SAC

        return SAC
