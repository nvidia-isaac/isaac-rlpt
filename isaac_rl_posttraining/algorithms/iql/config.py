# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Implicit Q-Learning over a trainable base model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_rl_posttraining.algorithms.base import RUNTIME_INJECTED, BaseAlgorithmConfig

__all__ = ["IQLConfig"]


@dataclass
class IQLConfig(BaseAlgorithmConfig):
    """Hyperparameters for :class:`~isaac_rl_posttraining.algorithms.iql.IQL`.

    The learner never imports a model stack. It calls
    :meth:`~isaac_rl_posttraining.policies.flow_sde.FlowSDEActorCritic.feature_tokens`
    and :meth:`~isaac_rl_posttraining.policies.flow_sde.FlowSDEActorCritic.actor_bc_loss`
    on the injected policy.

    Attributes:
        policy: Trainable base model whose action head IQL updates.
        offline_source: Object with ``sample(batch_size) -> dict`` producing
            ``inputs_cur``, ``inputs_next``, ``action``, ``reward``, ``done``.
        expectile: Expectile for the V regression (Kostrikov IQL).
        advantage_temp: Temperature inside the AWR exponent. With a sparse
            terminal reward, advantages are small, and a temperature near 1
            weights every sample almost equally, which reduces IQL to BC.
        clip_score: Upper bound on AWR weights.
        gamma: Discount factor applied to the chunk return.
        tau: Polyak coefficient for the target critic.
        batch_size: Transitions drawn per gradient step.
        num_q: Q-ensemble size.
        n_q_to_sample: Ensemble members used for the V/AWR target (all heads
            still regress).
        q_reduce: Reduction over the sampled members (``min`` / ``mean`` / ``max``).
        num_updates_per_train_step: Gradient steps per ``train_step`` call.
        bootstrap_horizon: Exponent on ``gamma`` for the V(s') bootstrap.
        critic_hidden: Hidden width of each token critic MLP.
        critic_n_heads: Attention heads in each token critic.
        actor_optimizer_lr: Peak learning rate of the policy (action-head) optimizer.
        critic_optimizer_lr: Learning rate of the Q/V optimizer.
        actor_optimizer_weight_decay: AdamW weight decay on the policy.
        critic_optimizer_weight_decay: AdamW weight decay on the critics.
        actor_lr_schedule: ``cosine`` or ``constant`` decay of the policy
            learning rate after warmup.
        actor_lr_warmup_ratio: Fraction of the run spent in linear warmup.
        actor_max_grad_norm: Global gradient-norm clip on the policy; 0 disables it.
        learning_starts: Gradient steps skipped at the start of a run (0 for
            pure offline).
        total_updates: Run length in gradient steps, set by the launcher from
            ``training.num_rollouts``; the policy schedule spans it.
    """

    policy: Any = field(default=None, metadata=RUNTIME_INJECTED)
    offline_source: Any = field(default=None, metadata=RUNTIME_INJECTED)
    expectile: float = 0.7
    advantage_temp: float = 10.0
    clip_score: float = 20.0
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    num_q: int = 5
    n_q_to_sample: int = 2
    q_reduce: str = "min"
    num_updates_per_train_step: int = 1
    bootstrap_horizon: int = 1
    critic_hidden: int = 1024
    critic_n_heads: int = 8
    actor_optimizer_lr: float = 1e-4
    critic_optimizer_lr: float = 3e-4
    actor_optimizer_weight_decay: float = 1e-5
    critic_optimizer_weight_decay: float = 0.0
    actor_lr_schedule: str = "cosine"
    actor_lr_warmup_ratio: float = 0.05
    actor_max_grad_norm: float = 1.0
    learning_starts: int = 0
    total_updates: int = field(default=0, metadata=RUNTIME_INJECTED)

    def __post_init__(self) -> None:
        if not 0.0 <= self.actor_lr_warmup_ratio < 1.0:
            raise ValueError("actor_lr_warmup_ratio must lie in [0, 1)")
        if self.num_q < 1:
            raise ValueError("num_q must be a positive integer")
        if self.n_q_to_sample < 1 or self.n_q_to_sample > self.num_q:
            raise ValueError("n_q_to_sample must lie in [1, num_q]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must lie in (0, 1)")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.num_updates_per_train_step < 1:
            raise ValueError("num_updates_per_train_step must be a positive integer")
        if self.q_reduce not in ("min", "mean", "max"):
            raise ValueError("q_reduce must be min, mean, or max")

    @property
    def algorithm_class(self) -> type:
        from isaac_rl_posttraining.algorithms.iql.iql import IQL

        return IQL
