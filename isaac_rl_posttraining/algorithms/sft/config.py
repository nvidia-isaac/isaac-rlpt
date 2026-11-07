# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for supervised fine-tuning over a trainable base model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_rl_posttraining.algorithms.base import RUNTIME_INJECTED, BaseAlgorithmConfig

__all__ = ["SFTConfig"]


@dataclass
class SFTConfig(BaseAlgorithmConfig):
    """Hyperparameters for :class:`~isaac_rl_posttraining.algorithms.sft.SFT`.

    The learner never imports a model stack. It calls
    :meth:`~isaac_rl_posttraining.policies.flow_sde.FlowSDEActorCritic.actor_bc_loss`
    on the injected policy.

    Attributes:
        policy: Trainable base model whose action head SFT updates.
        offline_source: Object with ``sample(batch_size) -> dict`` producing
            ``inputs_cur`` and ``action`` (other keys are ignored).
        batch_size: Transitions drawn per gradient step.
        num_updates_per_train_step: Gradient steps per ``train_step`` call.
        actor_optimizer_lr: Peak learning rate of the policy (action-head) optimizer.
        actor_optimizer_weight_decay: AdamW weight decay on the policy.
        actor_lr_schedule: ``cosine`` or ``constant`` decay after warmup.
        actor_lr_warmup_ratio: Fraction of the run spent in linear warmup.
        actor_max_grad_norm: Global gradient-norm clip on the policy; 0 disables it.
        learning_starts: Gradient steps skipped at the start of a run (0 for
            pure offline).
        total_updates: Run length in gradient steps, set by the launcher from
            ``training.num_rollouts``; the schedule spans it.
    """

    policy: Any = field(default=None, metadata=RUNTIME_INJECTED)
    offline_source: Any = field(default=None, metadata=RUNTIME_INJECTED)
    batch_size: int = 256
    num_updates_per_train_step: int = 1
    actor_optimizer_lr: float = 1e-4
    actor_optimizer_weight_decay: float = 1e-5
    actor_lr_schedule: str = "cosine"
    actor_lr_warmup_ratio: float = 0.05
    actor_max_grad_norm: float = 1.0
    learning_starts: int = 0
    total_updates: int = field(default=0, metadata=RUNTIME_INJECTED)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.num_updates_per_train_step < 1:
            raise ValueError("num_updates_per_train_step must be a positive integer")
        if self.actor_optimizer_lr < 0:
            raise ValueError("actor_optimizer_lr must be non-negative")
        if not 0.0 <= self.actor_lr_warmup_ratio < 1.0:
            raise ValueError("actor_lr_warmup_ratio must lie in [0, 1)")

    @property
    def algorithm_class(self) -> type:
        from isaac_rl_posttraining.algorithms.sft.sft import SFT

        return SFT
