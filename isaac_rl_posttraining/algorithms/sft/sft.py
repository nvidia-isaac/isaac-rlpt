# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised fine-tuning over a trainable base model.

The learner owns an :class:`~isaac_rl_posttraining.algorithms.actor_optimizer.ActorOptimizer`
over the base model's trainable parameters and minimizes the per-sample
``actor_bc_loss``. It never imports a model stack: the action reconstruction
loss comes from the injected base model.
"""

from __future__ import annotations

from typing import Any

from isaac_rl_posttraining.algorithms.actor_optimizer import ActorOptimizer
from isaac_rl_posttraining.algorithms.base import BaseAlgorithm
from isaac_rl_posttraining.algorithms.sft.config import SFTConfig

__all__ = ["SFT"]


class SFT(BaseAlgorithm):
    """Offline SFT: unweighted mean of ``actor_bc_loss``."""

    def __init__(self, config: SFTConfig) -> None:
        super().__init__(config)
        if config.policy is None:
            raise ValueError("SFTConfig.policy must be the trainable base model")
        if config.offline_source is None:
            raise ValueError("SFTConfig.offline_source must provide sample(batch_size)")
        self.policy = config.policy
        self.offline_source = config.offline_source
        actor_params = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not actor_params:
            raise ValueError("SFT needs at least one trainable parameter on the policy")
        self.actor_optimizer = ActorOptimizer(
            actor_params,
            lr=config.actor_optimizer_lr,
            weight_decay=config.actor_optimizer_weight_decay,
            warmup_steps=int(config.actor_lr_warmup_ratio * config.total_updates),
            schedule=config.actor_lr_schedule,
            total_steps=config.total_updates,
            max_grad_norm=config.actor_max_grad_norm,
        )

    def train_step(self, rollout_step: int | None = None) -> dict[str, Any]:
        """Run ``num_updates_per_train_step`` gradient steps on offline batches; the loop logs the metrics."""
        if rollout_step is not None and rollout_step < self.config.learning_starts:
            return {}
        metrics: dict[str, Any] = {}
        for _ in range(self.config.num_updates_per_train_step):
            batch = self.offline_source.sample(self.config.batch_size)
            metrics = self._update(batch)
            self.step_counter += 1
        return metrics

    def _update(self, batch: dict[str, Any]) -> dict[str, Any]:
        actor_inputs = dict(batch["inputs_cur"])
        actor_inputs["action"] = batch["action"]
        self.policy.train()
        try:
            bc_loss = self.policy.actor_bc_loss(actor_inputs)
        finally:
            self.policy.eval()
        actor_loss = bc_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        step = self.actor_optimizer.step()
        return {
            "actor_loss": float(actor_loss.detach()),
            "bc_loss": float(bc_loss.mean().detach()),
            "actor_grad_norm": step["grad_norm"],
            "actor_lr": step["lr"],
        }
