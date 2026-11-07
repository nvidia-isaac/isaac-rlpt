# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AdamW for a large low-precision action head, shared by the offline learners.

Trainable base models keep their weights in bfloat16. Stepping AdamW on bfloat16
weights rounds most updates away (8 mantissa bits), so the optimizer keeps a
float32 master copy of every parameter, steps that copy, and writes it back.
The model, its snapshots and inference stay in the base model's own dtype. The
schedule is linear warmup then cosine (or constant) decay, with global
gradient-norm clipping, as in GR00T's own fine-tuning recipe.

The update runs one parameter at a time, so beyond the float32 master and
moments only a single parameter's float32 gradient is ever resident.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

__all__ = ["LR_SCHEDULES", "ActorOptimizer"]

LR_SCHEDULES = ("constant", "cosine")


class ActorOptimizer:
    """Mixed-precision AdamW with warmup, cosine decay and gradient clipping.

    Args:
        parameters: Trainable parameters of the policy.
        lr: Peak learning rate.
        weight_decay: Decoupled (AdamW) weight decay.
        warmup_steps: Linear warmup length in optimizer steps.
        schedule: ``constant`` or ``cosine`` decay after warmup.
        total_steps: Run length in optimizer steps; the cosine decays to 0 there.
        max_grad_norm: Global gradient-norm clip; ``0`` disables clipping.
        betas: Adam moment coefficients.
        eps: Adam denominator epsilon.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        warmup_steps: int = 0,
        schedule: str = "constant",
        total_steps: int = 0,
        max_grad_norm: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if schedule not in LR_SCHEDULES:
            raise ValueError(f"schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
        if schedule == "cosine" and total_steps < 1:
            raise ValueError("a cosine schedule needs total_steps >= 1")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("ActorOptimizer needs at least one trainable parameter")
        self.masters = [
            parameter if parameter.dtype == torch.float32 else parameter.detach().float().clone()
            for parameter in self.parameters
        ]
        self.exp_avgs = [torch.zeros_like(master) for master in self.masters]
        self.exp_avg_sqs = [torch.zeros_like(master) for master in self.masters]
        self.peak_lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = betas
        self.eps = float(eps)
        self.warmup_steps = int(warmup_steps)
        self.schedule = schedule
        self.total_steps = int(total_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.steps_taken = 0

    def lr_at(self, step: int) -> float:
        """Learning rate for the ``step``-th optimizer step (0-based)."""
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if self.schedule == "cosine":
            span = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, (step - self.warmup_steps) / span)
            return self.peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        """Apply one AdamW update from the gradients on the policy parameters."""
        grads = [parameter.grad for parameter in self.parameters]
        present = [grad for grad in grads if grad is not None]
        if not present:
            raise RuntimeError("ActorOptimizer.step() found no gradients; call backward() first")
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(grad, dtype=torch.float32) for grad in present])
        )
        if not torch.isfinite(grad_norm):
            # A non-finite gradient would enter the moments and the master weights for good.
            self.zero_grad()
            return {"grad_norm": float(grad_norm), "lr": self.lr_at(self.steps_taken)}
        scale = 1.0
        if self.max_grad_norm > 0 and float(grad_norm) > self.max_grad_norm:
            scale = self.max_grad_norm / (float(grad_norm) + 1e-6)
        lr = self.lr_at(self.steps_taken)
        self.steps_taken += 1
        beta1, beta2 = self.betas
        bias1 = 1.0 - beta1**self.steps_taken
        bias2 = 1.0 - beta2**self.steps_taken
        for index, parameter in enumerate(self.parameters):
            grad = grads[index]
            if grad is None:
                continue
            master = self.masters[index]
            grad = grad.float().mul_(scale) if master is not parameter else grad.mul(scale)
            parameter.grad = None
            exp_avg, exp_avg_sq = self.exp_avgs[index], self.exp_avg_sqs[index]
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            del grad
            master.mul_(1.0 - lr * self.weight_decay)
            denom = (exp_avg_sq / bias2).sqrt_().add_(self.eps)
            master.addcdiv_(exp_avg, denom, value=-lr / bias1)
            del denom
            if master is not parameter:
                parameter.copy_(master)
        return {"grad_norm": float(grad_norm), "lr": lr}
