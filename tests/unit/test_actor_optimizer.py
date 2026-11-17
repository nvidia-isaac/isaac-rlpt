# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ActorOptimizer: float32 master weights for bf16 heads, warmup + cosine, clipping."""

from __future__ import annotations

import math

import pytest
import torch

from isaac_rl_posttraining.algorithms.actor_optimizer import ActorOptimizer


def _step(
    optimizer: ActorOptimizer, parameter: torch.nn.Parameter, grad: float
) -> dict:
    optimizer.zero_grad()
    (parameter * grad).sum().backward()
    return optimizer.step()


def test_small_updates_accumulate_on_a_bfloat16_parameter():
    # 1e-4 relative steps round away on bf16 weights near 1.0 (spacing 2^-7).
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = ActorOptimizer([parameter], lr=1e-4, weight_decay=0.0)
    for _ in range(200):
        _step(optimizer, parameter, 1.0)
    assert parameter.dtype == torch.bfloat16
    assert float(optimizer.masters[0][0]) == pytest.approx(1.0 - 200 * 1e-4, abs=1e-5)
    assert float(parameter.detach()[0]) < 1.0


def test_float32_parameters_are_stepped_in_place():
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = ActorOptimizer([parameter], lr=0.1, weight_decay=0.0)
    assert optimizer.masters[0] is parameter
    _step(optimizer, parameter, 1.0)
    assert float(parameter.detach()[0]) == pytest.approx(0.9, abs=1e-6)


def test_matches_torch_adamw_on_float32_parameters():
    torch.manual_seed(0)
    ours = torch.nn.Parameter(torch.randn(5, 3))
    reference = torch.nn.Parameter(ours.detach().clone())
    optimizer = ActorOptimizer([ours], lr=1e-2, weight_decay=0.1)
    torch_optimizer = torch.optim.AdamW([reference], lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        grad = torch.randn(5, 3)
        optimizer.zero_grad()
        ours.grad = grad.clone()
        optimizer.step()
        torch_optimizer.zero_grad()
        reference.grad = grad.clone()
        torch_optimizer.step()
    torch.testing.assert_close(ours.detach(), reference.detach())


def test_warmup_then_cosine_reaches_zero_at_the_end():
    optimizer = ActorOptimizer(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1.0,
        weight_decay=0.0,
        warmup_steps=10,
        schedule="cosine",
        total_steps=110,
    )
    assert optimizer.lr_at(0) == pytest.approx(0.1)
    assert optimizer.lr_at(9) == pytest.approx(1.0)
    assert optimizer.lr_at(60) == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 0.5)))
    assert optimizer.lr_at(110) == pytest.approx(0.0, abs=1e-12)


def test_gradient_norm_is_reported_before_clipping():
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    optimizer = ActorOptimizer([parameter], lr=0.0, weight_decay=0.0, max_grad_norm=1.0)
    metrics = _step(optimizer, parameter, 3.0)
    assert metrics["grad_norm"] == pytest.approx(6.0)
    assert parameter.grad is None


def test_cosine_needs_a_run_length():
    with pytest.raises(ValueError, match="total_steps"):
        ActorOptimizer(
            [torch.nn.Parameter(torch.ones(1))],
            lr=1.0,
            weight_decay=0.0,
            schedule="cosine",
        )
