# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT learner: exact mean of actor_bc_loss on a stub backend; no critics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.sft import SFT, SFTConfig

BATCH, HORIZON, ACTION = 4, 2, 3


class _StubPolicy(nn.Module):
    """Squared-parameter BC loss. feature_tokens must not be called."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def feature_tokens(self, inputs):
        del inputs
        raise AssertionError("SFT must not call feature_tokens")

    def actor_bc_loss(self, inputs):
        assert "action" in inputs
        return (self.scale * self.scale).expand(BATCH)


class _StubSource:
    def __init__(self, batch: dict) -> None:
        self.batch = batch
        self.requested: list[int] = []

    def sample(self, batch_size: int) -> dict:
        self.requested.append(int(batch_size))
        return self.batch


def _batch() -> dict:
    return {
        "inputs_cur": {"obs": torch.zeros(BATCH, 1)},
        "inputs_next": {"obs": torch.zeros(BATCH, 1), "next": True},
        "action": torch.zeros(BATCH, HORIZON, ACTION),
        "reward": torch.zeros(BATCH, HORIZON),
        "done": torch.zeros(BATCH, HORIZON),
    }


def test_sft_train_step_is_the_mean_of_actor_bc_loss():
    policy = _StubPolicy()
    source = _StubSource(_batch())
    learner = SFT(
        SFTConfig(
            name="sft",
            policy=policy,
            offline_source=source,
            batch_size=BATCH,
            num_updates_per_train_step=1,
            actor_optimizer_lr=0.0,
            total_updates=1,
        )
    )
    metrics = learner.train_step()
    expected = float((policy.scale * policy.scale).mean().detach())
    assert metrics["actor_loss"] == pytest.approx(expected)
    assert metrics["bc_loss"] == pytest.approx(expected)
    assert source.requested == [BATCH]
    assert learner.step_counter == 1


def test_sft_a_gradient_step_reduces_the_squared_parameter():
    policy = _StubPolicy()
    before = float(policy.scale.detach())
    learner = SFT(
        SFTConfig(
            name="sft",
            policy=policy,
            offline_source=_StubSource(_batch()),
            batch_size=BATCH,
            actor_optimizer_lr=0.1,
            actor_lr_warmup_ratio=0.0,
            total_updates=10,
        )
    )
    learner.train_step()
    assert float(policy.scale.detach()) < before
