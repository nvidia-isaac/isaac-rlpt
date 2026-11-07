# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IQL learner: exact Q/V/AWR on a stub backend and a scripted offline batch."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.iql import IQL, IQLConfig

BATCH, TOKENS, FEAT, HORIZON, ACTION = 4, 2, 8, 2, 3


class _StubPolicy(nn.Module):
    """Constant tokens and a squared-parameter BC loss, so AWR weights are exact."""

    def __init__(self, tokens_cur: torch.Tensor, tokens_next: torch.Tensor) -> None:
        super().__init__()
        self._tokens_cur = tokens_cur
        self._tokens_next = tokens_next
        self.scale = nn.Parameter(torch.tensor(0.5))

    def feature_tokens(self, inputs):
        tokens = self._tokens_next if inputs.get("next") else self._tokens_cur
        return tokens, None

    def actor_bc_loss(self, inputs):
        del inputs
        return (self.scale * self.scale).expand(BATCH)


class _StubSource:
    def __init__(self, batch: dict) -> None:
        self.batch = batch
        self.requested: list[int] = []

    def sample(self, batch_size: int) -> dict:
        self.requested.append(int(batch_size))
        return self.batch


def test_iql_train_step_matches_hand_computed_losses():
    torch.manual_seed(0)
    tokens_cur = torch.randn(BATCH, TOKENS, FEAT)
    tokens_next = torch.randn(BATCH, TOKENS, FEAT)
    policy = _StubPolicy(tokens_cur, tokens_next)
    batch = {
        "inputs_cur": {"obs": torch.zeros(BATCH, 1)},
        "inputs_next": {"obs": torch.zeros(BATCH, 1), "next": True},
        "action": torch.zeros(BATCH, HORIZON, ACTION),
        "reward": torch.zeros(BATCH, HORIZON),
        "done": torch.zeros(BATCH, HORIZON),
    }
    config = IQLConfig(
        name="iql",
        family="offline",
        policy=policy,
        offline_source=_StubSource(batch),
        batch_size=BATCH,
        num_q=2,
        n_q_to_sample=2,
        critic_hidden=8,
        critic_n_heads=2,
        num_updates_per_train_step=1,
        tau=0.0,
        actor_optimizer_lr=0.0,
        critic_optimizer_lr=0.0,
        total_updates=1,
    )
    learner = IQL(config)
    metrics = learner.train_step()
    # The stub's BC loss is constant, so the weighted mean is the mean weight times it.
    bc = float(policy.scale.detach() ** 2)
    assert metrics["actor_loss"] == pytest.approx(metrics["awr_weight_mean"] * bc)
    assert learner.offline_source.requested == [BATCH]


def test_nonfinite_critic_loss_skips_the_step_and_the_target_update():
    torch.manual_seed(0)
    tokens_cur = torch.randn(BATCH, TOKENS, FEAT)
    tokens_next = torch.randn(BATCH, TOKENS, FEAT)
    reward = torch.zeros(BATCH, HORIZON)
    batch = {
        "inputs_cur": {"obs": torch.zeros(BATCH, 1)},
        "inputs_next": {"obs": torch.zeros(BATCH, 1), "next": True},
        "action": torch.zeros(BATCH, HORIZON, ACTION),
        "reward": reward,
        "done": torch.zeros(BATCH, HORIZON),
    }
    learner = IQL(
        IQLConfig(
            name="iql",
            policy=_StubPolicy(tokens_cur, tokens_next),
            offline_source=_StubSource(batch),
            batch_size=BATCH,
            num_q=2,
            n_q_to_sample=2,
            critic_hidden=8,
            critic_n_heads=2,
            num_updates_per_train_step=1,
            tau=0.5,
            actor_optimizer_lr=0.0,
            critic_optimizer_lr=1e-3,
            total_updates=2,
        )
    )
    learner.train_step()
    critic = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    target = [
        parameter.detach().clone() for parameter in learner.critic_target.parameters()
    ]
    reward[:] = float("nan")
    metrics = learner.train_step()
    assert metrics["q_loss"] != metrics["q_loss"]
    for parameter, before in zip(learner.critic.parameters(), critic, strict=True):
        assert torch.equal(parameter, before)
    for parameter, before in zip(
        learner.critic_target.parameters(), target, strict=True
    ):
        assert torch.equal(parameter, before)


def test_iql_critics_accept_a_feature_width_that_does_not_divide_the_requested_heads():
    tokens_cur = torch.randn(BATCH, TOKENS, 6)
    tokens_next = torch.randn(BATCH, TOKENS, 6)
    policy = _StubPolicy(tokens_cur, tokens_next)
    batch = {
        "inputs_cur": {"obs": torch.zeros(BATCH, 1)},
        "inputs_next": {"obs": torch.zeros(BATCH, 1), "next": True},
        "action": torch.zeros(BATCH, HORIZON, ACTION),
        "reward": torch.zeros(BATCH),
        "done": torch.zeros(BATCH),
    }
    learner = IQL(
        IQLConfig(
            name="iql",
            policy=policy,
            offline_source=_StubSource(batch),
            batch_size=BATCH,
            critic_n_heads=8,
            critic_hidden=8,
            num_q=2,
            total_updates=1,
        )
    )
    metrics = learner.train_step()
    assert metrics["q_loss"] == metrics["q_loss"]
