# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bucketed gradient averaging matches one all-reduce per tensor, with one all-reduce per backward."""

from __future__ import annotations

import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from isaac_rl_posttraining.runtimes import gradient_averaging
from isaac_rl_posttraining.runtimes.learner import average_gradients_across_ranks

UPDATES = 6


def _per_tensor_average(parameters) -> None:
    """The previous scheme: a blocking all-reduce per tensor as its gradient is accumulated."""

    def average(parameter) -> None:
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(dist.get_world_size())

    for parameter in parameters:
        parameter.register_post_accumulate_grad_hook(average)


class _SacLike:
    """An actor, a critic and a learned temperature, updated like SAC (clipping included)."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(5, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
        self.log_ent_coef = torch.zeros(1, requires_grad=True)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        self.ent_optimizer = torch.optim.Adam([self.log_ent_coef], lr=1e-2)

    def parameters(self):
        return (*self.actor.parameters(), *self.critic.parameters(), self.log_ent_coef)

    def update(self, obs: torch.Tensor, target: torch.Tensor) -> None:
        action = self.actor(obs)
        q = self.critic(torch.cat([obs, action.detach()], dim=1))
        critic_loss = ((q - target) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        log_prob = -(action**2).sum(dim=1)
        ent_loss = -(self.log_ent_coef * (log_prob + 3.0).detach()).mean()
        self.ent_optimizer.zero_grad()
        ent_loss.backward()
        self.ent_optimizer.step()

        # The actor loss reaches the critic's parameters too, as in SAC.
        q_pi = self.critic(torch.cat([obs, self.actor(obs)], dim=1)).min(dim=1).values
        actor_loss = (
            self.log_ent_coef.exp().detach() * log_prob.detach() - q_pi
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()


def _train_on_rank(rank: int, directory: str) -> None:
    init_method = f"file://{os.path.join(directory, 'rendezvous')}"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    reductions: list[int] = []
    all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *args, **kwargs):
        reductions.append(tensor.numel())
        return all_reduce(tensor, *args, **kwargs)

    try:
        bucketed, per_tensor = _SacLike(), _SacLike()
        average_gradients_across_ranks(bucketed.parameters())
        _per_tensor_average(per_tensor.parameters())
        generator = torch.Generator().manual_seed(100 + rank)
        counts = {"bucketed": [], "per_tensor": []}
        dist.all_reduce = counting_all_reduce
        for _ in range(UPDATES):
            obs = torch.randn(16, 5, generator=generator)
            target = torch.randn(16, 2, generator=generator)
            for name, learner in (("bucketed", bucketed), ("per_tensor", per_tensor)):
                reductions.clear()
                learner.update(obs, target)
                counts[name].append(len(reductions))
        dist.all_reduce = all_reduce
        identical = all(
            torch.equal(a, b) and torch.equal(a.grad, b.grad)
            for a, b in zip(bucketed.parameters(), per_tensor.parameters(), strict=True)
        )
        state = [p.detach().flatten().tolist() for p in bucketed.parameters()]
        with open(os.path.join(directory, f"rank_{rank}.json"), "w") as stream:
            json.dump(
                {"identical": identical, "counts": counts, "state": state}, stream
            )
    finally:
        dist.all_reduce = all_reduce
        dist.destroy_process_group()


def test_bucketed_averaging_is_bit_identical_to_per_tensor_and_needs_one_all_reduce_per_backward():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_train_on_rank, args=(directory,), nprocs=2, join=True)
        results = []
        for rank in range(2):
            with open(os.path.join(directory, f"rank_{rank}.json")) as stream:
                results.append(json.load(stream))
    for result in results:
        assert result["identical"]
        # Three backwards per update: critic, temperature, actor (which also reaches the critic).
        assert result["counts"]["bucketed"] == [3] * UPDATES
        assert result["counts"]["per_tensor"] == [4 + 1 + 8] * UPDATES
    assert results[0]["state"] == results[1]["state"]


def test_buckets_split_by_dtype_and_size(monkeypatch):
    monkeypatch.setattr(gradient_averaging, "BUCKET_BYTES", 24)
    grads = [
        torch.zeros(2),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(3),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(4),
    ]
    buckets = gradient_averaging._buckets(grads)
    assert [[tuple(g.shape) for g in bucket] for bucket in buckets] == [
        [(2,), (3,)],
        [(4,)],
        [(2,), (1,)],
    ]
    assert [bucket[0].dtype for bucket in buckets] == [
        torch.float32,
        torch.float32,
        torch.float64,
    ]
