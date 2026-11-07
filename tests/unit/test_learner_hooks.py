# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed learner hooks: local and RLInf groups, defaults for hooks a worker leaves out, and publishing."""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef
from isaac_rl_posttraining.runtimes.learner import (
    LearnerHooks,
    LocalLearnerGroup,
    LocalLearnerHooks,
    run_hook,
)
from isaac_rl_posttraining.runtimes.rlinf.learner import RLInfLearnerGroup


class _Worker:
    def __init__(self, context, modules) -> None:
        modules.prepare(nn.Linear(2, 2), name="actor", publish=True)
        self.rewards: list[tuple] = []
        self.target: int | None = None

    def train_step(self, batch):
        return {}

    def checkpoint_state(self):
        return {}

    def load_checkpoint_state(self, state) -> None:
        return None

    def close(self) -> None:
        return None

    def gradient_steps(self):
        return 7.0

    def target_update_count(self, env_steps):
        return self.target if self.target is None else self.target + env_steps

    def observe_rewards(self, rewards, dones, truncateds, episode_ids):
        self.rewards.append((rewards, dones, truncateds, episode_ids))

    def on_evaluation(self, success_rate):
        return {} if success_rate is None else {"best": success_rate}


def test_local_hooks_forward_to_the_worker():
    group = LocalLearnerGroup(_Worker, device="cpu")
    hooks = group.hooks()
    assert isinstance(hooks, LocalLearnerHooks) and isinstance(hooks, LearnerHooks)
    worker = group._worker
    assert hooks.gradient_steps() == 7 and isinstance(hooks.gradient_steps(), int)
    assert hooks.target_update_count(10) is None
    worker.target = 5
    assert hooks.target_update_count(10) == 15
    hooks.observe_rewards([1.0], [False], [True], [4])
    assert worker.rewards == [([1.0], [False], [True], [4])]
    assert hooks.on_evaluation(None) == {}
    assert hooks.on_evaluation(0.5) == {"best": 0.5}
    group.close()


def test_hooks_left_out_by_a_worker_fall_back_to_defaults():
    class _Minimal:
        def gradient_steps(self):
            return 3

    worker = _Minimal()
    assert run_hook(worker, "gradient_steps") == 3
    assert run_hook(worker, "target_update_count", 10) is None
    assert run_hook(worker, "observe_rewards", [1.0], [False], [False], [0]) is None
    assert run_hook(worker, "on_evaluation", 0.5) == {}
    with pytest.raises(ValueError, match="unknown learner hook"):
        run_hook(worker, "train_step")
    with pytest.raises(TypeError, match="gradient_steps"):
        run_hook(object(), "gradient_steps")


class _Handle:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


class _FakeRLInfWorkers:
    """RLInf's worker-group surface: ``execute_on(rank)`` for one rank, the group itself for all."""

    def __init__(self, num_ranks: int) -> None:
        self.num_ranks = num_ranks
        self.calls: list[tuple[str, tuple[int, ...], tuple]] = []

    def _answer(self, rank, name, args):
        if name == "gradient_steps":
            return 10 + rank
        if name == "target_update_count":
            return args[0] + rank
        if name == "on_evaluation":
            return {"rank": float(rank)}
        return None

    def execute_on(self, rank):
        workers = self

        class _Rank:
            def run_hook(self, name, *args):
                workers.calls.append((name, (rank,), args))
                return _Handle([workers._answer(rank, name, args)])

        return _Rank()

    def run_hook(self, name, *args):
        ranks = tuple(range(self.num_ranks))
        self.calls.append((name, ranks, args))
        return _Handle([self._answer(rank, name, args) for rank in ranks])

    def policy_state(self, version):
        return _Handle(
            [
                PolicyState(version=version, tensors={"w": np.ones(2)}),
                *([None] * (self.num_ranks - 1)),
            ]
        )


def test_rlinf_hooks_query_rank_zero_and_notify_every_rank():
    workers = _FakeRLInfWorkers(num_ranks=3)
    group = RLInfLearnerGroup(workers, num_learner_ranks=3)
    hooks = group.hooks()
    assert isinstance(hooks, LearnerHooks)

    assert hooks.gradient_steps() == 10
    assert hooks.target_update_count(5) == 5
    hooks.observe_rewards([1.0, 0.0], [False, True], [False, False], [0, 0])
    assert hooks.on_evaluation(0.25) == {"rank": 0.0}
    assert workers.calls == [
        ("gradient_steps", (0,), ()),
        ("target_update_count", (0,), (5,)),
        (
            "observe_rewards",
            (0, 1, 2),
            ([1.0, 0.0], [False, True], [False, False], [0, 0]),
        ),
        ("on_evaluation", (0, 1, 2), (0.25,)),
    ]


def test_publish_returns_the_weights_inline():
    group = LocalLearnerGroup(_Worker, device="cpu")
    weights = group.publish(1)
    assert isinstance(weights, WeightsRef) and weights.version == 1
    assert set(weights.resolve().tensors) == {"weight", "bias"}
    with pytest.raises(ValueError, match="advance"):
        group.publish(1)
    group.close()

    rlinf = RLInfLearnerGroup(_FakeRLInfWorkers(num_ranks=2), num_learner_ranks=2)
    assert rlinf.publish(4).resolve().version == 4
