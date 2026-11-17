# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drawing the next learner batch while the current one trains keeps the draws and their order."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from conftest import RecordingBuffer, ScriptedLearnerGroup, ScriptedRuntimeSource

from isaac_rl_posttraining.runtimes.learner import LearnerBatch
from isaac_rl_posttraining.training import OffPolicyConfig, OffPolicyTrainer
from isaac_rl_posttraining.training.prefetch import prefetched


def test_prefetched_yields_every_draw_in_order_and_draws_no_more():
    calls = []

    def draw():
        calls.append(len(calls))
        return calls[-1]

    assert list(prefetched(draw, 4)) == [0, 1, 2, 3]
    assert calls == [0, 1, 2, 3]
    assert list(prefetched(draw, 0)) == []
    assert len(calls) == 4


def test_prefetched_starts_the_next_draw_before_the_current_one_is_used():
    started = [threading.Event() for _ in range(3)]

    def draw():
        index = next(i for i, event in enumerate(started) if not event.is_set())
        started[index].set()
        return index

    for index in prefetched(draw, 3):
        if index + 1 < 3:
            assert started[index + 1].wait(timeout=10)
        assert not any(event.is_set() for event in started[index + 2 :])


def test_prefetched_raises_the_draw_error():
    def draw():
        raise RuntimeError("replay is empty")

    with pytest.raises(RuntimeError, match="replay is empty"):
        list(prefetched(draw, 2))


def test_the_trainer_trains_on_the_same_draws_in_the_same_order():
    draws: list[int] = []

    def draw() -> LearnerBatch:
        draws.append(len(draws))
        return LearnerBatch(
            tensors={"0/replay/reward": np.full(2, float(draws[-1]), np.float32)},
            samples=2,
        )

    source = ScriptedRuntimeSource(
        [
            {"steps": 8, "episodes": 1, "decisions": 3},
            {"steps": 8, "episodes": 1, "decisions": 2},
        ],
        action_exec_len=8,
    )
    learner = ScriptedLearnerGroup()
    trained: list[float] = []
    original = learner.train_step

    def train_step(batches):
        trained.append(float(batches[0].tensors["0/replay/reward"][0]))
        return original(batches)

    learner.train_step = train_step
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(),
        draw_batch=draw,
        evaluator=None,
        config=OffPolicyConfig(num_rollouts=16, learning_starts=0),
    )
    trainer.run()
    assert trained == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert draws == [0, 1, 2, 3, 4]
