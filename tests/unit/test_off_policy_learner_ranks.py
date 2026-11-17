# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Off-policy learning on several ranks: one draw cut into rank slices, gradients averaged across ranks."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
)

from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerStepResult,
    average_gradients_across_ranks,
)
from isaac_rl_posttraining.training import OffPolicyConfig, OffPolicyTrainer


def _draw() -> LearnerBatch:
    return LearnerBatch(
        tensors={
            "0/replay/obs": {"state": np.arange(8, dtype=np.float32).reshape(4, 2)},
            "0/replay/reward": np.arange(4, dtype=np.float32),
            "0/expert/reward": np.arange(2, dtype=np.float32) + 10,
        },
        samples=6,
    )


def test_split_cuts_every_tensor_along_the_sample_axis():
    first, second = _draw().split(2)
    assert first.samples == second.samples == 3
    assert first.tensors["0/replay/reward"].tolist() == [0.0, 1.0]
    assert second.tensors["0/replay/reward"].tolist() == [2.0, 3.0]
    assert second.tensors["0/replay/obs"]["state"].tolist() == [[4.0, 5.0], [6.0, 7.0]]
    assert (
        first.tensors["0/expert/reward"].tolist(),
        second.tensors["0/expert/reward"].tolist(),
    ) == (
        [10.0],
        [11.0],
    )
    batch = _draw()
    assert batch.split(1) == (batch,)
    with pytest.raises(ValueError, match="does not split evenly"):
        batch.split(4)


def test_env_steps_stamp_keeps_the_arrays():
    batch = _draw()
    stamped = batch.at_env_steps(120)
    assert stamped.env_steps == 120 and batch.env_steps is None
    assert stamped.tensors is batch.tensors
    assert all(part.env_steps == 120 for part in stamped.split(2))
    with pytest.raises(ValueError, match="non-negative"):
        batch.at_env_steps(-1)


class _TwoRankLearner(ScriptedLearnerGroup):
    num_learner_ranks = 2

    def __init__(self):
        super().__init__()
        self.rank_batches: list[tuple[LearnerBatch, ...]] = []

    def train_step(self, batches):
        self.rank_batches.append(tuple(batches))
        self.train_step_calls.append(None)
        return tuple(
            LearnerStepResult(
                learner_rank=rank,
                step=len(self.train_step_calls),
                samples=batch.samples,
                metrics={"loss": rank},
            )
            for rank, batch in enumerate(batches)
        )


def test_the_trainer_sends_each_rank_its_slice_of_one_draw():
    source = ScriptedRuntimeSource(
        [{"steps": 8, "episodes": 1, "decisions": 2}], action_exec_len=8
    )
    learner = _TwoRankLearner()
    records: list[dict] = []
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(),
        draw_batch=_draw,
        evaluator=None,
        config=OffPolicyConfig(num_rollouts=8, learning_starts=0),
        metric_sink=lambda metrics, step: records.append(dict(metrics)),
    )
    trainer.run()

    assert len(learner.rank_batches) == 2
    for batches in learner.rank_batches:
        assert len(batches) == 2 and all(batch.env_steps == 8 for batch in batches)
        joined = np.concatenate([batch.tensors["0/replay/reward"] for batch in batches])
        assert joined.tolist() == [0.0, 1.0, 2.0, 3.0]
    learner_records = [metrics for metrics in records if "learner/loss" in metrics]
    assert learner_records[-1]["learner/loss"] == 0.5  # the mean over ranks


def _average_on_rank(rank: int, directory: str) -> None:
    init_method = f"file://{os.path.join(directory, 'rendezvous')}"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        layer = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        temperature = torch.zeros((), requires_grad=True)
        average_gradients_across_ranks((*layer.parameters(), temperature))
        inputs = torch.full((1, 2), float(rank + 1))
        (layer(inputs).sum() + temperature * (rank + 1)).backward()
        with open(os.path.join(directory, f"rank_{rank}.json"), "w") as stream:
            json.dump([layer.weight.grad.tolist(), float(temperature.grad)], stream)
    finally:
        dist.destroy_process_group()


def test_gradients_are_averaged_across_ranks():
    """Two CPU ranks with different data end with the same, averaged gradients."""
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_average_on_rank, args=(directory,), nprocs=2, join=True)
        gathered = []
        for rank in range(2):
            with open(os.path.join(directory, f"rank_{rank}.json")) as stream:
                gathered.append(json.load(stream))
    assert gathered[0] == gathered[1] == [[[1.5, 1.5]], 1.5]


def _average_twice_on_rank(rank: int, directory: str) -> None:
    init_method = f"file://{os.path.join(directory, 'rendezvous')}"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    reductions = []
    all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *args, **kwargs):
        reductions.append(tuple(tensor.shape))
        return all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = counting_all_reduce
    try:
        layer = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        # A module manager and the learner both ask for the same parameter.
        average_gradients_across_ranks(tuple(layer.parameters()))
        average_gradients_across_ranks(tuple(layer.parameters()))
        layer(torch.full((1, 2), float(rank + 1))).sum().backward()
        with open(os.path.join(directory, f"rank_{rank}.json"), "w") as stream:
            json.dump([layer.weight.grad.tolist(), len(reductions)], stream)
    finally:
        dist.all_reduce = all_reduce
        dist.destroy_process_group()


def test_a_parameter_is_averaged_once_however_many_callers_ask():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_average_twice_on_rank, args=(directory,), nprocs=2, join=True)
        gathered = []
        for rank in range(2):
            with open(os.path.join(directory, f"rank_{rank}.json")) as stream:
                gathered.append(json.load(stream))
    assert gathered[0] == gathered[1] == [[[1.5, 1.5]], 1]


def test_averaging_is_a_no_op_on_one_process():
    layer = torch.nn.Linear(2, 1)
    average_gradients_across_ranks(tuple(layer.parameters()))
    layer(torch.ones(1, 2)).sum().backward()
    assert layer.weight.grad.tolist() == [[1.0, 1.0]]
