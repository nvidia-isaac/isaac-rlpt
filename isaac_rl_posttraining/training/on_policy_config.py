# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The on-policy loop's configuration and iteration bookkeeping.

:class:`OnPolicyConfig` is the iteration, evaluation and checkpoint schedule;
:class:`OnPolicyAlgorithm` the algorithm's half of the loop; the helpers name
iteration checkpoints and derive each iteration's reset seed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from isaac_rl_posttraining.runtimes.base import SEED_MODULUS
from isaac_rl_posttraining.runtimes.learner import LearnerBatch
from isaac_rl_posttraining.training.schedule import (
    ScheduleSpec,
    as_schedule_spec,
    on_policy_schedule,
)
from isaac_rl_posttraining.training.trajectories import ChunkTrajectory, DiscountPer

__all__ = [
    "ITERATION_CHECKPOINT_PREFIX",
    "OnPolicyAlgorithm",
    "OnPolicyConfig",
    "checkpoint_iteration",
    "checkpoint_manifest",
    "training_reset_seed",
]

#: On-policy checkpoints are tagged by iteration: ``iteration_<n>`` directories.
ITERATION_CHECKPOINT_PREFIX = "iteration_"


class OnPolicyAlgorithm(Protocol):
    """The algorithm-specific half of the on-policy loop.

    The loop collects episodes and publishes weights; the algorithm decides how
    a chunk's rewards are discounted (``gamma`` per ``discount_per``) and how
    trajectories become the one learner batch each rank trains on. Its learner
    worker runs on the learner group the loop was given.
    """

    @property
    def gamma(self) -> float: ...

    @property
    def discount_per(self) -> DiscountPer: ...

    def build_batches(
        self,
        trajectories: Sequence[ChunkTrajectory],
        *,
        num_learner_ranks: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[LearnerBatch, ...], Mapping[str, float]]:
        """One learner batch per rank, plus batch statistics to log."""
        ...

@dataclass(frozen=True)
class OnPolicyConfig:
    """Iteration, evaluation, and checkpoint schedule of the on-policy loop.

    Attributes:
        iterations: PPO iterations to run (a resumed run continues to this total).
        episodes_per_environment: Episodes every environment plays per iteration.
        eval_episodes_per_environment: Episodes per environment in one
            deterministic evaluation; ``0`` disables evaluation.
        eval_every: Evaluate after every this many iterations (and before
            training starts).
        checkpoint_every: Save a checkpoint (learner state, policy snapshot,
            manifest) every this many iterations through the trainer's
            :class:`CheckpointStore`; ``0`` disables periodic checkpoints.
        seed: Seed of the batch shuffling and of the training reset seeds.
        eval_seed: Reset seed shared by every evaluation, so evaluations replay
            the same initial states.
        schedule: ``sync`` (collect, train, publish), ``overlapped`` (collect
            the next iteration's rollouts while the learner updates on the
            current one, so an iteration costs about ``max(collect, train)``;
            training data then comes from the policy one update behind,
            reported as ``system/policy_lag``) or ``async`` (re-submit the
            collection as soon as one returns, publish after every update,
            or every ``publish_every`` updates when set; data lags at most one
            publish).
            Evaluations drain the pipeline and always score the newest policy.
    """

    iterations: int
    episodes_per_environment: int
    eval_episodes_per_environment: int = 0
    eval_every: int = 10
    checkpoint_every: int = 0
    seed: int = 0
    eval_seed: int = 100_000
    schedule: ScheduleSpec | str = "sync"

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", on_policy_schedule(as_schedule_spec(self.schedule)))
        for name in ("iterations", "episodes_per_environment"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "eval_episodes_per_environment",
            "eval_every",
            "checkpoint_every",
            "seed",
            "eval_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.eval_episodes_per_environment and self.eval_every < 1:
            raise ValueError("eval_every must be positive when evaluation is enabled")

    @property
    def schedule_spec(self) -> ScheduleSpec:
        """``schedule`` as a :class:`ScheduleSpec` (resolved at construction)."""
        assert isinstance(self.schedule, ScheduleSpec)
        return self.schedule


def checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    """The ``manifest.json`` a checkpoint was written with, or ``{}`` when it has none."""
    path = Path(directory) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def checkpoint_iteration(directory: str | Path) -> int:
    """The iteration a checkpoint holds: its ``iteration_<n>`` name, else its manifest's ``iteration``."""
    name = Path(directory).name
    digits = name[len(ITERATION_CHECKPOINT_PREFIX) :]
    if name.startswith(ITERATION_CHECKPOINT_PREFIX) and digits.isdigit():
        return int(digits)
    iteration = checkpoint_manifest(directory).get("iteration")
    if iteration is None:
        raise ValueError(
            f"expected an iteration_<n> checkpoint directory or one whose manifest.json records "
            f"its iteration, got {str(directory)!r}"
        )
    return int(iteration)


def training_reset_seed(seed: int, iteration: int) -> int:
    """The reset seed of training iteration ``iteration`` (1-based).

    Iterations are spread ~10M seeds apart so their episodes never share a seed; the sum
    wraps at 2**32 because environments accept 32-bit seeds only (a 500-iteration run
    crosses that limit at iteration 430 otherwise).
    """
    return (seed + (iteration - 1) * 10_000_019) % SEED_MODULUS
