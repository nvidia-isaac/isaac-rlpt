# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The handle of one training run: its plan, its loop, and step-wise control.

:func:`~isaac_rl_posttraining.launch` returns a :class:`Run`. ``run.plan`` is
what the dry run prints (resolved placement and data specs). ``run.loop()``
runs the configured schedule to the budget, exactly as ``scripts/train.py``
does. The step methods (``collect``, ``train``, ``evaluate``, ``publish``,
``save``, ``load``) drive the same pieces one call at a time, for every
built-in algorithm, on the runtime the loop would use. An off-policy step collects rounds into
the replay buffer and trains any number of updates on it; an on-policy step is
one iteration: ``collect()`` plays its episodes, ``train()`` updates on them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from isaac_rl_posttraining.runners.build import (
    TrainPlan,
    build_run,
    open_session,
    run_config,
    train_plan,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from isaac_rl_posttraining.buffers import ReplayBuffer
    from isaac_rl_posttraining.data.spec import SpecComparison
    from isaac_rl_posttraining.runners.launchers import StepSession
    from isaac_rl_posttraining.runtimes import PolicyState
    from isaac_rl_posttraining.runtimes.placement import PlacementPlan

__all__ = ["Run", "RunTelemetry"]


@dataclass(frozen=True)
class RunTelemetry:
    """The telemetry of one run: the ``transitions``, ``episodes``, ``rounds`` and ``learner`` tables under ``<run_dir>/telemetry``."""

    run_dir: str

    def table(self, name: str) -> pa.Table:
        """One telemetry table as an Arrow table (``.to_pandas()`` for a DataFrame)."""
        from isaac_rl_posttraining.training.telemetry import read_telemetry

        return read_telemetry(self.run_dir, name)

    def dashboard(self, path: str, *, compare: Sequence[str] = ()) -> str:
        """Write the HTML dashboard of this run (overlaid with the run directories in ``compare``); returns its path."""
        from isaac_rl_posttraining.viz import write_dashboard

        return str(write_dashboard([self.run_dir, *compare], path))


class Run:
    """One training run, planned from a composed config and not yet started.

    Use it as a context manager (or call :meth:`close`) when driving it step
    by step, so rollout workers and evaluators are released.
    """

    def __init__(self, cfg: DictConfig, *, dry_run: bool = False) -> None:
        self._composed = cfg
        self.plan: TrainPlan = train_plan(cfg)
        self.config: DictConfig = run_config(cfg)
        self.dry_run = dry_run
        self._state = "planned"
        self._session: StepSession | None = None
        #: The run's output directory, once it is known (the run has started).
        self.run_dir: str | None = None

    # -- what the run will do -------------------------------------------------
    @property
    def placement(self) -> PlacementPlan:
        """The resolved placement (which role runs on which GPUs)."""
        return self.plan.placement_plan

    @property
    def data_specs(self) -> SpecComparison:
        """What the plan compared between the simulator's and the policy's data specs."""
        return self.plan.spec_report.comparison

    def __repr__(self) -> str:
        return f"Run(algorithm={self.plan.algorithm!r}, runtime={self.plan.runtime!r}, state={self._state!r})"

    # -- the whole loop -------------------------------------------------------
    def loop(self) -> None:
        """Run the configured schedule to the budget (what ``scripts/train.py`` runs)."""
        self._require_runnable("loop")
        if self._state != "planned":
            raise RuntimeError(f"loop() runs a fresh run; this one is {self._state}")
        self._state = "looping"
        try:
            self.run_dir = build_run(self._composed)()
        finally:
            self._state = "finished"

    # -- step by step ---------------------------------------------------------
    def collect(self, rounds: int = 1) -> Mapping[str, float]:
        """Collect rollouts with the published weights; returns the last round's metrics.

        Off-policy: ``rounds`` rounds into the replay buffer. On-policy: the
        next iteration's episodes (``rounds=1``), which :meth:`train` updates on.
        """
        return self._open("collect").collect(rounds)

    def train(self, steps: int = 1) -> Mapping[str, float]:
        """Update the learner, then publish the new weights; returns the last update's metrics.

        Off-policy: ``steps`` updates on replay draws, metrics averaged over
        learner ranks (what the run logs under ``learner/``). On-policy: the
        algorithm's update on the last collected iteration (``steps=1``),
        with every metric the run logs for that iteration.
        """
        return self._open("train").train(steps)

    def evaluate(self) -> Mapping[str, float]:
        """Score the latest published weights on the evaluation workers."""
        return self._open("evaluate").evaluate()

    def publish(self) -> PolicyState:
        """Send the learner's current weights to the rollout workers."""
        return self._open("publish").publish()

    def save(self, path: str) -> None:
        """Write the learner's checkpoint to directory ``path``."""
        self._open("save").save(path)

    def load(self, path: str) -> None:
        """Load a learner checkpoint from directory ``path`` and publish it."""
        self._open("load").load(path)

    @property
    def buffer(self) -> ReplayBuffer:
        """The replay buffer the rollouts fill (off-policy algorithms)."""
        return self._open("buffer").buffer

    @property
    def telemetry(self) -> RunTelemetry:
        """The run's telemetry tables (rows written so far; the run must have started)."""
        if self._session is not None and self._session.telemetry is not None:
            self._session.telemetry.flush()
        if self.run_dir is None:
            raise RuntimeError("the run has written no telemetry yet; drive it with loop() or a step method")
        return RunTelemetry(self.run_dir)

    def close(self) -> None:
        """Release rollout workers, evaluators, the learner and the logger (idempotent)."""
        session, self._session = self._session, None
        if self._state == "stepping":
            self._state = "closed"
        if session is not None:
            session.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals ------------------------------------------------------------
    def _require_runnable(self, method: str) -> None:
        if self.dry_run:
            raise RuntimeError(
                f"Run.{method}() on a dry run; launch(..., dry_run=False) to run it"
            )

    def _open(self, method: str) -> StepSession:
        self._require_runnable(method)
        if self._session is not None:
            return self._session
        if self._state != "planned":
            raise RuntimeError(
                f"Run.{method}() needs an open run; this one is {self._state}"
            )
        self._state = "stepping"
        try:
            self._session = open_session(self._composed)
        except BaseException:
            self._state = "closed"
            raise
        self.run_dir = self._session.run_dir
        return self._session
