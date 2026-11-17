# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking wrappers over rollout and learner groups.

Controllers that overlap collection with training submit a call, keep working,
and wait for the result later. Every rollout group offers ``collect_async``:
a runtime implements it natively (RLInf's rollout group returns a future) or
inherits :class:`RolloutGroupBase`, which runs the blocking call on a helper
thread, so controller code is identical across runtimes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, Protocol, TypeVar

from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    RolloutRequest,
    RolloutResult,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerGroup,
    LearnerStepResult,
)

__all__ = [
    "AsyncCall",
    "RolloutGroupBase",
    "ThreadCall",
    "train_step_async",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class AsyncCall(Protocol[T_co]):
    """A submitted call whose result is awaited later.

    ``finished_at`` is the wall-clock time (``time.time()``) at which the work
    completed, stamped where it ran, so a controller that reads the result late
    still knows how long the work itself took; ``None`` until the result has
    been retrieved.
    """

    def result(self) -> T_co: ...

    def finished_at(self) -> float | None: ...


class ThreadCall(Generic[T]):
    """Run one blocking call on a dedicated thread and expose its result."""

    def __init__(self, function: Callable[[], T]) -> None:
        self._finished_at: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future[T] = self._executor.submit(self._run, function)

    def _run(self, function: Callable[[], T]) -> T:
        value = function()
        self._finished_at = time.time()
        return value

    def result(self) -> T:
        try:
            return self._future.result()
        finally:
            self._executor.shutdown(wait=False)

    def finished_at(self) -> float | None:
        return self._finished_at


class RolloutGroupBase:
    """Defaults for a :class:`~isaac_rl_posttraining.runtimes.base.RolloutGroup`.

    ``collect_async`` runs ``collect`` on a helper thread and ``set_weights``
    resolves the reference and sends the state through ``set_policy_state``;
    a runtime overrides either when it does better (a native future, weights
    fetched by reference on the workers).
    """

    def collect(self, requests: Sequence[RolloutRequest]) -> tuple[RolloutResult, ...]:
        raise NotImplementedError

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        raise NotImplementedError

    def collect_async(
        self, requests: Sequence[RolloutRequest]
    ) -> AsyncCall[tuple[RolloutResult, ...]]:
        """Start a collection without waiting for it."""
        return ThreadCall(lambda: self.collect(requests))

    def set_weights(self, weights: WeightsRef) -> tuple[int, ...]:
        """Send published weights to every worker; returns the versions they acknowledged."""
        return tuple(self.set_policy_state(weights.resolve()))


def train_step_async(
    group: LearnerGroup,
    batches: Sequence[LearnerBatch],
) -> AsyncCall[tuple[LearnerStepResult, ...]]:
    """Start a learner update on ``group`` without waiting for it."""
    native = getattr(group, "train_step_async", None)
    if callable(native):
        return native(batches)
    return ThreadCall(lambda: group.train_step(batches))
