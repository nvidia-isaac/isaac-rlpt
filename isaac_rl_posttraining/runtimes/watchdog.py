# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A time limit on rollout collections, so a frozen simulator ends the run instead of hanging it.

A collection that has not finished ``runtime.rollout_timeout_seconds`` after it
started, for each episode its workers were asked for, raises :class:`RolloutTimeoutError` (a
:class:`~isaac_rl_posttraining.runtimes.base.RolloutInfrastructureError`)
naming the rollout worker and its device; the launcher then writes the
``aborted`` checkpoint and the run ends. Multi-process runtimes bound their
wait on the worker processes. In one process the collecting thread may be
stuck inside the simulator, where no exception reaches it: the watchdog then
runs the handlers registered with :func:`on_frozen_rollout` (the launchers
write the aborted checkpoint there) and exits the process.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

from loguru import logger

from isaac_rl_posttraining.runtimes.base import (
    RolloutInfrastructureError,
    RolloutRequest,
)

__all__ = [
    "DEFAULT_ROLLOUT_TIMEOUT_SECONDS",
    "FROZEN_ROLLOUT_EXIT_CODE",
    "CollectionWatchdog",
    "RolloutTimeoutError",
    "collection_timeout",
    "on_frozen_rollout",
    "rollout_timeout_message",
]

T = TypeVar("T")

#: Default ``runtime.rollout_timeout_seconds``: far above any healthy round (a 448-step
#: Isaac Lab episode of 16 scenes takes about two minutes), far below a night lost to a hang.
DEFAULT_ROLLOUT_TIMEOUT_SECONDS = 1800.0
#: Longest time a timed-out run gets to unwind before the process exits.
EXIT_GRACE_SECONDS = 120.0
#: Exit status of a process ended by a frozen collection.
FROZEN_ROLLOUT_EXIT_CODE = 3


class RolloutTimeoutError(RolloutInfrastructureError):
    """A rollout collection did not finish within ``runtime.rollout_timeout_seconds``."""


class _Interrupted(BaseException):
    """Raised asynchronously in a collecting thread whose collection ran out of time."""


def rollout_timeout_message(what: str, late: str, timeout_seconds: float) -> str:
    return (
        f"{what}: no result within {timeout_seconds:g} s from {late}; its simulator may be frozen "
        "(runtime.rollout_timeout_seconds sets this limit)"
    )


def collection_timeout(
    timeout_seconds: float | None, requests: Sequence[RolloutRequest]
) -> float | None:
    """The limit on one collection: ``timeout_seconds`` for each episode a worker must finish."""
    if timeout_seconds is None:
        return None
    episodes = max(
        (request.episodes_per_environment or 1 for request in requests), default=1
    )
    return timeout_seconds * episodes


_handlers: list[Callable[[BaseException], None]] = []
_handlers_lock = threading.Lock()


@contextmanager
def on_frozen_rollout(handler: Callable[[BaseException], None]) -> Iterator[None]:
    """Run ``handler(error)`` before the process exits over a frozen collection (while this block runs)."""
    with _handlers_lock:
        _handlers.append(handler)
    try:
        yield
    finally:
        with _handlers_lock:
            _handlers.remove(handler)


def _run_handlers(error: BaseException) -> None:
    with _handlers_lock:
        handlers = list(_handlers)
    for handler in reversed(handlers):
        try:
            handler(error)
        except Exception as exc:  # noqa: BLE001 - the process exits either way
            logger.error(f"frozen-rollout handler failed: {exc!r}")


class CollectionWatchdog:
    """Runs collections on the calling thread and acts when one exceeds ``timeout_seconds`` (``None``: no limit).

    On time-out the collecting thread receives :class:`RolloutTimeoutError`
    as soon as it runs Python code again. If it is still inside the collection
    ``grace_seconds`` later (stuck in native code), the registered handlers
    run and the process exits with :data:`FROZEN_ROLLOUT_EXIT_CODE`.
    """

    def __init__(
        self,
        timeout_seconds: float | None,
        *,
        grace_seconds: float | None = None,
        exit_process: Callable[[int], object] = os._exit,
    ) -> None:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool) or not timeout_seconds > 0
        ):
            raise ValueError(
                "rollout_timeout_seconds must be a positive number or None"
            )
        self.timeout_seconds = timeout_seconds
        self.grace_seconds = (
            min(EXIT_GRACE_SECONDS, timeout_seconds or EXIT_GRACE_SECONDS)
            if grace_seconds is None
            else grace_seconds
        )
        self._exit_process = exit_process

    def run(
        self,
        collect: Callable[[], T],
        *,
        describe: Callable[[], str],
        what: str,
        requests: Sequence[RolloutRequest] = (),
    ) -> T:
        """``collect()``; ``describe()`` names the workers still collecting when the time runs out."""
        timeout = collection_timeout(self.timeout_seconds, requests)
        if timeout is None:
            return collect()
        watch = _Watch(threading.get_ident())
        thread = threading.Thread(
            target=self._watch,
            args=(watch, describe, what, timeout),
            name="rollout-watchdog",
            daemon=True,
        )
        thread.start()
        try:
            try:
                return collect()
            finally:
                watch.finish()
        except _Interrupted:
            assert watch.error is not None
            raise watch.error from None

    def _watch(
        self, watch: _Watch, describe: Callable[[], str], what: str, timeout: float
    ) -> None:
        if watch.finished.wait(timeout):
            return
        error = RolloutTimeoutError(rollout_timeout_message(what, describe(), timeout))
        if not watch.interrupt(error):
            return
        logger.error(str(error))
        if watch.finished.wait(self.grace_seconds):
            return
        logger.error(
            f"the collecting thread is still stuck {self.grace_seconds:g} s later; writing what "
            f"can be saved and exiting with status {FROZEN_ROLLOUT_EXIT_CODE}"
        )
        _run_handlers(error)
        sys.stdout.flush()
        sys.stderr.flush()
        self._exit_process(FROZEN_ROLLOUT_EXIT_CODE)


class _Watch:
    """One watched collection: whether it finished, and the error it was interrupted with."""

    def __init__(self, thread_id: int) -> None:
        self.thread_id = thread_id
        self.finished = threading.Event()
        self.error: RolloutTimeoutError | None = None
        self._lock = threading.Lock()

    def finish(self) -> None:
        with self._lock:
            self.finished.set()

    def interrupt(self, error: RolloutTimeoutError) -> bool:
        """Raise :class:`_Interrupted` in the collecting thread unless the collection already finished."""
        with self._lock:
            if self.finished.is_set():
                return False
            self.error = error
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self.thread_id), ctypes.py_object(_Interrupted)
            )
            return True
