# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy RLInf imports, cluster lifetime, and accelerator ownership."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from isaac_rl_posttraining.runtimes.ray_driver import keep_driver_interpreter
from isaac_rl_posttraining.runtimes.rlinf.config import RLInfRuntimeConfig
from isaac_rl_posttraining.utils.compat import add_note

RLINF_VERSION = "0.3.0"


def load_scheduler() -> Any:
    """Import the validated RLInf scheduler only when the runtime is opened."""
    try:
        installed = version("rlinf")
    except PackageNotFoundError as exc:
        raise ImportError(
            "RLInfRuntime requires the optional dependency; install "
            "isaac-rl-posttraining[rlinf] in a dedicated environment"
        ) from exc
    if installed != RLINF_VERSION:
        raise ImportError(
            f"RLInfRuntime requires rlinf=={RLINF_VERSION}, found {installed}"
        )
    return importlib.import_module("rlinf.scheduler")


def rlinf_worker_class(
    name: str, methods: type, worker_base: type, **attributes: Any
) -> Any:
    """Subclass RLInf's ``worker_base`` with the methods of ``methods`` and class ``attributes``.

    The methods, inherited ones too, are copied into the class body so RLInf's
    worker metaclass wraps them as it wraps methods written there.
    """
    namespace = {
        key: value
        for klass in reversed(methods.__mro__[:-1])
        for key, value in vars(klass).items()
        if callable(value)
    }
    namespace.update(attributes, __module__=methods.__module__)
    return type(worker_base)(name, (methods, worker_base), namespace)


def terminate_worker_group(
    group: Any, *, close_delegates: bool = True
) -> Exception | None:
    """Close delegates and kill every Ray actor, retaining the first error.

    ``close_delegates=False`` only kills them (workers stuck in a frozen collection never answer).

    Calling ``WorkerGroup.close().wait()`` is unsafe for library integration:
    RLInf's result waiter sends ``SIGUSR1`` to the controller whenever an actor
    has already died.  Invoke actor methods directly so a failed worker remains
    an ordinary Python exception that the caller can report and recover from.
    """
    ray = importlib.import_module("ray")
    actor_missing_error = getattr(
        getattr(ray, "exceptions", None),
        "ActorHandleNotFoundError",
        (),
    )
    workers = tuple(group.worker_info_list)
    error = (
        _close_delegates(ray, workers, actor_missing_error) if close_delegates else None
    )
    for worker_info in workers:
        try:
            ray.kill(worker_info.worker)
        except Exception as exc:  # noqa: BLE001
            if actor_missing_error and isinstance(exc, actor_missing_error):
                continue
            error = _first_error(error, exc, "RLInf actor cleanup")
    return error


def _close_delegates(
    ray: Any, workers: Sequence[Any], actor_missing_error: Any
) -> Exception | None:
    """Close every worker's delegate and wait for them; return the first failure."""
    error: Exception | None = None
    close_refs = []
    for worker_info in workers:
        close_method = getattr(worker_info.worker, "close", None)
        if close_method is not None:
            try:
                close_refs.append(close_method.remote())
            except Exception as exc:  # noqa: BLE001
                error = error or exc
    if close_refs:
        try:
            ray.get(close_refs)
        except Exception as exc:  # noqa: BLE001
            if not (actor_missing_error and isinstance(exc, actor_missing_error)):
                error = _first_error(error, exc, "RLInf delegate cleanup")
    return error


def _first_error(error: Exception | None, exc: Exception, step: str) -> Exception:
    """Keep the first cleanup failure; a later one becomes a note on it."""
    if error is None:
        return exc
    add_note(error, f"{step} also failed: {exc!r}")
    return error


def terminate_channel(channel: Any) -> Exception | None:
    """Release the actor group owned by an RLInf 0.3 channel."""
    channel_group = getattr(channel, "_channel_worker_group", None)
    if channel_group is None:
        return None
    try:
        channel_group._close()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


class RLInfCluster:
    """Own one process-wide RLInf cluster and its accelerator reservations."""

    def __init__(self, config: RLInfRuntimeConfig) -> None:
        self.config = config
        self.scheduler: Any = None
        self.cluster: Any = None
        self._accelerator_claims: dict[int, str] = {}
        self._ray_started_here = False

    def ensure(self) -> tuple[Any, Any]:
        if self.cluster is not None:
            return self.scheduler, self.cluster
        scheduler = load_scheduler()
        cluster_class = scheduler.Cluster
        already_initialized = cluster_class.has_initialized()
        if already_initialized:
            ray = importlib.import_module("ray")
            if not ray.is_initialized():
                raise RuntimeError(
                    "RLInf retains stale process-wide state after ray.shutdown(); "
                    "start a fresh controller process or set "
                    "shutdown_ray_on_close=false"
                )
        if not already_initialized:
            keep_driver_interpreter()  # under `uv run`, workers must not rebuild the project environment
        self.cluster = (
            cluster_class()
            if already_initialized
            else cluster_class(num_nodes=self.config.num_nodes)
        )
        self.scheduler = scheduler
        self._ray_started_here = not already_initialized
        return scheduler, self.cluster

    def claim_accelerators(
        self,
        start: int,
        count: int,
        owner: str,
        *,
        exclusive: bool,
    ) -> tuple[int, ...]:
        """Reserve ``count`` accelerators from ``start`` for ``owner``.

        The registry only protects RLInf-placed groups from each other; the
        controller-owned learner never registers. A shared runtime therefore
        records nothing: with sharing allowed there is no overlap to catch.
        """
        return self.claim_accelerator_list(
            range(start, start + count), owner, exclusive=exclusive
        )

    def claim_accelerator_list(
        self, indices: Iterable[int], owner: str, *, exclusive: bool
    ) -> tuple[int, ...]:
        """Reserve the accelerators ``indices`` for ``owner`` (see :meth:`claim_accelerators`)."""
        accelerator_indices = tuple(indices)
        if not exclusive:
            return accelerator_indices
        conflicts = {
            index: self._accelerator_claims[index]
            for index in accelerator_indices
            if index in self._accelerator_claims
        }
        if conflicts:
            detail = ", ".join(
                f"accelerator {index} is already reserved by {claimed_owner}"
                for index, claimed_owner in conflicts.items()
            )
            raise ValueError(f"RLInf placement overlap for {owner}: {detail}")
        for index in accelerator_indices:
            self._accelerator_claims[index] = owner
        return accelerator_indices

    def release_accelerators(self, indices: Sequence[int]) -> None:
        for index in indices:
            self._accelerator_claims.pop(index, None)

    def close(self) -> None:
        if self.config.shutdown_ray_on_close and self._ray_started_here:
            ray = importlib.import_module("ray")
            ray.shutdown()
            self._ray_started_here = False
            self.cluster = None
            self.scheduler = None
