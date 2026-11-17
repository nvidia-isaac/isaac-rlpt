# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf rollout workers, channel transport, and group proxy."""

from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
    RolloutWorkerSpec,
    WeightsRef,
    check_rollout_results,
)
from isaac_rl_posttraining.runtimes.concurrency import RolloutGroupBase
from isaac_rl_posttraining.runtimes.rlinf.cluster import (
    terminate_channel,
    terminate_worker_group,
)
from isaac_rl_posttraining.runtimes.serving.hosted import HostedPolicy
from isaac_rl_posttraining.runtimes.watchdog import (
    RolloutTimeoutError,
    collection_timeout,
    rollout_timeout_message,
)


def make_rollout_worker(worker_base: type) -> Any:
    """Create the provider wrapper after RLInf's optional base class is available."""

    class RLPTRolloutWorker(worker_base):
        def __init__(
            self,
            spec: RolloutWorkerSpec,
            device: str,
            assignments: tuple[tuple[int, ...], ...],
        ) -> None:
            super().__init__()
            # "gpu" is the one card RLInf left visible to this process.
            if device not in ("cpu", "gpu"):
                raise ValueError(f"rollout device must be cpu or gpu, got {device!r}")
            runtime_device = "cuda:0" if device == "gpu" else "cpu"
            self.delegate = spec(
                RolloutWorkerContext(
                    worker_index=int(self._rank),
                    num_rollout_workers=int(self._world_size),
                    environment_indices=assignments[int(self._rank)],
                    device=runtime_device,
                )
            )

        def ping(self) -> int:
            return int(self._rank)

        def collect(self, request: RolloutRequest) -> RolloutResult:
            return self.delegate.collect(request)

        def collect_to_channel(
            self,
            request: RolloutRequest,
            output_channel: Any,
            key: str,
        ) -> tuple[int, float]:
            """Collect into the channel; returns the worker index and the wall-clock time the round ended here."""
            result = self.delegate.collect(request)
            finished_at = time.time()
            output_channel.put(result, key=key, weight=max(1, result.env_steps))
            return result.worker_index, finished_at

        def set_policy_state(self, state: PolicyState) -> int:
            return self.delegate.set_policy_state(state)

        def close(self) -> None:
            self.delegate.close()

    RLPTRolloutWorker.__name__ = "RLPTRolloutWorker"
    RLPTRolloutWorker.__qualname__ = "RLPTRolloutWorker"
    return RLPTRolloutWorker


def wait_or_time_out(
    ray: Any,
    calls: list[Any],
    timeout_seconds: float | None,
    *,
    what: str,
    describe_worker: Callable[[int], str],
    on_timeout: Callable[[], None],
) -> None:
    """Wait for every worker's call; raise :class:`RolloutTimeoutError` naming the late workers."""
    if timeout_seconds is None:
        return
    _, pending = ray.wait(calls, num_returns=len(calls), timeout=timeout_seconds)
    if pending:
        on_timeout()
        late = ", ".join(describe_worker(calls.index(call)) for call in pending)
        raise RolloutTimeoutError(rollout_timeout_message(what, late, timeout_seconds))


class RLInfCollectFuture:
    """One nonblocking rollout submission backed by an RLInf Channel."""

    def __init__(
        self,
        handles: Sequence[Any],
        channel: Any,
        keys: Sequence[str],
        assignments: tuple[tuple[int, ...], ...],
        *,
        timeout_seconds: float | None = None,
        describe_worker: Callable[[int], str] = lambda index: (
            f"RLInf rollout worker {index}"
        ),
        on_timeout: Callable[[], None] = lambda: None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._describe_worker = describe_worker
        self._on_timeout = on_timeout
        self._handles = tuple(handles)
        self._channel = channel
        self._keys = tuple(keys)
        self._assignments = assignments
        self._result: tuple[RolloutResult, ...] | None = None
        self._finished_at: float | None = None

    def result(self) -> tuple[RolloutResult, ...]:
        if self._result is None:
            ray = importlib.import_module("ray")
            wait_or_time_out(
                ray,
                list(self._handles),
                self._timeout_seconds,
                what="RLInf rollout group collecting",
                describe_worker=self._describe_worker,
                on_timeout=self._on_timeout,
            )
            try:
                # ray.get accepts a list of handles, not a tuple.
                completions = tuple(ray.get(list(self._handles)))  # NOSONAR
                results = tuple(self._channel.get(key=key) for key in self._keys)
            except Exception as exc:
                raise RolloutInfrastructureError(
                    "RLInf rollout workers or result transport failed before "
                    "collection completed"
                ) from exc
            worker_indices = tuple(int(index) for index, _ in completions)
            expected = tuple(range(len(worker_indices)))
            if tuple(sorted(worker_indices)) != expected:
                raise RuntimeError(
                    f"RLInf rollout worker indices are inconsistent: {worker_indices}"
                )
            # The round ends when its last worker does; the actors stamp their own clocks.
            self._finished_at = max(
                float(finished_at) for _, finished_at in completions
            )
            self._result = tuple(
                sorted(results, key=lambda result: result.worker_index)
            )
            check_rollout_results(self._result, self._assignments)
        return self._result

    def finished_at(self) -> float | None:
        return self._finished_at


class RLInfRolloutGroup(RolloutGroupBase):
    def __init__(
        self,
        group,
        num_rollout_workers: int,
        *,
        num_envs: int,
        assignments: tuple[tuple[int, ...], ...],
        output_channel: Any,
        policy: HostedPolicy | None = None,
        servers: HostedPolicy | None = None,
        rollout_timeout_seconds: float | None = None,
        accelerators: Sequence[int] = (),
    ) -> None:
        self._rollout_timeout_seconds = rollout_timeout_seconds
        #: The accelerator of each worker; none when the workers run on the CPU.
        self.accelerators = tuple(accelerators)
        self._worker_devices = (
            tuple(f"accelerator {index}" for index in self.accelerators)
            or ("cpu",) * num_rollout_workers
        )
        self._group = group
        self.num_rollout_workers = num_rollout_workers
        self.num_envs = num_envs
        self._assignments = assignments
        self._output_channel = output_channel
        #: This group's policy on the run's policy servers, when it is served.
        self._policy = policy
        #: The policy (or base-model) servers this group calls, read for their counters.
        self._servers = servers
        self._closed = False
        #: A collection timed out: close kills the workers without asking them to close.
        self._frozen = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RLInf rollout group is closed")

    def collect_async(self, requests: Sequence[RolloutRequest]) -> RLInfCollectFuture:
        self._ensure_open()
        if len(requests) != self.num_rollout_workers:
            raise ValueError(
                "expected one rollout request per worker "
                f"({self.num_rollout_workers}), got {len(requests)}"
            )
        call_id = uuid.uuid4().hex
        keys = tuple(
            f"{call_id}:{worker_index}"
            for worker_index in range(self.num_rollout_workers)
        )
        workers = tuple(self._group.worker_info_list)
        if len(workers) != self.num_rollout_workers:
            raise RuntimeError(
                "RLInf rollout group exposes "
                f"{len(workers)} actors, expected {self.num_rollout_workers}"
            )
        handles = tuple(
            worker_info.worker.collect_to_channel.remote(
                request,
                self._output_channel,
                keys[worker_index],
            )
            for worker_index, (worker_info, request) in enumerate(
                zip(workers, requests, strict=True)
            )
        )
        return RLInfCollectFuture(
            handles,
            self._output_channel,
            keys,
            self._assignments,
            timeout_seconds=collection_timeout(self._rollout_timeout_seconds, requests),
            describe_worker=self.describe_worker,
            on_timeout=self._mark_frozen,
        )

    def _mark_frozen(self) -> None:
        self._frozen = True

    def describe_worker(self, index: int) -> str:
        device = (
            self._worker_devices[index]
            if index < len(self._worker_devices)
            else "device unknown"
        )
        return f"RLInf rollout worker {index} ({device})"

    def collect(self, requests: Sequence[RolloutRequest]) -> tuple[RolloutResult, ...]:
        return self.collect_async(requests).result()

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        """Publish to the workers, which take it after a collection in flight, then to the servers."""
        self._ensure_open()
        ray = importlib.import_module("ray")
        calls = [
            worker_info.worker.set_policy_state.remote(state)
            for worker_info in self._group.worker_info_list
        ]
        wait_or_time_out(
            ray,
            calls,
            self._rollout_timeout_seconds,
            what="RLInf rollout group publishing weights",
            describe_worker=self.describe_worker,
            on_timeout=self._mark_frozen,
        )
        versions = tuple(int(version) for version in ray.get(calls))
        if self._policy is not None:
            self._policy.set_weights(WeightsRef.of(state))
        return versions

    def policy_server_stats(self) -> tuple[dict[str, Any], ...]:
        """The counters of the policy (or base-model) servers this group calls."""
        self._ensure_open()
        if self._servers is None:
            return ()
        return tuple(self._servers.stats()["replicas"])

    def reset_policy_server_stats(self) -> None:
        """Clear service counters after benchmark warmup."""
        self._ensure_open()
        if self._servers is not None:
            self._servers.reset_stats()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        provider_group = self._group
        policy = self._policy
        output_channel = self._output_channel
        errors: list[Exception] = []
        try:
            error = terminate_worker_group(
                provider_group, close_delegates=not self._frozen
            )
            if error is not None:
                errors.append(error)
            if policy is not None:
                # The servers belong to the runtime; only this group's policy leaves them.
                try:
                    policy.close()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
            error = terminate_channel(output_channel)
            if error is not None:
                errors.append(error)
        finally:
            self._group = None
            self._policy = None
            self._output_channel = None
        if errors:
            raise RuntimeError(
                f"{len(errors)} RLInf rollout resource(s) failed to close"
            ) from errors[0]
