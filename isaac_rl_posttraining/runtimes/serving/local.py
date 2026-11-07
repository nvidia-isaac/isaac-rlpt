# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process policy services and connecting a rollout worker to a service handle."""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from isaac_rl_posttraining.runtimes.base import PolicyContext, WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher
from isaac_rl_posttraining.runtimes.serving.clients import (
    InProcessTransport,
    PolicyTransport,
    ServedPolicy,
)
from isaac_rl_posttraining.runtimes.serving.engine import make_engine
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.queues import ThreadRequestQueue
from isaac_rl_posttraining.runtimes.serving.ray_client import (
    RayServiceHandle,
    RayTransport,
    ray_replica_for,
)
from isaac_rl_posttraining.runtimes.serving.server import PolicyServer, replica_stats

__all__ = [
    "InProcessPolicyService",
    "connect_policy",
    "connect_transport",
    "find_local_service",
]

#: The in-process services of this process, by name.
_LOCAL_SERVICES: dict[str, InProcessPolicyService] = {}


class InProcessPolicyService:
    """Policy-server replicas in this process, shared by every rollout worker that connects.

    Workers collecting on several threads have their requests batched
    together; workers collecting one after another see batches of their own.
    """

    def __init__(
        self,
        spec: Callable[[PolicyContext], Any],
        *,
        device: str,
        batcher: Batcher,
        name: str | None = None,
        replicas: int = 1,
        engine: str = "builtin",
        collect_metrics: bool = False,
    ) -> None:
        self.name = name or f"policy_{uuid.uuid4().hex[:8]}"
        if self.name in _LOCAL_SERVICES:
            raise ValueError(
                f"an in-process policy service named {self.name!r} is open"
            )
        self._handle = ServiceHandle(self.name, replicas)
        self._device = device
        self._queues: list[ThreadRequestQueue] = []
        self._initialization_seconds: list[float] = []
        try:
            for replica in range(replicas):
                started = time.perf_counter()
                serving_engine = make_engine(engine)
                serving_engine.load(
                    spec,
                    PolicyContext(
                        policy_index=replica,
                        num_policies=replicas,
                        device=device,
                        placement="policy_server",
                    ),
                )
                self._queues.append(
                    ThreadRequestQueue(
                        PolicyServer(
                            serving_engine, batcher, collect_metrics=collect_metrics
                        )
                    )
                )
                self._initialization_seconds.append(time.perf_counter() - started)
        except BaseException:
            for queue in self._queues:
                queue.close()
            raise
        _LOCAL_SERVICES[self.name] = self
        self._closed = False

    def handle(self) -> ServiceHandle:
        return self._handle

    def queue_for(self, route_index: int) -> ThreadRequestQueue:
        return self._queues[self._handle.replica_for(route_index)]

    def client(self, route_index: int = 0) -> RolloutPolicy:
        return connect_policy(self._handle, route_index=route_index)

    def ping(self) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError(f"policy service {self.name!r} is closed")
        return tuple(range(len(self._queues)))

    def features(self) -> PolicyFeatures:
        return self._queues[0].server.features

    def set_weights(self, ref: WeightsRef) -> tuple[int, ...]:
        return tuple(
            queue.exclusive(functools.partial(queue.server.set_weights, ref))
            for queue in self._queues
        )

    def stats(self) -> Mapping[str, Any]:
        return {
            "replicas": tuple(
                queue.exclusive(
                    functools.partial(
                        replica_stats,
                        queue.server,
                        index=index,
                        initialization_seconds=seconds,
                        device=self._device,
                    )
                )
                for index, (queue, seconds) in enumerate(
                    zip(self._queues, self._initialization_seconds, strict=True)
                )
            )
        }

    def reset_stats(self) -> None:
        for queue in self._queues:
            queue.exclusive(queue.server.reset_stats)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _LOCAL_SERVICES.pop(self.name, None)
        errors: list[Exception] = []
        for queue in self._queues:
            try:
                queue.close()
            except Exception as exc:  # noqa: BLE001 - close every replica, report after
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} policy-server replica(s) failed to close"
            ) from errors[0]


def find_local_service(handle: ServiceHandle) -> InProcessPolicyService | None:
    """The in-process service a handle names, if it lives in this process."""
    if isinstance(handle, RayServiceHandle):
        return None
    return _LOCAL_SERVICES.get(handle.name)


def connect_transport(handle: ServiceHandle, *, route_index: int) -> PolicyTransport:
    """The transport to the replica ``route_index`` routes to (an environment or worker index)."""
    if isinstance(handle, RayServiceHandle):
        actor = ray_replica_for(handle, route_index)
        return RayTransport(
            actor, timeout_seconds=handle.timeout_seconds, policy=handle.policy
        )
    service = _LOCAL_SERVICES.get(handle.name)
    if service is None:
        raise LookupError(
            f"no in-process policy service named {handle.name!r} in this process; "
            "an in-process service only serves rollout workers of the native runtime"
        )
    return InProcessTransport(service.queue_for(route_index))


def connect_policy(handle: ServiceHandle, *, route_index: int) -> RolloutPolicy:
    """A client of the replica ``route_index`` routes to (an environment or worker index)."""
    return ServedPolicy(connect_transport(handle, route_index=route_index))
