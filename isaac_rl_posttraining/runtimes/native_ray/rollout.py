# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout workers as Ray actors: one actor per worker, built on its node from today's worker specs."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from isaac_rl_posttraining.runtimes._serialization import serialize_specs_by_value
from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
    RolloutWorkerSpec,
    WeightsRef,
    check_rollout_results,
    partition_environments,
)
from isaac_rl_posttraining.runtimes.base_part import with_served_base_part
from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec
from isaac_rl_posttraining.runtimes.concurrency import RolloutGroupBase
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
from isaac_rl_posttraining.runtimes.native_ray.cluster import (
    ActorFailedError,
    ActorPlacement,
    ActorTimeoutError,
    RayCluster,
    close_actors,
    import_ray,
    remote_class,
    seed_process,
    wait_all,
    weights_by_reference,
)
from isaac_rl_posttraining.runtimes.serving import RayServiceHandle, ServiceHandle
from isaac_rl_posttraining.runtimes.watchdog import (
    RolloutTimeoutError,
    rollout_timeout_message,
)
from isaac_rl_posttraining.utils.compat import add_note

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.native import NativeRuntimeConfig
    from isaac_rl_posttraining.runtimes.serving.hosted import HostedPolicy

__all__ = ["RayRolloutGroup", "RolloutWorkerActor"]


class RolloutWorkerActor:
    """One rollout worker in its own process; Ray runs its calls one at a time, in order."""

    def __init__(
        self, spec: RolloutWorkerSpec, context: RolloutWorkerContext, seed: int | None
    ) -> None:
        import_ray()  # registers the resolver of weights published by reference
        seed_process(seed)
        self._index = context.worker_index
        self._worker = spec(context)

    def ready(self) -> int:
        return self._index

    def collect(self, request: RolloutRequest) -> tuple[RolloutResult, float]:
        """One bounded collection and the wall-clock time it ended here."""
        result = self._worker.collect(request)
        return result, time.time()

    def set_weights(self, weights: WeightsRef, version_only: bool) -> int:
        """Apply published weights; a worker acting through a policy server only takes the version."""
        state = (
            PolicyState(version=weights.version, tensors={})
            if version_only
            else weights.resolve()
        )
        return int(self._worker.set_policy_state(state))

    def close(self) -> None:
        self._worker.close()


class _RayCollect:
    """A collection in flight on every worker of a group (an ``AsyncCall``)."""

    def __init__(self, group: RayRolloutGroup, calls: list[Any]) -> None:
        self._group = group
        self._calls = calls
        self._results: tuple[RolloutResult, ...] | None = None
        self._finished_at: float | None = None

    def result(self) -> tuple[RolloutResult, ...]:
        if self._results is None:
            group = self._group
            what = f"rollout group {group.name!r} collecting"
            timeout = group.config.rollout_timeout_seconds
            try:
                answers = wait_all(
                    self._calls,
                    describe=group.describe_worker,
                    timeout_seconds=timeout,
                    what=what,
                )
            except ActorTimeoutError as exc:
                raise group.timed_out(exc.late, what) from exc
            except ActorFailedError as exc:
                raise RolloutInfrastructureError(str(exc)) from exc
            finally:
                self._group.collection_ended(self._calls)
            results = tuple(result for result, _ in answers)
            check_rollout_results(results, self._group.assignments)
            self._finished_at = max(float(finished) for _, finished in answers)
            self._results = results
        return self._results

    def finished_at(self) -> float | None:
        return self._finished_at


class RayRolloutGroup(RolloutGroupBase):
    """Persistent rollout-worker actors on the plan's ``envs`` GPUs, results in worker order.

    The policy is either a copy in every worker (``placement.policy`` local),
    or policy-server actors on the plan's ``policy`` GPUs that every worker
    calls, or a copy of the actor in every worker over a served base model
    (``placement.parts.base``). Weights reach the workers by Ray object
    reference; workers of a served policy receive only the version, the
    servers the weights; a served frozen base model takes none.
    """

    def __init__(
        self,
        cluster: RayCluster,
        spec: RolloutWorkerSpec,
        *,
        num_rollout_workers: int,
        num_envs: int,
        name: str,
        config: NativeRuntimeConfig,
        open_service: Any,
        open_base_service: Any,
    ) -> None:
        self.num_rollout_workers = num_rollout_workers
        self.num_envs = num_envs
        self.name = name
        self.config = config
        self.assignments = partition_environments(num_envs, num_rollout_workers)
        self._lock = threading.Lock()
        #: Collect calls not yet answered, each with its worker's index.
        self._in_flight: dict[Any, int] = {}
        #: Workers whose collection timed out: killed on close instead of closed.
        self.frozen: set[int] = set()
        self._actors: list[Any] = []
        self._placements: list[ActorPlacement] = []
        self._cluster: RayCluster | None = None
        self._slots: list[int] = []
        self._owned_service: HostedPolicy | None = None
        self._base_service: HostedPolicy | None = None
        self._closed = False
        try:
            spec = self._connect_policy(cluster, spec, open_service, open_base_service)
            self._served = isinstance(getattr(spec, "policy", None), RayServiceHandle)
            self._start_workers(cluster, spec)
        except BaseException as exc:
            self._close_after_failure(exc)
            raise

    def _connect_policy(
        self,
        cluster: RayCluster,
        spec: RolloutWorkerSpec,
        open_service: Any,
        open_base_service: Any,
    ) -> RolloutWorkerSpec:
        """Serve the spec's policy, or only its base model, from policy-server actors when the plan serves it."""
        if not isinstance(spec, (GymWorkerSpec, BatchedWorkerSpec)):
            return spec
        if isinstance(spec.policy, ServiceHandle):
            if not isinstance(spec.policy, RayServiceHandle):
                raise TypeError(
                    "rollout workers in their own processes cannot reach an in-process policy "
                    "service; open it with runtime.native.processes=ray (or replicas or gpus) "
                    "so its replicas run as Ray actors"
                )
            return spec
        services = [role for role in cluster.roles().values() if role.served]
        if not services:
            return spec
        if any(role.parts == ("base",) for role in services):
            # The actor stays in every worker; the base-model servers belong to the runtime.
            spec, self._base_service = with_served_base_part(spec, open_base_service)
            logger.info(
                f"rollout group {self.name!r}: every worker runs its actor; the base model "
                "is served by policy-server actors"
            )
            return spec
        servable = (
            spec.supports_batched_policy
            if isinstance(spec, GymWorkerSpec)
            else spec.supports_served_policy
        )
        if not servable:
            raise ValueError(
                f"the placement serves the policy but the {type(spec).__name__} of {self.name!r} "
                "keeps it local (supports_batched_policy / supports_served_policy is false)"
            )
        self._owned_service = open_service(spec.policy, name=f"{self.name}_policy")
        return replace(spec, policy=self._owned_service.handle())

    def _start_workers(self, cluster: RayCluster, spec: RolloutWorkerSpec) -> None:
        actor_class = remote_class(RolloutWorkerActor)
        with serialize_specs_by_value(
            spec, enabled=self.config.serialize_specs_by_value
        ):
            spec_ref = import_ray().put(spec)
        token = uuid.uuid4().hex[:6]
        self._cluster = cluster
        self._slots = cluster.claim_slots("envs", len(self.assignments))
        for index, environment_indices in enumerate(self.assignments):
            placement = cluster.place(
                "envs",
                index,
                slot=self._slots[index],
                name=f"{self.name}_worker{index}_{token}",
            )
            context = RolloutWorkerContext(
                worker_index=index,
                num_rollout_workers=self.num_rollout_workers,
                environment_indices=environment_indices,
                device=placement.device,
            )
            seed = None if self.config.seed is None else self.config.seed + 7919 * index
            self._placements.append(placement)
            self._actors.append(
                actor_class.options(**placement.options).remote(spec_ref, context, seed)
            )
        wait_all(
            [actor.ready.remote() for actor in self._actors],
            describe=self.describe_worker,
            timeout_seconds=self.config.startup_timeout_seconds,
            what=f"starting rollout group {self.name!r}",
        )

    def describe_worker(self, index: int) -> str:
        return self._placements[index].describe("rollout worker")

    def collection_ended(self, calls: Sequence[Any]) -> None:
        with self._lock:
            for call in calls:
                self._in_flight.pop(call, None)

    def timed_out(self, late: Iterable[int], what: str) -> RolloutTimeoutError:
        """Mark the ``late`` workers frozen and name them in the error."""
        timeout = self.config.rollout_timeout_seconds
        assert timeout is not None
        late = sorted(late)
        self.frozen.update(late)
        names = ", ".join(self.describe_worker(index) for index in late)
        return RolloutTimeoutError(rollout_timeout_message(what, names, timeout))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"rollout group {self.name!r} is closed")

    def collect_async(self, requests: Sequence[RolloutRequest]) -> _RayCollect:
        self._require_open()
        if len(requests) != self.num_rollout_workers:
            raise ValueError(
                "expected one rollout request per worker "
                f"({self.num_rollout_workers}), got {len(requests)}"
            )
        with self._lock:
            calls = [
                actor.collect.remote(request)
                for actor, request in zip(self._actors, requests, strict=True)
            ]
            self._in_flight.update({call: index for index, call in enumerate(calls)})
        return _RayCollect(self, calls)

    def collect(self, requests: Sequence[RolloutRequest]) -> tuple[RolloutResult, ...]:
        return self.collect_async(requests).result()

    def set_weights(self, weights: WeightsRef) -> tuple[int, ...]:
        """Publish ``weights`` once and let every worker (or policy server) fetch them.

        Waits for a collection in flight to end first, so weights change only
        between collections, as on one process.
        """
        self._require_open()
        with self._lock:
            in_flight = dict(self._in_flight)
        if in_flight:
            _, pending = import_ray().wait(
                list(in_flight),
                num_returns=len(in_flight),
                timeout=self.config.rollout_timeout_seconds,
            )
            if pending:
                raise self.timed_out(
                    (in_flight[call] for call in pending),
                    f"rollout group {self.name!r} publishing weights",
                )
        weights = weights_by_reference(weights)
        if self._owned_service is not None:
            self._owned_service.set_weights(weights)
        served = self._served
        versions = wait_all(
            [actor.set_weights.remote(weights, served) for actor in self._actors],
            describe=self.describe_worker,
            timeout_seconds=self.config.control_timeout_seconds,
            what=f"rollout group {self.name!r} taking weights version {weights.version}",
        )
        return tuple(int(version) for version in versions)

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        return self.set_weights(WeightsRef.of(state))

    def policy_server_stats(self) -> tuple[dict[str, Any], ...]:
        """The counters of the policy (or base-model) servers this group uses, when metrics are collected."""
        service = self._owned_service or self._base_service
        if service is None or not self.config.collects_server_metrics:
            return ()
        return tuple(service.stats()["replicas"])

    def reset_policy_server_stats(self) -> None:
        service = self._owned_service or self._base_service
        if service is not None and self.config.collects_server_metrics:
            service.reset_stats()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = self._release()
        if errors:
            raise RuntimeError(
                f"{len(errors)} rollout worker(s) of {self.name!r} failed to close"
            ) from errors[0]

    def _release(self) -> list[Exception]:
        # A frozen worker would not answer ``close`` behind its stuck collection.
        healthy = [
            index for index in range(len(self._actors)) if index not in self.frozen
        ]
        for index in sorted(self.frozen):
            import_ray().kill(self._actors[index])
        errors = close_actors(
            [self._actors[index] for index in healthy],
            describe=lambda position: self.describe_worker(healthy[position]),
            timeout_seconds=self.config.control_timeout_seconds,
        )
        self._actors.clear()
        if self._cluster is not None:
            self._cluster.release_slots("envs", self._slots)
            self._cluster = None
        if self._owned_service is not None:
            try:
                self._owned_service.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        return errors

    def _close_after_failure(self, exc: BaseException) -> None:
        self._closed = True
        errors = self._release()
        if errors:
            add_note(exc, f"{len(errors)} resource(s) also failed to close")
