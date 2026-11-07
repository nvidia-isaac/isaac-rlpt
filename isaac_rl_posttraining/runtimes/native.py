# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native runtime: every role in this process, or each role in its own processes on Ray.

One process is the default and needs no optional dependency. When the
placement asks for more processes (several learner ranks, several policy-server
replicas or GPUs for them, GPUs for the rollout workers, or several nodes),
the runtime runs rollout workers, policy servers and learner ranks as Ray
actors on the GPUs the :class:`~isaac_rl_posttraining.runtimes.placement.PlacementPlan`
assigns (``runtimes/native_ray/``; the ``ray`` extra).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from isaac_rl_posttraining.runtimes.base import (
    PolicyContext,
    PolicyState,
    RolloutRequest,
    RolloutResult,
    RolloutWorker,
    RolloutWorkerContext,
    RolloutWorkerSpec,
    RuntimeFeatures,
    WeightsRef,
    check_rollout_results,
    partition_environments,
)
from isaac_rl_posttraining.runtimes.base_part import with_served_base_part
from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec
from isaac_rl_posttraining.runtimes.concurrency import (
    AsyncCall,
    RolloutGroupBase,
    ThreadCall,
)
from isaac_rl_posttraining.runtimes.frozen_base_model import FrozenBaseModelSpec
from isaac_rl_posttraining.runtimes.gpu_pool import gpu_count
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
from isaac_rl_posttraining.runtimes.learner import LearnerGroup, LearnerWorkerSpec
from isaac_rl_posttraining.runtimes.native_config import NativeRuntimeConfig
from isaac_rl_posttraining.runtimes.placement import (
    GpuRequest,
    Placement,
    PlacementPlan,
    Resources,
    RolePlacement,
    Served,
    plan_policy_service,
    served_role,
)
from isaac_rl_posttraining.runtimes.serving import (
    Batcher,
    HostedPolicy,
    InProcessPolicyService,
    PolicyService,
    ServiceHandle,
)
from isaac_rl_posttraining.runtimes.serving.local import find_local_service
from isaac_rl_posttraining.runtimes.watchdog import (
    CollectionWatchdog,
    RolloutTimeoutError,
)
from isaac_rl_posttraining.utils.compat import add_note

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.native_ray.cluster import RayCluster
    from isaac_rl_posttraining.runtimes.native_ray.serving import RayPolicyService

__all__ = ["NativeRuntime", "NativeRuntimeConfig", "wants_processes"]


def wants_processes(placement: Placement, *, num_nodes: int = 1) -> bool:
    """Whether a placement asks for more than this process: the native runtime's ``processes=auto`` rule.

    It does when it asks for several learner ranks (or learner GPUs), several
    policy-server replicas, any GPUs for them or several per replica, any GPUs for the rollout
    workers, or several nodes. A served policy without such requests stays an
    in-process policy server.
    """
    if num_nodes > 1 or len(placement.nodes or ()) > 1:
        return True
    learner = placement.learner
    if (learner.ranks or 1) > 1 or (gpu_count(learner.gpus) or 0) > 1:
        return True
    if gpu_count(placement.envs.gpus):
        return True
    for part in (placement.policy, *placement.parts.values()):
        if isinstance(part, Served) and (
            (part.replicas or 1) > 1
            or part.gpus is not None
            or part.gpus_per_replica > 1
        ):
            return True
    return False


def _driver_gpus(ray: Any) -> int:
    """The GPUs Ray sees on this process's node."""
    driver = ray.get_runtime_context().get_node_id()
    for node in ray.nodes():
        if node["NodeID"] == driver:
            return int(node.get("Resources", {}).get("GPU", 0))
    return 0


def _close_partial_group(
    exc: BaseException,
    workers: Sequence[RolloutWorker],
    service: InProcessPolicyService | None,
) -> None:
    """Close what a failed group construction built, noting cleanup failures on ``exc``."""
    cleanup_errors: list[Exception] = []
    for worker in reversed(workers):
        try:
            worker.close()
        except Exception as close_exc:  # noqa: BLE001
            cleanup_errors.append(close_exc)
    if cleanup_errors:
        add_note(
            exc,
            f"{len(cleanup_errors)} partially constructed worker(s) also failed to close",
        )
    if service is not None:
        try:
            service.close()
        except Exception as close_exc:  # noqa: BLE001
            add_note(
                exc,
                "policy-server cleanup after construction failure also "
                f"failed: {close_exc!r}",
            )


class _NativeRolloutGroup(RolloutGroupBase):
    """Persistent in-process workers with deterministic result ordering."""

    def __init__(
        self,
        spec: RolloutWorkerSpec,
        *,
        num_rollout_workers: int,
        num_envs: int,
        device: str,
        config: NativeRuntimeConfig,
        name: str,
        open_base_service: Callable[[FrozenBaseModelSpec], Any],
    ) -> None:
        assignments = partition_environments(num_envs, num_rollout_workers)
        self.num_rollout_workers = num_rollout_workers
        self.num_envs = num_envs
        self._assignments = assignments
        self._collect_metrics = config.collects_server_metrics
        concurrent = config.collects_concurrently(num_rollout_workers)
        self.name = name
        self._device = device
        self._watchdog = CollectionWatchdog(config.rollout_timeout_seconds)
        #: Workers inside a collection right now; a timed-out collection leaves them frozen.
        self._active: set[int] = set()
        self._frozen: set[int] = set()
        workers: list[RolloutWorker] = []
        owned_service: InProcessPolicyService | None = None
        self._base_service: Any = None
        try:
            if config.serves_policy(spec):
                assert isinstance(spec, (GymWorkerSpec, BatchedWorkerSpec))
                assert callable(spec.policy) and not isinstance(
                    spec.policy, ServiceHandle
                )
                owned_service = InProcessPolicyService(
                    spec.policy,
                    device=device,
                    batcher=config.batcher(concurrent=concurrent),
                    name=f"{name}_policy_{uuid.uuid4().hex[:8]}",
                    collect_metrics=self._collect_metrics,
                )
                spec = replace(spec, policy=owned_service.handle())
            elif config.base_mode == "served":
                # The runtime keeps the base-model service; every group of the run shares it.
                spec, self._base_service = with_served_base_part(
                    spec, open_base_service
                )
                logger.info(
                    f"rollout group {name!r}: every worker runs its actor; the base model "
                    "is served in this process"
                )
            for worker_index, environment_indices in enumerate(assignments):
                workers.append(
                    spec(
                        RolloutWorkerContext(
                            worker_index=worker_index,
                            num_rollout_workers=num_rollout_workers,
                            environment_indices=environment_indices,
                            device=device,
                        )
                    )
                )
        except BaseException as exc:
            _close_partial_group(exc, workers, owned_service)
            raise
        self._workers = tuple(workers)
        self._owned_service = owned_service
        # A service opened elsewhere and named by the spec also receives this
        # group's weights; it serves each version once.
        policy = (
            spec.policy
            if isinstance(spec, (GymWorkerSpec, BatchedWorkerSpec))
            else None
        )
        self._service = (
            find_local_service(policy) if isinstance(policy, ServiceHandle) else None
        )
        self._concurrent = concurrent and (
            self._service is not None or self._base_service is not None
        )
        self._closed = False
        # Held while a collection runs: the workers' weights change only between
        # collections, the ordering a remote actor gives by serving calls in turn.
        self._collecting = threading.Lock()

    def collect(self, requests: Sequence[RolloutRequest]) -> tuple[RolloutResult, ...]:
        with self._collecting:
            return self._watched_collect(requests)

    def collect_async(
        self, requests: Sequence[RolloutRequest]
    ) -> AsyncCall[tuple[RolloutResult, ...]]:
        """Start a collection on a helper thread and pin the workers' weights until it ends.

        The lock is taken on the caller's thread, so a ``set_policy_state``
        issued after this call waits for the collection instead of racing it.
        """
        self._collecting.acquire()

        def run() -> tuple[RolloutResult, ...]:
            try:
                return self._watched_collect(requests)
            finally:
                self._collecting.release()

        return ThreadCall(run)

    def _watched_collect(
        self, requests: Sequence[RolloutRequest]
    ) -> tuple[RolloutResult, ...]:
        late: set[int] = set()

        def describe() -> str:
            late.update(self._active)
            return ", ".join(
                f"rollout worker {index} ({self._device})" for index in sorted(late)
            )

        try:
            return self._watchdog.run(
                lambda: self._collect(requests),
                describe=describe,
                what=f"rollout group {self.name!r} collecting",
                requests=requests,
            )
        except RolloutTimeoutError:
            self._frozen |= late
            raise

    def _collect_one(self, index: int, request: RolloutRequest) -> RolloutResult:
        self._active.add(index)
        try:
            return self._workers[index].collect(request)
        finally:
            self._active.discard(index)

    def _collect(self, requests: Sequence[RolloutRequest]) -> tuple[RolloutResult, ...]:
        if self._closed:
            raise RuntimeError("rollout group is closed")
        if len(requests) != self.num_rollout_workers:
            raise ValueError(
                "expected one rollout request per worker "
                f"({self.num_rollout_workers}), got {len(requests)}"
            )
        if self._concurrent:
            # Workers step their environments on threads so the shared policy
            # server batches their requests; results keep worker order.
            with ThreadPoolExecutor(max_workers=len(self._workers)) as pool:
                results = tuple(
                    pool.map(self._collect_one, range(len(self._workers)), requests)
                )
        else:
            results = tuple(
                self._collect_one(index, request)
                for index, request in enumerate(requests)
            )
        check_rollout_results(results, self._assignments)
        return results

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("rollout group is closed")
        with self._collecting:
            if self._service is not None:
                served = self._service.set_weights(WeightsRef.of(state))
                if served != (state.version,) * len(served):
                    raise RuntimeError(
                        f"policy server acknowledged versions {served}, "
                        f"expected {state.version}"
                    )
            return tuple(worker.set_policy_state(state) for worker in self._workers)

    def policy_server_stats(self) -> tuple[dict[str, Any], ...]:
        """The counters of the policy (or base-model) server this group uses, when metrics are collected."""
        service = self._owned_service or self._base_service
        if service is None or not self._collect_metrics:
            return ()
        return tuple(service.stats()["replicas"])

    def reset_policy_server_stats(self) -> None:
        service = self._owned_service or self._base_service
        if service is not None and self._collect_metrics:
            service.reset_stats()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        if self._frozen:
            logger.warning(
                f"rollout group {self.name!r}: not closing frozen rollout worker(s) {sorted(self._frozen)}"
            )
        for index, worker in reversed(list(enumerate(self._workers))):
            if index in self._frozen:
                continue
            try:
                worker.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if self._owned_service is not None:
            try:
                self._owned_service.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} rollout worker(s) failed to close"
            ) from errors[0]


class NativeRuntime:
    """RLPT's default runtime: one process, or Ray actors placed by a :class:`PlacementPlan`.

    Without ``plan`` every role runs in this process (today's runtime). With
    a plan, rollout workers, policy servers and learner ranks of more than one
    rank run as Ray actors on the plan's nodes and GPUs; a single learner rank
    stays in this process.
    """

    name = "native"
    # Policy services serve both rollout worker kinds; several processes add nodes and learner ranks.
    features = RuntimeFeatures(
        name=name,
        multi_node=True,
        policy_service=True,
        served_batched_collectors=True,
        distributed_learner=True,
        async_collect=True,
        weights_by_reference=True,
        serving_engines=("builtin",),
        served_parts=True,
        standalone_policy_service=True,
    )

    def __init__(
        self,
        config: NativeRuntimeConfig | None = None,
        *,
        plan: PlacementPlan | None = None,
    ) -> None:
        self.config = config or NativeRuntimeConfig()
        if plan is not None and self.config.processes == "single":
            raise ValueError(
                "a placement plan runs Ray actors; processes='single' forbids them"
            )
        self.plan = plan
        if plan is None:
            # Nodes and learner ranks come with a plan, which runs the roles as Ray actors.
            self.features = replace(
                NativeRuntime.features, multi_node=False, distributed_learner=False
            )
        self._cluster: RayCluster | None = None
        self._started_ray = False
        self._groups: list[Any] = []
        self._services: list[Any] = []
        self._base_services: dict[str, Any] = {}
        self._servers: dict[str, RayPolicyService] = {}
        self._learner_groups: list[LearnerGroup] = []

    @property
    def multi_process(self) -> bool:
        """Whether the roles run as Ray actors."""
        return self.plan is not None

    def cluster(self) -> RayCluster:
        """The Ray cluster of a multi-process run, connected (and its GPUs reserved) on first use."""
        if self.plan is None:
            raise RuntimeError("this native runtime runs every role in this process")
        if self._cluster is None:
            self._cluster = self._reserve(self.plan.resources, self.plan.roles())
            for line in self._cluster.describe():
                logger.info(f"native runtime on Ray, {line}")
        return self._cluster

    def _connect_ray(self) -> Any:
        """Ray, joined or started on first use; the runtime stops a Ray it started when it closes."""
        from isaac_rl_posttraining.runtimes.native_ray.cluster import (
            connect_ray,
            import_ray,
        )

        self._started_ray |= connect_ray(self.config.ray_address)
        return import_ray()

    def _reserve(
        self, resources: Resources, roles: Mapping[str, RolePlacement]
    ) -> RayCluster:
        from isaac_rl_posttraining.runtimes.native_ray.cluster import RayCluster

        self._connect_ray()
        return RayCluster(
            resources,
            roles,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            node_join_timeout_seconds=self.config.node_join_timeout_seconds,
        )

    def open_rollout_group(
        self,
        spec: RolloutWorkerSpec,
        *,
        num_rollout_workers: int,
        num_envs: int,
        name: str,
    ) -> Any:
        if self.plan is not None:
            from isaac_rl_posttraining.runtimes.native_ray.rollout import (
                RayRolloutGroup,
            )

            group: Any = RayRolloutGroup(
                self.cluster(),
                spec,
                num_rollout_workers=num_rollout_workers,
                num_envs=num_envs,
                name=name,
                config=self.config,
                open_service=self._serve_policy,
                open_base_service=self.open_base_service,
            )
        else:
            group = _NativeRolloutGroup(
                spec,
                num_rollout_workers=num_rollout_workers,
                num_envs=num_envs,
                device=self.config.resolved_device(),
                config=self.config,
                name=name,
                open_base_service=self.open_base_service,
            )
        self._groups.append(group)
        return group

    def open_policy_service(
        self,
        spec: Callable[[PolicyContext], Any],
        *,
        name: str,
        replicas: int | None = None,
        gpus: GpuRequest = None,
        max_batch: int | None = None,
        max_wait_ms: float | None = None,
        engine: str = "builtin",
    ) -> PolicyService:
        """Serve ``spec`` from ``replicas`` policy servers of their own, apart from any rollout group.

        One replica without ``gpus`` runs in this process (on ``device``);
        otherwise every replica is a Ray actor on its own GPU: ``gpus`` lists
        them (indices of the GPUs this process sees), or the first free ones
        are taken, or the CPU without GPUs. ``replicas`` defaults to one per
        listed GPU. Unset batching options take ``policy_max_batch`` and
        ``policy_max_wait_ms``; statistics are always collected. Rollout
        workers connect with ``service.client(i)`` or take ``service.handle()``
        as a worker spec's ``policy``. The runtime closes the service on close.
        """
        batcher = self._batcher(max_batch, max_wait_ms)
        in_process = self.config.processes == "single" or (
            self.config.processes == "auto"
            and self.plan is None
            and (replicas or 1) == 1
            and gpus is None
        )
        if in_process and gpus is not None:
            raise ValueError(
                "runtime.native.processes=single serves every replica in this process; "
                "leave gpus unset or set processes=auto"
            )
        service: PolicyService
        if in_process:
            service = InProcessPolicyService(
                spec,
                device=self.config.resolved_device(),
                batcher=batcher,
                name=name,
                replicas=replicas or 1,
                engine=engine,
                collect_metrics=True,
            )
        else:
            service = self._open_service_actors(
                spec,
                Served(
                    replicas=replicas,
                    gpus=gpus,
                    max_batch=batcher.max_batch,
                    max_wait_ms=batcher.max_wait_ms,
                    engine=engine,
                ),
                name=name,
                batcher=batcher,
            )
        self._services.append(service)
        return service

    def _batcher(
        self,
        max_batch: int | None,
        max_wait_ms: float | None,
        *,
        concurrent: bool = True,
    ) -> Batcher:
        """The runtime's batching, with the options a caller set."""
        batcher = self.config.batcher(concurrent=concurrent)
        return Batcher(
            max_batch=batcher.max_batch if max_batch is None else max_batch,
            max_wait_ms=batcher.max_wait_ms if max_wait_ms is None else max_wait_ms,
            serve_eval_unbatched=batcher.serve_eval_unbatched,
        )

    def _open_service_actors(
        self,
        spec: Callable[[PolicyContext], Any],
        served: Served,
        *,
        name: str,
        batcher: Batcher,
    ) -> HostedPolicy:
        """A standalone service's replicas as Ray actors, on GPUs planned for it alone."""
        from isaac_rl_posttraining.runtimes.native_ray.serving import RayPolicyService

        ray = self._connect_ray()
        # device=cpu keeps a service that names no GPUs off them.
        on_cpu = served.gpus is None and self.config.device == "cpu"
        resources = Resources.uniform(1, 0 if on_cpu else _driver_gpus(ray))
        role = plan_policy_service(resources, served)
        logger.info(f"policy service {name!r}: {role.describe()}")
        cluster = self._reserve(resources, {"policy": role})
        try:
            servers = RayPolicyService(
                cluster,
                role="policy",
                name=name,
                replicas=role.processes,
                gpus_per_replica=served.gpus_per_replica,
                routing=served.routing,
                max_batch=batcher.max_batch,
                config=self.config,
            )
        except BaseException:
            cluster.close()
            raise

        def release() -> None:
            try:
                servers.close()
            finally:
                cluster.close()

        try:
            return servers.add_policy(
                spec,
                name=name,
                batcher=batcher,
                engine=served.engine,
                collect_metrics=True,
                release=release,
            )
        except BaseException:
            release()
            raise

    def _serve_policy(
        self,
        spec: Callable[[PolicyContext], Any],
        *,
        name: str,
    ) -> HostedPolicy:
        """A rollout group's policy (or a base model) on the server actors of the plan's served role.

        One set of server actors per served role serves every group's policy.
        """
        from isaac_rl_posttraining.runtimes.native_ray.serving import RayPolicyService

        key, role = served_role(
            self.cluster().roles(), base_part=isinstance(spec, FrozenBaseModelSpec)
        )
        assert role.served is not None
        # Policy-server actors always see concurrent callers.
        batcher = self._batcher(None, None)
        servers = self._servers.get(key)
        if servers is None:
            servers = self._servers[key] = RayPolicyService(
                self.cluster(),
                role=key,
                name="base_model" if role.parts == ("base",) else "policy",
                replicas=role.processes,
                gpus_per_replica=role.served.gpus_per_replica,
                routing=role.served.routing,
                max_batch=batcher.max_batch,
                config=self.config,
            )
        hosted = servers.add_policy(
            spec,
            name=f"{name}_{uuid.uuid4().hex[:6]}",
            batcher=batcher,
            engine=role.served.engine,
            collect_metrics=self.config.collects_server_metrics,
        )
        self._services.append(hosted)
        return hosted

    def open_base_service(self, base_model: FrozenBaseModelSpec) -> Any:
        """The service of a frozen base model, opened once per run and shared by every rollout group."""
        key = base_model.key()
        service = self._base_services.get(key)
        if service is None:
            name = f"base_model_{uuid.uuid4().hex[:6]}"
            if self.plan is not None:
                service = self._serve_policy(base_model, name=name)
            else:
                service = InProcessPolicyService(
                    base_model,
                    device=self.config.resolved_device(),
                    batcher=self._batcher(
                        None, None, concurrent=self.config.base_service_waits
                    ),
                    name=name,
                    collect_metrics=self.config.collects_server_metrics,
                )
                self._services.append(service)
            self._base_services[key] = service
        return service

    def open_learner_group(
        self,
        spec: LearnerWorkerSpec,
        *,
        num_learner_ranks: int,
        name: str,
    ) -> LearnerGroup:
        """Learner-rank actors forming one process group (several processes only)."""
        if self.plan is None:
            raise RuntimeError(
                "the native runtime runs one learner rank in this process; a placement with "
                "several learner ranks runs them as Ray actors"
            )
        from isaac_rl_posttraining.runtimes.native_ray.learner import RayLearnerGroup

        group = RayLearnerGroup(
            self.cluster(),
            spec,
            num_learner_ranks=num_learner_ranks,
            name=name,
            config=self.config,
        )
        self._learner_groups.append(group)
        return group

    def close(self) -> None:
        errors: list[Exception] = []
        for learner_group in reversed(self._learner_groups):
            try:
                learner_group.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._learner_groups.clear()
        for group in reversed(self._groups):
            try:
                group.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._groups.clear()
        for service in reversed(self._services):
            try:
                service.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._services.clear()
        self._base_services.clear()
        for servers in reversed(list(self._servers.values())):
            try:
                servers.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._servers.clear()
        if self._cluster is not None:
            try:
                self._cluster.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            self._cluster = None
        if self._started_ray:
            from isaac_rl_posttraining.runtimes.native_ray.cluster import import_ray

            import_ray().shutdown()
            self._started_ray = False
        if errors:
            raise RuntimeError(
                f"{len(errors)} rollout group(s) or policy service(s) failed to close"
            ) from errors[0]
