# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf rollout runtime assembly."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from isaac_rl_posttraining.runtimes._serialization import (
    serialize_specs_by_value,
)
from isaac_rl_posttraining.runtimes.base import (
    PolicyContext,
    RolloutWorkerSpec,
    RuntimeFeatures,
    WeightsRef,
    partition_environments,
)
from isaac_rl_posttraining.runtimes.base_part import with_served_base_part
from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec
from isaac_rl_posttraining.runtimes.frozen_base_model import FrozenBaseModelSpec
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
from isaac_rl_posttraining.runtimes.learner import LearnerWorkerSpec
from isaac_rl_posttraining.runtimes.placement import (
    GpuRequest,
    Resources,
    Served,
    plan_policy_service,
)
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.rlinf.cluster import (
    RLInfCluster,
    terminate_channel,
    terminate_worker_group,
)
from isaac_rl_posttraining.runtimes.rlinf.config import RLInfRuntimeConfig
from isaac_rl_posttraining.runtimes.rlinf.layout import (
    policy_server_accelerators,
    rollout_worker_accelerators,
)
from isaac_rl_posttraining.runtimes.rlinf.learner import (
    RLInfLearnerGroup,
    make_learner_worker,
)
from isaac_rl_posttraining.runtimes.rlinf.rollout import (
    RLInfRolloutGroup,
    make_rollout_worker,
)
from isaac_rl_posttraining.runtimes.rlinf.serving import make_policy_worker
from isaac_rl_posttraining.runtimes.serving import (
    Batcher,
    HostedPolicy,
    RayServiceHandle,
    ServiceHandle,
)
from isaac_rl_posttraining.utils.compat import add_note

__all__ = ["RLInfRuntime"]


def _validate_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _terminate_launched(output_channel: Any, *groups: Any) -> list[Exception]:
    """Release the channel and worker groups a failed launch created; return cleanup errors."""
    errors = [terminate_channel(output_channel)] if output_channel is not None else []
    errors += [terminate_worker_group(group) for group in groups if group is not None]
    return [error for error in errors if error is not None]


#: The name a policy-server group launches its first policy under.
_FIRST_POLICY = "policy"


@dataclass(frozen=True)
class _PolicyServerGroup:
    """Policy servers the runtime owns, of a frozen base model or of whole policies (a ``PolicyServers``)."""

    group: Any
    claimed: tuple[int, ...]
    service_handle: RayServiceHandle

    def handle(self) -> RayServiceHandle:
        return self.service_handle

    def ping(self) -> list[int]:
        return self.group.ping().wait()

    def features(self, *, policy: str) -> PolicyFeatures:
        return self.group.features(policy=policy).wait()[0]

    def set_weights(self, ref: WeightsRef, *, policy: str) -> list[int]:
        return self.group.set_weights(ref, policy=policy).wait()

    def stats(self, *, policy: str) -> list[dict[str, Any]]:
        return self.group.stats(policy=policy).wait()

    def reset_stats(self, *, policy: str) -> None:
        self.group.reset_stats(policy=policy).wait()

    def remove_policy(self, name: str) -> None:
        self.group.remove_policy(name).wait()


class RLInfRuntime:
    """Distribute provider-neutral rollout workers with RLInf and Ray."""

    name = "rlinf"
    # Policy servers serve both rollout worker kinds (runtimes/rlinf/layout.py places them).
    features = RuntimeFeatures(
        name=name,
        multi_node=True,
        policy_service=True,
        served_batched_collectors=True,
        distributed_learner=True,
        async_collect=True,
        weights_by_reference=False,
        serving_engines=("builtin",),
        served_parts=True,
        standalone_policy_service=True,
    )

    def __init__(self, config: RLInfRuntimeConfig | None = None) -> None:
        self.config = config or RLInfRuntimeConfig()
        self._cluster_owner = RLInfCluster(self.config)
        self._groups: list[RLInfRolloutGroup] = []
        self._learner_groups: list[RLInfLearnerGroup] = []
        self._base_services: dict[str, _PolicyServerGroup] = {}
        #: The whole-policy servers every rollout group with a served policy adds its policy to.
        self._policy_servers: _PolicyServerGroup | None = None
        #: The servers of services opened with open_policy_service.
        self._services: list[_PolicyServerGroup] = []

    def open_rollout_group(
        self,
        spec: RolloutWorkerSpec,
        *,
        num_rollout_workers: int,
        num_envs: int,
        name: str,
    ) -> RLInfRolloutGroup:
        _validate_positive_integer(
            num_rollout_workers,
            name="num_rollout_workers",
        )
        _validate_positive_integer(num_envs, name="num_envs")
        assignments = partition_environments(num_envs, num_rollout_workers)
        scheduler, cluster = self._cluster_owner.ensure()

        if isinstance(spec, GymWorkerSpec) and any(
            len(indices) != 1 for indices in assignments
        ):
            raise ValueError(
                "GymWorkerSpec requires num_rollout_workers == num_envs; "
                "use a BatchedWorkerSpec to host multiple environments per worker"
            )
        served_spec: GymWorkerSpec | BatchedWorkerSpec | None = None
        if isinstance(spec, GymWorkerSpec):
            served_spec = spec
            servable = spec.supports_batched_policy
        elif isinstance(spec, BatchedWorkerSpec):
            served_spec = spec
            servable = spec.supports_served_policy
        else:
            servable = False
        serve_base = (
            self.config.base_mode == "served"
            and served_spec is not None
            and not isinstance(served_spec.policy, ServiceHandle)
        )
        if served_spec is not None and isinstance(served_spec.policy, ServiceHandle):
            # The workers connect to a service opened elsewhere.
            served_policy = False
        else:
            served_policy = self.config.serves_policy(
                supports_batched_policy=servable,
                batched_worker=isinstance(spec, BatchedWorkerSpec),
            )

        # Workers holding a policy part, or simulating on a GPU, get accelerators (runtimes/rlinf/layout.py).
        holds_policy = not served_policy and not (
            served_spec is not None and isinstance(served_spec.policy, ServiceHandle)
        )
        needs_accelerators = self.config.device == "gpu" and (
            holds_policy
            or (served_spec is not None and served_spec.entry.simulates_on_gpu)
        )
        sizing = {
            "num_envs": num_envs,
            "num_rollout_workers": num_rollout_workers,
            "workers_need_accelerators": needs_accelerators,
        }
        hosted: HostedPolicy | None = None
        provider_group: Any = None
        output_channel: Any = None
        num_policy_servers = 0
        servers: HostedPolicy | None = None

        try:
            if serve_base:
                # The actor runs in every worker; the base-model servers belong to the runtime.
                spec, service = with_served_base_part(
                    spec,
                    lambda base_model: self._base_service(
                        scheduler, cluster, base_model, **sizing
                    ),
                )
                if service is not None:
                    servers = HostedPolicy(
                        service, _FIRST_POLICY, service.service_handle
                    )
                    num_policy_servers = service.service_handle.replicas
            elif served_policy:
                assert served_spec is not None and callable(served_spec.policy)
                hosted = self._serve_policy(
                    scheduler, cluster, served_spec.policy, name=name, **sizing
                )
                servers = hosted
                num_policy_servers = hosted.handle().replicas
                spec = replace(served_spec, policy=hosted.handle())

            placement, accelerators = self._rollout_placement(
                scheduler,
                cluster,
                num_rollout_workers=num_rollout_workers,
                needs_accelerators=needs_accelerators,
            )
            worker_class = make_rollout_worker(scheduler.Worker)
            with serialize_specs_by_value(
                spec,
                enabled=self.config.serialize_specs_by_value,
            ):
                provider_group = worker_class.create_group(
                    spec,
                    "gpu" if accelerators else "cpu",
                    assignments,
                ).launch(
                    cluster=cluster,
                    name=f"{name}_{uuid.uuid4().hex[:8]}",
                    placement_strategy=placement,
                )
            worker_indices = tuple(int(index) for index in provider_group.ping().wait())
            expected_indices = tuple(range(num_rollout_workers))
            if worker_indices != expected_indices:
                raise RuntimeError(
                    f"RLInf launched rollout workers {worker_indices}, expected "
                    f"{expected_indices}"
                )

            output_channel = scheduler.Channel.create(
                name=f"{name}_results_{uuid.uuid4().hex[:8]}",
                maxsize=max(2, num_rollout_workers * 2),
                distributed=self.config.num_nodes > 1,
            )
            # Wait for the channel actor: RLInf exits the process when one is closed before it runs.
            output_channel.qsize()
            group = RLInfRolloutGroup(
                provider_group,
                num_rollout_workers,
                num_envs=num_envs,
                assignments=assignments,
                output_channel=output_channel,
                policy=hosted,
                servers=servers,
                rollout_timeout_seconds=self.config.rollout_timeout_seconds,
                accelerators=accelerators,
            )
        except BaseException as exc:
            cleanup_errors = _terminate_launched(output_channel, provider_group)
            if hosted is not None:
                try:
                    hosted.close()
                except Exception as close_exc:  # noqa: BLE001
                    cleanup_errors.append(close_exc)
            if cleanup_errors:
                add_note(
                    exc, f"{len(cleanup_errors)} RLInf resource(s) also failed to close"
                )
            raise

        self._groups.append(group)
        mode = (
            "local actor, served base model"
            if serve_base
            else "served"
            if served_policy
            else "local"
        )
        logger.info(
            "RLInf rollout topology: {} environment(s), {} rollout worker(s), "
            "policy {}, {} policy server(s)",
            num_envs,
            num_rollout_workers,
            mode,
            num_policy_servers,
        )
        return group

    def _serve_policy(
        self,
        scheduler: Any,
        cluster: Any,
        policy_spec: Any,
        *,
        name: str,
        **sizing: Any,
    ) -> HostedPolicy:
        """Serve a rollout group's whole policy from the run's policy servers, started by the first group.

        Training and evaluation groups add their policies to the same servers,
        so a fresh evaluation group claims no accelerator of its own.
        """
        policy_name = f"{name}_{uuid.uuid4().hex[:6]}"
        servers = self._policy_servers
        if servers is None:
            group, claimed, replicas = self._open_policy_group(
                scheduler,
                cluster,
                policy_spec,
                name="policy",
                policy_name=policy_name,
                **sizing,
            )
            servers = self._policy_servers = _PolicyServerGroup(
                group,
                claimed,
                RayServiceHandle(
                    name="policy",
                    replicas=replicas,
                    actors=tuple(worker.worker for worker in group.worker_info_list),
                    timeout_seconds=self.config.policy_request_timeout_seconds,
                ),
            )
        else:
            with serialize_specs_by_value(
                policy_spec, enabled=self.config.serialize_specs_by_value
            ):
                servers.group.add_policy(
                    policy_name,
                    policy_spec,
                    self._batcher(
                        self.config.policy_batch_size,
                        self.config.policy_batch_wait_seconds,
                    ),
                    "builtin",
                    self.config.collect_policy_server_metrics is not False,
                ).wait()
        return HostedPolicy(
            servers,
            policy_name,
            replace(servers.service_handle, name=policy_name, policy=policy_name),
        )

    def _base_service(
        self,
        scheduler: Any,
        cluster: Any,
        base_model: FrozenBaseModelSpec,
        **sizing: Any,
    ) -> _PolicyServerGroup:
        """The policy servers of a frozen base model, started once and shared by every rollout group."""
        key = base_model.key()
        service = self._base_services.get(key)
        if service is None:
            group, claimed, replicas = self._open_policy_group(
                scheduler, cluster, base_model, name="base_model", **sizing
            )
            service = _PolicyServerGroup(
                group,
                claimed,
                RayServiceHandle(
                    name=f"base_model_{uuid.uuid4().hex[:6]}",
                    replicas=replicas,
                    actors=tuple(worker.worker for worker in group.worker_info_list),
                    timeout_seconds=self.config.policy_request_timeout_seconds,
                ),
            )
            self._base_services[key] = service
        return service

    def _open_policy_group(
        self,
        scheduler: Any,
        cluster: Any,
        policy_spec: Any,
        *,
        name: str,
        policy_name: str = _FIRST_POLICY,
        num_envs: int,
        num_rollout_workers: int,
        workers_need_accelerators: bool,
    ) -> tuple[Any, tuple[int, ...], int]:
        claimed: tuple[int, ...] = ()
        if self.config.device == "gpu":
            servers = policy_server_accelerators(
                self.config,
                accelerators=int(cluster.num_accelerators),
                num_envs=num_envs,
                num_rollout_workers=num_rollout_workers,
                workers_need_accelerators=workers_need_accelerators,
            )
            num_policy_servers = len(servers)
            first = servers[0]
            placement = scheduler.PackedPlacementStrategy(first, servers[-1])
            claimed = self._cluster_owner.claim_accelerators(
                first,
                num_policy_servers,
                f"policy servers {name!r}",
                exclusive=self.config.exclusive_accelerators,
            )
        else:
            num_policy_servers = self.config.num_policy_servers or 1
            node_indices = [
                index % self.config.num_nodes for index in range(num_policy_servers)
            ]
            placement = scheduler.NodePlacementStrategy(node_indices)

        policy_group = self._launch_policy_servers(
            scheduler,
            cluster,
            policy_spec,
            group_name=f"{name}_policy_{uuid.uuid4().hex[:8]}",
            policy_name=policy_name,
            placement=placement,
            num_policy_servers=num_policy_servers,
            device=self.config.device,
            claimed=claimed,
            batch_size=self.config.policy_batch_size,
            batch_wait_seconds=self.config.policy_batch_wait_seconds,
            collect_metrics=self.config.collect_policy_server_metrics is not False,
        )
        return policy_group, claimed, num_policy_servers

    def _launch_policy_servers(
        self,
        scheduler: Any,
        cluster: Any,
        policy_spec: Any,
        *,
        group_name: str,
        policy_name: str,
        placement: Any,
        num_policy_servers: int,
        device: str,
        claimed: tuple[int, ...],
        batch_size: int | None,
        batch_wait_seconds: float,
        collect_metrics: bool,
    ) -> Any:
        """Launch policy-server workers and wait for them; a failure releases ``claimed``."""
        worker_class = make_policy_worker(scheduler.Worker)
        policy_group: Any = None
        try:
            with serialize_specs_by_value(
                policy_spec,
                enabled=self.config.serialize_specs_by_value,
            ):
                policy_group = worker_class.create_group(
                    policy_spec,
                    device,
                    self._batcher(batch_size, batch_wait_seconds),
                    collect_metrics,
                    policy_name,
                ).launch(
                    cluster=cluster,
                    name=group_name,
                    placement_strategy=placement,
                    max_concurrency=max(2, (batch_size or 32) * 2),
                )
            indices = tuple(int(index) for index in policy_group.ping().wait())
            expected = tuple(range(num_policy_servers))
            if indices != expected:
                raise RuntimeError(
                    f"RLInf launched policy servers {indices}, expected {expected}"
                )
        except BaseException as exc:
            self._cluster_owner.release_accelerators(claimed)
            if policy_group is not None:
                error = terminate_worker_group(policy_group)
                if error is not None:
                    add_note(exc, f"RLInf policy-server cleanup also failed: {error!r}")
            raise
        return policy_group

    def _batcher(self, batch_size: int | None, batch_wait_seconds: float) -> Batcher:
        """Microbatching of a policy on the servers (``batch_size=None``: no cap)."""
        return Batcher(
            max_batch=batch_size,
            max_wait_ms=batch_wait_seconds * 1000.0,
            serve_eval_unbatched=self.config.deterministic_evaluation,
        )

    def _rollout_placement(
        self,
        scheduler: Any,
        cluster: Any,
        *,
        num_rollout_workers: int,
        needs_accelerators: bool,
    ) -> tuple[Any, tuple[int, ...]]:
        """Where the workers run, and each one's accelerator (none: on the CPU)."""
        if not needs_accelerators:
            node_indices = self.config.node_ranks
            if node_indices is None:
                node_indices = tuple(
                    index % self.config.num_nodes
                    for index in range(num_rollout_workers)
                )
            if len(node_indices) != num_rollout_workers:
                raise ValueError(
                    "node_ranks must contain one entry per rollout worker "
                    f"({num_rollout_workers}), got {len(node_indices)}"
                )
            invalid = [
                index
                for index in node_indices
                if index < 0 or index >= self.config.num_nodes
            ]
            if invalid:
                raise ValueError(
                    f"node_ranks contains invalid node indices {invalid} for "
                    f"num_nodes={self.config.num_nodes}"
                )
            return scheduler.NodePlacementStrategy(list(node_indices)), ()
        servers = [
            index
            for service in (
                *self._base_services.values(),
                self._policy_servers,
                *self._services,
            )
            if service is not None
            for index in service.claimed
        ]
        accelerators = rollout_worker_accelerators(
            self.config,
            accelerators=int(cluster.num_accelerators),
            num_rollout_workers=num_rollout_workers,
            servers=servers,
            busy=Counter(
                index
                for group in self._groups
                if not group.closed
                for index in group.accelerators
            ),
        )
        # Each worker sees only its own card; several workers may share one.
        placement = scheduler.FlexiblePlacementStrategy(
            [[index] for index in accelerators]
        )
        return placement, accelerators

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
    ) -> HostedPolicy:
        """Serve ``spec`` from ``replicas`` RLInf policy-server workers of their own, apart from any rollout group.

        ``gpus`` lists the accelerators the replicas run on, one each;
        without it they take the first free ones when ``device=gpu``, and
        run on the CPU otherwise. ``replicas`` defaults to one per listed
        accelerator. Unset batching options take ``policy_batch_size`` and
        ``policy_batch_wait_seconds``; statistics are always collected. The
        runtime closes the service on close.
        """
        if engine not in self.features.serving_engines:
            raise ValueError(
                f"runtime 'rlinf' serves with {self.features.serving_engines}, not {engine!r}"
            )
        scheduler, cluster = self._cluster_owner.ensure()
        num_nodes = self.config.num_nodes
        per_node = int(cluster.num_accelerators) // num_nodes
        on_cpu = gpus is None and self.config.device == "cpu"
        resources = Resources.uniform(num_nodes, 0 if on_cpu else per_node)
        batch_size = self.config.policy_batch_size if max_batch is None else max_batch
        wait_ms = (
            self.config.policy_batch_wait_seconds * 1000.0
            if max_wait_ms is None
            else max_wait_ms
        )
        role = plan_policy_service(
            resources,
            Served(
                replicas=replicas,
                gpus=gpus,
                max_batch=batch_size,
                max_wait_ms=wait_ms,
                engine=engine,
            ),
        )
        logger.info(f"policy service {name!r}: {role.describe()}")
        accelerators = [
            resources.nodes.index(node) * per_node + index
            for node, indices in role.gpus
            for index in indices
        ]
        if accelerators:
            placement = scheduler.FlexiblePlacementStrategy(
                [[index] for index in accelerators]
            )
            claimed = self._cluster_owner.claim_accelerator_list(
                accelerators,
                f"policy service {name!r}",
                exclusive=self.config.exclusive_accelerators,
            )
        else:
            placement = scheduler.NodePlacementStrategy(
                [index % num_nodes for index in range(role.processes)]
            )
            claimed = ()
        group = self._launch_policy_servers(
            scheduler,
            cluster,
            spec,
            group_name=name,
            policy_name=name,
            placement=placement,
            num_policy_servers=role.processes,
            device="gpu" if accelerators else "cpu",
            claimed=claimed,
            batch_size=batch_size,
            batch_wait_seconds=wait_ms / 1000.0,
            collect_metrics=True,
        )
        servers = _PolicyServerGroup(
            group,
            claimed,
            RayServiceHandle(
                name=name,
                replicas=role.processes,
                actors=tuple(worker.worker for worker in group.worker_info_list),
                timeout_seconds=self.config.policy_request_timeout_seconds,
                policy=name,
            ),
        )
        self._services.append(servers)

        def release() -> None:
            self._services.remove(servers)
            self._stop_servers(servers)

        return HostedPolicy(servers, name, servers.service_handle, release=release)

    def _stop_servers(self, servers: _PolicyServerGroup) -> None:
        self._cluster_owner.release_accelerators(servers.claimed)
        error = terminate_worker_group(servers.group)
        if error is not None:
            raise error

    def open_learner_group(
        self,
        spec: LearnerWorkerSpec,
        *,
        num_learner_ranks: int,
        name: str,
    ) -> RLInfLearnerGroup:
        """Launch algorithm-owned learners with PyTorch FSDP sharding."""
        _validate_positive_integer(
            num_learner_ranks,
            name="num_learner_ranks",
        )
        if not callable(spec):
            raise TypeError("learner spec must be callable")
        scheduler, cluster = self._cluster_owner.ensure()
        available = int(cluster.num_accelerators)
        first = self.config.learner_first_accelerator
        end = first + num_learner_ranks - 1
        if end >= available:
            raise RuntimeError(
                f"FSDP learner needs accelerators {first}..{end}, but RLInf "
                f"detected {available}"
            )
        claimed = self._cluster_owner.claim_accelerators(
            first,
            num_learner_ranks,
            f"learner group {name!r}",
            exclusive=self.config.exclusive_accelerators,
        )
        worker_class = make_learner_worker(scheduler.Worker)
        provider_group: Any = None
        try:
            placement = scheduler.PackedPlacementStrategy(first, end)
            with serialize_specs_by_value(
                spec,
                enabled=self.config.serialize_specs_by_value,
            ):
                provider_group = worker_class.create_group(
                    spec,
                    self.config.learner_mixed_precision,
                    self.config.learner_seed,
                ).launch(
                    cluster=cluster,
                    name=f"{name}_{uuid.uuid4().hex[:8]}",
                    placement_strategy=placement,
                )
            learner_ranks = tuple(
                int(rank) for rank in provider_group.initialize().wait()
            )
            expected = tuple(range(num_learner_ranks))
            if learner_ranks != expected:
                raise RuntimeError(
                    f"RLInf initialized learner ranks {learner_ranks}, "
                    f"expected {expected}"
                )
            group = RLInfLearnerGroup(
                provider_group,
                num_learner_ranks,
                release_accelerators=lambda: self._cluster_owner.release_accelerators(
                    claimed
                ),
            )
        except BaseException as exc:
            self._cluster_owner.release_accelerators(claimed)
            if provider_group is not None:
                error = terminate_worker_group(provider_group)
                if error is not None:
                    add_note(exc, f"RLInf learner cleanup also failed: {error!r}")
            raise
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
        for rollout_group in reversed(self._groups):
            try:
                rollout_group.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._groups.clear()
        servers = [*self._services, *self._base_services.values()]
        if self._policy_servers is not None:
            servers.append(self._policy_servers)
        for service in servers:
            try:
                self._stop_servers(service)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._services.clear()
        self._base_services.clear()
        self._policy_servers = None
        self._cluster_owner.close()
        if errors:
            raise RuntimeError(
                f"{len(errors)} RLInf rollout group(s) or policy service(s) failed to close"
            ) from errors[0]
