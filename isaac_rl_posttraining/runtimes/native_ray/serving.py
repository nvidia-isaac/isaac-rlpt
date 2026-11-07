# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy servers as Ray actors: ``replicas`` x ``gpus_per_replica`` on the plan's policy GPUs.

Each served role of the plan starts one set of policy-server actors for the
whole run. Every rollout group adds its policy to them (a run's training and
evaluation policies, or a base model served alone), so the groups share the
GPUs and the base model, which a server process builds once
(:class:`~isaac_rl_posttraining.runtimes.frozen_base_model.FrozenBaseModelSpec`).
Each policy has its own :class:`~isaac_rl_posttraining.runtimes.serving.PolicyServer`
with a serving engine behind an asyncio request queue, so requests from many
rollout workers are batched as in-process. Rollout workers reach a replica
through a :class:`~isaac_rl_posttraining.runtimes.serving.RayServiceHandle`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from isaac_rl_posttraining.runtimes._serialization import serialize_specs_by_value
from isaac_rl_posttraining.runtimes.base import PolicyContext, WeightsRef
from isaac_rl_posttraining.runtimes.native_ray.cluster import (
    ActorFailedError,
    ActorPlacement,
    RayCluster,
    close_actors,
    import_ray,
    remote_class,
    seed_process,
    wait_all,
    weights_by_reference,
    writable_state,
)
from isaac_rl_posttraining.runtimes.placement import Routing
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher
from isaac_rl_posttraining.runtimes.serving.hosted import HostedPolicy
from isaac_rl_posttraining.runtimes.serving.ray_client import RayServiceHandle
from isaac_rl_posttraining.runtimes.serving.replica import PolicyReplica

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.native import NativeRuntimeConfig

__all__ = ["PolicyServerActor", "RayPolicyService", "ReplicaRouter"]


class ReplicaRouter:
    """Assigns each new client the replica with the fewest clients (least-loaded routing), one at a time."""

    def __init__(self, replicas: int) -> None:
        self._clients = [0] * replicas

    def assign(self) -> int:
        replica = min(range(len(self._clients)), key=self._clients.__getitem__)
        self._clients[replica] += 1
        return replica

    def clients(self) -> list[int]:
        return list(self._clients)


class PolicyServerActor(PolicyReplica):
    """One policy-server replica in its own process (an asyncio Ray actor)."""

    def __init__(self, context: PolicyContext, seed: int | None) -> None:
        import_ray()  # registers the resolver of weights published by reference
        seed_process(seed)
        super().__init__(context)

    async def set_weights(self, weights: WeightsRef, policy: str | None = None) -> int:
        """Fetch published weights without blocking the event loop, then serve them."""
        server = self._server(policy)
        if weights.handle is not None and weights.version != server.version:
            state = await weights.handle
            weights = WeightsRef.of(writable_state(state), parts=weights.parts)
        return server.set_weights(weights)


class RayPolicyService:
    """The policy-server actors of one served role; rollout groups add their policies to them."""

    def __init__(
        self,
        cluster: RayCluster,
        *,
        role: str,
        name: str,
        replicas: int,
        gpus_per_replica: int,
        routing: Routing,
        max_batch: int | None,
        config: NativeRuntimeConfig,
    ) -> None:
        self.name = name
        self.config = config
        self._actors: list[Any] = []
        self._placements: list[ActorPlacement] = []
        self._router: Any = None
        self._routing = routing
        self._policies: dict[str, HostedPolicy] = {}
        actor_class = remote_class(PolicyServerActor)
        token = uuid.uuid4().hex[:6]
        try:
            for index in range(replicas):
                placement = cluster.place(
                    role,
                    index,
                    gpus_per_process=gpus_per_replica,
                    name=f"{name}_server{index}_{token}",
                    max_concurrency=max(8, 2 * (max_batch or 64)),
                )
                context = PolicyContext(
                    policy_index=index,
                    num_policies=replicas,
                    device=placement.device,
                    placement="policy_server",
                )
                seed = (
                    None if config.seed is None else config.seed + 104_729 * (index + 1)
                )
                self._placements.append(placement)
                self._actors.append(
                    actor_class.options(**placement.options).remote(context, seed)
                )
            self._wait(
                [actor.ping.remote() for actor in self._actors],
                f"starting policy servers {name!r}",
                config.startup_timeout_seconds,
            )
            if routing == "least_loaded":
                self._router = (
                    remote_class(ReplicaRouter)
                    .options(num_cpus=0, max_restarts=0)
                    .remote(replicas)
                )
        except BaseException:
            self._kill()
            raise
        self._closed = False

    def describe_replica(self, index: int) -> str:
        return self._placements[index].describe("policy server")

    def _wait(
        self, calls: Sequence[Any], what: str, timeout: float | None
    ) -> list[Any]:
        return wait_all(
            calls, describe=self.describe_replica, timeout_seconds=timeout, what=what
        )

    def add_policy(
        self,
        spec: Callable[[PolicyContext], Any],
        *,
        name: str,
        batcher: Batcher,
        engine: str = "builtin",
        collect_metrics: bool,
        release: Callable[[], None] | None = None,
    ) -> HostedPolicy:
        """Serve ``spec`` from every replica under ``name``; the returned view publishes its weights.

        ``release`` is what closing the view does instead of removing the
        policy (a standalone service stopping its own servers).
        """
        if self._closed:
            raise RuntimeError(f"policy servers {self.name!r} are closed")
        if name in self._policies:
            raise ValueError(f"policy servers {self.name!r} already serve {name!r}")
        with serialize_specs_by_value(
            spec, enabled=self.config.serialize_specs_by_value
        ):
            spec_ref = import_ray().put(spec)
        self._wait(
            [
                actor.add_policy.remote(
                    name,
                    spec_ref,
                    batcher,
                    engine,
                    collect_metrics,
                )
                for actor in self._actors
            ],
            f"policy servers {self.name!r} loading {name!r}",
            self.config.startup_timeout_seconds,
        )
        hosted = HostedPolicy(
            self,
            name,
            RayServiceHandle(
                name=name,
                replicas=len(self._actors),
                routing=self._routing,
                actors=tuple(self._actors),
                timeout_seconds=self.config.policy_request_timeout_seconds,
                router=self._router,
                policy=name,
            ),
            release=release,
        )
        self._policies[name] = hosted
        return hosted

    def call(
        self, method: str, *args: Any, what: str, timeout: float | None, **kwargs: Any
    ) -> list[Any]:
        return self._wait(
            [getattr(actor, method).remote(*args, **kwargs) for actor in self._actors],
            f"policy servers {self.name!r}: {what}",
            timeout,
        )

    def ping(self) -> list[int]:
        return self.call(
            "ping", what="ping", timeout=self.config.control_timeout_seconds
        )

    def features(self, *, policy: str) -> PolicyFeatures:
        return self.call(
            "features",
            policy=policy,
            what=f"{policy!r} features",
            timeout=self.config.control_timeout_seconds,
        )[0]

    def set_weights(self, ref: WeightsRef, *, policy: str) -> list[int]:
        """Every replica fetches the weights published once by reference."""
        ref = weights_by_reference(ref)
        return self.call(
            "set_weights",
            ref,
            policy=policy,
            what=f"{policy!r} taking weights version {ref.version}",
            timeout=self.config.control_timeout_seconds,
        )

    def stats(self, *, policy: str) -> list[dict[str, Any]]:
        return self.call(
            "stats",
            policy=policy,
            what=f"{policy!r} statistics",
            timeout=self.config.control_timeout_seconds,
        )

    def reset_stats(self, *, policy: str) -> None:
        self.call(
            "reset_stats",
            policy=policy,
            what=f"{policy!r} statistics reset",
            timeout=self.config.control_timeout_seconds,
        )

    def remove_policy(self, name: str) -> None:
        if self._closed or self._policies.pop(name, None) is None:
            return
        try:
            self.call(
                "remove_policy",
                name,
                what=f"removing {name!r}",
                timeout=self.config.control_timeout_seconds,
            )
        except ActorFailedError as exc:
            # A replica that died ends the run; the servers are killed when it closes.
            logger.warning(
                f"policy servers {self.name!r} could not remove {name!r}: {exc}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._policies.clear()
        errors = close_actors(
            self._actors,
            describe=self.describe_replica,
            timeout_seconds=self.config.control_timeout_seconds,
        )
        self._actors.clear()
        self._kill()
        if errors:
            raise errors[0]

    def _kill(self) -> None:
        ray = import_ray()
        for actor in [
            *self._actors,
            *([self._router] if self._router is not None else []),
        ]:
            ray.kill(actor)
        self._actors.clear()
        self._router = None
