# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One policy-server process hosting one or more policies; Ray actors (native) and RLInf workers both run it.

Each policy has its own :class:`PolicyServer` behind an asyncio request queue,
so a run's training and evaluation rollout groups share the process (and,
through :class:`~isaac_rl_posttraining.runtimes.frozen_base_model.FrozenBaseModelSpec`,
one copy of the base model). Methods take ``policy`` (its name); it may be
omitted while the process hosts one policy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from isaac_rl_posttraining.runtimes.base import PolicyContext, WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher
from isaac_rl_posttraining.runtimes.serving.engine import make_engine
from isaac_rl_posttraining.runtimes.serving.messages import (
    PolicyRequest,
    PolicyResponse,
)
from isaac_rl_posttraining.runtimes.serving.queues import AsyncRequestQueue
from isaac_rl_posttraining.runtimes.serving.server import PolicyServer, replica_stats

__all__ = ["PolicyReplica"]


class PolicyReplica:
    """One policy-server replica: the policies it serves, each behind its own request queue."""

    def __init__(self, context: PolicyContext) -> None:
        self._context = context
        self._index = context.policy_index
        self._device = context.device
        self._policies: dict[str, tuple[PolicyServer, AsyncRequestQueue, float]] = {}

    def ping(self) -> int:
        return self._index

    def add_policy(
        self,
        name: str,
        spec: Callable[[PolicyContext], Any],
        batcher: Batcher,
        engine: str,
        collect_metrics: bool,
    ) -> int:
        """Build a policy (or a base model alone) from its spec and serve it under ``name``."""
        if name in self._policies:
            raise ValueError(f"policy server {self._index} already serves {name!r}")
        started = time.perf_counter()
        serving_engine = make_engine(engine)
        serving_engine.load(spec, self._context)
        server = PolicyServer(serving_engine, batcher, collect_metrics=collect_metrics)
        self._policies[name] = (
            server,
            AsyncRequestQueue(server),
            time.perf_counter() - started,
        )
        return self._index

    async def remove_policy(self, name: str) -> None:
        """Stop serving ``name``; the others keep their servers."""
        _, requests, _ = self._policies.pop(name)
        await requests.close()

    def _name(self, policy: str | None) -> str:
        if policy is not None:
            return policy
        if len(self._policies) != 1:
            raise ValueError(
                f"policy server {self._index} serves {sorted(self._policies)}; name the policy"
            )
        return next(iter(self._policies))

    def _server(self, policy: str | None) -> PolicyServer:
        return self._policies[self._name(policy)][0]

    def features(self, policy: str | None = None) -> PolicyFeatures:
        return self._server(policy).features

    def connect(self, policy: str | None = None) -> int:
        """Register a client of ``policy``; it passes the number with its requests."""
        return self._policies[self._name(policy)][1].connect()

    def disconnect(self, client: int, policy: str | None = None) -> None:
        # A client may outlive its policy, or name none on a server hosting one.
        entry = self._policies.get(policy or next(iter(self._policies), ""))
        if entry is not None:
            entry[1].disconnect(client)

    async def serve_requests(
        self,
        requests: Sequence[PolicyRequest],
        policy: str | None = None,
        client: int | None = None,
    ) -> list[PolicyResponse]:
        """Answer one client call; its requests are queued together."""
        return await self._policies[self._name(policy)][1].submit(requests, client)

    def reset_sessions(
        self,
        session_ids: Sequence[Hashable],
        seeds: Sequence[int | None],
        policy: str | None = None,
    ) -> None:
        self._server(policy).reset_sessions(session_ids, seeds)

    async def set_weights(self, weights: WeightsRef, policy: str | None = None) -> int:
        return self._server(policy).set_weights(weights)

    def stats(self, policy: str | None = None) -> dict[str, Any]:
        """The policy's counters since the last reset, and the process's peak GPU memory since then."""
        server, _, initialization_seconds = self._policies[self._name(policy)]
        return replica_stats(
            server,
            index=self._index,
            initialization_seconds=initialization_seconds,
            device=self._device,
        )

    def reset_stats(self, policy: str | None = None) -> None:
        self._server(policy).reset_stats()
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats()

    async def close(self) -> None:
        for name in list(self._policies):
            await self.remove_policy(name)
