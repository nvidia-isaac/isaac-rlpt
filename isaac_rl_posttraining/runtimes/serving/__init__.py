# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy servers: one policy answering many rollout workers, independent of the runtime.

A :class:`PolicyServer` answers typed :class:`PolicyRequest`s with a serving
engine (``builtin`` runs today's acting policies), grouping them with a
:class:`Batcher`. Runtimes host servers (in-process for ``native``, Ray
actors for ``rlinf``) and rollout workers reach them through clients that
look like any local :class:`~isaac_rl_posttraining.runtimes.policy.RolloutPolicy`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from isaac_rl_posttraining.runtimes.base import DecisionMode, PolicyContext, WeightsRef
from isaac_rl_posttraining.runtimes.placement import ROUTINGS, Routing
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.messages import (
    BASE_MODEL_KINDS,
    REQUEST_KINDS,
    BasePartInputs,
    DecodeOptions,
    PolicyRequest,
    PolicyResponse,
    RequestKind,
)

__all__ = [
    "BASE_MODEL_KINDS",
    "REQUEST_KINDS",
    "ROUTINGS",
    "BasePartInputs",
    "Batcher",
    "BuiltinEngine",
    "DecodeOptions",
    "HostedPolicy",
    "InProcessPolicyService",
    "PolicyRequest",
    "PolicyResponse",
    "PolicyServer",
    "PolicyServerTimeoutError",
    "PolicyServers",
    "PolicyService",
    "RayServiceHandle",
    "RequestKind",
    "Routing",
    "ServedPolicy",
    "ServiceHandle",
    "ServingEngine",
    "connect_policy",
]


@runtime_checkable
class ServingEngine(Protocol):
    """What runs the model inside a policy server (``builtin`` wraps today's acting policies)."""

    @property
    def features(self) -> PolicyFeatures:
        """What the loaded policy can answer."""
        ...

    @property
    def version(self) -> int:
        """The weight version being served."""
        ...

    def load(self, spec: Any, context: PolicyContext) -> None:
        """Build the model from a policy spec on this server's device."""
        ...

    def run(
        self,
        kind: RequestKind,
        observations: Sequence[Any],
        sessions: Sequence[Any],
        modes: Sequence[DecisionMode | None],
        inputs: Sequence[BasePartInputs | None] | None = None,
    ) -> list[Any]:
        """Answer one batch of same-kind requests, one result per session.

        ``inputs`` carries base-model requests' inputs (``decode``, ``reference``).
        """
        ...

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        """Start new episodes for sessions (``seed`` seeds evaluation randomness)."""
        ...

    def load_weights(self, ref: WeightsRef) -> int:
        """Apply published weights; returns the version now served."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Facts a benchmark reports next to the server's timings."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class PolicyService(Protocol):
    """A running group of policy-server replicas, the same on every runtime.

    ``RolloutRuntime.open_policy_service`` returns one, and a training run's
    served policy is one too.
    """

    def handle(self) -> ServiceHandle:
        """The picklable address rollout workers use to send requests."""
        ...

    def client(self, route_index: int = 0) -> RolloutPolicy:
        """A client of the replica ``route_index`` routes to (``route_index % replicas``)."""
        ...

    def ping(self) -> tuple[int, ...]:
        """Every replica's index, once each has answered."""
        ...

    def features(self) -> PolicyFeatures:
        """What the served policy can answer."""
        ...

    def set_weights(self, ref: WeightsRef) -> tuple[int, ...]:
        """Publish weights to every replica; returns each replica's served version."""
        ...

    def stats(self) -> Mapping[str, Any]:
        """``{"replicas": (...)}``: per replica, batch sizes, latencies and timings since the last reset."""
        ...

    def reset_stats(self) -> None: ...

    def close(self) -> None: ...


# The implementations import the protocols above.
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher
from isaac_rl_posttraining.runtimes.serving.clients import (
    ServedPolicy,
)
from isaac_rl_posttraining.runtimes.serving.engine import BuiltinEngine
from isaac_rl_posttraining.runtimes.serving.hosted import HostedPolicy, PolicyServers
from isaac_rl_posttraining.runtimes.serving.local import (
    InProcessPolicyService,
    connect_policy,
)
from isaac_rl_posttraining.runtimes.serving.ray_client import (
    PolicyServerTimeoutError,
    RayServiceHandle,
)
from isaac_rl_posttraining.runtimes.serving.server import PolicyServer
