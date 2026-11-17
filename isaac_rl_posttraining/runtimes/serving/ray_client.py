# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A policy client over Ray: requests travel to a policy-server actor and back."""

from __future__ import annotations

import importlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.messages import (
    PolicyRequest,
    PolicyResponse,
)

__all__ = [
    "PolicyServerTimeoutError",
    "RayServiceHandle",
    "RayTransport",
    "ray_replica_for",
]


class PolicyServerTimeoutError(TimeoutError):
    """A policy server did not answer within the client's request timeout."""


@dataclass(frozen=True)
class RayServiceHandle(ServiceHandle):
    """A service handle that carries its Ray policy-server actors (picklable)."""

    actors: tuple[Any, ...] = ()
    #: Seconds a client waits for one call; ``None`` waits forever.
    timeout_seconds: float | None = None
    #: The actor that assigns clients to replicas under ``least_loaded`` routing.
    router: Any = None
    #: The policy a client addresses on actors that host several (``None``: actors hosting one).
    policy: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "actors", tuple(self.actors))
        if len(self.actors) != self.replicas:
            raise ValueError(
                f"a Ray service with {self.replicas} replica(s) needs as many actors, "
                f"got {len(self.actors)}"
            )
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number or None")


class RayTransport:
    """Requests to one policy-server actor; ``policy`` names one policy of actors that host several."""

    def __init__(
        self, actor: Any, *, timeout_seconds: float | None, policy: str | None = None
    ) -> None:
        self._actor = actor
        self._timeout_seconds = timeout_seconds
        self._policy = policy
        self._client = self._get(
            actor.connect.remote(**self._addressed()), "a client connection"
        )

    def _addressed(self) -> dict[str, str]:
        return {} if self._policy is None else {"policy": self._policy}

    def _get(self, reference: Any, what: str) -> Any:
        ray = importlib.import_module("ray")
        try:
            return ray.get(reference, timeout=self._timeout_seconds)
        except ray.exceptions.GetTimeoutError as exc:
            raise PolicyServerTimeoutError(
                f"the policy server did not answer {what} within "
                f"{self._timeout_seconds} s; it may be overloaded or stuck (check its log), "
                "or raise runtime.native.policy_request_timeout_seconds "
                "(runtime.rlinf.policy_request_timeout_seconds under runtime=rlinf)"
            ) from exc

    def submit(self, requests: Sequence[PolicyRequest]) -> list[PolicyResponse]:
        kinds = sorted({request.kind for request in requests})
        return list(
            self._get(
                self._actor.serve_requests.remote(
                    list(requests), client=self._client, **self._addressed()
                ),
                f"{len(requests)} {'/'.join(kinds)} request(s)",
            )
        )

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        self._get(
            self._actor.reset_sessions.remote(
                list(session_ids), list(seeds), **self._addressed()
            ),
            f"an episode reset of {len(session_ids)} session(s)",
        )

    def features(self) -> PolicyFeatures:
        return self._get(
            self._actor.features.remote(**self._addressed()), "a capability query"
        )

    def close(self) -> None:
        # Not awaited: a server that is gone has no client to forget.
        self._actor.disconnect.remote(self._client, **self._addressed())


def ray_replica_for(handle: RayServiceHandle, route_index: int) -> Any:
    """The policy-server actor a client with ``route_index`` talks to.

    ``modulo`` routing maps the index onto a replica; ``least_loaded`` asks
    the service's router for the replica with the fewest clients so far (a
    client keeps its replica, which holds its sessions' state).
    """
    if handle.routing == "modulo":
        return handle.actors[handle.replica_for(route_index)]
    if handle.router is None:
        raise ValueError(
            f"{handle.routing!r} routing needs a service with a replica router (runtime=native)"
        )
    ray = importlib.import_module("ray")
    return handle.actors[
        int(ray.get(handle.router.assign.remote(), timeout=handle.timeout_seconds))
    ]
