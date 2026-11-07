# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One policy on a runtime's policy-server processes, as the trainer sees it (a :class:`PolicyService`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from isaac_rl_posttraining.runtimes.base import WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.local import connect_policy

__all__ = ["HostedPolicy", "PolicyServers"]


class PolicyServers(Protocol):
    """Policy-server processes that host named policies; each call returns one answer per replica."""

    def ping(self) -> Sequence[int]: ...

    def features(self, *, policy: str) -> PolicyFeatures: ...

    def set_weights(self, ref: WeightsRef, *, policy: str) -> Sequence[int]: ...

    def stats(self, *, policy: str) -> Sequence[Mapping[str, Any]]: ...

    def reset_stats(self, *, policy: str) -> None: ...

    def remove_policy(self, name: str) -> None: ...


class HostedPolicy:
    """The policy ``name`` on ``servers``; rollout workers reach it through ``handle``.

    By default the servers belong to the runtime and outlive the policy:
    closing this removes the policy from them, and the runtime stops them when
    it closes. A service opened on servers of its own passes ``release``,
    which closing calls instead to stop them.
    """

    def __init__(
        self,
        servers: PolicyServers,
        name: str,
        handle: ServiceHandle,
        *,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.servers = servers
        self.name = name
        self._handle = handle
        self._release = release
        self._closed = False

    def handle(self) -> ServiceHandle:
        return self._handle

    def client(self, route_index: int = 0) -> RolloutPolicy:
        return connect_policy(self._handle, route_index=route_index)

    def ping(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.servers.ping())

    def features(self) -> PolicyFeatures:
        return self.servers.features(policy=self.name)

    def set_weights(self, ref: WeightsRef) -> tuple[int, ...]:
        """Every replica takes the published weights; returns the versions they serve."""
        served = tuple(
            int(version) for version in self.servers.set_weights(ref, policy=self.name)
        )
        if served != (ref.version,) * len(served):
            raise RuntimeError(
                f"policy servers of {self.name!r} acknowledged versions {served}, expected {ref.version}"
            )
        return served

    def stats(self) -> Mapping[str, Any]:
        return {"replicas": tuple(self.servers.stats(policy=self.name))}

    def reset_stats(self) -> None:
        self.servers.reset_stats(policy=self.name)

    def close(self) -> None:
        """Stop serving this policy (and stop its own servers, if it has them)."""
        if self._closed:
            return
        self._closed = True
        if self._release is None:
            self.servers.remove_policy(self.name)
        else:
            self._release()
