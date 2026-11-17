# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The picklable address of a policy service."""

from __future__ import annotations

from dataclasses import dataclass

from isaac_rl_posttraining.runtimes.placement import ROUTINGS, Routing

__all__ = ["ServiceHandle"]


@dataclass(frozen=True)
class ServiceHandle:
    """A picklable address of a policy service that rollout workers connect to."""

    name: str
    replicas: int
    routing: Routing = "modulo"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("service name must be a non-empty string")
        if (
            isinstance(self.replicas, bool)
            or not isinstance(self.replicas, int)
            or self.replicas < 1
        ):
            raise ValueError("service replicas must be a positive integer")
        if self.routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}, got {self.routing!r}")

    def replica_for(self, environment_index: int) -> int:
        """The replica an environment uses under ``modulo`` routing."""
        if self.routing != "modulo":
            raise ValueError(f"{self.routing!r} routing picks replicas at request time")
        if environment_index < 0:
            raise ValueError("environment_index must be non-negative")
        return environment_index % self.replicas
