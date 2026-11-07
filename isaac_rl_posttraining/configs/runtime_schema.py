# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured schema of the ``runtime`` and ``placement`` sections (re-exported by :mod:`.schema`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "EnvsPlacementConfig",
    "LearnerPlacementConfig",
    "NativeRuntimeSettings",
    "PlacementConfig",
    "PolicyPlacementConfig",
    "RLInfRuntimeSettings",
    "RuntimeConfig",
]


@dataclass
class NativeRuntimeSettings:
    """Settings used only when the native rollout runtime is selected.

    Whether workers share an in-process policy server, and its batching, are
    ``placement.policy.*`` (mode, max_batch, max_wait_ms).
    """

    device: str = "auto"
    # Batch evaluation requests only within one worker's call, so request order cannot change a score.
    serve_evaluation_unbatched: bool = True
    # Workers of a group collect on threads so a served policy batches their requests
    # (auto: when the placement serves the policy and a group has several workers).
    concurrent_collection: Any = "auto"
    # Record policy-server batch sizes and latencies (auto: when the placement serves the policy).
    collect_policy_server_metrics: Any = "auto"
    # When the roles run as Ray actors (auto: when the placement asks for several processes).
    processes: str = "auto"
    num_nodes: int = 1
    ray_address: Optional[str] = None
    startup_timeout_seconds: float = 900.0
    node_join_timeout_seconds: float = 120.0
    control_timeout_seconds: float = 300.0
    call_timeout_seconds: Optional[float] = None
    policy_request_timeout_seconds: Optional[float] = 600.0
    collective_timeout_seconds: Optional[float] = 1800.0
    learner_mixed_precision: str = "none"
    learner_seed: int = 0
    seed: Optional[int] = None
    serialize_specs_by_value: bool = True


@dataclass
class RLInfRuntimeSettings:
    """Settings used only when the RLInf rollout runtime is selected."""

    num_nodes: int = 1
    device: str = "cpu"
    node_ranks: Optional[list[int]] = None
    first_accelerator: int = 1
    learner_first_accelerator: int = 0
    exclusive_accelerators: bool = True
    rollout_workers_share_accelerator: bool = False
    learner_mixed_precision: str = "none"
    learner_seed: int = 0
    # Record policy-server batch sizes and latencies (auto: when the runtime serves the policy).
    collect_policy_server_metrics: Any = "auto"
    # Serve evaluation requests one at a time so batching cannot change a score.
    serve_evaluation_unbatched: bool = True
    # Deprecated aliases: policy_serving, num_policy_servers, policy_batch_size and
    # policy_batch_wait_seconds are placement.policy.* (mode, replicas, max_batch, max_wait_ms);
    # deterministic_evaluation is serve_evaluation_unbatched.
    policy_serving: str = "auto"
    num_policy_servers: Optional[int] = None
    policy_batch_size: int = 32
    policy_batch_wait_seconds: float = 0.005
    deterministic_evaluation: bool = True
    # Seconds a rollout worker waits for its policy server to answer one call (null: no limit).
    policy_request_timeout_seconds: Optional[float] = 600.0
    shutdown_ray_on_close: bool = False
    serialize_specs_by_value: bool = True


@dataclass
class RuntimeConfig:
    """Rollout execution selection, independent of simulator and algorithm."""

    name: str = "native"
    # Deprecated alias of placement.envs.workers.
    num_rollout_workers: Optional[int] = None
    # Seconds one collection may take, per episode each worker was asked for, before the run fails with the late rollout workers named and
    # writes its aborted checkpoint (a frozen simulator otherwise hangs the run); null: no limit.
    rollout_timeout_seconds: Optional[float] = 1800.0
    native: NativeRuntimeSettings = field(default_factory=NativeRuntimeSettings)
    rlinf: RLInfRuntimeSettings = field(default_factory=RLInfRuntimeSettings)

    def __post_init__(self) -> None:
        if isinstance(self.native, dict):
            self.native = NativeRuntimeSettings(**self.native)
        if isinstance(self.rlinf, dict):
            self.rlinf = RLInfRuntimeSettings(**self.rlinf)
        runtimes = ["native", "rlinf"]
        if self.name not in runtimes:
            raise ValueError(
                f"runtime.name must be one of {', '.join(runtimes)}; got {self.name!r}"
            )
        timeout = self.rollout_timeout_seconds
        if timeout is not None and (isinstance(timeout, bool) or not timeout > 0):
            raise ValueError("runtime.rollout_timeout_seconds must be null or positive")


@dataclass
class EnvsPlacementConfig:
    """Rollout-worker processes and their GPUs (``placement.envs``)."""

    workers: Optional[int] = field(
        default=None, metadata={"help": "Rollout workers (null: the simulator entry's layout decides)"}
    )
    gpus: Any = field(default=None, metadata={"help": "A GPU count or ['node:index', ...] (null: planner)"})


@dataclass
class PolicyPlacementConfig:
    """Where a policy (or one policy part) runs (``placement.policy``, ``placement.parts.<part>``)."""

    mode: str = field(
        default="auto",
        metadata={"help": "local (a copy in every rollout worker), served (policy servers) or auto (runtime decides)"},
    )
    replicas: Optional[int] = field(default=None, metadata={"help": "Policy-server replicas (served)"})
    gpus: Any = field(default=None, metadata={"help": "A GPU count or ['node:index', ...] (served)"})
    gpus_per_replica: int = 1
    max_batch: Optional[int] = field(
        default=None, metadata={"help": "Largest batch a policy server runs (null: the runtime's default)"}
    )
    max_wait_ms: Optional[float] = field(
        default=None, metadata={"help": "How long a policy server waits to fill a batch (null: the runtime's default)"}
    )
    engine: str = "builtin"
    routing: str = "modulo"

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "local", "served"):
            raise ValueError(f"placement policy mode must be auto, local or served, got {self.mode!r}")


@dataclass
class LearnerPlacementConfig:
    """The learner's ranks and GPUs (``placement.learner``)."""

    ranks: int = field(
        default=1,
        metadata={
            "help": "Data-parallel learner processes (FSDP ranks under runtime=rlinf, or runtime=native with "
            "runtime.native.processes=auto|ray); 0 evaluates an on-policy checkpoint without a learner"
        },
    )
    gpus: Any = field(default=None, metadata={"help": "A GPU count or ['node:index', ...] (null: planner)"})

    def __post_init__(self) -> None:
        if isinstance(self.ranks, bool) or not isinstance(self.ranks, int) or self.ranks < 0:
            raise ValueError("placement.learner.ranks must be a non-negative integer")


@dataclass
class PlacementConfig:
    """Which role runs where (``placement``); the dry run prints the resolved plan.

    Unset values are the planner's to decide. ``parts`` overrides single policy
    parts (``base``, ``actor``); ``gpus_per_node`` overrides the GPUs the
    planner assumes per node (null: the GPUs visible here).
    """

    envs: EnvsPlacementConfig = field(default_factory=EnvsPlacementConfig)
    policy: PolicyPlacementConfig = field(default_factory=PolicyPlacementConfig)
    parts: dict[str, PolicyPlacementConfig] = field(default_factory=dict)
    learner: LearnerPlacementConfig = field(default_factory=LearnerPlacementConfig)
    nodes: Optional[list[str]] = None
    gpus_per_node: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.envs, dict):
            self.envs = EnvsPlacementConfig(**self.envs)
        if isinstance(self.policy, dict):
            self.policy = PolicyPlacementConfig(**self.policy)
        if isinstance(self.learner, dict):
            self.learner = LearnerPlacementConfig(**self.learner)
        self.parts = {
            name: PolicyPlacementConfig(**part) if isinstance(part, dict) else part
            for name, part in dict(self.parts or {}).items()
        }
