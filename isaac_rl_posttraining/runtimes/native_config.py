# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings of the native runtime (:class:`NativeRuntimeConfig`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from isaac_rl_posttraining.runtimes.base import BaseMode, PolicyMode, check_policy_modes
from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
from isaac_rl_posttraining.runtimes.serving.batcher import (
    DEFAULT_MAX_WAIT_MS,
    Batcher,
)
from isaac_rl_posttraining.runtimes.watchdog import DEFAULT_ROLLOUT_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.base import RolloutWorkerSpec

__all__ = ["NativeRuntimeConfig", "Processes"]

#: ``auto`` runs several processes when the placement asks for them
#: (:func:`~isaac_rl_posttraining.runtimes.native.wants_processes`), ``single``
#: keeps every role in this process, ``ray`` always runs the roles as Ray actors.
Processes = Literal["auto", "single", "ray"]

#: ``auto`` turns a setting on for a served policy (``placement.policy`` or
#: ``placement.parts.base`` served) and off otherwise.
Auto = Literal["auto"]


def _auto_or_bool(value: object, name: str) -> None:
    if value != "auto" and not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean or 'auto', got {value!r}")


@dataclass(frozen=True)
class NativeRuntimeConfig:
    """Placement and policy serving for the in-process runtime.

    A served policy is a policy server that batches the requests of rollout
    workers that act at the same time: up to ``policy_max_batch`` requests
    (``None``: no limit; the engine still splits a decode by the policy's own
    decode batch size), waiting up to ``policy_max_wait_ms`` after the first
    (default :data:`DEFAULT_MAX_WAIT_MS`). Policy-server actors and
    concurrent collection (workers of a group collect on threads) wait;
    in-process workers collecting one after another have nothing to wait for
    and are served at once. ``policy_serve_eval_unbatched`` batches evaluation
    requests only within one worker's call, so the order requests arrive in
    cannot change a checkpoint's score.

    ``concurrent_collection``: ``auto`` collects on threads when the placement
    serves the policy (or its base model) and a group has several rollout
    workers, and one worker after another otherwise; ``true``/``false`` force it.
    ``collect_policy_server_metrics``: ``auto`` records the servers' batch sizes
    and latencies (``policy_service/*``) when the placement serves the policy.

    ``policy_mode="auto"`` serves one policy to unbatched (Gym) workers whose
    spec allows it and gives batched workers their own; ``served`` serves every
    worker kind from one in-process policy server. ``base_mode="served"``
    serves only the base model (``placement.parts.base``):
    every worker keeps its own actor and sends decodes to one base-model server
    that the run's training and evaluation groups share.

    The remaining fields apply when the roles run in their own processes:
    ``processes`` chooses that (see :data:`Processes`), ``num_nodes`` is how
    many Ray nodes the placement plans onto, ``ray_address`` names a cluster
    to join (``None``: the running one, else a local one is started).
    ``startup_timeout_seconds`` bounds starting an actor or joining the
    learner process group, ``node_join_timeout_seconds`` waiting for the
    plan's Ray nodes to join, ``control_timeout_seconds`` a weight update,
    hook or close, ``call_timeout_seconds`` an update or checkpoint
    (``None``: no limit; a dead actor fails the call at once either way),
    ``policy_request_timeout_seconds`` one policy-server call and
    ``collective_timeout_seconds`` one collective among learner ranks.
    ``seed`` seeds each actor's random generators from its role and index.

    ``rollout_timeout_seconds`` bounds one collection, per episode each worker was asked for, in either mode (``None``:
    no limit): past it the collection raises
    :class:`~isaac_rl_posttraining.runtimes.watchdog.RolloutTimeoutError`
    naming the late rollout workers and their devices (one process: see
    :class:`~isaac_rl_posttraining.runtimes.watchdog.CollectionWatchdog`).
    """

    device: Literal["auto", "cpu", "gpu"] = "auto"
    policy_mode: PolicyMode = "auto"
    base_mode: BaseMode = "local"
    policy_max_batch: int | None = None
    policy_max_wait_ms: float = DEFAULT_MAX_WAIT_MS
    policy_serve_eval_unbatched: bool = True
    concurrent_collection: bool | Auto = "auto"
    collect_policy_server_metrics: bool | Auto = "auto"
    processes: Processes = "auto"
    num_nodes: int = 1
    ray_address: str | None = None
    startup_timeout_seconds: float = 900.0
    node_join_timeout_seconds: float = 120.0
    control_timeout_seconds: float = 300.0
    call_timeout_seconds: float | None = None
    rollout_timeout_seconds: float | None = DEFAULT_ROLLOUT_TIMEOUT_SECONDS
    policy_request_timeout_seconds: float | None = 600.0
    collective_timeout_seconds: float | None = 1800.0
    learner_mixed_precision: str = "none"
    learner_seed: int = 0
    seed: int | None = None
    serialize_specs_by_value: bool = True

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        check_policy_modes(self.policy_mode, self.base_mode)
        for name in ("policy_serve_eval_unbatched", "serialize_specs_by_value"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        _auto_or_bool(self.concurrent_collection, "concurrent_collection")
        _auto_or_bool(
            self.collect_policy_server_metrics, "collect_policy_server_metrics"
        )
        if self.processes not in ("auto", "single", "ray"):
            raise ValueError("processes must be 'auto', 'single', or 'ray'")
        if (
            isinstance(self.num_nodes, bool)
            or not isinstance(self.num_nodes, int)
            or self.num_nodes < 1
        ):
            raise ValueError("num_nodes must be a positive integer")
        if self.learner_mixed_precision not in ("none", "bf16"):
            raise ValueError("learner_mixed_precision must be 'none' or 'bf16'")
        for name in (
            "startup_timeout_seconds",
            "node_join_timeout_seconds",
            "control_timeout_seconds",
            "call_timeout_seconds",
            "rollout_timeout_seconds",
            "policy_request_timeout_seconds",
            "collective_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not value > 0):
                raise ValueError(f"{name} must be a positive number")
        self.batcher()

    @property
    def placement_serves(self) -> bool:
        """Whether the placement serves the policy or its base model (not the ``auto`` in-process sharing)."""
        return self.policy_mode == "served" or self.base_mode == "served"

    def collects_concurrently(self, num_rollout_workers: int) -> bool:
        """Whether a group of ``num_rollout_workers`` in-process workers collects on threads."""
        if self.concurrent_collection == "auto":
            return self.placement_serves and num_rollout_workers > 1
        return bool(self.concurrent_collection)

    @property
    def collects_server_metrics(self) -> bool:
        """Whether policy servers record batch sizes and latencies."""
        if self.collect_policy_server_metrics == "auto":
            return self.placement_serves
        return bool(self.collect_policy_server_metrics)

    @property
    def base_service_waits(self) -> bool:
        """Whether the in-process base-model service, which every group shares, waits for concurrent requests."""
        if self.concurrent_collection == "auto":
            return self.placement_serves
        return bool(self.concurrent_collection)

    def resolved_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "native device=gpu requires an available CUDA device"
                )
            return "cuda:0"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def batcher(self, *, concurrent: bool = True) -> Batcher:
        """The server batching; ``concurrent=False`` (requests arrive one at a time) waits for nothing."""
        return Batcher(
            max_batch=self.policy_max_batch,
            max_wait_ms=self.policy_max_wait_ms if concurrent else 0.0,
            serve_eval_unbatched=self.policy_serve_eval_unbatched,
        )

    def serves_policy(self, spec: RolloutWorkerSpec) -> bool:
        """Whether a group opened from ``spec`` builds an in-process policy server for its workers."""
        if self.policy_mode == "local" or (
            self.policy_mode == "auto" and self.base_mode == "served"
        ):
            return False
        if isinstance(spec, GymWorkerSpec):
            servable = spec.supports_batched_policy
            batched = False
        elif isinstance(spec, BatchedWorkerSpec):
            servable = spec.supports_served_policy
            batched = True
        else:
            if self.policy_mode == "served":
                raise ValueError(
                    "policy_mode='served' needs a GymWorkerSpec or BatchedWorkerSpec"
                )
            return False
        if self.policy_mode == "served":
            if not servable:
                raise ValueError(
                    "policy_mode='served' needs a worker spec whose policy may be "
                    "served (supports_batched_policy / supports_served_policy)"
                )
            return True
        return servable and not batched
