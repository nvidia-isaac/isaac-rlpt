# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf rollout placement and policy-serving configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from isaac_rl_posttraining.runtimes.base import BaseMode, PolicyMode, check_policy_modes
from isaac_rl_posttraining.runtimes.watchdog import DEFAULT_ROLLOUT_TIMEOUT_SECONDS

__all__ = ["RLInfRuntimeConfig"]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class RLInfRuntimeConfig:
    """RLInf cluster, rollout placement, and hidden serving policy."""

    num_nodes: int = 1
    device: Literal["cpu", "gpu"] = "cpu"
    node_ranks: tuple[int, ...] | None = None
    first_accelerator: int = 0
    learner_first_accelerator: int = 0
    #: Whether learner ranks and policy-server groups refuse an accelerator another group
    #: holds. Rollout workers go where :mod:`~isaac_rl_posttraining.runtimes.rlinf.layout`
    #: puts them, sharing cards when there are fewer than workers.
    exclusive_accelerators: bool = True
    #: Whether every GPU rollout worker of a group runs on ``first_accelerator`` instead of
    #: one card each: several batched workers (one per simulator server) on the learner's card.
    rollout_workers_share_accelerator: bool = False
    learner_mixed_precision: Literal["none", "bf16"] = "none"
    learner_seed: int = 0
    #: ``placement.policy.mode``; ``auto`` serves unbatched (Gym) workers on GPU.
    policy_mode: PolicyMode = "auto"
    #: ``served``: serve only the base model of a steering or edit policy from
    #: policy servers shared by the run's rollout groups; the workers (on the
    #: CPU) run the actor.
    base_mode: BaseMode = "local"
    num_policy_servers: int | None = None
    #: Record policy-server batch sizes and latencies (``auto``: when the runtime serves a policy).
    collect_policy_server_metrics: bool | Literal["auto"] = "auto"
    deterministic_evaluation: bool = True
    policy_batch_size: int = 32
    policy_batch_wait_seconds: float = 0.005
    #: Seconds a rollout worker waits for its policy server to answer one call.
    policy_request_timeout_seconds: float | None = 600.0
    #: Seconds a collection may take before it fails as frozen (``runtime.rollout_timeout_seconds``).
    rollout_timeout_seconds: float | None = DEFAULT_ROLLOUT_TIMEOUT_SECONDS
    shutdown_ray_on_close: bool = False
    serialize_specs_by_value: bool = True

    def __post_init__(self) -> None:
        _positive_integer(self.num_nodes, name="num_nodes")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        _non_negative_integer(self.first_accelerator, name="first_accelerator")
        _non_negative_integer(
            self.learner_first_accelerator, name="learner_first_accelerator"
        )
        if not isinstance(self.exclusive_accelerators, bool):
            raise TypeError("exclusive_accelerators must be a boolean")
        if not isinstance(self.rollout_workers_share_accelerator, bool):
            raise TypeError("rollout_workers_share_accelerator must be a boolean")
        if self.learner_mixed_precision not in ("none", "bf16"):
            raise ValueError("learner_mixed_precision must be 'none' or 'bf16'")
        _non_negative_integer(self.learner_seed, name="learner_seed")
        check_policy_modes(self.policy_mode, self.base_mode)
        if self.num_policy_servers is not None:
            _positive_integer(
                self.num_policy_servers,
                name="num_policy_servers",
            )
        if (
            self.policy_mode == "local"
            and self.base_mode == "local"
            and self.num_policy_servers
        ):
            raise ValueError("num_policy_servers only applies to a served policy")
        if self.collect_policy_server_metrics != "auto" and not isinstance(
            self.collect_policy_server_metrics, bool
        ):
            raise TypeError("collect_policy_server_metrics must be a boolean or 'auto'")
        if not isinstance(self.deterministic_evaluation, bool):
            raise TypeError("deterministic_evaluation must be a boolean")
        _positive_integer(self.policy_batch_size, name="policy_batch_size")
        if (
            isinstance(self.policy_batch_wait_seconds, bool)
            or not isinstance(self.policy_batch_wait_seconds, (int, float))
            or self.policy_batch_wait_seconds < 0
        ):
            raise ValueError("policy_batch_wait_seconds must be non-negative")
        if self.policy_request_timeout_seconds is not None and (
            isinstance(self.policy_request_timeout_seconds, bool)
            or not isinstance(self.policy_request_timeout_seconds, (int, float))
            or self.policy_request_timeout_seconds <= 0
        ):
            raise ValueError("policy_request_timeout_seconds must be positive or None")
        if self.rollout_timeout_seconds is not None and (
            isinstance(self.rollout_timeout_seconds, bool)
            or not isinstance(self.rollout_timeout_seconds, (int, float))
            or self.rollout_timeout_seconds <= 0
        ):
            raise ValueError("rollout_timeout_seconds must be positive or None")
        if not isinstance(self.shutdown_ray_on_close, bool):
            raise TypeError("shutdown_ray_on_close must be a boolean")
        if not isinstance(self.serialize_specs_by_value, bool):
            raise TypeError("serialize_specs_by_value must be a boolean")

    def serves_policy(
        self, *, supports_batched_policy: bool, batched_worker: bool = False
    ) -> bool:
        """Whether rollout workers call policy servers for the whole policy.

        ``auto`` serves unbatched (Gym) workers on GPU; batched workers use
        policy servers only when ``policy_mode="served"``.
        """
        if self.policy_mode == "served":
            if not supports_batched_policy:
                raise ValueError(
                    "a served policy requires a worker factory whose policy "
                    "supports batched inference"
                )
            return True
        if self.policy_mode == "local" or self.base_mode == "served":
            return False
        return self.device == "gpu" and supports_batched_policy and not batched_worker
