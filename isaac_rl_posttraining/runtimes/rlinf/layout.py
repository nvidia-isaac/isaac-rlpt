# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which accelerators a ``runtime=rlinf`` rollout group's policy servers and rollout workers use.

One rule, applied by the runtime when it opens a group and by the dry run
(:mod:`isaac_rl_posttraining.runners.rlinf_placement`). The rollout
accelerators are ``first_accelerator`` onwards. Policy servers take the first
of them: one per four environments, at most one per rollout worker (a server
no worker routes to would sit idle), and, when the rollout workers need
accelerators too, the policy's share of a split with them (envs 3 : policy 2).
Rollout workers need an accelerator when they hold a policy part (a local
policy, or the actor next to a served base model) or simulate on one
(:attr:`SimulatorEntry.simulates_on_gpu`). They take one each from the
accelerators the servers leave, share them evenly when there are fewer than
workers, and share the servers' when the servers took them all.
``rollout_workers_share_accelerator`` puts every worker on ``first_accelerator``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from isaac_rl_posttraining.runtimes.placement import policy_server_share
from isaac_rl_posttraining.runtimes.rlinf.config import RLInfRuntimeConfig

__all__ = [
    "policy_server_accelerators",
    "resolve_policy_server_count",
    "rollout_worker_accelerators",
]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_policy_server_count(
    *,
    num_envs: int,
    available_accelerators: int,
    requested: int | None,
    num_rollout_workers: int | None = None,
) -> int:
    """How many policy servers serve ``num_envs`` environments on ``available_accelerators``.

    Four environments per policy server is the measured throughput-oriented
    operating point for GR00T N1.7 on LIBERO with L40S GPUs; no more servers
    than rollout workers, since each worker's requests go to one server.
    ``requested`` overrides the count.
    """
    num_envs = _positive_integer(num_envs, name="num_envs")
    available_accelerators = _positive_integer(
        available_accelerators, name="available_accelerators"
    )
    if requested is not None:
        requested = _positive_integer(requested, name="placement.policy.replicas")
        if requested > available_accelerators:
            raise ValueError(
                f"placement.policy.replicas={requested} asks for more policy servers than the "
                f"{available_accelerators} GPU(s) the rollout groups use; lower placement.policy.replicas "
                "or give the run more GPUs (placement.gpus_per_node)"
            )
        return requested
    count = min(available_accelerators, max(1, math.ceil(num_envs / 4)))
    return count if num_rollout_workers is None else min(count, num_rollout_workers)


def _rollout_accelerators(config: RLInfRuntimeConfig, accelerators: int) -> list[int]:
    rollout = list(range(config.first_accelerator, accelerators))
    if not rollout:
        raise ValueError(
            f"RLInf rollout groups start at runtime.rlinf.first_accelerator={config.first_accelerator}, but the run "
            f"sees {accelerators} GPU(s); give the run more GPUs or set runtime.rlinf.first_accelerator=0 (with "
            "runtime.rlinf.exclusive_accelerators=false) to share GPU 0 with the learner"
        )
    return rollout


def policy_server_accelerators(
    config: RLInfRuntimeConfig,
    *,
    accelerators: int,
    num_envs: int,
    num_rollout_workers: int,
    workers_need_accelerators: bool,
) -> tuple[int, ...]:
    """The accelerators of the policy servers, one per replica (see the module rule)."""
    rollout = _rollout_accelerators(config, accelerators)
    available = len(rollout)
    if workers_need_accelerators and config.num_policy_servers is None:
        available = policy_server_share(available)
    count = resolve_policy_server_count(
        num_envs=num_envs,
        available_accelerators=available,
        requested=config.num_policy_servers,
        num_rollout_workers=num_rollout_workers,
    )
    return tuple(rollout[:count])


def rollout_worker_accelerators(
    config: RLInfRuntimeConfig,
    *,
    accelerators: int,
    num_rollout_workers: int,
    servers: Sequence[int] = (),
    busy: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """The accelerator of each rollout worker that needs one, next to policy servers on ``servers``.

    ``busy`` counts the workers of groups already open on each accelerator; a
    later group (the evaluation workers) starts on the least busy ones.
    """
    if config.rollout_workers_share_accelerator:
        return (config.first_accelerator,) * num_rollout_workers
    rollout = _rollout_accelerators(config, accelerators)
    left = [index for index in rollout if index not in servers] or list(servers)
    left.sort(key=lambda index: (busy or {}).get(index, 0))
    per_card = math.ceil(num_rollout_workers / len(left))
    return tuple(left[worker // per_card] for worker in range(num_rollout_workers))
