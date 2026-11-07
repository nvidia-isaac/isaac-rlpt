# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The placement plan of a ``runtime=rlinf`` run, resolved by RLInf's own rules, and its accelerator check.

An off-policy run's single learner rank trains in the controller process
(GPU 0); RLInf learner ranks start at ``runtime.rlinf.learner_first_accelerator``.
Policy servers and rollout workers follow one rule from
``runtime.rlinf.first_accelerator`` on (:mod:`~isaac_rl_posttraining.runtimes.rlinf.layout`,
which the runtime applies too). :func:`rlinf_plan` describes exactly that;
:func:`check_rlinf_accelerators` reports learner ranks that would share an
accelerator while ``exclusive_accelerators`` is on.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from omegaconf import DictConfig

from isaac_rl_posttraining.runners.run_placement import placement_config_of
from isaac_rl_posttraining.runtimes import policy_serving_mode
from isaac_rl_posttraining.runtimes.placement import (
    Placement,
    PlacementPlan,
    Resources,
    RolePlacement,
    Served,
)
from isaac_rl_posttraining.runtimes.rlinf.config import RLInfRuntimeConfig
from isaac_rl_posttraining.runtimes.rlinf.layout import (
    policy_server_accelerators,
    rollout_worker_accelerators,
)

__all__ = [
    "check_rlinf_accelerators",
    "reject_unused_server_settings",
    "rlinf_plan",
]


def reject_unused_server_settings(placement: Placement) -> None:
    """Refuse the served settings RLInf would ignore: server GPUs, GPUs per replica, routing and engine."""
    for name, part in {
        "placement.policy": placement.policy,
        **{f"placement.parts.{key}": value for key, value in placement.parts.items()},
    }.items():
        if not isinstance(part, Served):
            continue
        if part.gpus is not None:
            raise ValueError(
                f"runtime=rlinf starts policy servers at runtime.rlinf.first_accelerator; leave "
                f"{name}.gpus unset ({name}.replicas sets how many)"
            )
        ignored = {
            "gpus_per_replica": (part.gpus_per_replica, 1),
            "routing": (part.routing, "modulo"),
            "engine": (part.engine, "builtin"),
        }
        for key, (value, supported) in ignored.items():
            if value != supported:
                raise ValueError(
                    f"runtime=rlinf runs each policy server on one GPU with the builtin engine and "
                    f"modulo routing; set {name}.{key}={supported} (got {value}) or use runtime=native"
                )


def _grouped(
    indices: Sequence[int], resources: Resources
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """RLInf's cluster-wide accelerator indices as ``(node, local indices)``."""
    per_node = max(resources.gpus_per_node[0], 1)
    nodes: dict[str, list[int]] = {}
    for index in indices:
        position = min(index // per_node, len(resources.nodes) - 1)
        nodes.setdefault(resources.nodes[position], []).append(
            index - position * per_node
        )
    return tuple((node, tuple(found)) for node, found in nodes.items())


def _span(first: int, count: int) -> list[int]:
    return list(range(first, first + count))


def rlinf_plan(
    cfg: DictConfig,
    checked: PlacementPlan,
    resources: Resources,
    *,
    gpu_simulator: bool = False,
) -> PlacementPlan:
    """The accelerators RLInf will give each role, from a plan already checked against the algorithm.

    ``checked`` carries the roles, worker count and served parts (planned
    without GPUs); one learner rank trains in the controller process, several
    in an RLInf group; ``gpu_simulator``: the rollout workers simulate on a GPU. The policy
    servers and rollout workers follow :mod:`~isaac_rl_posttraining.runtimes.rlinf.layout`.
    """
    settings = cfg.runtime.rlinf
    config = _layout_config(cfg)
    on_gpus = config.device == "gpu"
    workers = checked.envs.processes
    ranks = int(cfg.placement.learner.ranks)
    needed = max(
        config.first_accelerator + max(workers, 1),
        int(settings.learner_first_accelerator) + ranks,
    )
    if cfg.placement.gpus_per_node is None and sum(resources.gpus_per_node) < needed:
        # Too few GPUs here (none, or a laptop's one) to place the run: plan onto the GPUs it will have.
        # The launch still refuses to start on too few; a stated placement.gpus_per_node is always checked.
        resources = Resources(resources.nodes, (0,) * len(resources.nodes))
    total = sum(resources.gpus_per_node)
    holds_policy = not checked.services or checked.base_served_alone
    needs_accelerators = on_gpus and (holds_policy or gpu_simulator)
    # A dry run on a machine without the run's GPUs places onto the GPUs the run will have.
    accelerators = total or config.first_accelerator + max(workers, 1)
    policy = []
    server_indices: list[int] = []
    for role in checked.policy:
        if role.served is None:
            policy.append(role)
            continue
        if on_gpus:
            server_indices = list(
                policy_server_accelerators(
                    config,
                    accelerators=accelerators,
                    num_envs=int(cfg.training.num_envs),
                    num_rollout_workers=workers,
                    workers_need_accelerators=needs_accelerators,
                )
            )
        replicas = len(server_indices) or config.num_policy_servers or 1
        served = dataclasses.replace(
            role.served, replicas=replicas, gpus=len(server_indices)
        )
        policy.append(
            dataclasses.replace(
                role,
                gpus=_grouped(server_indices, resources),
                processes=replicas,
                served=served,
            )
        )
    env_indices = (
        list(
            rollout_worker_accelerators(
                config,
                accelerators=accelerators,
                num_rollout_workers=workers,
                servers=server_indices,
            )
        )
        if needs_accelerators
        else []
    )
    if ranks == 0:
        learner_indices: list[int] = []
    elif ranks == 1:
        learner_indices = [0] if total else []
    else:
        learner_indices = _span(int(settings.learner_first_accelerator), ranks)
    roles = {
        "rollout workers": env_indices,
        "learner": learner_indices,
        "policy servers": server_indices,
    }
    if total:
        for name, indices in roles.items():
            if indices and max(indices) >= total:
                raise ValueError(
                    f"runtime=rlinf puts the {name} on accelerator(s) {min(indices)}-{max(indices)}, "
                    f"but the run sees {total} GPU(s); lower runtime.rlinf.first_accelerator / "
                    "learner_first_accelerator or the counts"
                )
    used = [*set(env_indices), *learner_indices, *server_indices]
    return PlacementPlan(
        resources=resources,
        envs=RolePlacement(
            role="envs",
            gpus=_grouped(sorted(set(env_indices)), resources),
            processes=workers,
        ),
        policy=tuple(policy),
        learner=RolePlacement(
            role="learner", gpus=_grouped(learner_indices, resources), processes=ranks
        ),
        shared=len(used) != len(set(used)) or len(set(env_indices)) < len(env_indices),
    )


def _layout_config(cfg: DictConfig) -> RLInfRuntimeConfig:
    """The runtime settings the layout rule reads, as the runtime will see them."""
    settings = cfg.runtime.rlinf
    return RLInfRuntimeConfig(
        device=str(settings.device),
        first_accelerator=int(settings.first_accelerator),
        rollout_workers_share_accelerator=bool(settings.rollout_workers_share_accelerator),
        # The servers' count as the runtime reads it: placement.policy.replicas (or parts.base's).
        num_policy_servers=policy_serving_mode(placement_config_of(cfg))[1].replicas,
    )


def check_rlinf_accelerators(cfg: DictConfig, plan: PlacementPlan) -> None:
    """With ``exclusive_accelerators`` on, RLInf learner ranks may not share an accelerator.

    The rollout workers and policy servers go where the layout rule puts them
    (sharing cards when there are fewer than processes); the learner ranks
    start at ``learner_first_accelerator`` on their own. The controller's
    learner rank claims nothing.
    """
    settings = cfg.runtime.rlinf
    if not bool(settings.exclusive_accelerators) or str(settings.device) != "gpu":
        return
    if plan.learner.processes == 1:
        return
    per_node = max(plan.resources.gpus_per_node[0], 1)

    def indices(role: RolePlacement) -> set[int]:
        return {
            plan.resources.nodes.index(node) * per_node + index
            for node, local in role.gpus
            for index in local
        }

    learner = indices(plan.learner)
    for owner, role in (("rollout workers", plan.envs), *(("policy servers", r) for r in plan.services)):
        overlap = sorted(learner & indices(role))
        if overlap:
            raise ValueError(
                f"runtime=rlinf would place the learner ranks and the {owner} on accelerator(s) "
                f"{overlap} while runtime.rlinf.exclusive_accelerators=true; move "
                "runtime.rlinf.first_accelerator past the learner ranks (learner_first_accelerator "
                "+ placement.learner.ranks) or set runtime.rlinf.exclusive_accelerators=false to let them share"
            )
