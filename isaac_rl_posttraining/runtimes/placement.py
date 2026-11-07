# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where each role of a run executes: user-facing placement and the resolved plan.

A :class:`Placement` says what the user wants (``None`` fields are left to the
planner); :func:`plan_placement` resolves it against the :class:`Resources` a
runtime sees into a :class:`PlacementPlan` with concrete GPUs per role.

GPU requests are either a count (the planner picks the devices) or a sequence
of GPU indices or ``"node:index"`` strings (an index alone means node 0;
``node`` is a node name or its position in :attr:`Resources.nodes`).

Default layout when nothing is requested, per 8 GPUs: a served policy gets
envs 3, policy 2, learner 3; a local policy gets envs 5, learner 3. A learner
that cannot shard takes one GPU and the envs take the rest. No role takes more
GPUs than its processes use (one per rollout worker, ``gpus_per_replica`` per
policy-server replica, at most one replica per rollout worker); what none can
use stays free. When the GPUs left after the explicit requests cannot give
every remaining role one of its own, the roles that named no GPUs share them:
the k-th such role takes the k-th GPU still free, wrapping around, or the run's
GPUs when none is free (one GPU: every role on it, like a single-process run).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from isaac_rl_posttraining.runtimes.gpu_pool import (
    GpuPool,
    gpu_count,
    grouped,
    requested_count,
    whole_groups,
)
from isaac_rl_posttraining.runtimes.serving.batcher import DEFAULT_MAX_WAIT_MS

__all__ = [
    "ROUTINGS",
    "Envs",
    "GpuRequest",
    "Learner",
    "Local",
    "Placement",
    "PlacementPlan",
    "PolicyPlacement",
    "Resources",
    "RolePlacement",
    "Routing",
    "Served",
    "plan_placement",
    "plan_policy_service",
    "policy_server_share",
    "served_role",
]

#: A GPU count (the planner picks devices) or explicit GPU indices / ``"node:index"`` strings.
GpuRequest = int | Sequence[int | str] | None
#: How an environment picks a policy-server replica: ``modulo`` sends environment
#: ``i`` to replica ``i % replicas``; ``least_loaded`` picks the replica with the
#: fewest requests in flight.
Routing = Literal["modulo", "least_loaded"]
ROUTINGS: tuple[Routing, ...] = ("modulo", "least_loaded")

#: Default GPU shares (out of 8) for roles the user left unspecified.
_SERVED_WEIGHTS = {"envs": 3, "policy": 2, "learner": 3}
_LOCAL_WEIGHTS = {"envs": 5, "learner": 3}


def policy_server_share(gpus: int) -> int:
    """How many of ``gpus`` the policy servers take when GPU rollout workers split them too (envs 3 : policy 2)."""
    served = _SERVED_WEIGHTS["policy"]
    return max(1, gpus * served // (served + _SERVED_WEIGHTS["envs"]))


def _check_count(value: object, *, name: str, minimum: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_gpus(value: GpuRequest, *, name: str, minimum: int) -> GpuRequest:
    if value is None or isinstance(value, int):
        _check_count(value, name=name, minimum=minimum)
        return value
    if isinstance(value, str):
        raise TypeError(
            f"{name} must be a count or a sequence of GPU strings, not one string"
        )
    devices = tuple(str(device) if _is_index(device) else device for device in value)
    if not devices or not all(isinstance(device, str) and device for device in devices):
        raise ValueError(
            f"{name} must list GPU indices or 'node:index' strings, got {value!r}"
        )
    if len(set(devices)) != len(devices):
        raise ValueError(f"{name} lists a GPU twice: {devices}")
    return devices


@dataclass(frozen=True)
class Envs:
    """Rollout-worker processes and the GPUs their simulators use."""

    workers: int | None = None
    gpus: GpuRequest = None

    def __post_init__(self) -> None:
        _check_count(self.workers, name="envs.workers", minimum=1)
        object.__setattr__(
            self, "gpus", _normalize_gpus(self.gpus, name="envs.gpus", minimum=0)
        )


@dataclass(frozen=True)
class Local:
    """A copy of the policy part inside every rollout worker."""


@dataclass(frozen=True)
class Served:
    """The policy part runs in policy-server replicas that rollout workers call.

    ``max_batch`` caps the requests answered together (``None``: no cap; one
    call of a batched rollout worker is never split and the serving engine
    still decodes in chunks of the policy's own decode batch size);
    ``max_wait_ms`` is how long a server waits after the first request for
    other rollout workers' requests to join the batch.
    ``routing`` picks the replica of an environment: ``modulo`` maps environment
    ``i`` to replica ``i % replicas``. Parts with equal ``Served`` values share
    one policy server (one round trip for all of them).
    """

    replicas: int | None = None
    gpus: GpuRequest = None
    gpus_per_replica: int = 1
    max_batch: int | None = None
    max_wait_ms: float = DEFAULT_MAX_WAIT_MS
    engine: str = "builtin"
    routing: Routing = "modulo"

    def __post_init__(self) -> None:
        _check_count(self.replicas, name="served.replicas", minimum=1)
        _check_count(self.gpus_per_replica, name="served.gpus_per_replica", minimum=1)
        _check_count(self.max_batch, name="served.max_batch", minimum=1)
        object.__setattr__(
            self, "gpus", _normalize_gpus(self.gpus, name="served.gpus", minimum=0)
        )
        if isinstance(self.max_wait_ms, bool) or not isinstance(
            self.max_wait_ms, int | float
        ):
            raise TypeError("served.max_wait_ms must be a number")
        if self.max_wait_ms < 0:
            raise ValueError("served.max_wait_ms must be non-negative")
        if not isinstance(self.engine, str) or not self.engine:
            raise ValueError("served.engine must name a serving engine")
        if self.routing not in ROUTINGS:
            raise ValueError(
                f"served.routing must be one of {ROUTINGS}, got {self.routing!r}"
            )


PolicyPlacement = Local | Served


@dataclass(frozen=True)
class Learner:
    """Learner ranks (one GPU each; more than one rank means FSDP)."""

    ranks: int | None = None
    gpus: GpuRequest = None

    def __post_init__(self) -> None:
        _check_count(self.ranks, name="learner.ranks", minimum=1)
        object.__setattr__(
            self, "gpus", _normalize_gpus(self.gpus, name="learner.gpus", minimum=0)
        )


@dataclass(frozen=True)
class Placement:
    """Which role and policy part runs where; ``None`` and defaults are planner-decided.

    ``policy`` places every policy part; ``parts`` overrides single parts, for
    example ``{"actor": Local(), "base": Served(gpus=2)}``. ``nodes`` restricts
    the run to some of the runtime's nodes (names or positions).
    """

    envs: Envs = field(default_factory=Envs)
    policy: PolicyPlacement = field(default_factory=Local)
    parts: Mapping[str, PolicyPlacement] = field(default_factory=dict)
    learner: Learner = field(default_factory=Learner)
    nodes: Sequence[str] | None = None

    def __post_init__(self) -> None:
        for name, part in {"policy": self.policy, **dict(self.parts)}.items():
            if not isinstance(part, Local | Served):
                raise TypeError(
                    f"placement of {name!r} must be Local() or Served(...), got {part!r}"
                )
        object.__setattr__(self, "parts", dict(self.parts))
        if self.nodes is not None:
            object.__setattr__(self, "nodes", tuple(str(node) for node in self.nodes))


@dataclass(frozen=True)
class Resources:
    """What a runtime can place onto: node names with their GPU and CPU counts."""

    nodes: tuple[str, ...]
    gpus_per_node: tuple[int, ...]
    cpus_per_node: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "gpus_per_node", tuple(self.gpus_per_node))
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("resources need at least one node and unique node names")
        if len(self.gpus_per_node) != len(self.nodes):
            raise ValueError("resources need one GPU count per node")
        for count in self.gpus_per_node:
            _check_count(count, name="gpus_per_node", minimum=0)
        if self.cpus_per_node is not None:
            object.__setattr__(self, "cpus_per_node", tuple(self.cpus_per_node))
            if len(self.cpus_per_node) != len(self.nodes):
                raise ValueError("resources need one CPU count per node")

    @classmethod
    def uniform(
        cls,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        cpus_per_node: int | None = None,
    ) -> Resources:
        """``num_nodes`` identical nodes named ``node0``, ``node1``, ..."""
        _check_count(num_nodes, name="num_nodes", minimum=1)
        return cls(
            nodes=tuple(f"node{index}" for index in range(num_nodes)),
            gpus_per_node=(gpus_per_node,) * num_nodes,
            cpus_per_node=None
            if cpus_per_node is None
            else (cpus_per_node,) * num_nodes,
        )

    @property
    def num_gpus(self) -> int:
        return sum(self.gpus_per_node)

    def node(self, token: str) -> str:
        """Resolve a node name or position to the node name."""
        if token in self.nodes:
            return token
        if token.isdigit() and int(token) < len(self.nodes):
            return self.nodes[int(token)]
        raise ValueError(f"unknown node {token!r}; nodes are {self.nodes}")


@dataclass(frozen=True)
class RolePlacement:
    """The concrete assignment of one role: GPUs per node and how many processes.

    ``parts`` and ``served`` are set for policy roles: the parts this role hosts
    and, for a policy server, its resolved settings (``served.replicas`` equals
    ``processes``).
    """

    role: str
    gpus: tuple[tuple[str, tuple[int, ...]], ...]
    processes: int
    parts: tuple[str, ...] = ()
    served: Served | None = None

    @property
    def num_gpus(self) -> int:
        return sum(len(indices) for _, indices in self.gpus)

    def by_node(self) -> dict[str, tuple[int, ...]]:
        """GPU indices keyed by node name."""
        return dict(self.gpus)

    def devices(self) -> tuple[str, ...]:
        """The assigned GPUs as ``"node:index"`` strings."""
        return tuple(
            f"{node}:{index}" for node, indices in self.gpus for index in indices
        )

    def describe(self, *, in_process: bool = False) -> str:
        """One plan line; ``in_process``: the role runs in the launching process."""
        where = (
            "; ".join(f"{node} gpus {list(indices)}" for node, indices in self.gpus)
            or "the cpu"
        )
        if self.served is not None:
            served = self.served
            batching = f"max_batch={served.max_batch or 'unbounded'}, max_wait_ms={served.max_wait_ms}"
            if in_process:
                return f"served by one in-process policy server on {where}, {batching}"
            replicas = f"{self.processes} replica(s)"
            if self.gpus:
                replicas += f" x {served.gpus_per_replica} gpu"
            return (
                f"served by {replicas} on {where}, "
                f"engine={served.engine}, routing={served.routing}, {batching}"
            )
        if self.role == "policy":
            return "local (a copy in every rollout worker)"
        unit = "rank(s)" if self.role == "learner" else "worker(s)"
        return f"{self.processes} {unit} {'in this process ' if in_process else ''}on {where}"


@dataclass(frozen=True)
class PlacementPlan:
    """A resolved placement: frozen, picklable, printed by the dry run."""

    resources: Resources
    envs: RolePlacement
    policy: tuple[RolePlacement, ...]
    learner: RolePlacement
    shared: bool = False
    #: Every role runs in the launching process (the native runtime without Ray).
    in_process: bool = False

    def part(self, name: str) -> RolePlacement:
        """The policy role hosting part ``name``."""
        for role in self.policy:
            if name in role.parts:
                return role
        raise KeyError(
            f"no policy part {name!r}; parts are {[p for r in self.policy for p in r.parts]}"
        )

    @property
    def services(self) -> tuple[RolePlacement, ...]:
        """The policy roles that run as servers."""
        return tuple(role for role in self.policy if role.served is not None)

    @property
    def base_served_alone(self) -> bool:
        """Whether the base model is served while the actor runs in the rollout workers."""
        return any(role.parts == ("base",) for role in self.services)

    def roles(self) -> dict[str, RolePlacement]:
        """The process roles by key: ``envs``, ``learner`` and ``policy`` (``policy1``, ... for further served groups)."""
        roles = {"envs": self.envs, "learner": self.learner}
        for position, role in enumerate(self.services):
            roles["policy" if position == 0 else f"policy{position}"] = role
        return roles

    def lines(self) -> list[str]:
        """``plan.placement.<role>=<summary>`` lines for the dry run."""
        nodes = ", ".join(
            f"{node} ({gpus} gpus)"
            for node, gpus in zip(self.resources.nodes, self.resources.gpus_per_node)
        )
        used = {
            (node, index)
            for role in (self.envs, *self.policy, self.learner)
            for node, indices in role.gpus
            for index in indices
        }
        unused = self.resources.num_gpus - len(used)
        lines = [
            f"plan.placement.resources={nodes}"
            + ("; roles share gpus" if self.shared else "")
            + ("; every role in this process" if self.in_process else "")
            + (f"; {unused} gpu(s) unused" if unused > 0 else "")
        ]
        lines.append(
            f"plan.placement.envs={self.envs.describe(in_process=self.in_process)}"
        )
        for role in self.policy:
            label = "" if role.parts == ("policy",) else "." + "+".join(role.parts)
            lines.append(
                f"plan.placement.policy{label}={role.describe(in_process=self.in_process)}"
            )
        lines.append(
            f"plan.placement.learner={self.learner.describe(in_process=self.in_process)}"
        )
        return lines

    def to_json(self) -> str:
        """The plan as JSON (written to ``placement.json`` in the run directory)."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def served_role(
    roles: Mapping[str, RolePlacement], *, base_part: bool
) -> tuple[str, RolePlacement]:
    """The served role (by its key in ``roles``) hosting a base model alone, or else a whole policy."""
    served = [(key, role) for key, role in roles.items() if role.served is not None]
    if not served:
        raise ValueError(
            "the placement keeps the policy local; set placement.policy.mode=served "
            "(or placement.parts.base.mode=served) to run policy servers"
        )
    for key, role in served:
        if base_part == (role.parts == ("base",)):
            return key, role
    if base_part:
        raise ValueError(
            "the placement serves the whole policy, not its base model alone"
        )
    raise ValueError(
        "the placement serves only the base model; the actor runs in the rollout workers "
        "(placement.policy.mode=served serves the whole policy)"
    )


def plan_policy_service(resources: Resources, served: Served) -> RolePlacement:
    """Where a policy service opened on its own runs; it plans no other role.

    Explicit ``served.gpus`` are taken as given; otherwise each replica takes
    the next free ``gpus_per_replica`` GPUs, and without GPUs every replica
    runs on the CPU. ``served.replicas`` defaults to one per requested GPU
    group, or one.
    """
    pool = GpuPool(resources, resources.nodes)
    label = "the policy service"
    per_replica = served.gpus_per_replica
    if served.gpus is None:
        replicas = served.replicas or 1
        wanted = replicas * per_replica if pool.free else 0
        gpus = pool.take(label, wanted, group=per_replica)
    else:
        count = requested_count(label, served.gpus, served.replicas, per_replica)
        if isinstance(served.gpus, int):
            gpus = pool.take(label, count or 0, group=per_replica)
        else:
            gpus = pool.claim(label, served.gpus)
        replicas = served.replicas or max(1, len(gpus) // per_replica)
    return RolePlacement(
        role="policy",
        gpus=grouped(gpus),
        processes=replicas,
        parts=("policy",),
        served=Served(**{**asdict(served), "replicas": replicas, "gpus": len(gpus)}),
    )


def _policy_groups(
    placement: Placement, policy_parts: Sequence[str]
) -> list[tuple[PolicyPlacement, tuple[str, ...]]]:
    unknown = sorted(set(placement.parts) - set(policy_parts))
    if unknown:
        named = ", ".join(f"placement.parts.{part}" for part in unknown)
        whole = tuple(policy_parts) == ("policy",)
        parts = (
            "is one part that it trains as a whole, with no separate base model"
            if whole
            else f"has the parts {list(policy_parts)} (placement.parts.<part>)"
        )
        raise ValueError(
            f"{named} names a policy part this algorithm does not have: its policy {parts}; "
            "serve the whole policy with placement.policy.mode=served"
        )
    groups: dict[PolicyPlacement, list[str]] = {}
    for part in policy_parts:
        groups.setdefault(placement.parts.get(part, placement.policy), []).append(part)
    return [(spec, tuple(parts)) for spec, parts in groups.items()]


def _check_part_placement(
    groups: Sequence[tuple[PolicyPlacement, tuple[str, ...]]],
    served: Sequence[tuple[Served, tuple[str, ...]]],
    *,
    served_parts: bool,
) -> None:
    """Parts are local together, served together, or the base model is served alone."""
    if len(served) > 1:
        raise ValueError(
            f"policy parts {[parts for _, parts in served]} ask for different policy servers; "
            "serve the base model alone (placement.parts.base) or every part with the same settings"
        )
    if not served or len(groups) == 1:
        return
    ((_, parts),) = served
    if parts != ("base",):
        raise ValueError(
            f"policy parts {list(parts)} cannot be served while others stay local: only the base "
            "model is served alone (the actor follows the observations in the rollout workers)"
        )
    if not served_parts:
        raise ValueError(
            "this runtime serves whole policies only; serve the whole policy "
            "(placement.policy.mode=served) or keep the base model local"
        )


@dataclass(frozen=True)
class _Request:
    """One GPU-holding role while planning: what it asked for and how its GPUs group."""

    key: str
    gpus: GpuRequest
    count: int | None
    weight: str
    group: int = 1


def _role_label(key: str, served: Sequence[tuple[Served, tuple[str, ...]]]) -> str:
    """A role as the user configures it: ``learner (placement.learner.gpus)``."""
    if key in ("envs", "learner"):
        return f"{key} (placement.{key}.gpus)"
    parts = served[int(key.removeprefix("policy"))][1]
    section = "placement.parts.base" if parts == ("base",) else "placement.policy"
    return f"policy server of {'+'.join(parts)} ({section}.gpus)"


def _explicit_conflicts(
    requests: Sequence[_Request], labels: Mapping[str, str], resources: Resources
) -> list[str]:
    """Explicit GPU lists that name a GPU a node does not have, or that two roles both claim."""
    problems: list[str] = []
    owners: dict[tuple[str, int], list[str]] = {}
    for request in requests:
        if request.gpus is None or isinstance(request.gpus, int):
            continue
        for device in request.gpus:
            node_token, _, index_token = str(device).rpartition(":")
            if not index_token.isdigit():
                continue  # the claim reports the malformed string
            node = resources.node(node_token or "0")
            total = resources.gpus_per_node[resources.nodes.index(node)]
            if int(index_token) >= total:
                problems.append(
                    f"{labels[request.key]} asks for GPU {device}, which does not exist: "
                    f"{node} has {total} GPU(s) (0-{total - 1})"
                )
                continue
            owners.setdefault((node, int(index_token)), []).append(labels[request.key])
    shared: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for gpu, roles in owners.items():
        if len(roles) > 1:
            shared.setdefault(tuple(roles), []).append(gpu)
    for claimants, gpus in shared.items():
        where = ", ".join(f"{node}:{index}" for node, index in gpus)
        problems.append(f"{' and '.join(claimants)} both ask for GPU(s) {where}")
    return problems


def plan_placement(
    placement: Placement,
    resources: Resources,
    *,
    policy_parts: Sequence[str] = ("policy",),
    served_supported: bool,
    distributed_learner: bool,
    served_parts: bool = False,
    runtime: str = "this runtime",
    evaluation_workers: int = 0,
) -> PlacementPlan:
    """Resolve ``placement`` against ``resources``; explicit values always win.

    ``policy_parts`` names the parts of the algorithm's policy (for example
    ``("base", "actor")``). ``served_supported``, ``served_parts`` (the base
    model served alone) and ``distributed_learner`` are what the runtime and
    algorithm allow; asking for more raises before anything starts, naming
    ``runtime``. ``evaluation_workers`` run next to the training workers on
    the ``envs`` GPUs (an evaluation group of their own); with a local policy
    each holds a copy, so they may use GPUs the training workers leave free.
    """
    if not policy_parts or len(set(policy_parts)) != len(policy_parts):
        raise ValueError(
            f"policy_parts must be unique and non-empty, got {list(policy_parts)}"
        )
    groups = _policy_groups(placement, policy_parts)
    served = [(spec, parts) for spec, parts in groups if isinstance(spec, Served)]
    if served and not served_supported:
        names = [part for _, parts in served for part in parts]
        raise ValueError(
            f"{runtime} has no policy server, so it cannot serve the policy parts {names}; "
            "set placement.policy.mode=local (and placement.parts.<part>.mode=local), "
            "or serve the policy with runtime=native or runtime=rlinf"
        )
    _check_part_placement(groups, served, served_parts=served_parts)
    envs, learner = placement.envs, placement.learner
    if (
        max(learner.ranks or 1, gpu_count(learner.gpus) or 1) > 1
        and not distributed_learner
    ):
        raise ValueError(
            "more than one learner rank needs a runtime with a distributed (FSDP) learner"
        )

    nodes = tuple(resources.node(node) for node in placement.nodes or resources.nodes)
    pool = GpuPool(resources, nodes)
    # Without GPUs, process counts (replicas, ranks) imply no GPU request.
    on_gpus = bool(pool.free)
    requests = [
        _Request("envs", envs.gpus, gpu_count(envs.gpus), "envs"),
        *(
            _Request(
                f"policy{position}",
                spec.gpus,
                requested_count(
                    "policy",
                    spec.gpus,
                    spec.replicas if on_gpus else None,
                    spec.gpus_per_replica,
                ),
                "policy",
                spec.gpus_per_replica if on_gpus else 1,
            )
            for position, (spec, _) in enumerate(served)
        ),
        _Request(
            "learner",
            learner.gpus,
            requested_count(
                "learner", learner.gpus, learner.ranks if on_gpus else None, 1
            ),
            "learner",
        ),
    ]
    available = len(pool.free)
    explicit = sum(request.count for request in requests if request.count is not None)
    labels = {request.key: _role_label(request.key, served) for request in requests}
    conflicts = _explicit_conflicts(requests, labels, resources)
    if explicit > available:
        asked = ", ".join(
            f"{labels[request.key].split(' (')[0]} {request.count}"
            for request in requests
            if request.count is not None
        )
        conflicts.append(f"requested {explicit} ({asked}), available {available}")
    if conflicts:
        raise ValueError("placement over-subscribes the GPUs: " + "; ".join(conflicts))

    weights = dict(_SERVED_WEIGHTS if served else _LOCAL_WEIGHTS)
    if not distributed_learner:
        weights["learner"] = 0  # a single rank takes one GPU
    # A role asking for nothing, or for one process without naming GPUs, takes GPUs of its own while
    # every role asking for nothing can have one; otherwise these roles share what the others leave.
    flexible = [
        request.key
        for request in requests
        if request.gpus is None and (request.count or 0) <= 1 and request.group == 1
    ]
    # An undecided served role needs at least one whole replica.
    shared = available - explicit < sum(
        request.group for request in requests if request.count is None
    )
    gpus: dict[str, list[tuple[str, int]]] = {}
    for request in requests:  # explicit devices first, so defaults cannot take them
        if request.gpus is not None and not isinstance(request.gpus, int):
            gpus[request.key] = pool.claim(request.key, request.gpus)
    if shared:
        for request in requests:
            if request.key not in gpus and request.key not in flexible:
                count = request.group if request.count is None else request.count
                gpus[request.key] = pool.take(request.key, count, group=request.group)
        sharing = pool.free or pool.all
        for position, key in enumerate(flexible):
            gpus[key] = [sharing[position % len(sharing)]] if sharing else []
    else:
        counts = _default_counts(
            {request.key: request.count for request in requests},
            {request.key: weights[request.weight] for request in requests},
            available - explicit,
            _process_caps(envs.workers, served, evaluation_workers),
            {request.key: request.group for request in requests},
        )
        for request in requests:
            if request.key not in gpus:
                gpus[request.key] = pool.take(
                    request.key, counts[request.key], group=request.group
                )

    env_workers = envs.workers or max(1, 1 if shared else len(gpus["envs"]))
    policy_roles = []
    for spec, parts in groups:
        if isinstance(spec, Local):
            policy_roles.append(
                RolePlacement(
                    role="policy", gpus=(), processes=env_workers, parts=parts
                )
            )
            continue
        assigned = gpus[f"policy{[group for group, _ in served].index(spec)}"]
        replicas = spec.replicas or (
            1 if shared else max(1, len(assigned) // spec.gpus_per_replica)
        )
        resolved = Served(
            **{**asdict(spec), "replicas": replicas, "gpus": len(assigned)}
        )
        policy_roles.append(
            RolePlacement(
                role="policy",
                gpus=grouped(assigned),
                processes=replicas,
                parts=parts,
                served=resolved,
            )
        )
    return PlacementPlan(
        resources=resources,
        envs=RolePlacement(
            role="envs", gpus=grouped(gpus["envs"]), processes=env_workers
        ),
        policy=tuple(policy_roles),
        learner=RolePlacement(
            role="learner",
            gpus=grouped(gpus["learner"]),
            processes=learner.ranks or max(1, 1 if shared else len(gpus["learner"])),
        ),
        shared=shared,
    )


def _process_caps(
    workers: int | None,
    served: Sequence[tuple[Served, tuple[str, ...]]],
    evaluation_workers: int = 0,
) -> dict[str, int | None]:
    """The most GPUs a role can use: one per rollout worker (evaluation ones too with a local policy), one replica per rollout worker."""
    envs = None if workers is None else workers + (0 if served else evaluation_workers)
    caps: dict[str, int | None] = {"envs": envs}
    for position, (spec, _) in enumerate(served):
        caps[f"policy{position}"] = (
            None if workers is None else workers * spec.gpus_per_replica
        )
    return caps


def _default_counts(
    requests: Mapping[str, int | None],
    weights: Mapping[str, int],
    remaining: int,
    caps: Mapping[str, int | None],
    groups: Mapping[str, int],
) -> dict[str, int]:
    """GPU counts for roles without an explicit request, at least one each.

    A role of weight 0 (a single-rank learner) takes one GPU; the other
    undecided roles split the rest in proportion to their weights, envs taking
    what rounding leaves. No role takes more than its cap (``caps``); what one
    cannot use goes to the other capped roles, and what none can use stays free.
    A role whose GPUs come in groups (``groups``: a served role's replicas) takes
    whole groups, at least one; envs, or else the largest other role, evens out.
    """
    counts = {role: count for role, count in requests.items() if count is not None}
    undecided = [role for role, count in requests.items() if count is None]
    for role in undecided:
        if weights[role] == 0:
            counts[role] = 1
            remaining -= 1
    proportional = [role for role in undecided if weights[role] > 0]
    total = sum(weights[role] for role in proportional)
    for role in proportional:
        if role != "envs":
            counts[role] = max(remaining, 0) * weights[role] // total
    if "envs" in proportional:
        counts["envs"] = remaining - sum(
            counts[role] for role in proportional if role != "envs"
        )
    # Rounding down can leave a role with none while another holds several; move one over.
    for role in proportional:
        donor = max(proportional, key=lambda other: counts[other])
        if counts[role] < 1 and counts[donor] > 1:
            counts[donor] -= 1
            counts[role] += 1
    limits = {
        role: limit for role in proportional if (limit := caps.get(role)) is not None
    }
    surplus = 0
    for role, limit in limits.items():
        surplus += max(counts[role] - limit, 0)
        counts[role] = min(counts[role], limit)
    for role, limit in limits.items():
        extra = min(surplus, limit - counts[role])
        counts[role] += extra
        surplus -= extra
    for role in proportional:
        group = groups.get(role, 1)
        if group > 1:
            counts[role] = whole_groups(counts, role, group, proportional, groups)
    return {role: min(count, limits.get(role, count)) for role, count in counts.items()}
