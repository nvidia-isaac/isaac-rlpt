# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Ray cluster of a multi-process native run: connect, reserve each role's GPUs, wait on actors.

:func:`connect_ray` joins a running Ray cluster (``ray.init(address="auto")``)
or starts a local one. :class:`RayCluster` matches the planned nodes to Ray
nodes (the driver's node is ``node0``) and reserves one Ray placement group per
role per node from the planned roles (a
:class:`~isaac_rl_posttraining.runtimes.placement.PlacementPlan`'s, or a
policy service's alone): one bundle per GPU the role holds there (a GPU two
roles share is split between them).
Actors are scheduled into their bundle and see exactly the plan's GPUs
through ``CUDA_VISIBLE_DEVICES``; Ray does not pick devices for them.

:func:`wait_all` waits on actor calls and fails as soon as one of them fails
or its actor dies, naming the actor, instead of waiting for the others.
"""

from __future__ import annotations

import importlib
import random
import re
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from isaac_rl_posttraining.runtimes.base import (
    ActorFailedError,
    PolicyState,
    WeightsRef,
    register_weights_resolver,
)
from isaac_rl_posttraining.runtimes.placement import Resources, RolePlacement
from isaac_rl_posttraining.runtimes.ray_driver import (
    keep_driver_interpreter,
    visible_devices,
)

__all__ = [
    "ActorFailedError",
    "ActorPlacement",
    "ActorTimeoutError",
    "RayCluster",
    "close_actors",
    "connect_ray",
    "device_names",
    "free_port",
    "import_ray",
    "node_ip",
    "remote_class",
    "seed_process",
    "wait_all",
    "weights_by_reference",
    "writable_state",
]

#: Ray's own CUDA_VISIBLE_DEVICES handling is off: actors get the plan's GPUs instead.
_NOSET_CUDA = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
#: A sliver of a node's ``node:<ip>`` resource pins a bundle to that node.
_NODE_SLIVER = 0.001


def import_ray() -> Any:
    """Import Ray, the optional dependency of the multi-process native runtime."""
    try:
        ray = importlib.import_module("ray")
    except ImportError as exc:
        raise ImportError(
            "runtime=native runs several processes through Ray when the placement asks for "
            "them; install the ray extra (pip install 'isaac-rl-posttraining[ray]') or set "
            "runtime.native.processes=single"
        ) from exc
    register_weights_resolver(ray.ObjectRef, _resolve_object_ref)
    return ray


def _resolve_object_ref(handle: Any) -> PolicyState:
    return writable_state(importlib.import_module("ray").get(handle))


def writable_state(state: Any) -> PolicyState:
    """A private, writable copy of weights read from the object store (whose arrays are read-only)."""
    if not isinstance(state, PolicyState):
        raise TypeError(f"a weights handle resolved to {type(state).__name__}")
    return PolicyState(
        version=state.version, tensors=state.tensors, dtypes=state.dtypes
    )


_REMOTE_CLASSES: dict[type, Any] = {}


def remote_class(cls: type) -> Any:
    """``ray.remote(cls)``, made once per class (the classes stay importable without Ray)."""
    if cls not in _REMOTE_CLASSES:
        _REMOTE_CLASSES[cls] = import_ray().remote(cls)
    return _REMOTE_CLASSES[cls]


def seed_process(seed: int | None) -> None:
    """Seed this process's Python, NumPy and torch generators (no-op for ``None``)."""
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def weights_by_reference(weights: WeightsRef) -> WeightsRef:
    """Weights behind a Ray object reference: inline weights are put in the object store once."""
    if weights.inline is None:
        return weights
    handle = import_ray().put(weights.inline)
    return WeightsRef(version=weights.version, handle=handle, parts=weights.parts)


def node_ip() -> str:
    """This process's node address as Ray reports it."""
    return str(import_ray().util.get_node_ip_address())


def free_port() -> int:
    """A TCP port nothing listens on right now (the learner ranks' rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class ActorTimeoutError(ActorFailedError):
    """Some calls of a :func:`wait_all` did not answer in time; ``late`` holds their indices."""

    def __init__(self, message: str, late: tuple[int, ...]) -> None:
        super().__init__(message)
        self.late = late


def wait_all(
    calls: Sequence[Any],
    *,
    describe: Callable[[int], str],
    timeout_seconds: float | None,
    what: str,
) -> list[Any]:
    """The results of ``calls`` (Ray object references) in order.

    Raises :class:`ActorFailedError` as soon as one call fails or its actor
    dies, naming that actor (``describe(index)``), and when ``timeout_seconds``
    pass before every call answered.
    """
    ray = import_ray()
    index_of = {ref: index for index, ref in enumerate(calls)}
    results: dict[int, Any] = {}
    pending = list(calls)
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while pending:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        ready, pending = ray.wait(pending, num_returns=1, timeout=remaining)
        if not ready:
            late = ", ".join(describe(index_of[ref]) for ref in pending[:4])
            raise ActorTimeoutError(
                f"{what}: no answer within {timeout_seconds} s from {late}"
                + (" and others" if len(pending) > 4 else ""),
                tuple(sorted(index_of[ref] for ref in pending)),
            )
        for ref in ready:
            index = index_of[ref]
            try:
                results[index] = ray.get(ref)
            except Exception as exc:
                raise ActorFailedError(
                    f"{what}: {describe(index)} failed: {_cause(exc)}"
                ) from exc
    return [results[index] for index in range(len(calls))]


def close_actors(
    actors: Sequence[Any],
    *,
    describe: Callable[[int], str],
    timeout_seconds: float | None,
) -> list[Exception]:
    """Call ``close`` on every actor, then kill it; returns the failures of live actors.

    An actor that already died (the failure that ended the run) has nothing
    to close and is not reported again.
    """
    ray = import_ray()
    calls = [actor.close.remote() for actor in actors]
    errors: list[Exception] = []
    for index, call in enumerate(calls):
        try:
            ray.get(call, timeout=timeout_seconds)
        except ray.exceptions.RayActorError:
            continue
        except Exception as exc:  # noqa: BLE001 - every actor is closed and killed
            errors.append(
                ActorFailedError(f"closing {describe(index)} failed: {_cause(exc)}")
            )
    for actor in actors:
        ray.kill(actor)
    return errors


def _cause(exc: BaseException) -> str:
    """The error's last line, and the dead actor it names (Ray reports the innermost one last)."""
    text = str(exc).strip()
    lines = text.splitlines()
    summary = f"{type(exc).__name__}: {lines[-1]}" if lines else type(exc).__name__
    dead = re.findall(r"^\s*name: (\S+)$", text, flags=re.MULTILINE)
    return f"{summary} (dead actor {dead[-1]})" if dead else summary


@dataclass(frozen=True)
class ActorPlacement:
    """Where one actor of a role runs: its node, its GPUs, and the Ray options that put it there."""

    role: str
    index: int
    node: str
    gpus: tuple[int, ...]
    options: Mapping[str, Any]

    @property
    def device(self) -> str:
        """The actor's torch device: its first visible GPU, or the CPU."""
        return "cuda:0" if self.gpus else "cpu"

    def describe(self, label: str) -> str:
        """``<label> <index> (<node>, gpus [...])``, e.g. ``rollout worker 2 (node0, gpus [3])``."""
        where = f"gpus {list(self.gpus)}" if self.gpus else "cpu"
        return f"{label} {self.index} ({self.node}, {where})"


@dataclass
class _RoleReservation:
    group: Any
    #: (node, gpu index or None for a CPU bundle) in bundle order.
    bundles: list[tuple[str, int | None]]
    #: GPU slots in plan order: (node, gpu index).
    slots: list[tuple[str, int]]


class RayCluster:
    """The planned nodes on a connected Ray cluster, and one placement group per role per node."""

    def __init__(
        self,
        resources: Resources,
        roles: Mapping[str, RolePlacement],
        *,
        startup_timeout_seconds: float = 900.0,
        node_join_timeout_seconds: float = 120.0,
    ) -> None:
        ray = import_ray()
        if not ray.is_initialized():
            raise RuntimeError(
                "connect to Ray (connect_ray) before reserving its resources"
            )
        self.resources = resources
        self._roles = dict(roles)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._node_join_timeout_seconds = node_join_timeout_seconds
        self._driver_devices = visible_devices()
        self._node_resources = self._match_nodes(ray)
        self._reservations: dict[str, _RoleReservation] = {}
        #: Live processes per GPU slot of each role, so a later group starts on the least used GPUs.
        self._slot_load: dict[str, list[int]] = {}
        try:
            for key, role in self._roles.items():
                self._reservations[key] = self._reserve(ray, key, role)
        except BaseException:
            self.close()
            raise

    def roles(self) -> dict[str, RolePlacement]:
        """The planned process roles by key (``envs``, ``learner``, ``policy``, ...)."""
        return dict(self._roles)

    def _match_nodes(self, ray: Any) -> dict[str, str]:
        """The ``node:<ip>`` resource of every planned node, waiting for nodes that have not joined yet.

        The driver's node is the first planned node; the others follow in address order.
        """
        wanted = self.resources.nodes
        deadline = time.monotonic() + self._node_join_timeout_seconds
        driver = ray.get_runtime_context().get_node_id()
        while True:
            alive = [node for node in ray.nodes() if node.get("Alive")]
            alive.sort(
                key=lambda node: (
                    node["NodeID"] != driver,
                    str(node.get("NodeManagerAddress")),
                )
            )
            if len(alive) >= len(wanted) or time.monotonic() > deadline:
                break
            time.sleep(2.0)
        if len(alive) < len(wanted):
            raise ActorFailedError(
                f"the placement needs {len(wanted)} node(s) but only {len(alive)} joined the Ray "
                f"cluster within {self._node_join_timeout_seconds:g} s "
                "(runtime.native.node_join_timeout_seconds); start Ray on every node "
                "(ray start --address=<head>:6379) before the run"
            )
        matched: dict[str, str] = {}
        for name, gpus, node in zip(
            wanted, self.resources.gpus_per_node, alive, strict=False
        ):
            have = int(node.get("Resources", {}).get("GPU", 0))
            if have < gpus:
                raise ActorFailedError(
                    f"plan node {name} needs {gpus} GPU(s) but Ray node "
                    f"{node.get('NodeManagerAddress')} has {have}"
                )
            matched[name] = f"node:{node['NodeManagerAddress']}"
        return matched

    def _reserve(self, ray: Any, key: str, role: RolePlacement) -> _RoleReservation:
        sharers = self._gpu_sharers()
        bundles: list[tuple[str, int | None]] = []
        resources: list[dict[str, float]] = []
        for node, indices in role.gpus:
            for index in indices:
                bundles.append((node, index))
                resources.append(
                    {
                        "GPU": 1.0 / sharers[(node, index)],
                        self._node_resources[node]: _NODE_SLIVER,
                    }
                )
        if not bundles:
            # CPU-only processes go round-robin over the plan's nodes.
            for node in self._nodes_in_use():
                bundles.append((node, None))
                resources.append({self._node_resources[node]: _NODE_SLIVER})
        placement = importlib.import_module("ray.util.placement_group")
        group = placement.placement_group(resources, strategy="PACK")
        try:
            ray.get(group.ready(), timeout=self._startup_timeout_seconds)
        except Exception as exc:
            placement.remove_placement_group(group)
            raise ActorFailedError(
                f"Ray could not reserve the {key} role's resources {resources} within "
                f"{self._startup_timeout_seconds} s (are the GPUs free?): {_cause(exc)}"
            ) from exc
        slots = [(node, index) for node, index in bundles if index is not None]
        return _RoleReservation(group, bundles, slots)

    def _gpu_sharers(self) -> dict[tuple[str, int], int]:
        counts: dict[tuple[str, int], int] = {}
        for role in self.roles().values():
            for node, indices in role.gpus:
                for index in indices:
                    counts[(node, index)] = counts.get((node, index), 0) + 1
        return counts

    def _nodes_in_use(self) -> list[str]:
        nodes = self.resources.nodes
        restricted = {node for role in self.roles().values() for node, _ in role.gpus}
        return [node for node in nodes if not restricted or node in restricted] or [
            nodes[0]
        ]

    def claim_slots(self, key: str, count: int) -> list[int]:
        """Slot numbers for ``count`` new one-GPU (or CPU) processes of role ``key``, least used first.

        A training group's workers spread over the role's GPUs as before; an
        evaluation group opened next to them starts on the GPUs they leave free.
        Pass the numbers to :meth:`place` as ``slot`` and back to :meth:`release_slots`.
        """
        reservation = self._reservations[key]
        load = self._slot_load.setdefault(
            key, [0] * len(reservation.slots or reservation.bundles)
        )
        claimed = []
        for _ in range(count):
            slot = load.index(min(load))
            load[slot] += 1
            claimed.append(slot)
        return claimed

    def release_slots(self, key: str, slots: Sequence[int]) -> None:
        """Return slots from :meth:`claim_slots` once their processes are gone."""
        load = self._slot_load[key]
        for slot in slots:
            load[slot] -= 1

    def place(
        self,
        key: str,
        index: int,
        *,
        slot: int | None = None,
        gpus_per_process: int = 1,
        env_vars: Mapping[str, str] | None = None,
        **options: Any,
    ) -> ActorPlacement:
        """Where actor ``index`` of role ``key`` runs, with the Ray options for ``.options(...)``.

        Process ``i`` takes the role's GPU slots ``i * gpus_per_process`` onwards
        (wrapping around, so more processes than GPUs share them), or slot
        ``slot`` when given; a role without GPUs spreads its processes over the nodes.
        """
        reservation = self._reservations[key]
        position = index if slot is None else slot
        if reservation.slots:
            groups = max(1, len(reservation.slots) // gpus_per_process)
            first = (position % groups) * gpus_per_process
            chosen = reservation.slots[first : first + gpus_per_process]
            node = chosen[0][0]
            gpus = tuple(gpu for _, gpu in chosen)
            bundle = reservation.bundles.index(chosen[0])
        else:
            bundle = position % len(reservation.bundles)
            node = reservation.bundles[bundle][0]
            gpus = ()
        strategies = importlib.import_module("ray.util.scheduling_strategies")
        on_driver = node == self.resources.nodes[0]
        devices = device_names(gpus, self._driver_devices if on_driver else None)
        variables = {
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
            _NOSET_CUDA: "1",
            **dict(env_vars or {}),
        }
        return ActorPlacement(
            role=key,
            index=index,
            node=node,
            gpus=gpus,
            options={
                "num_cpus": 0,
                "num_gpus": 0,
                "max_restarts": 0,
                "scheduling_strategy": strategies.PlacementGroupSchedulingStrategy(
                    reservation.group, placement_group_bundle_index=bundle
                ),
                "runtime_env": {"env_vars": variables},
                **options,
            },
        )

    def describe(self) -> list[str]:
        """One line per role: the nodes and GPUs its placement group holds."""
        lines = []
        for key, reservation in self._reservations.items():
            where = sorted({node for node, _ in reservation.bundles})
            gpus = [f"{node}:{gpu}" for node, gpu in reservation.slots]
            lines.append(
                f"{key}: {', '.join(gpus) if gpus else 'cpu'} on {', '.join(where)}"
            )
        return lines

    def close(self) -> None:
        """Release the placement groups."""
        placement = importlib.import_module("ray.util.placement_group")
        for reservation in self._reservations.values():
            try:
                placement.remove_placement_group(reservation.group)
            except Exception as exc:  # noqa: BLE001 - the cluster may already be gone
                logger.warning(f"could not release a Ray placement group: {exc!r}")
        self._reservations.clear()


def device_names(gpus: Sequence[int], visible: Sequence[str] | None) -> tuple[str, ...]:
    """The devices behind plan GPU indices, which count the GPUs the driver sees (``visible``)."""
    if visible is None:
        return tuple(str(gpu) for gpu in gpus)
    return tuple(visible[gpu] if gpu < len(visible) else str(gpu) for gpu in gpus)


def connect_ray(address: str | None) -> bool:
    """Join a running cluster (``address``, ``RAY_ADDRESS`` or the local one), else start one; True when started here."""
    ray = import_ray()
    if ray.is_initialized():
        return False
    keep_driver_interpreter()
    try:
        ray.init(address=address or "auto", log_to_driver=True)
        return False
    except ConnectionError:
        if address is not None:
            raise
    ray.init(include_dashboard=False, log_to_driver=True)
    return True
