# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free GPUs while planning a placement, and the counts a role asks for."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.placement import GpuRequest, Resources

__all__ = ["GpuPool", "gpu_count", "grouped", "requested_count", "whole_groups"]


class GpuPool:
    """Free GPUs in (node, index) order across the allowed nodes."""

    def __init__(self, resources: Resources, nodes: Sequence[str]) -> None:
        self.resources = resources
        self.all: tuple[tuple[str, int], ...] = tuple(
            (node, index)
            for node, count in zip(resources.nodes, resources.gpus_per_node)
            if node in nodes
            for index in range(count)
        )
        self.free: list[tuple[str, int]] = list(self.all)

    def claim(self, role: str, devices: Sequence[int | str]) -> list[tuple[str, int]]:
        claimed = []
        for device in devices:
            node_token, _, index_token = str(device).rpartition(":")
            node = self.resources.node(node_token or "0")
            if not index_token.isdigit():
                raise ValueError(
                    f"{role}: GPU {device!r} is not 'node:index' or 'index'"
                )
            gpu = (node, int(index_token))
            if gpu not in self.free:
                total = self.resources.gpus_per_node[self.resources.nodes.index(node)]
                reason = (
                    "does not exist"
                    if gpu[1] >= total
                    else "is not free (taken by another role or outside placement.nodes)"
                )
                raise ValueError(f"{role}: GPU {device!r} {reason}")
            self.free.remove(gpu)
            claimed.append(gpu)
        return claimed

    def take(self, role: str, count: int, *, group: int = 1) -> list[tuple[str, int]]:
        """Take ``count`` free GPUs in groups of ``group`` that each stay on one node."""
        taken: list[tuple[str, int]] = []
        while len(taken) < count:
            for node in dict.fromkeys(node for node, _ in self.free):
                on_node = [gpu for gpu in self.free if gpu[0] == node]
                if len(on_node) >= group:
                    chunk = on_node[:group]
                    break
            else:
                raise ValueError(
                    f"placement over-subscribes the GPUs: {role} needs {count} "
                    f"(in groups of {group} per node) but only {len(self.free) + len(taken)} are free"
                )
            for gpu in chunk:
                self.free.remove(gpu)
            taken.extend(chunk)
        return taken


def grouped(
    gpus: Sequence[tuple[str, int]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    nodes: dict[str, list[int]] = {}
    for node, index in gpus:
        nodes.setdefault(node, []).append(index)
    return tuple((node, tuple(sorted(indices))) for node, indices in nodes.items())


def gpu_count(gpus: GpuRequest) -> int | None:
    return gpus if gpus is None or isinstance(gpus, int) else len(gpus)


def requested_count(
    role: str, gpus: GpuRequest, processes: int | None, per_process: int
) -> int | None:
    """The GPU count a role asks for explicitly (``None``: planner decides)."""
    if gpus is None:
        return None if processes is None else processes * per_process
    count = gpus if isinstance(gpus, int) else len(gpus)
    if processes is not None and processes * per_process != count:
        raise ValueError(
            f"{role}: {processes} process(es) x {per_process} gpu disagree with {count} gpus"
        )
    return count


def whole_groups(
    counts: dict[str, int],
    role: str,
    group: int,
    proportional: Sequence[str],
    groups: Mapping[str, int],
) -> int:
    """Round ``role``'s count to whole groups, moving the difference to envs (or the largest other role)."""
    whole = max(group, counts[role] // group * group)
    others = [other for other in proportional if groups.get(other, 1) == 1]
    if not others:
        return whole
    taker = "envs" if "envs" in others else max(others, key=counts.__getitem__)
    counts[taker] += counts[role] - whole
    while counts[taker] < 1 and whole > group:
        whole -= group
        counts[taker] += group
    for donor in sorted(others, key=counts.__getitem__, reverse=True):
        while counts[taker] < 1 and donor != taker and counts[donor] > 1:
            counts[donor] -= 1
            counts[taker] += 1
    return whole
