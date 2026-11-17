# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move policy states between torch modules, transport arrays, and snapshot files.

A :class:`~isaac_rl_posttraining.runtimes.base.PolicyState` carries NumPy arrays
because rollout workers and learners may live in different processes. These
helpers do the torch side: bfloat16 parameters travel as their raw ``int16``
bits (half the bytes of a float32 copy) and come back as bfloat16, and a state
can be written to a ``policy_state.pt`` snapshot that later evaluates or
deploys without a learner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from isaac_rl_posttraining.runtimes.base import PolicyState

__all__ = [
    "SNAPSHOT_FILE_NAME",
    "apply_policy_state",
    "load_policy_snapshot",
    "policy_state_from_tensors",
    "policy_state_tensors",
    "save_policy_snapshot",
]

#: File name of the policy snapshot written next to a learner checkpoint.
SNAPSHOT_FILE_NAME = "policy_state.pt"


def policy_state_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    version: int,
    exclude_prefixes: Iterable[str] = (),
) -> PolicyState:
    """Pack torch tensors (any device) into a policy state.

    Tensors whose name starts with one of ``exclude_prefixes`` are left out;
    learners use this to keep frozen submodules out of what they publish.
    """
    prefixes = tuple(exclude_prefixes)
    packed: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for name, value in tensors.items():
        if prefixes and name.startswith(prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            packed[name] = tensor.view(torch.int16).numpy()
            dtypes[name] = "bfloat16"
        else:
            packed[name] = tensor.numpy()
    return PolicyState(version=version, tensors=packed, dtypes=dtypes)


def policy_state_tensors(state: PolicyState) -> dict[str, torch.Tensor]:
    """The state's tensors as CPU torch tensors in their original dtypes."""
    tensors: dict[str, torch.Tensor] = {}
    for name, array in state.tensors.items():
        tensor = torch.from_numpy(array)
        dtype_name = state.dtypes.get(name)
        if dtype_name is not None:
            tensor = tensor.view(getattr(torch, dtype_name))
        tensors[name] = tensor
    return tensors


def apply_policy_state(
    module: torch.nn.Module, state: PolicyState, *, current_version: int
) -> int:
    """Replace ``module``'s whole state dict with a newer published state; returns its version.

    The published names must match the module's exactly, so a learner and its
    rollout copies cannot drift apart silently.
    """
    if state.version <= current_version:
        raise ValueError(
            f"policy state version must increase beyond {current_version}, got {state.version}"
        )
    current = module.state_dict()
    missing = sorted(set(current) - set(state.tensors))
    unexpected = sorted(set(state.tensors) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"policy state keys differ; missing={missing}, unexpected={unexpected}"
        )
    published = policy_state_tensors(state)
    converted = {
        name: published[name].to(device=tensor.device, dtype=tensor.dtype, copy=True)
        for name, tensor in current.items()
    }
    module.load_state_dict(converted, strict=True)
    return state.version


def save_policy_snapshot(state: PolicyState, path: str | Path) -> Path:
    """Write the state's tensors to ``path`` (a file, or a directory that gets
    :data:`SNAPSHOT_FILE_NAME`); returns the file written."""
    file = Path(path)
    if file.is_dir() or not file.suffix:
        file = file / SNAPSHOT_FILE_NAME
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".tmp")
    torch.save(policy_state_tensors(state), temporary)
    temporary.replace(file)
    return file


def load_policy_snapshot(path: str | Path, *, version: int = 1) -> PolicyState:
    """Load a snapshot written by :func:`save_policy_snapshot`.

    ``path`` is the snapshot file or the checkpoint directory holding it. The
    result can be published to a rollout group directly, so a saved policy
    evaluates without a learner group.
    """
    file = Path(path)
    if file.is_dir():
        file = file / SNAPSHOT_FILE_NAME
    if not file.is_absolute():
        raise ValueError("policy snapshot path must be absolute")
    if not file.is_file():
        raise FileNotFoundError(f"policy snapshot not found: {file}")
    tensors = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError(
            f"policy snapshot {file} must hold a non-empty mapping of tensors"
        )
    return policy_state_from_tensors(tensors, version=version)
