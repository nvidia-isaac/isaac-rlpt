# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic dataset-adapter seam for simulator embodiments.

An embodiment adapter turns a raw LeRobot episode into the contiguous
state/action arrays an offline buffer normalizes, and carries the modality
layout used to slice them. The generic GR00T policy stays embodiment-agnostic
and receives one of these through the :class:`DatasetAdapter` seam; each
simulator owns its own adapter (e.g. ``simulators/g1_arena/embodiment.py``).
:func:`get_dataset_adapter` resolves the ``dataset_adapter`` config name the
env-aware launcher passes in, importing the owning simulator lazily so this
module never imports a simulator at load time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

__all__ = ["DatasetAdapter", "get_dataset_adapter", "register_dataset_adapter", "resolve_adapter_option"]


@dataclass(frozen=True)
class DatasetAdapter:
    """How an embodiment turns a raw LeRobot episode into buffer arrays.

    The seam the generic
    :class:`~isaac_rl_posttraining.policies.gr00t.data.Gr00tOfflineBuffer`
    consumes without importing any embodiment code: it calls :attr:`assemble`
    per episode and slices the result with :attr:`modality_json`. All fields
    are optional so the buffer's own generic path (raw ``observation.state`` /
    ``action`` and the dataset's ``meta/modality.json``) is the ``None``
    adapter.

    Attributes:
        assemble: ``assemble(df) -> (state, action)`` float32 arrays.
        modality_json: Flat layout slicing the assembled arrays; ``None``
            keeps the dataset's own ``meta/modality.json``.
        fallback_state_groups: State groups a raw base model (no baked config)
            is registered for; ``None`` registers every group.
        fallback_action_groups: Same, for action groups.
        decode_horizon: Executed action steps a raw base model is registered
            for; ``None`` uses the base model's action window.
    """

    assemble: Callable[[Any], tuple[Any, Any]]
    modality_json: Optional[dict] = None
    fallback_state_groups: Optional[Sequence[str]] = None
    fallback_action_groups: Optional[Sequence[str]] = None
    decode_horizon: Optional[int] = None


# Names that select the buffer's built-in generic path (no adapter): raw
# observation.state/action sliced by the dataset's own modality.json.
_GENERIC_NAMES = {"", "none", "passthrough", "generic"}

# name -> adapter constructor, or a lazy ``"module:attr"`` string pointing at the
# simulator that owns the adapter. The string form keeps importing this module
# free of any simulator import; the simulator loads only on first use.
_AdapterTarget = Union[str, Callable[[], "DatasetAdapter"]]
_ADAPTERS: dict[str, _AdapterTarget] = {
    "g1_wbc": "simulators.g1_arena.embodiment:g1_wbc_adapter",
}


def register_dataset_adapter(name: str, target: _AdapterTarget) -> None:
    """Register a simulator's adapter constructor (a callable or ``"module:attr"``)."""
    _ADAPTERS[name.strip().lower()] = target


def get_dataset_adapter(name: Optional[str]) -> Optional[DatasetAdapter]:
    """Resolve a ``dataset_adapter`` config name to an adapter (or ``None``).

    Args:
        name: Adapter name (e.g. ``"g1_wbc"``), or a generic name
            (``None``/``"none"``/``"passthrough"``) selecting the buffer's
            raw-passthrough path.

    Returns:
        The adapter instance, or ``None`` for the generic path.

    Raises:
        KeyError: For an unknown adapter name.
    """
    key = (name or "").strip().lower()
    if key in _GENERIC_NAMES:
        return None
    try:
        target = _ADAPTERS[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset_adapter {name!r}; "
            f"known: {sorted(_ADAPTERS)} or one of {sorted(_GENERIC_NAMES - {''})}"
        )
    if isinstance(target, str):
        module_path, attr = target.split(":", 1)
        target = getattr(importlib.import_module(module_path), attr)
    return target()


def resolve_adapter_option(options: dict[str, Any]) -> dict[str, Any]:
    """Replace a policy spec's ``dataset_adapter`` name with the adapter it names.

    Policy stacks never import this layer, so launchers resolve the name
    before the spec reaches ``load``.
    """
    if not isinstance(options.get("dataset_adapter"), str):
        return options
    return {**options, "dataset_adapter": get_dataset_adapter(options["dataset_adapter"])}
