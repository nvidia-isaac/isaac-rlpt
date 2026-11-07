# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra/OmegaConf helpers for validating configs against dataclass schemas."""

from __future__ import annotations

import dataclasses
import json
import typing
from collections.abc import Sequence
from typing import Any, TypeVar

from omegaconf import OmegaConf

__all__ = [
    "REMOVED_KEYS",
    "REMOVED_VALUES",
    "RENAMED_GROUPS",
    "RemovedConfigKeyError",
    "convert_hydra_to_dataclass",
    "display_config",
    "refuse_removed_overrides",
]

T = TypeVar("T")


def display_config(cfg: Any) -> None:
    """Pretty-print a config (dataclass or mapping) as indented JSON."""
    try:
        d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
        print("[config]\n" + json.dumps(d, indent=2, default=str))
    except (TypeError, ValueError):
        print("[config]", cfg)


def _unwrap_optional(t: Any) -> Any:
    """Return ``X`` for ``Optional[X]`` annotations, else ``t`` unchanged."""
    if typing.get_origin(t) is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


#: Top-level keys the conversion knowingly leaves to Hydra (config-group
#: nodes consumed directly from the composed config, not the dataclasses).
_EXPECTED_EXTRA_KEYS = {"algorithm", "base_model", "simulator", "policy", "value_function"}

#: Settings that no longer exist, with what replaced them; a config that still sets one is refused.
REMOVED_KEYS: dict[str, str] = {
    "buffer.split": (
        "the pool layouts are gone: buffer.success_fraction=<share> replaces split=outcome, "
        "buffer.demonstrations=<saved buffer dir> with buffer.demonstration_fraction=<share> replaces split=source"
    ),
    "buffer.fractions": "give the share directly as buffer.success_fraction or buffer.demonstration_fraction",
    "buffer.save_buffer_images": "removed; saved buffers always keep every stored observation component",
    "buffer.save_buffer_image_keys": "removed; saved buffers always keep every stored observation component",
    "runtime.rlinf.serialize_factory_by_value": "renamed to runtime.rlinf.serialize_specs_by_value",
    "env.use_gt_rewards": (
        "removed; nothing read it, the run always trains on the environment reward "
        "(env.sparse_success_reward replaces it with a terminal-outcome reward)"
    ),
    "env.extra_keys_to_drop": "removed; nothing read it, the networks drop observation keys themselves",
    "debug": "removed; nothing read it (logging.log_level=DEBUG raises the console log level)",
    "training.schedule": "removed; use schedule=sync (was serial), schedule=overlapped (was lookahead) or schedule=async",
    "ppo.overlap_collection": "removed; use schedule=overlapped (was true) or schedule=sync (was false)",
    "learner.num_ranks": "moved to placement.learner.ranks",
    **{
        f"algorithm.{key}": "removed; nothing read it"
        for key in (
            "action_noise",
            "actor_scheduler_name",
            "actor_scheduler_warmup_steps",
            "critic_scheduler_name",
            "critic_scheduler_warmup_steps",
        )
    },
}

_SCHEDULE_RENAMES = {"serial": "sync", "lookahead": "overlapped"}
#: Renamed values by key (or config group): ``{key: {old value: new value}}``.
REMOVED_VALUES: dict[str, dict[str, str]] = {
    "schedule": _SCHEDULE_RENAMES,
    "schedule.kind": _SCHEDULE_RENAMES,
}

#: Renamed config groups: ``{old group: (new group, {old key: new key})}``; unlisted keys keep their name.
RENAMED_GROUPS: dict[str, tuple[str, dict[str, str]]] = {
    "backend": ("base_model", {"base_ckpt": "ckpt"}),
}


class RemovedConfigKeyError(ValueError):
    """A config sets a key the schema does not have (removed, renamed or mistyped)."""


def _renamed_key(key: str) -> str | None:
    """The current spelling of ``key`` when its config group was renamed, else ``None``."""
    group, dot, rest = key.partition(".")
    if group not in RENAMED_GROUPS:
        return None
    new_group, keys = RENAMED_GROUPS[group]
    if not dot:
        return new_group
    head, sep, tail = rest.partition(".")
    return f"{new_group}.{keys.get(head, head)}{sep}{tail}"


def refuse_removed_overrides(overrides: Sequence[str]) -> None:
    """Refuse command-line overrides of :data:`REMOVED_KEYS`, :data:`REMOVED_VALUES` and :data:`RENAMED_GROUPS` with what replaced them.

    Hydra itself would only say the key or option is not in the config; this names the replacement.
    """
    refused = []
    for override in overrides:
        key, _, value = override.lstrip("+~").partition("=")
        renamed = REMOVED_VALUES.get(key, {}).get(value)
        if key in REMOVED_KEYS:
            refused.append(f"{key}: {REMOVED_KEYS[key]}")
        elif (current := _renamed_key(key)) is not None:
            refused.append(f"{key}: renamed to {current}")
        elif renamed is not None:
            refused.append(f"{key}={value}: renamed to {key}={renamed}")
    if refused:
        raise RemovedConfigKeyError("config keys this version does not accept: " + "; ".join(refused))


def _refuse_undeclared_keys(dc_type: type, data: dict, path: str) -> None:
    """Raise :class:`RemovedConfigKeyError` naming every key ``dc_type`` does not declare."""
    declared = {f.name for f in dataclasses.fields(dc_type)}
    undeclared = [
        k
        for k in data
        if k not in declared and not (path == "" and k in _EXPECTED_EXTRA_KEYS)
    ]
    if undeclared:
        refused = []
        for key in sorted(undeclared):
            dotted = f"{path}{key}"
            if dotted in REMOVED_KEYS:
                refused.append(f"{dotted}: {REMOVED_KEYS[dotted]}")
            elif (current := _renamed_key(dotted)) is not None:
                refused.append(f"{dotted}: renamed to {current}")
            else:
                refused.append(
                    f"{dotted}: not a setting of {dc_type.__name__}; its settings are {sorted(declared)}"
                )
        raise RemovedConfigKeyError(
            "config keys this version does not accept: " + "; ".join(refused)
        )


def _build(dc_type: type[T], data: Any, _path: str = "") -> Any:
    """Recursively instantiate ``dc_type`` from a plain dict.

    Nested dataclass fields are constructed recursively. A key the dataclass
    does not declare is refused: a removed or renamed one (:data:`REMOVED_KEYS`)
    with its replacement, any other with the group's real settings, so a typo
    in a yaml or a ``+group.key`` override fails at launch instead of
    silently running the default. Only the known Hydra config-group nodes at
    the root are left to their own consumers.
    """
    if not (dataclasses.is_dataclass(dc_type) and isinstance(data, dict)):
        return data
    try:
        hints = typing.get_type_hints(dc_type)
    except Exception:
        hints = {}
    _refuse_undeclared_keys(dc_type, data, _path)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        if f.name not in data:
            continue
        field_type = _unwrap_optional(hints.get(f.name))
        value = data[f.name]
        if field_type is not None and dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _build(field_type, value, _path=f"{_path}{f.name}.")
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def convert_hydra_to_dataclass(cfg: Any, dataclass_type: type[T]) -> T:
    """Resolve a Hydra/OmegaConf config and build the matching dataclass.

    Args:
        cfg: An OmegaConf config (interpolations are resolved) or plain dict.
        dataclass_type: Root dataclass schema to instantiate.

    Returns:
        A populated ``dataclass_type`` instance. Unknown keys raise
        :class:`RemovedConfigKeyError`; unknown *values* for declared fields
        raise via the dataclass itself.
    """
    container = cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)
    return _build(dataclass_type, container)
