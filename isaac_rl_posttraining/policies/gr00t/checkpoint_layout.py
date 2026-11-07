# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GR00T checkpoint's state, action and horizon sizes, read from its JSON metadata.

Loading a checkpoint registers a modality layout that fixes the state width
the policy slices and the action width and steps it decodes
(:meth:`~.policy.Gr00tPolicy.register_baked_embodiment`,
:meth:`~.policy.Gr00tPolicy.make_offline_buffer`). The same numbers are in
plain JSON next to the weights: ``config.json`` (flow horizon, head width),
``processor_config.json`` (the baked modality keys and decode window) and
``statistics.json`` (one statistics vector per group, whose length is the
group's width). This module reads them, following the load's own order of
sources, so the launch plan can compare them with the simulator before any
weights load.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import Any

from isaac_rl_posttraining.policies.geometry import PolicyGeometry

__all__ = ["DEFAULT_EMBODIMENT_TAG", "checkpoint_geometry"]

#: The tag the steering base model registers when ``base_model.embodiment_tag`` is unset.
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
_N15_MODEL_TYPE = "gr00t_n1_5"
#: Statistics whose vector length is a group's width, in the order the load reads them.
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _processor_dir(base_ckpt: str) -> str:
    """Where the processor files are: the root, or ``processor/`` when the root has none (as at load)."""
    subdir = os.path.join(base_ckpt, "processor")
    if os.path.isdir(subdir) and not os.path.exists(
        os.path.join(base_ckpt, "processor_config.json")
    ):
        return subdir
    return base_ckpt


def _unresolved(reason: str) -> PolicyGeometry:
    return PolicyGeometry(
        source="GR00T checkpoint", resizes_images=True, unknown=reason
    )


def _stat_width(group: Any) -> int | None:
    if isinstance(group, Mapping):
        for name in _STAT_KEYS:
            if name in group:
                return len(group[name])
    return None


def _baked_geometry(
    processor: str, baked: Mapping[str, Any], tag: str, base: dict[str, Any]
) -> PolicyGeometry:
    """Widths of a baked embodiment: its modality keys, sized by the baked statistics."""
    statistics_path = os.path.join(processor, "statistics.json")
    if not os.path.exists(statistics_path):
        return PolicyGeometry(**base, unknown=f"{statistics_path} is missing")
    statistics = _read_json(statistics_path).get(tag, {})
    widths: dict[str, int] = {}
    for modality in ("state", "action"):
        total = 0
        for key in baked[modality]["modality_keys"]:
            width = _stat_width(statistics.get(modality, {}).get(key))
            if width is None:
                return PolicyGeometry(
                    **base,
                    unknown=f"{statistics_path} has no {modality} statistics for {key!r}",
                )
            total += width
        widths[modality] = total
    return PolicyGeometry(
        **{**base, "source": f"{base['source']}, baked embodiment {tag!r}"},
        state_width=widths["state"],
        action_width=widths["action"],
        decoded_horizon=len(baked["action"]["delta_indices"]),
    )


def _dataset_geometry(
    layout: Mapping[str, Any],
    origin: str,
    horizon: int | None,
    action_groups: Any,
    base: dict[str, Any],
) -> PolicyGeometry:
    """Widths of a raw base model registered from a dataset layout (the adapter's, or the dataset's own)."""
    state = layout.get("state", {})
    actions = layout.get("action", {})
    keys = (
        list(actions)
        if action_groups is None
        else [key for key in action_groups if key in actions]
    )
    return PolicyGeometry(
        **{**base, "source": f"{base['source']}, layout of {origin}"},
        state_width=max((int(group["end"]) for group in state.values()), default=None),
        action_width=sum(
            int(actions[key]["end"]) - int(actions[key]["start"]) for key in keys
        )
        or None,
        decoded_horizon=horizon,
    )


def checkpoint_geometry(
    base_ckpt: str | None,
    embodiment_tag: str,
    *,
    dataset_dir: str | None = None,
    multitask: bool = False,
    dataset_adapter: Any = None,
) -> PolicyGeometry:
    """The geometry a GR00T steering base model will register for ``embodiment_tag``, from JSON only.

    Mirrors the load: a checkpoint that bakes the tag defines the layout;
    otherwise the dataset adapter's layout, else the dataset's own
    ``meta/modality.json``, with a decode window as long as the flow horizon.
    Whatever cannot be read here is left ``None``, with the reason in ``unknown``.
    """
    if not base_ckpt:
        return _unresolved("base_model.ckpt is unset")
    if not os.path.isdir(base_ckpt):
        return _unresolved(
            f"base_model.ckpt={base_ckpt} is not a local directory (a model id)"
        )
    config_path = os.path.join(base_ckpt, "config.json")
    if not os.path.exists(config_path):
        return _unresolved(f"{config_path} is missing")
    config = _read_json(config_path)
    if config.get("model_type") == _N15_MODEL_TYPE:
        return _unresolved("an N1.5 checkpoint's data-config layout binds at load")
    horizon = config.get("action_horizon")
    base: dict[str, Any] = {
        "source": f"checkpoint {base_ckpt}",
        "latent_horizon": None if horizon is None else int(horizon),
        "latent_width": None
        if config.get("max_action_dim") is None
        else int(config["max_action_dim"]),
        "horizon_key": f"the checkpoint's action_horizon ({config_path})",
        "resizes_images": True,
    }
    geometry = _layout_geometry(
        _processor_dir(base_ckpt),
        embodiment_tag,
        base,
        dataset_dir=dataset_dir,
        multitask=multitask,
        dataset_adapter=dataset_adapter,
    )
    missing = [
        key for key in ("action_horizon", "max_action_dim") if config.get(key) is None
    ]
    if missing and not geometry.unknown:
        return dataclasses.replace(
            geometry, unknown=f"{config_path} declares no {' or '.join(missing)}"
        )
    return geometry


def _layout_geometry(
    processor: str,
    embodiment_tag: str,
    base: dict[str, Any],
    *,
    dataset_dir: str | None,
    multitask: bool,
    dataset_adapter: Any,
) -> PolicyGeometry:
    """The state and action layout the load registers, from the first source that has one."""
    processor_config = os.path.join(processor, "processor_config.json")
    if not os.path.exists(processor_config):
        return PolicyGeometry(**base, unknown=f"{processor_config} is missing")
    configs = (
        _read_json(processor_config)["processor_kwargs"].get("modality_configs") or {}
    )
    baked = configs.get(embodiment_tag)
    if isinstance(baked, Mapping) and baked.get("action", {}).get("modality_keys"):
        return _baked_geometry(processor, baked, embodiment_tag, base)
    adapter_layout = None if dataset_adapter is None else dataset_adapter.modality_json
    action_groups = (
        None if dataset_adapter is None else dataset_adapter.fallback_action_groups
    )
    if adapter_layout is not None:
        return _dataset_geometry(
            adapter_layout,
            "the dataset adapter",
            base["latent_horizon"],
            action_groups,
            base,
        )
    if dataset_dir and not multitask:
        modality = os.path.join(str(dataset_dir), "meta", "modality.json")
        if os.path.exists(modality):
            return _dataset_geometry(
                _read_json(modality),
                modality,
                base["latent_horizon"],
                action_groups,
                base,
            )
        return PolicyGeometry(**base, unknown=f"{modality} is missing")
    return PolicyGeometry(
        **base,
        unknown=(
            f"the checkpoint bakes no {embodiment_tag!r} layout (base_model.embodiment_tag) and no single "
            "dataset names one"
        ),
    )
