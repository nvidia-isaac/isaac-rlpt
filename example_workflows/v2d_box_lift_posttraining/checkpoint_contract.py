# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate complete GR00T 58-D joint checkpoint inputs before GPU startup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from simulators.v2d.contract import (
    ACTION_KEYS,
    STATE_KEYS,
    VIDEO_KEYS,
    WIDTHS,
    JsonValue,
)


def _fingerprint(path: Path) -> dict[str, JsonValue]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_statistics(group: str, keys: tuple[str, ...], stats: dict) -> None:
    for key in keys:
        for statistic in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[group][key][statistic])
            if len(values) != WIDTHS[key] or not np.isfinite(values).all():
                raise ValueError(f"Invalid statistics: {group}.{key}.{statistic}")
        if np.any(np.asarray(stats[group][key]["std"]) < 0):
            raise ValueError("Negative normalization standard deviation")
        if np.any(np.asarray(stats[group][key]["q01"]) > stats[group][key]["q99"]):
            raise ValueError("Reversed normalization quantiles")


def _validate_modalities(modalities: dict, stats: dict) -> None:
    for group, keys in (
        ("state", STATE_KEYS),
        ("action", ACTION_KEYS),
        ("video", VIDEO_KEYS),
    ):
        if tuple(modalities[group]["modality_keys"]) != keys:
            raise ValueError(f"Wrong {group} modality order")
        expected_delta = list(range(16)) if group == "action" else [0]
        if modalities[group]["delta_indices"] != expected_delta:
            raise ValueError(f"Wrong {group} horizon")
        if group != "video":
            _validate_statistics(group, keys, stats)


def _validate_action_configs(action_configs: list) -> None:
    expected_action_config = {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": None,
    }
    if len(action_configs) != 4 or any(
        item != expected_action_config for item in action_configs
    ):
        raise ValueError("Only absolute NON_EEF joint checkpoints are supported")


def _weight_files(path: Path, index: Path) -> tuple[dict | None, list[str]]:
    """The shard index (``None`` for a single file) and the weight file names."""
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        weights = sorted(set(weight_map.values()))
    else:
        weight_map = None
        weights = ["model.safetensors"]
    if any(Path(name).name != name for name in weights):
        raise ValueError("Checkpoint index references a shard outside the checkpoint")
    if not weights or any(
        not (path / name).is_file() or (path / name).stat().st_size == 0
        for name in weights
    ):
        raise ValueError("Checkpoint weights are missing or incomplete")
    return weight_map, weights


def _validate_shards(path: Path, weight_map: dict | None, weights: list[str]) -> None:
    from safetensors import safe_open

    for name in weights:
        with safe_open(str(path / name), framework="numpy") as shard:
            keys = set(shard.keys())
            if not keys:
                raise ValueError(f"Empty weight shard: {name}")
            if weight_map is not None:
                expected = {
                    key for key, filename in weight_map.items() if filename == name
                }
                if not expected <= keys:
                    raise ValueError(f"Weight index entries missing in shard: {name}")


def validate_checkpoint(path: Path) -> dict[str, JsonValue]:
    """Reject EEF, incomplete, or differently ordered checkpoints."""
    config = json.loads((path / "config.json").read_text())
    processor = json.loads((path / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    stats = json.loads((path / "statistics.json").read_text())["new_embodiment"]
    horizons = (
        config["action_horizon"],
        config["max_action_dim"],
        processor["max_action_horizon"],
        processor["max_action_dim"],
    )
    if horizons != (40, 132, 40, 132):
        raise ValueError("Expected flow horizon 40 and padded width 132")
    modalities = processor["modality_configs"]["new_embodiment"]
    _validate_modalities(modalities, stats)
    _validate_action_configs(modalities["action"]["action_configs"])
    index = path / "model.safetensors.index.json"
    weight_map, weights = _weight_files(path, index)
    _validate_shards(path, weight_map, weights)
    metadata = ["config.json", "processor_config.json", "statistics.json"]
    if index.exists():
        metadata.append(index.name)
    return {
        "checkpoint": str(path.resolve()),
        "weight_shards": len(weights),
        "decoded_horizon": 16,
        "flow_horizon": 40,
        "action_dim": 58,
        "files": [_fingerprint(path / name) for name in [*metadata, *weights]],
    }
