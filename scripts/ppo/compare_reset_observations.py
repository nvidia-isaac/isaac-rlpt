#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare captured simulator reset observations at pixel level."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_KEYS = {"schema_version", "seed", "num_envs", "reset_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--assert-identical",
        action="store_true",
        help="exit nonzero unless metadata and every captured array match exactly",
    )
    parser.add_argument("--output", type=Path, help="also write the comparison as JSON")
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.array(payload[name], copy=True) for name in payload.files}


def _scalar(payload: dict[str, np.ndarray], name: str) -> int:
    if name not in payload:
        raise ValueError(f"capture has no {name!r} metadata")
    value = payload[name]
    if value.size != 1:
        raise ValueError(f"capture metadata {name!r} must be scalar")
    return int(value.reshape(-1)[0])


def _with_env_axis(array: np.ndarray, num_envs: int) -> np.ndarray:
    if array.ndim > 0 and array.shape[0] == num_envs:
        return array
    if num_envs == 1:
        return np.expand_dims(array, axis=0)
    raise ValueError(
        f"array shape {array.shape} has no leading num_envs={num_envs} axis"
    )


def _array_comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_envs: int,
) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    report: dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "shape_matches": same_shape,
        "dtype_matches": same_dtype,
        "exact": bool(same_shape and same_dtype and np.array_equal(left, right)),
    }
    if (
        not same_shape
        or num_envs <= 0
        or not np.issubdtype(left.dtype, np.number)
        or not np.issubdtype(right.dtype, np.number)
    ):
        return report

    left_envs = _with_env_axis(left, num_envs)
    right_envs = _with_env_axis(right, num_envs)
    difference = np.abs(left_envs.astype(np.float64) - right_envs.astype(np.float64))
    flat = difference.reshape(num_envs, -1)
    per_env_mean = flat.mean(axis=1)
    mean_square = float(np.square(difference).mean())
    report.update(
        exact_envs=int(np.count_nonzero(np.all(flat == 0, axis=1))),
        changed_values=int(np.count_nonzero(difference)),
        total_values=int(difference.size),
        changed_fraction=float(np.count_nonzero(difference) / difference.size),
        mean_absolute_error=float(difference.mean()),
        root_mean_square_error=float(math.sqrt(mean_square)),
        maximum_absolute_error=float(difference.max()),
        per_env_mean_absolute_error={
            "minimum": float(per_env_mean.min()),
            "median": float(np.median(per_env_mean)),
            "p95": float(np.percentile(per_env_mean, 95)),
            "maximum": float(per_env_mean.max()),
        },
    )
    if np.issubdtype(left.dtype, np.integer) and mean_square > 0:
        peak = float(np.iinfo(left.dtype).max - np.iinfo(left.dtype).min)
        report["peak_signal_to_noise_db"] = float(
            20.0 * math.log10(peak / math.sqrt(mean_square))
        )
    else:
        report["peak_signal_to_noise_db"] = None
    return report


def compare_captures(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    left_seed = _scalar(left, "seed")
    right_seed = _scalar(right, "seed")
    left_num_envs = _scalar(left, "num_envs")
    right_num_envs = _scalar(right, "num_envs")
    metadata_matches = bool(
        _scalar(left, "schema_version") == _scalar(right, "schema_version")
        and left_seed == right_seed
        and left_num_envs == right_num_envs
    )
    left_keys = set(left) - _METADATA_KEYS
    right_keys = set(right) - _METADATA_KEYS
    array_keys_match = left_keys == right_keys
    comparisons = {
        name: _array_comparison(
            left[name],
            right[name],
            num_envs=left_num_envs if left_num_envs == right_num_envs else 0,
        )
        for name in sorted(left_keys & right_keys)
    }
    identical = bool(
        metadata_matches
        and array_keys_match
        and all(result["exact"] for result in comparisons.values())
    )
    return {
        "schema_version": 1,
        "metadata_matches": metadata_matches,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "left_num_envs": left_num_envs,
        "right_num_envs": right_num_envs,
        "left_reset_index": _scalar(left, "reset_index"),
        "right_reset_index": _scalar(right, "reset_index"),
        "array_keys_match": array_keys_match,
        "left_array_keys": sorted(left_keys),
        "right_array_keys": sorted(right_keys),
        "arrays": comparisons,
        "identical": identical,
    }


def main() -> int:
    args = parse_args()
    report = compare_captures(_load(args.left), _load(args.right))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.assert_identical and not report["identical"])


if __name__ == "__main__":
    raise SystemExit(main())
