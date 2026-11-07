# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for reset-observation comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ppo"
    / "compare_reset_observations.py"
)


def _capture(
    path: Path, *, image: np.ndarray, state: np.ndarray, num_envs: int = 2
) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        seed=np.asarray(100000, dtype=np.int64),
        num_envs=np.asarray(num_envs, dtype=np.int64),
        reset_index=np.asarray(0, dtype=np.int64),
        ego_view=image,
        state=state,
    )


def test_comparator_accepts_identical_captures(tmp_path: Path) -> None:
    image = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _capture(left, image=image, state=state)
    _capture(right, image=image, state=state)

    result = subprocess.run(
        [sys.executable, SCRIPT, left, right, "--assert-identical"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["identical"] is True
    assert report["arrays"]["ego_view"]["exact_envs"] == 2
    assert report["arrays"]["ego_view"]["mean_absolute_error"] == pytest.approx(0.0)


def test_comparator_quantifies_pixel_differences(tmp_path: Path) -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    changed = image.copy()
    changed[1, 0, 0, 0] = 10
    state = np.zeros((2, 3), dtype=np.float32)
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _capture(left, image=image, state=state)
    _capture(right, image=changed, state=state)

    result = subprocess.run(
        [sys.executable, SCRIPT, left, right, "--assert-identical"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    image_report = report["arrays"]["ego_view"]
    assert result.returncode == 1
    assert report["identical"] is False
    assert image_report["exact_envs"] == 1
    assert image_report["changed_values"] == 1
    assert image_report["maximum_absolute_error"] == pytest.approx(10.0)
    assert image_report["mean_absolute_error"] == 10 / image.size
    assert report["arrays"]["state"]["exact"] is True


def test_comparator_reports_env_count_mismatch_without_crashing(tmp_path: Path) -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    state = np.zeros((2, 3), dtype=np.float32)
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _capture(left, image=image, state=state)
    _capture(right, image=image, state=state, num_envs=3)

    result = subprocess.run(
        [sys.executable, SCRIPT, left, right, "--assert-identical"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["metadata_matches"] is False
    assert report["identical"] is False
    assert report["arrays"]["ego_view"]["exact"] is True
    assert "exact_envs" not in report["arrays"]["ego_view"]
