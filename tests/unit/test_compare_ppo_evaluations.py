# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the retained PPO evaluation comparator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "ppo" / "compare_evaluations.py"
)


def _record(*, iteration: int, policy: str, success: bool = True) -> dict:
    return {
        "schema_version": 3,
        "iteration": iteration,
        "policy_version": iteration,
        "policy_sha256": policy,
        "episodes": [
            {
                "environment_index": 0,
                "episode_index": 0,
                "reset_seed": 100000,
                "reset_seed_scope": "environment_batch",
                "success": success,
                "env_steps": 10,
                "return": 1.0 if success else 0.0,
                "initial_observation_sha256": "observation",
                "initial_observation_sha256_by_key": {
                    "dino_embedding": "dino",
                    "state": "state",
                },
                "initial_raw_observation_sha256": "raw-observation",
                "initial_raw_observation_sha256_by_key": {
                    "ego_view": "raw-image",
                    "state": "raw-state",
                },
                "first_execution_action_sha256": "first-action",
                "execution_action_sha256": "actions",
                "observation_sha256_by_decision": ["observation"],
                "execution_action_sha256_by_decision": ["actions"],
            }
        ],
    }


def _write(path: Path, *records: dict) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_cli_asserts_identical_retained_evaluations(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(
        left, _record(iteration=0, policy="old"), _record(iteration=1, policy="same")
    )
    _write(right, _record(iteration=1, policy="same"))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(left),
            str(right),
            "--left-iteration",
            "1",
            "--assert-identical",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["identical"] is True


def test_cli_exposes_policy_and_outcome_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, _record(iteration=1, policy="left", success=True))
    _write(right, _record(iteration=1, policy="right", success=False))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left), str(right), "--assert-identical"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["policy_sha256_matches"] is False
    assert report["matches"]["success"] == 0
    assert report["identical"] is False


def test_cli_exposes_raw_camera_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left_record = _record(iteration=1, policy="same")
    right_record = _record(iteration=1, policy="same")
    right_record["episodes"][0]["initial_raw_observation_sha256_by_key"]["ego_view"] = (
        "different-image"
    )
    _write(left, left_record)
    _write(right, right_record)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left), str(right), "--assert-identical"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["matches"]["initial_raw_observation_sha256_by_key"]["ego_view"] == 0
    assert report["identical"] is False


def _rendered_differently(record: dict) -> dict:
    """The same snapshot and scene in another process: cameras, actions and outcome differ."""
    episode = record["episodes"][0]
    episode["initial_raw_observation_sha256_by_key"]["ego_view"] = "other-render"
    episode["execution_action_sha256"] = "other-actions"
    episode["success"] = not episode["success"]
    return record


def test_cli_accepts_the_same_setup_rendered_differently(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, _record(iteration=1, policy="same"))
    _write(right, _rendered_differently(_record(iteration=1, policy="same")))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left), str(right), "--assert-same-setup"],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["same_setup"] is True and report["identical"] is False
    assert (report["left_success_rate"], report["right_success_rate"]) == (1.0, 0.0)


def test_cli_same_setup_rejects_another_policy_or_robot_state(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    _write(left, _record(iteration=1, policy="same"))
    other_state = _record(iteration=1, policy="same")
    other_state["episodes"][0]["initial_raw_observation_sha256_by_key"]["state"] = (
        "moved"
    )
    for right_record in (_record(iteration=1, policy="other"), other_state):
        right = tmp_path / "right.jsonl"
        _write(right, right_record)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(left), str(right), "--assert-same-setup"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert json.loads(result.stdout)["same_setup"] is False
