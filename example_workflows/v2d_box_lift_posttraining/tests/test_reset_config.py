# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision validation is explicit and does not alter historical reset paths."""

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from example_workflows.v2d_box_lift_posttraining.capability import (
    _configure_trajectory_reset,
)
from example_workflows.v2d_box_lift_posttraining.config import TaskConfig, load_task
from example_workflows.v2d_box_lift_posttraining.fixed_reset import (
    reset_to_fixed_trajectory_start,
)
from example_workflows.v2d_box_lift_posttraining.launch import parser, resolve_plan
from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

TASKS = Path(__file__).resolve().parents[1] / "configs/tasks"
BASE = TASKS / "tissue_box_lift_hold_xy_1cm_550_fixed.json"


def _reset_event(func: object) -> "EventTermCfg":
    """A stand-in for Isaac Lab's reset event term, which needs a simulator to import."""
    return cast(
        "EventTermCfg", SimpleNamespace(func=func, params={"command_name": "motion"})
    )


@pytest.fixture
def collision_task_path(tmp_path: Path) -> Path:
    config = json.loads(BASE.read_text())
    config["reset_validation"] = ResetValidationConfig(True, 32, 0.001, ()).as_dict()
    path = tmp_path / "collision.json"
    path.write_text(json.dumps(config))
    return path


def test_collision_config_preserves_fixed_reset_and_scientific_settings(
    collision_task_path,
):
    base = load_task(BASE)
    task = load_task(collision_task_path)
    assert replace(task, reset_validation=None) == base
    assert task.reset_validation == ResetValidationConfig(True, 32, 0.001, ())
    assert task.sha256 != base.sha256
    historical = asdict(base)
    del historical["reset_validation"]
    assert (
        base.sha256
        == hashlib.sha256(json.dumps(historical, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", "false"),
        ("max_attempts", 0),
        ("max_attempts", True),
        ("max_penetration_m", float("nan")),
        ("max_penetration_m", -1),
        ("max_penetration_m", True),
        ("allowed_contact_pairs", [["robot", "robot"]]),
        ("allowed_contact_pairs", [["robot"]]),
        ("allowed_contact_pairs", ["robot"]),
    ],
)
def test_invalid_collision_configuration_fails(key, value):
    config = ResetValidationConfig(True, 32, 0.001, ()).as_dict()
    config[key] = value
    with pytest.raises((ValueError, TypeError)):
        ResetValidationConfig.from_json(config)


def test_contact_pair_roundtrip_and_missing_fields():
    config = ResetValidationConfig(True, 3, 0.001, (("left", "right"),))
    assert ResetValidationConfig.from_json(config.as_dict()) == config
    missing = config.as_dict()
    del missing["max_attempts"]
    with pytest.raises(ValueError, match="fields"):
        ResetValidationConfig.from_json(missing)


@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("explicit_disabled", [True, False])
def test_disabled_validation_keeps_original_reset_function_and_parameters(
    fixed, explicit_disabled
):
    config = ResetValidationConfig(False, 32, 0.001, ()) if explicit_disabled else None
    task = cast(
        TaskConfig,
        replace(load_task(BASE), fixed_reset=fixed, reset_validation=config),
    )

    def original():
        """Stand-in for the task's configured reset function; never called."""

    event = _reset_event(original)
    _configure_trajectory_reset(event, task, 17)
    if fixed:
        assert event.func is reset_to_fixed_trajectory_start
        assert event.params == {"command_name": "motion", "seed": 17}
    else:
        assert event.func is original
        assert event.params == {"command_name": "motion"}


def test_collision_event_installed_before_environment_construction(
    monkeypatch, collision_task_path
):
    name = "example_workflows.v2d_box_lift_posttraining.collision_reset"
    module = ModuleType(name)
    marker = type("CollisionValidatedReset", (), {})
    module.CollisionValidatedReset = marker
    monkeypatch.setitem(sys.modules, name, module)
    event = _reset_event(None)
    task = load_task(collision_task_path)
    _configure_trajectory_reset(event, task, 17)
    assert event.func is marker
    assert event.params == {
        "command_name": "motion",
        "seed": 17,
        "fixed_reset": True,
        "reset_validation": task.reset_validation,
    }


def test_launch_records_validation_and_rejects_training_evaluation_mismatch(
    tmp_path, collision_task_path
):
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path),
            "--evaluation-mode",
            "enabled",
            "--train-task-config",
            str(collision_task_path),
            "--eval-task-config",
            str(collision_task_path),
        ]
    )
    plan = resolve_plan(args)
    assert plan["resolved_settings"]["+simulator.reset_validation"]["enabled"] is True
    assert plan["task"]["fixed_reset"] is True
    args.eval_task_config = BASE
    with pytest.raises(ValueError, match="may differ only"):
        resolve_plan(args)
