# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native evaluation script's pure parts: the task-selection rule, the report, the command line."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EVAL_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval.py"


@pytest.fixture(scope="module")
def eval_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("libero_eval_script", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(module: ModuleType, task_id: int, rate: float, episodes: int = 20):
    return module.TaskResult(
        task_id=task_id,
        task_name=f"task_{task_id}",
        description=f"description {task_id}",
        episodes=episodes,
        successes=round(rate * episodes),
    )


def test_target_is_the_worst_task_inside_the_band(eval_script: ModuleType) -> None:
    results = [
        _result(eval_script, 0, 0.90),
        _result(eval_script, 1, 0.45),
        _result(eval_script, 2, 0.35),
        _result(eval_script, 3, 0.75),
    ]
    pick, note = eval_script.pick_dsrl_target(results)
    assert pick.task_id == 2 and "worst" in note


def test_near_zero_tasks_are_never_picked(eval_script: ModuleType) -> None:
    results = [
        _result(eval_script, 0, 0.0),
        _result(eval_script, 1, 0.55),
        _result(eval_script, 2, 0.95),
    ]
    pick, note = eval_script.pick_dsrl_target(results)
    assert pick.task_id == 1 and "task_0" in note
    none_pick, none_note = eval_script.pick_dsrl_target([_result(eval_script, 0, 0.0)])
    assert none_pick is None and "no viable" in none_note


def test_report_lists_every_task_and_the_pick(eval_script: ModuleType) -> None:
    results = [_result(eval_script, 0, 0.9), _result(eval_script, 1, 0.4)]
    text = eval_script.format_markdown(results, "/ckpt", "libero_10", "20 eps")
    assert "| 0 | description 0 | 18/20 (90%) |" in text
    assert "| 1 | description 1 | 8/20 (40%) |" in text
    assert "Overall: 26/40 episodes; mean per-task SR 65.0%." in text
    assert "DSRL target: task 1" in text


def test_command_line_defaults_are_the_reference_protocol(
    eval_script: ModuleType,
) -> None:
    args = eval_script.build_parser().parse_args(["--checkpoint", "/ckpt/libero_10"])
    assert (args.suite, args.episodes, args.max_episode_steps, args.n_action_steps) == (
        "libero_10",
        20,
        720,
        8,
    )
    assert args.embodiment_tag == "libero_sim" and args.tasks is None
