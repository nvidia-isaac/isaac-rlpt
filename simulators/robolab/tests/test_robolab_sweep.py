# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task discovery and sweep reporting remain usable without Isaac Sim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulators.robolab.sweep import (
    ResultStatus,
    TaskValidationError,
    discover_tasks,
    read_sweep_results,
    select_dsrl_candidates,
    validate_task_names,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "\n".join(json.dumps(row) for row in rows) + "\n")


def test_discover_tasks_uses_direct_class_definitions_not_filenames(tmp_path: Path):
    task_dir = tmp_path / "tasks"
    _write(
        task_dir / "bagel_on_plate_task.py",
        """\
from abc import abstractmethod
from robolab.core.task import Task as RoboTask

class AbstractPlacementTask(RoboTask):
    @abstractmethod
    def target(self):
        raise NotImplementedError

class BagelsOnPlateTask(RoboTask):
    def target(self):
        return 'plate'

class UnfinishedTask(RoboTask):
    @abstractmethod
    def target(self):
        raise NotImplementedError

class IndirectTask(AbstractPlacementTask):
    pass
""",
    )
    _write(
        task_dir / "banana.py",
        """\
import robolab.core.task as task

class BananaInBowlTask(task.Task):
    def target(self):
        return 'bowl'
""",
    )

    tasks = discover_tasks(task_dir)

    assert [(task.name, task.path.name) for task in tasks] == [
        ("BagelsOnPlateTask", "bagel_on_plate_task.py"),
        ("BananaInBowlTask", "banana.py"),
    ]
    assert validate_task_names(task_dir, ["BagelsOnPlateTask"]) == (tasks[0],)


def test_validate_task_names_rejects_unknown_and_ambiguous_names(tmp_path: Path):
    _write(tmp_path / "one.py", "class SharedTask(Task):\n    pass\n")
    _write(tmp_path / "two.py", "class SharedTask(Task):\n    pass\n")

    with pytest.raises(TaskValidationError) as error:
        validate_task_names(tmp_path, ["MissingTask", "SharedTask"])

    message = str(error.value)
    assert "Unknown task 'MissingTask'" in message
    assert "Ambiguous task 'SharedTask'" in message
    assert "one.py" in message
    assert "two.py" in message


def test_read_sweep_results_preserves_paths_metadata_and_zero_success(tmp_path: Path):
    zero = tmp_path / "zero.jsonl"
    _write_jsonl(
        zero,
        [
            {
                "env_name": "ZeroSuccessTask",
                "success": False,
                "policy": "nvidia/GR00T-N1.7-DROID",
                "open_loop_horizon": 8,
            },
            {
                "env_name": "ZeroSuccessTask",
                "success": False,
                "policy": "nvidia/GR00T-N1.7-DROID",
                "open_loop_horizon": 8,
            },
        ],
    )
    middle = tmp_path / "middle.jsonl"
    _write_jsonl(
        middle,
        [
            {
                "robolab_episode": {
                    "env_name": "MiddleTask",
                    "success": True,
                    "policy": "nvidia/GR00T-N1.7-DROID",
                },
                "config": {"open_loop_horizon": 8, "num_envs": 10},
            },
            {
                "robolab_episode": {
                    "env_name": "MiddleTask",
                    "success": False,
                    "policy": "nvidia/GR00T-N1.7-DROID",
                },
                "config": {"open_loop_horizon": 8, "num_envs": 10},
            },
        ],
    )
    empty = tmp_path / "empty.jsonl"
    _write(empty, "\n")
    invalid = tmp_path / "invalid.jsonl"
    _write(invalid, '{"env_name":"BadTask","success": 1}\n')
    missing = tmp_path / "missing.jsonl"

    report = read_sweep_results([zero, middle, empty, invalid, missing])

    assert [source.status for source in report.sources] == [
        ResultStatus.COMPLETED,
        ResultStatus.COMPLETED,
        ResultStatus.EMPTY,
        ResultStatus.INVALID,
        ResultStatus.MISSING,
    ]
    assert [
        (task.task_name, task.successes, task.episodes, task.success_rate)
        for task in report.tasks
    ] == [
        ("MiddleTask", 1, 2, 0.5),
        ("ZeroSuccessTask", 0, 2, 0.0),
    ]
    middle_result = report.tasks[0]
    assert middle_result.result_paths == (middle.resolve(),)
    assert middle_result.config_metadata == {
        "episode": {"policy": "nvidia/GR00T-N1.7-DROID"},
        "trace": {"config": {"open_loop_horizon": 8, "num_envs": 10}},
    }
    assert select_dsrl_candidates(
        report, min_success_rate=0.25, max_success_rate=0.80
    ) == (middle_result,)


def test_read_sweep_results_does_not_merge_configuration_variants(tmp_path: Path):
    results = tmp_path / "variants.jsonl"
    _write_jsonl(
        results,
        [
            {"env_name": "VariantTask", "success": True, "open_loop_horizon": 4},
            {"env_name": "VariantTask", "success": False, "open_loop_horizon": 8},
        ],
    )

    report = read_sweep_results([results])

    assert [
        (task.episodes, task.success_rate, task.config_metadata)
        for task in report.tasks
    ] == [
        (1, 1.0, {"episode": {"open_loop_horizon": 4}}),
        (1, 0.0, {"episode": {"open_loop_horizon": 8}}),
    ]
