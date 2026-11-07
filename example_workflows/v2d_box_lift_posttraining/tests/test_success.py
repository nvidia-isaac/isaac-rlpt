# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from example_workflows.v2d_box_lift_posttraining.capability import evaluation_case_seed
from example_workflows.v2d_box_lift_posttraining.config import load_task
from example_workflows.v2d_box_lift_posttraining.success import CombinedSuccessTracker

TASKS = Path(__file__).resolve().parents[1] / "configs" / "tasks"


@dataclass
class _LiftResult:
    success: bool
    lift_hold_success: bool
    max_lift_m: float = 0.11
    max_hold_steps: int = 20
    sample_count: int = 550


class _LiftTracker:
    def __init__(self, results: list[_LiftResult]) -> None:
        self.results = results
        self.samples: list[np.ndarray] = []
        self.next_initial: list[tuple[int, float]] = []

    def update(self, object_z: np.ndarray) -> None:
        self.samples.append(object_z.copy())

    def complete(self, env_index: int, *, next_initial_z: float) -> _LiftResult:
        self.next_initial.append((env_index, next_initial_z))
        return self.results[env_index]


def _record(*heights: float) -> dict[str, np.ndarray]:
    positions: NDArray[np.float64] = np.zeros((len(heights), 3), dtype=np.float64)
    positions[:, 2] = heights
    return {"object_position_e": positions}


def _tracker(initial: tuple[float, ...], results: list[_LiftResult]):
    task = load_task(TASKS / "tissue_box_lift_hold_xy_1cm_550_fixed.json")
    lift = _LiftTracker(results)
    tracker = CombinedSuccessTracker(
        _record(*initial),
        object_index=0,
        definition=task.success_definition,
        lift_tracker=lift,
    )
    return tracker, lift


def test_task_contract_is_joint_550_step_fixed_reset_with_v2d_evaluator() -> None:
    task = load_task(TASKS / "tissue_box_lift_hold_xy_1cm_550_fixed.json")
    assert (task.episode_horizon, task.execution_length) == (550, 4)
    assert task.fixed_reset is True
    assert task.embodiment == "vega_sharpa_joint"
    assert task.reset_perturbation.object_xy_noise_m == pytest.approx(0.01)
    assert task.success_definition.evaluator == "lift_hold"
    assert task.success_definition.sha256 == (
        "4979ea4785fec56c8076eb6a48aab3f9f66356c06fd6b35d5849ca01992cb31e"
    )


@pytest.mark.parametrize(
    ("success", "lift_success", "final_z", "reason"),
    [
        (False, False, 0.80, "lift"),
        (True, True, 0.80, None),
        (True, True, 0.79, None),
        (True, True, 0.70, None),
        (False, True, 0.80, "final_height"),
    ],
)
def test_upstream_success_is_not_overridden_by_terminal_height(
    success: bool, lift_success: bool, final_z: float, reason: str | None
) -> None:
    tracker, lift = _tracker((0.80,), [_LiftResult(success, lift_success)])
    tracker.sample_pre_action(_record(0.85), np.asarray([True]))
    tracker.update_post_action(_record(final_z))
    result = tracker.complete(0, next_initial_record=_record(1.25))
    assert result.success is success
    assert result.combined_success is success
    assert result.lift_hold_success is lift_success
    assert result.final_height_ok == (final_z >= 0.79)
    assert result.failure_reason == reason
    assert lift.samples[0].tolist() == [0.85]
    assert lift.next_initial == [(0, 1.25)]


def test_terminal_capture_is_not_replaced_by_autoreset_state() -> None:
    tracker, _ = _tracker((0.80,), [_LiftResult(False, True)])
    tracker.update_post_action(_record(0.70))
    result = tracker.complete(0, next_initial_record=_record(1.40))
    assert result.final_object_z_m == pytest.approx(0.70)
    assert result.initial_object_z_m == pytest.approx(0.80)
    assert not result.success


def test_parallel_environments_keep_independent_initial_and_final_heights() -> None:
    tracker, _ = _tracker(
        (0.50, 1.00), [_LiftResult(True, True), _LiftResult(False, True)]
    )
    tracker.update_post_action(_record(0.50, 0.98))
    first = tracker.complete(0, next_initial_record=_record(1.50, 2.00))
    second = tracker.complete(1, next_initial_record=_record(1.50, 2.00))
    assert first.success
    assert not second.success
    assert (first.initial_object_z_m, second.initial_object_z_m) == (0.50, 1.00)


def test_evaluation_case_streams_are_fixed_and_distinct() -> None:
    first = [evaluation_case_seed(10000, 0, index) for index in range(32)]
    repeated = [evaluation_case_seed(10000, 0, index) for index in range(32)]
    second_round = [evaluation_case_seed(10000, 1, index) for index in range(32)]
    assert first == repeated
    assert len(set(first)) == 32
    assert not set(first) & set(second_round)
