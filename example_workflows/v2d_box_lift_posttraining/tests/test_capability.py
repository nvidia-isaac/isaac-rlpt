# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training consumes startup/autoreset states; seeded cases reset explicitly."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from example_workflows.v2d_box_lift_posttraining.capability import (
    NativeBatchStep,
    TissueBoxCapability,
    evaluation_case_seed,
)
from example_workflows.v2d_box_lift_posttraining.config import load_task


def record(value, num_envs=3):
    return {
        "arm_joint_pos": np.full((num_envs, 14), value, dtype=np.float32),
        "finger_joint_pos": np.full((num_envs, 44), value, dtype=np.float32),
        "object_position_e": np.full((num_envs, 3), value, dtype=np.float32),
        **{
            key: np.full((num_envs, 2, 2, 3), value, dtype=np.uint8)
            for key in ("image", "image_right_wrist", "image_left_wrist")
        },
    }


def capability(num_envs=3):
    task = load_task(
        Path(__file__).resolve().parents[1]
        / "configs/tasks/tissue_box_lift_hold_xy_1cm_550_fixed.json"
    )
    tracker = Mock()
    tracker.complete.return_value.success = False
    tracker.complete.return_value.task_metrics.return_value = {"success": False}
    return TissueBoxCapability(
        Mock(),
        tracker,
        record(1, num_envs),
        task=task,
        prompt="lift",
        default_seed=2,
        receipt={},
    )


def test_training_reuses_startup_and_terminal_autoreset_states():
    cap = capability()
    initial = cap.reset(None, None)
    cap.runtime.reset_cases.assert_not_called()
    cap.runtime.reset_envs.assert_not_called()
    assert initial["info"]["case_seed"] == [None] * 3
    np.testing.assert_array_equal(initial["state"], 1)
    cap.runtime.step_batch.return_value = NativeBatchStep(
        record(2),
        record(1),
        np.ones(3, dtype=bool),
        ((), (), ()),
    )
    terminal = cap.step(np.zeros((3, 58)), need_obs=True)
    assert terminal["truncated"].all()
    assert all(info["case_seed"] is None for info in terminal["info"])
    following = cap.reset(None, None)
    cap.runtime.reset_cases.assert_not_called()
    cap.runtime.reset_envs.assert_not_called()
    np.testing.assert_array_equal(following["state"], 2)
    assert following["info"]["round_index"] == 1
    assert (
        following["info"]["initial_state_fingerprint"]
        != initial["info"]["initial_state_fingerprint"]
    )
    assert cap.active.all() and not cap.steps.any()
    assert cap.tracker.complete.call_args.kwargs["next_initial_record"] is cap.record


def test_single_environment_reset_and_step_return_unbatched_observations():
    cap = capability(num_envs=1)
    initial = cap.reset(None, None)
    assert initial["state"].shape == (58,)
    assert initial["video.front"].shape == (2, 2, 3)
    cap.runtime.step_batch.return_value = NativeBatchStep(
        record(2, 1), record(1, 1), np.ones(1, dtype=bool), ((),)
    )
    assert cap.step(np.zeros(58), need_obs=True)["state"].shape == (58,)


def test_explicit_seed_repeats_case_identity_and_unseeded_reset_batches_active_slots():
    cap = capability()
    cap.runtime.reset_cases.return_value = record(3)
    first = cap.reset(90, None)
    second = cap.reset(90, None)
    expected = [(index, evaluation_case_seed(90, 0, index)) for index in range(3)]
    cap.runtime.reset_cases.assert_called_with(expected)
    assert first["info"] == second["info"]
    cap.runtime.step_batch.return_value = NativeBatchStep(
        record(4),
        record(3),
        np.asarray([True, False, False]),
        ((), (), ()),
    )
    cap.step(np.zeros((3, 58)), need_obs=True)
    cap.runtime.reset_envs.return_value = record(5)
    reset = cap.reset(None, [0, 2])
    cap.runtime.reset_envs.assert_called_once_with([2], seed=None)
    assert reset["info"]["case_seed"] == [None, expected[1][1], None]
    assert cap.steps.tolist() == [0, 1, 0]


def test_terminal_reset_diagnostics_follow_the_completed_episode():
    cap = capability()
    cap.reset(None, None)
    cap.runtime.step_batch.return_value = NativeBatchStep(
        record(2),
        record(1),
        np.asarray([True, False, True]),
        ((), (), ()),
        reset_diagnostics={"reset_rejections": np.asarray([2, 4, 0])},
    )
    result = cap.step(np.zeros((3, 58)), need_obs=True)
    assert result["info"][0]["episode_diagnostics"] == {"reset_rejections": 2.0}
    assert "episode_diagnostics" not in result["info"][1]
    assert result["info"][2]["episode_diagnostics"] == {"reset_rejections": 0.0}


def test_startup_reset_rejection_closes_simulator(monkeypatch, tmp_path):
    from example_workflows.v2d_box_lift_posttraining import capability as module

    runtime = Mock()
    runtime.reset_envs.side_effect = RuntimeError("No valid reset")
    monkeypatch.setattr(
        module, "_build_runtime", lambda *args: (runtime, None, "", 0, {})
    )
    with pytest.raises(RuntimeError, match="No valid reset"):
        module.create_capability(tmp_path, "cpu", 2, 2)
    runtime.close.assert_called_once_with()
