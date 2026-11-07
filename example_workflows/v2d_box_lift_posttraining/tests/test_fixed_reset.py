# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed physical resets preserve per-environment diversity and surrounding RNG."""

import pytest
import torch

from example_workflows.v2d_box_lift_posttraining.fixed_reset import repeat_reset_draws


def test_fixed_draws_repeat_across_masked_and_reordered_resets():
    poses = torch.empty(4, 8)

    def reset(ids):
        poses[ids] = torch.rand(len(ids), 8)

    before = torch.get_rng_state().clone()
    repeat_reset_draws(reset, torch.arange(4), seed=17, device=torch.device("cpu"))
    initial = poses.clone()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(initial[0], initial[1])

    torch.rand(71)
    before = torch.get_rng_state().clone()
    poses[[1, 3]] = -1
    repeat_reset_draws(reset, torch.tensor([3, 1]), seed=17, device=torch.device("cpu"))
    assert torch.equal(poses, initial)
    assert torch.equal(before, torch.get_rng_state())

    repeat_reset_draws(reset, torch.arange(4), seed=29, device=torch.device("cpu"))
    assert not torch.equal(poses, initial)


def test_fixed_draws_restore_rng_after_reset_failure():
    def reset(ids):
        torch.rand(8)
        raise RuntimeError("reset failed")

    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="reset failed"):
        repeat_reset_draws(
            reset, torch.tensor([0]), seed=17, device=torch.device("cpu")
        )
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.gpu
def test_fixed_cuda_draws_restore_cpu_and_device_rng():
    device = torch.device("cuda:0")
    poses = torch.empty(2, 8, device=device)

    def reset(ids):
        torch.rand(8)
        poses[ids] = torch.rand(len(ids), 8, device=device)

    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone()
    ids = torch.arange(2, device=device)
    repeat_reset_draws(reset, ids, seed=31, device=device)
    initial = poses.clone()
    repeat_reset_draws(reset, ids.flip(0), seed=31, device=device)
    assert torch.equal(poses, initial)
    assert not torch.equal(poses[0], poses[1])
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))


def test_fixed_task_only_changes_reset_behavior():
    from dataclasses import replace
    from pathlib import Path

    from example_workflows.v2d_box_lift_posttraining.config import load_task

    tasks = Path(__file__).resolve().parents[1] / "configs/tasks"
    fixed = load_task(tasks / "tissue_box_lift_hold_xy_1cm_550_fixed.json")
    normal = replace(fixed, fixed_reset=False)
    assert normal.fixed_reset is False
    assert fixed.fixed_reset is True
    assert fixed.sha256 != normal.sha256
    with pytest.raises(TypeError, match="fixed_reset must be a boolean"):
        replace(normal, fixed_reset="false")


@pytest.mark.parametrize("evaluation_mode", ["enabled", "disabled"])
def test_launch_logs_fixed_reset_and_allows_independent_evaluation(
    tmp_path, evaluation_mode
):
    import json
    from pathlib import Path

    from example_workflows.v2d_box_lift_posttraining.launch import parser, resolve_plan

    tasks = Path(__file__).resolve().parents[1] / "configs/tasks"
    evaluation_config = json.loads(
        (tasks / "tissue_box_lift_hold_xy_1cm_550_fixed.json").read_text()
    )
    evaluation_config["fixed_reset"] = False
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation_config))
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path),
            "--evaluation-mode",
            evaluation_mode,
            "--train-task-config",
            str(tasks / "tissue_box_lift_hold_xy_1cm_550_fixed.json"),
            "--eval-task-config",
            str(evaluation_path),
        ]
    )
    plan = resolve_plan(args)
    assert plan["resolved_settings"]["+simulator.training_fixed_reset"] is True
    expected = False if evaluation_mode == "enabled" else None
    assert plan["resolved_settings"]["+simulator.evaluation_fixed_reset"] is expected
    for key in ("simulator.training_fixed_reset", "+simulator.training_fixed_reset"):
        args.override = [key + "=false"]
        with pytest.raises(ValueError, match="pinned workflow contract"):
            resolve_plan(args)
