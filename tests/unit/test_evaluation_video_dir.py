# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation videos land in the run's output directory, next to its checkpoints."""

from __future__ import annotations

import os
from types import SimpleNamespace

from omegaconf import OmegaConf

from isaac_rl_posttraining.runners import off_policy_evaluation


def test_videos_go_to_the_run_output_directory(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        off_policy_evaluation, "resolve_num_rollout_workers", lambda *a, **k: 1
    )
    monkeypatch.setattr(off_policy_evaluation, "environment_spec", lambda *a, **k: None)
    monkeypatch.setattr(
        off_policy_evaluation,
        "resolve_image_keys",
        lambda *a, **k: SimpleNamespace(keys=["image"]),
    )
    monkeypatch.setattr(
        off_policy_evaluation,
        "RolloutEvaluator",
        lambda **kwargs: captured.update(kwargs),
    )
    cfg = OmegaConf.create(
        {
            "eval": {
                "eval_num_episodes": 2,
                "eval_max_steps": 10,
                "reset_seed": None,
                "eval_record_video": True,
                "video_fps": 20,
            }
        }
    )
    run_dir = tmp_path / "run_2026-10-06_00-00-00"

    off_policy_evaluation.open_evaluator(
        cfg,
        SimpleNamespace(evaluation="fresh_workers"),
        SimpleNamespace(open_rollout_group=lambda *a, **k: None),
        None,
        SimpleNamespace(),
        eval_num_envs=1,
        experiment_logger=SimpleNamespace(
            log_dir=str(tmp_path / "logs" / "wandb" / "run")
        ),
        reward_channel="sparse_outcome",
        reward_aggregation="last",
        output_dir=str(run_dir),
    )

    assert captured["video_dir"] == os.path.join(str(run_dir), "evaluation_videos")
