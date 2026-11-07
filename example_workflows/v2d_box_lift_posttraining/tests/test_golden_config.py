# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: every V2D launch, and its learner launched from Python, resolves to the pinned config."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden
from example_workflows.v2d_box_lift_posttraining import launch
from isaac_rl_posttraining.train import launch_overrides

WORKFLOW = "v2d_box_lift_posttraining"


def test_every_launch_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow(WORKFLOW, tmp_path)


def test_the_python_learner_is_the_launcher_learner(tmp_path: Path) -> None:
    args = launch.parser().parse_args(
        [
            "--v2d-root",
            "/dry-run/v2d",
            "--checkpoint",
            "/dry-run/checkpoint",
            "--motion-file",
            "/dry-run/tissue_box.parquet",
            "--sim-python",
            "/dry-run/isaac-python",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    algorithm, env, overrides = launch.learner_recipe(launch.resolve_plan(args))
    composed = launch_overrides(algorithm, env, overrides=overrides)
    golden.check_same_config(WORKFLOW, "train", composed, {str(tmp_path): "<work>"})
