# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: every isaac_rlpt command line, and the Python PPO recipe, resolves to the pinned config."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden
from example_workflows.isaaclab_gr00t_posttraining.run import ppo_recipe
from isaac_rl_posttraining.train import launch_overrides

WORKFLOW = "isaaclab_gr00t_posttraining"


def test_every_command_line_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow(WORKFLOW, tmp_path)


def test_the_python_ppo_recipe_is_the_launcher_recipe() -> None:
    algorithm, env, placement, overrides = ppo_recipe(
        "/ckpt/RLinf-Gr00t-SFT-Stack-cube"
    )
    composed = launch_overrides(
        algorithm,
        env,
        placement=placement,
        overrides=[*overrides, "logging.wandb_name=golden"],
    )
    golden.check_same_config(WORKFLOW, "train_ppo", composed, {})
