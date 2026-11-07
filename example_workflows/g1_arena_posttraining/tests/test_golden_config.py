# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: every G1 launcher recipe, and its Python form, resolves to the pinned config."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden
from example_workflows.g1_arena_posttraining.run import recipe
from example_workflows.g1_arena_posttraining.tests.golden_cases import base_checkpoint
from isaac_rl_posttraining.algorithms import DSRLPPO
from isaac_rl_posttraining.train import launch_overrides

WORKFLOW = "g1_arena_posttraining"


def test_every_launcher_recipe_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow(WORKFLOW, tmp_path)


def test_the_python_recipe_is_the_launcher_recipe(tmp_path: Path) -> None:
    algorithm, env = recipe("DrillLift", str(base_checkpoint(tmp_path)))
    overrides = launch_overrides(
        algorithm, env, overrides=["logging.wandb_name=golden"]
    )
    golden.check_same_config(
        WORKFLOW, "train_dsrl", overrides, {str(tmp_path): "<work>"}
    )


def test_dsrl_ppo_from_python_is_the_launcher_recipe(tmp_path: Path) -> None:
    """The dsrl_ppo recipe lives in its algorithm config, not in the launcher (review finding entry F14)."""
    algorithm, env = recipe("DrillLift", str(base_checkpoint(tmp_path)), DSRLPPO)
    overrides = launch_overrides(
        algorithm, env, overrides=["logging.wandb_name=golden"]
    )
    golden.check_same_config(
        WORKFLOW, "train_dsrl_ppo", overrides, {str(tmp_path): "<work>"}
    )
