# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: every LIBERO launcher recipe, and its Python form, resolves to the pinned config."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden
from example_workflows.libero_posttraining.run import recipe
from example_workflows.libero_posttraining.tests.golden_cases import ENV
from isaac_rl_posttraining.train import launch_overrides

WORKFLOW = "libero_posttraining"


def test_every_launcher_recipe_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow(WORKFLOW, tmp_path)


def test_the_python_recipe_is_the_launcher_recipe() -> None:
    algorithm, env, overrides = recipe("8", ENV["POSTTRAIN_BASE_CKPT"])
    composed = launch_overrides(
        algorithm, env, overrides=[*overrides, "logging.wandb_name=golden"]
    )
    golden.check_same_config(WORKFLOW, "train_gr00t", composed, {})
