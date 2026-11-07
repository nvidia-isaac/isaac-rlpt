# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The V2D box-lift recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``).

The launcher writes its learner command in Python (``launch.resolve_plan``); the recipes are its
command lines, and the overrides are what that command passes to ``scripts/train.py``.
"""

from __future__ import annotations

from pathlib import Path

from example_workflows.v2d_box_lift_posttraining import launch

#: name -> (launcher flags, which learner command of the plan).
RECIPES: dict[str, tuple[tuple[str, ...], str]] = {
    "train": ((), "learner"),
    "train_with_evaluation": (("--evaluation-mode", "enabled"), "learner"),
    "train_with_evaluation_native_baseline": (
        ("--evaluation-mode", "enabled"),
        "native_baseline_learner",
    ),
    "eval_native": (
        ("--evaluation-mode", "enabled", "--training-steps", "0", "--native"),
        "learner",
    ),
    "eval_steered": (
        (
            "--evaluation-mode",
            "enabled",
            "--training-steps",
            "0",
            "--steering-checkpoint",
            "/runs/v2d/checkpoints/final",
        ),
        "learner",
    ),
}


def overrides(flags: tuple[str, ...], learner: str, work: Path) -> list[str]:
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
            str(work / "output"),
            *flags,
        ]
    )
    plan = launch.resolve_plan(args)
    command = (
        plan["learner"] if learner == "learner" else plan["native_baseline_learner"]
    )
    assert command is not None, f"this launch has no {learner}"
    return list(command[2:])  # after the interpreter and scripts/train.py


def cases(work: Path) -> dict[str, list[str]]:
    return {
        name: overrides(flags, learner, work)
        for name, (flags, learner) in RECIPES.items()
    }
