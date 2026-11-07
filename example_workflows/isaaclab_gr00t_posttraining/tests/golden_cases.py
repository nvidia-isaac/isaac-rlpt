# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Isaac Lab recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``).

The ``isaac_rlpt`` module composes its overrides in Python (``isaac_rlpt/cli_args.py``), so the
recipes are its command lines, turned into overrides exactly as ``train_isaac_rlpt.py`` and
``play_isaac_rlpt.py`` do.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from example_workflows.isaaclab_gr00t_posttraining.isaac_rlpt import cli_args

STACK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
COMMON = ("--model_path", "/ckpt/RLinf-Gr00t-SFT-Stack-cube", "--run_name", "golden")

#: name -> (train | play, command-line flags).
RECIPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "train_dsrl": ("train", ("--task", STACK, "--algorithm", "dsrl_sac")),
    "train_dsrl_8_scenes": (
        "train",
        (
            "--task",
            STACK,
            "--algorithm",
            "dsrl_sac",
            "--num_envs",
            "8",
            "--eval_episodes",
            "25",
        ),
    ),
    "train_rfs": ("train", ("--task", STACK, "--algorithm", "rfs")),
    "train_rfs_pc": (
        "train",
        ("--task", STACK, "--algorithm", "rfs_pc", "--seed", "3"),
    ),
    "train_sac_edit": ("train", ("--task", STACK, "--algorithm", "sac_edit")),
    "train_dsrl_ppo": (
        "train",
        ("--task", STACK, "--algorithm", "dsrl_ppo", "--num_envs", "8"),
    ),
    "train_ppo": ("train", ("--task", STACK, "--algorithm", "ppo", "--num_envs", "16")),
    "train_ppo_8_gpus": (
        "train",
        (
            "--task",
            STACK,
            "--algorithm",
            "ppo",
            "--runtime",
            "rlinf",
            "--learner_gpus",
            "4",
            "--rollout_gpus",
            "4",
            "--num_envs",
            "64",
            "--iterations",
            "500",
        ),
    ),
    "play_base": ("play", ("--task", STACK, "--algorithm", "dsrl_sac")),
    "play_checkpoint": (
        "play",
        (
            "--task",
            STACK,
            "--algorithm",
            "rfs",
            "--checkpoint",
            "/runs/rfs/checkpoints/final",
        ),
    ),
    "play_dsrl_ppo": (
        "play",
        (
            "--task",
            STACK,
            "--algorithm",
            "dsrl_ppo",
            "--num_envs",
            "8",
            "--checkpoint",
            "/runs/dsrl_ppo/checkpoints/iteration_10",
        ),
    ),
    "play_ppo": (
        "play",
        (
            "--task",
            STACK,
            "--algorithm",
            "ppo",
            "--num_envs",
            "16",
            "--eval_episodes",
            "25",
        ),
    ),
}


def overrides(mode: str, flags: tuple[str, ...]) -> list[str]:
    """The overrides ``train_isaac_rlpt.py`` / ``play_isaac_rlpt.py`` compose for ``flags``."""
    parser = argparse.ArgumentParser()
    cli_args.add_isaac_rlpt_args(parser)
    (cli_args.add_train_args if mode == "train" else cli_args.add_play_args)(parser)
    args = parser.parse_args([*flags, *COMMON])
    recipe = cli_args.resolve_recipe(args.task)
    compose = cli_args.train_overrides if mode == "train" else cli_args.play_overrides
    return compose(args, recipe, args.run_name)


def cases(work: Path) -> dict[str, list[str]]:
    del work
    return {name: overrides(mode, flags) for name, (mode, flags) in RECIPES.items()}
