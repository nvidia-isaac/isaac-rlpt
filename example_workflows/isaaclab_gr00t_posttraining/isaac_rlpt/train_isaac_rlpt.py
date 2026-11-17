# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""isaac_rlpt training logic for the unified reinforcement learning entrypoint.

Post-trains a supervised-fine-tuned GR00T policy on an Isaac Lab task with
DSRL, RFS, SAC-edit (frozen policy) or PPO (RLinf's fine-tuning recipe) from
``isaac_rl_posttraining``::

    uv run isaaclab train --rl_library isaac_rlpt \\
        --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \\
        --model_path /path/to/RLinf-Gr00t-SFT-Stack-cube --algorithm dsrl_sac
    uv run isaaclab train --rl_library isaac_rlpt --list_tasks
    uv run isaaclab train --rl_library isaac_rlpt --task LMDrillPnP --algorithm ppo \\
        --model_path nvidia/G1-Factory-GR00T-1.7  # G1 Factory (g1_arena_posttraining)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))  # the pre-3.0.0 launcher loads this file by path
from _compat import add_isaaclab_launcher_args, import_local_module

CLI_ARGS = import_local_module(
    "isaaclab_isaac_rlpt_cli_args", MODULE_DIR / "cli_args.py"
)
LAUNCH = import_local_module("isaaclab_isaac_rlpt_launch", MODULE_DIR / "_launch.py")
G1_FACTORY = import_local_module(
    "isaaclab_isaac_rlpt_g1_factory", MODULE_DIR / "_g1_factory.py"
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="RL post-training of a GR00T policy with isaac_rlpt."
    )
    CLI_ARGS.add_isaac_rlpt_args(parser)
    CLI_ARGS.add_train_args(parser)
    add_isaaclab_launcher_args(parser)
    args_cli, hydra_overrides = parser.parse_known_args(argv)
    return args_cli, hydra_overrides


def run(argv: list[str]) -> None:
    """Train with isaac_rlpt (called by Isaac Lab's ``train`` entrypoint)."""
    args_cli, extra = _parse_args(argv)
    if args_cli.list_tasks:
        print(CLI_ARGS.format_task_table())
        print(G1_FACTORY.format_task_table())
        return
    CLI_ARGS.require_task_and_model(args_cli)
    if args_cli.task in G1_FACTORY.TASKS:
        # The G1 Factory launchers boot the simulator servers and run the learner.
        G1_FACTORY.run(args_cli, extra, "train", LAUNCH.output_root())
        return
    recipe = CLI_ARGS.resolve_recipe(args_cli.task)
    run_name = LAUNCH.default_run_name(args_cli, "train")
    overrides = CLI_ARGS.train_overrides(args_cli, recipe, run_name) + extra
    LAUNCH.launch(args_cli, overrides, run_name)


if __name__ == "__main__":
    import sys

    run(sys.argv[1:])
