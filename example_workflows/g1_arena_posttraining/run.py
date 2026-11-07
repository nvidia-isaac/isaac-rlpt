# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The G1 Factory DSRL recipe from Python: what ``scripts/train.sh TASK`` runs once its simulators listen.

``train.sh`` boots a training and an evaluation simulator, then starts the learner; this file is
the learner step as a Python call, with the same resolved config. Start the two simulators first
(the launcher's dry run prints their commands as ``launch.cmd.*`` lines), then::

    export ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301
    uv run python -m example_workflows.g1_arena_posttraining.run DrillLift --dry-run
    uv run python -m example_workflows.g1_arena_posttraining.run DrillLift training.num_rollouts=40000
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml

from isaac_rl_posttraining import Env, launch
from isaac_rl_posttraining.algorithms import DSRL, Algorithm, BaseModel

EPISODE_CAPS = (
    Path(__file__).resolve().parents[2]
    / "simulators/g1_arena/configs/simulator/g1_arena.yaml"
)
TASK_PROMPTS = Path(__file__).resolve().parent / "scripts/task_prompts.sh"


def task_prompt(task: str) -> str:
    """The task's language prompt, from ``scripts/task_prompts.sh`` (the launchers' one source)."""
    return subprocess.run(
        ["bash", "-c", 'source "$0" && task_prompt "$1"', str(TASK_PROMPTS), task],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def recipe(
    task: str, base_ckpt: str, method: type[Algorithm] = DSRL
) -> tuple[Algorithm, Env]:
    """The algorithm and environments ``train.sh TASK`` trains, for a task with or without its ``LM`` prefix.

    ``method`` is ``DSRL`` (``train.sh``), ``RFS``, ``SACEdit`` or ``DSRLPPO`` (``train_ppo.sh``).
    """
    task = task if task.startswith("LM") else f"LM{task}"
    cap = yaml.safe_load(EPISODE_CAPS.read_text(encoding="utf-8"))["max_episode_steps"][
        task
    ]
    # No dataset: the checkpoint's baked embodiment is authoritative and the prompt names the task.
    base = BaseModel(
        "gr00t",
        base_ckpt,
        bc_ckpt=base_ckpt,
        dataset_dir=None,
        task_instruction=task_prompt(task),
    )
    # The simulator server runs the task; env.env_name keeps the launcher's default and labels the run.
    return method(base=base), Env("arena/drill_lift", max_episode_steps=cap)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "task", help="DrillLift, PushButton, ... (with or without the LM prefix)"
    )
    parser.add_argument(
        "overrides", nargs="*", help="further overrides, as train.sh takes them"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and start nothing"
    )
    args = parser.parse_args()
    algorithm, env = recipe(args.task, os.environ["POSTTRAIN_BASE_CKPT"])
    run = launch(algorithm, env=env, overrides=args.overrides, dry_run=args.dry_run)
    print("\n".join(run.plan.lines()))
    if not args.dry_run:
        run.loop()


if __name__ == "__main__":
    main()
