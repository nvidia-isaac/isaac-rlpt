# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO fine-tuning on an Isaac Lab task from Python: the run ``uv run isaaclab train --rl_library isaac_rlpt
--algorithm ppo`` launches, built from the same task recipe (``isaac_rlpt/cli_args.py``).

PPO's rollout workers boot their own Isaac Lab simulations, so this runs under the interpreter
that hosts Isaac Sim without Isaac Lab's launcher (the steering algorithms step the simulator in
the launching process and go through ``uv run isaaclab``). From this folder::

    PYTHONPATH=../.. uv run python -m example_workflows.isaaclab_gr00t_posttraining.run /path/to/RLinf-Gr00t-SFT-Stack-cube --dry-run
"""

from __future__ import annotations

import argparse

from example_workflows.isaaclab_gr00t_posttraining.isaac_rlpt.cli_args import (
    resolve_recipe,
)
from isaac_rl_posttraining import Env, launch
from isaac_rl_posttraining.algorithms import PPO, BaseModel
from isaac_rl_posttraining.runtimes.placement import Envs, Learner, Placement

STACK_CUBE = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"


def ppo_recipe(
    model_path: str, task: str = STACK_CUBE, num_envs: int = 16, eval_episodes: int = 20
) -> tuple[PPO, Env, Placement, list[str]]:
    """``--algorithm ppo --num_envs <num_envs>`` on one GPU: one learner rank, one rollout worker of every scene."""
    recipe = resolve_recipe(task)
    base = BaseModel(
        "gr00t_n15",
        model_path,
        embodiment_tag=recipe.embodiment_tag,
        data_config=recipe.data_config,
        dataset_dir=None,
        dataset_adapter="none",
        task_instruction=recipe.instruction,
    )
    # Every scene evaluates in parallel, so the episode count rounds up to a whole number per scene.
    algorithm = PPO(
        base=base, eval_episodes_per_environment=-(-eval_episodes // num_envs)
    )
    env = Env(
        f"isaaclab/{task}",
        simulator=recipe.simulator,
        num_envs=num_envs,
        max_episode_steps=recipe.max_episode_steps,
    )
    placement = Placement(envs=Envs(workers=1), learner=Learner(ranks=1))
    return algorithm, env, placement, ["logging.wandb_project=isaac_rlpt"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_path", help="the SFT'd GR00T N1.5 checkpoint directory")
    parser.add_argument(
        "overrides", nargs="*", help="further overrides (ppo.iterations=50, ...)"
    )
    parser.add_argument("--task", default=STACK_CUBE)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and start nothing"
    )
    args = parser.parse_args()
    algorithm, env, placement, overrides = ppo_recipe(
        args.model_path, args.task, args.num_envs
    )
    run = launch(
        algorithm,
        env=env,
        placement=placement,
        overrides=[*overrides, *args.overrides],
        dry_run=args.dry_run,
    )
    print("\n".join(run.plan.lines()))
    if not args.dry_run:
        run.loop()


if __name__ == "__main__":
    main()
