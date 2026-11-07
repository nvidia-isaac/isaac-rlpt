# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launch recipes compose into library plans that resolve without Isaac Lab, GR00T or a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import isaac_rl_posttraining
from example_workflows.isaaclab_gr00t_posttraining.isaac_rlpt import cli_args
from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.runners.build import TrainPlan, train_plan

CONFIG_DIR = Path(isaac_rl_posttraining.__file__).resolve().parent / "configs"
STACK_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
TROCAR_TASK = "Isaac-Assemble-Trocar-G129-Dex3-v0"
#: ``--algorithm`` value -> the library algorithm it selects.
LIBRARY_ALGORITHM = {
    "dsrl_sac": "dsrl_sac",
    "rfs": "rfs",
    "rfs_pc": "rfs",
    "sac_edit": "sac_edit",
}


def _args(**overrides: object) -> argparse.Namespace:
    """The parsed train and play flags at their defaults, on the stack cube."""
    values: dict[str, object] = {
        "task": STACK_TASK,
        "list_tasks": False,
        "model_path": "/path/to/RLinf-Gr00t-SFT-Stack-cube",
        "algorithm": "dsrl_sac",
        "seed": 0,
        "eval_episodes": 25,
        "denoising_steps": None,
        "run_name": None,
        "wandb_offline": False,
        "dry_run": False,
        "runtime": None,
        "learner_gpus": None,
        "rollout_gpus": None,
        "iterations": None,
        "num_steps": None,
        "eval_freq": None,
        "save_interval": None,
        "eval_num_envs": None,
        "skip_first_eval": False,
        "num_envs": 1,
        "checkpoint": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


#: The mock base model in the simulator's action width.
MOCK = (
    "base_model=mock",
    "base_model.action_dim=${simulator.action_dim}",
    "dsrl.noise_dim=${simulator.action_dim}",
)


def _plan(overrides: list[str]) -> TrainPlan:
    """The dry run's work: compose the library config with ``overrides`` and resolve its plan."""
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg: DictConfig = compose(config_name="train", overrides=overrides)
    return train_plan(cfg)


@pytest.mark.parametrize(
    ("algorithm", "task"),
    # Every algorithm on the stack-cube recipe; the task axis only swaps recipe fields, so one algorithm
    # on the trocar recipe covers it.
    [
        *[(algorithm, STACK_TASK) for algorithm in cli_args.STEERING_ALGORITHMS],
        ("dsrl_sac", TROCAR_TASK),
    ],
)
def test_steering_train_and_play_plans_resolve(algorithm: str, task: str) -> None:
    recipe = cli_args.resolve_recipe(task)
    args = _args(task=task, algorithm=algorithm)
    train = cli_args.train_overrides(args, recipe, "train-run")
    play = cli_args.play_overrides(args, recipe, "play-run")
    for overrides in (train, play):
        # The mock base model stands in for the frozen GR00T policy, as a user's trailing override would,
        # in the simulator's action width (the mock steers exactly that many dimensions).
        plan = _plan([*overrides, *MOCK])
        assert plan.simulator == "isaaclab"
        assert plan.algorithm == LIBRARY_ALGORITHM[algorithm]
        assert plan.layout == "1 rollout worker(s) x 1 environment(s)"
    assert f"env.max_episode_steps={recipe.max_episode_steps}" in train
    assert "eval.eval_base=true" in play


def test_eight_scene_steering_run_evaluates_on_all_scenes() -> None:
    recipe = cli_args.resolve_recipe(STACK_TASK)
    overrides = cli_args.train_overrides(_args(num_envs=8), recipe, "eight")
    assert {"training.num_envs=8", "eval.eval_num_envs=8"} <= set(overrides)
    plan = _plan([*overrides, *MOCK])
    # The eight scenes are clones in one simulation, stepped by one batched worker.
    assert plan.layout == "1 rollout worker(s) x 8 environment(s), batched"


def test_eval_episodes_round_up_to_a_multiple_of_the_scenes() -> None:
    """25 episodes over 8 scenes becomes 32; the evaluator gives every scene the same count."""
    recipe = cli_args.resolve_recipe(STACK_TASK)
    assert "eval.eval_num_episodes=32" in cli_args.train_overrides(
        _args(num_envs=8), recipe, "eight"
    )
    assert "eval.eval_num_episodes=25" in cli_args.train_overrides(
        _args(num_envs=1), recipe, "one"
    )


def test_ppo_train_plan_resolves_on_one_gpu() -> None:
    recipe = cli_args.resolve_recipe(STACK_TASK)
    overrides = cli_args.train_overrides(
        _args(algorithm="ppo", num_envs=16, eval_episodes=20), recipe, "ppo-one-gpu"
    )
    assert {"algorithm=ppo", "base_model=gr00t_n15", "runtime=native"} <= set(overrides)
    plan = _plan(overrides)
    assert (plan.algorithm, plan.policy_family, plan.runtime) == (
        "ppo",
        "flow",
        "native",
    )
    assert plan.layout == "1 rollout worker(s) x 16 scene(s), 448-step episodes"
    assert "global batch 512 (rank minibatch 512, micro 32)" in plan.schedule


def test_ppo_rlinf_placement_and_snapshot_evaluation_resolve(tmp_path: Path) -> None:
    recipe = cli_args.resolve_recipe(STACK_TASK)
    train = cli_args.train_overrides(
        _args(
            algorithm="ppo",
            runtime="rlinf",
            learner_gpus=4,
            rollout_gpus=4,
            num_envs=64,
            iterations=50,
        ),
        recipe,
        "ppo-node",
    )
    plan = _plan(train)
    assert plan.layout == "4 rollout worker(s) x 16 scene(s), 448-step episodes"
    assert plan.placement.startswith("4 learner rank(s) on accelerator(s) 0-3")
    assert plan.schedule.startswith("50 iterations")

    checkpoint = tmp_path / "checkpoints" / "iteration_40"
    play = cli_args.play_overrides(
        _args(algorithm="ppo", checkpoint=str(checkpoint), num_envs=8, eval_episodes=8),
        recipe,
        "ppo-play",
    )
    plan = _plan(play)
    assert (
        plan.schedule
        == f"evaluation only: 1 episode(s) per environment of {checkpoint.resolve()}"
    )
    assert plan.placement.startswith("0 learner rank(s)")


@pytest.mark.parametrize(
    "flags,match",
    [
        ({"algorithm": "ppo", "num_steps": 1000}, "steering trainer"),
        ({"algorithm": "ppo", "rollout_gpus": 2}, "--runtime rlinf"),
        (
            {"algorithm": "ppo", "runtime": "rlinf", "rollout_gpus": 4, "num_envs": 30},
            "multiple of --rollout_gpus",
        ),
        ({"algorithm": "dsrl_sac", "iterations": 5}, "apply to --algorithm ppo"),
    ],
)
def test_incoherent_flags_fail_before_launch(
    flags: dict[str, object], match: str
) -> None:
    recipe = cli_args.resolve_recipe(STACK_TASK)
    with pytest.raises(SystemExit, match=match):
        cli_args.train_overrides(_args(**flags), recipe, "rejected")


def test_unknown_task_names_the_supported_ones() -> None:
    with pytest.raises(SystemExit, match="no launch recipe") as excinfo:
        cli_args.resolve_recipe("Isaac-Nope-v0")
    assert STACK_TASK in str(excinfo.value)


def test_task_table_lists_every_recipe() -> None:
    table = cli_args.format_task_table()
    for task_id, recipe in cli_args.TASK_RECIPES.items():
        assert task_id in table
        assert recipe.checkpoint in table and recipe.instruction in table
