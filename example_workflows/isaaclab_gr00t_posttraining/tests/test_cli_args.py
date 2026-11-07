# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The isaac_rlpt Isaac Lab module composes the documented launch overrides per algorithm."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from isaac_rl_posttraining.runners.flow_sde_ppo_launch import build_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = Path(__file__).resolve().parents[1] / "isaac_rlpt" / "cli_args.py"
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
TROCAR_TASK = "Isaac-Assemble-Trocar-G129-Dex3-v0"
FRANKA_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = (
        module  # dataclasses resolve annotations through sys.modules
    )
    spec.loader.exec_module(module)
    return module


def _cli():
    return _module(CLI_PATH, "isaac_rlpt_cli_args_under_test")


def _args(**kw):
    """A parsed namespace with the train and play flags at their defaults."""
    base = {
        "task": TROCAR_TASK,
        "list_tasks": False,
        "model_path": "/ckpt/Assemble_Trocar",
        "algorithm": "dsrl_sac",
        "seed": 3,
        "eval_episodes": 20,
        "denoising_steps": None,
        "run_name": None,
        "wandb_offline": False,
        "dry_run": False,
        "runtime": None,
        "learner_gpus": None,
        "rollout_gpus": None,
        "iterations": None,
        "num_steps": 40000,
        "eval_freq": 5000,
        "save_interval": 5000,
        "eval_num_envs": None,
        "skip_first_eval": False,
        "num_envs": 1,
        "checkpoint": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _compose(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.configs.normalize import normalize_train_config

    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        # Normalized as every run is (deprecated keys land on their replacements).
        return normalize_train_config(
            compose(config_name=config_name, overrides=overrides)
        )


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("dsrl_sac", {"algorithm=dsrl_sac", "training.chunk_size=16"}),
        ("rfs", {"algorithm=rfs", "training.chunk_size=16"}),
        (
            "rfs_pc",
            {
                "algorithm=rfs",
                "rfs.proposal_conditioned_residual=true",
                "rfs.reference_action_conditioning=true",
            },
        ),
        ("sac_edit", {"algorithm=sac_edit"}),
    ],
)
def test_algorithm_overrides(algorithm, expected):
    cli = _cli()
    got = set(cli.algorithm_overrides(algorithm, cli.resolve_recipe(TROCAR_TASK)))
    assert expected <= got, got
    assert not any(o.startswith("alg.online_alg_name") for o in got), (
        "the algorithm group is the only selector"
    )


def test_train_overrides_carry_task_base_model_and_protocol():
    cli = _cli()
    args = _args(algorithm="rfs", denoising_steps=4, wandb_offline=True)
    got = cli.train_overrides(args, cli.resolve_recipe(args.task), "run1")
    for needle in [
        "simulator=isaaclab_trocar",
        f"env.env_name=isaaclab/{TROCAR_TASK}",
        "env.max_episode_steps=1000",
        "base_model=gr00t",
        "base_model.ckpt=/ckpt/Assemble_Trocar",
        "base_model.embodiment_tag=new_embodiment",
        "base_model.dataset_dir=null",
        "base_model.dataset_adapter=none",
        "base_model.task_instruction='assemble trocar from tray'",
        "base_model.denoising_steps=4",
        "dsrl.noise_dim=32",
        "dsrl.action_exec_len=8",
        "training.num_envs=1",
        "training.seed=3",
        "training.num_rollouts=40000",
        "eval.eval_freq=5000",
        "eval.eval_num_episodes=20",
        "training.checkpoint_retention=best_last_final",
        "logging.wandb_offline=true",
        "logging.wandb_name=run1",
    ]:
        assert needle in got, needle
    assert any(
        o.startswith("base_model.data_config=") and o.endswith(":IsaacLabDataConfig")
        for o in got
    )


def test_steering_cadence_flags_default_when_omitted():
    cli = _cli()
    args = _args(num_steps=None, eval_freq=None, save_interval=None)
    got = cli.train_overrides(args, cli.resolve_recipe(args.task), "r")
    assert {
        "training.num_rollouts=40000",
        "eval.eval_freq=5000",
        "training.save_interval=5000",
    } <= set(got)


def test_play_overrides_base_vs_checkpoint():
    cli = _cli()
    recipe = cli.resolve_recipe(TROCAR_TASK)
    base = cli.play_overrides(_args(), recipe, "p")
    assert (
        "eval.eval_base=true" in base
        and "training.num_rollouts=0" in base
        and not any("load_dir" in o for o in base)
    )
    trained = cli.play_overrides(
        _args(checkpoint="/runs/x/checkpoints/5000"), recipe, "p"
    )
    assert (
        "training.load_dir=/runs/x/checkpoints/5000" in trained
        and "eval.eval_base=true" not in trained
    )
    # The guide passes the run's relative checkpoint path; the learner needs it absolute.
    relative = cli.play_overrides(
        _args(checkpoint="logs/isaac_rlpt/run/checkpoints/5000"), recipe, "p"
    )
    expected = Path("logs/isaac_rlpt/run/checkpoints/5000").resolve()
    assert f"training.load_dir={expected}" in relative


def test_unknown_task_and_algorithm_are_loud():
    cli = _cli()
    with pytest.raises(SystemExit, match="no launch recipe"):
        cli.resolve_recipe("Isaac-Nope-v0")
    with pytest.raises(SystemExit, match="unknown"):
        cli.algorithm_overrides("sac", cli.resolve_recipe(TROCAR_TASK))
    with pytest.raises(SystemExit, match="--task is required"):
        cli.require_task_and_model(_args(task=None))
    with pytest.raises(SystemExit, match="--model_path is required"):
        cli.require_task_and_model(_args(model_path=None))


def test_franka_stack_recipe_carries_the_rlinf_protocol():
    cli = _cli()
    args = _args(task=FRANKA_TASK, model_path="/ckpt/RLinf-Gr00t-SFT-Stack-cube")
    recipe = cli.resolve_recipe(FRANKA_TASK)
    assert (
        recipe.checkpoint == "RLinf/RLinf-Gr00t-SFT-Stack-cube"
        and recipe.embodiment_tag == "isaaclab_franka"
    )
    got = cli.train_overrides(args, recipe, "run2")
    for needle in [
        "simulator=isaaclab_franka_stack",
        f"env.env_name=isaaclab/{FRANKA_TASK}",
        "env.max_episode_steps=448",
        "base_model.embodiment_tag=isaaclab_franka",
        "base_model.data_config=simulators.isaaclab.gr00t_data_configs:IsaacLabFrankaDataConfig",
        "base_model.task_instruction='Stack the red block on the blue block, then stack the green block on the red block.'",
        "dsrl.noise_dim=32",
        "dsrl.action_exec_len=16",
    ]:
        assert needle in got, needle


@pytest.mark.parametrize(
    "task,steps,exec_len,tag,config_suffix",
    [
        (TROCAR_TASK, 1000, 8, "new_embodiment", ":IsaacLabDataConfig"),
        (FRANKA_TASK, 448, 16, "isaaclab_franka", ":IsaacLabFrankaDataConfig"),
    ],
)
@pytest.mark.parametrize("algorithm", ["dsrl_sac", "rfs", "rfs_pc", "sac_edit"])
def test_full_train_override_set_composes(
    algorithm, task, steps, exec_len, tag, config_suffix, tmp_path
):
    """Every override the Isaac Lab module emits resolves against the library's train config."""
    cli = _cli()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"model_type": "gr00t_n1_5"}')
    args = _args(task=task, algorithm=algorithm, model_path=str(ckpt))
    overrides = cli.train_overrides(args, cli.resolve_recipe(args.task), "compose-test")
    cfg = _compose("train", overrides)
    assert cfg.simulator.provider == "isaaclab"
    assert cfg.env.env_name == f"isaaclab/{task}"
    assert cfg.env.max_episode_steps == steps
    assert cfg.base_model.ckpt == str(ckpt) and cfg.base_model.data_config.endswith(
        config_suffix
    )
    assert cfg.base_model.embodiment_tag == tag
    assert cfg.dsrl.noise_dim == 32 and cfg.dsrl.action_exec_len == exec_len
    expected_name = {
        "dsrl_sac": "dsrl_sac",
        "rfs": "rfs",
        "rfs_pc": "rfs",
        "sac_edit": "sac_edit",
    }[algorithm]
    assert cfg.algorithm.name == expected_name
    assert cfg.training.chunk_size == (None if algorithm == "sac_edit" else 16)


def test_num_envs_reaches_the_learner_for_training_only():
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    one = cli.train_overrides(_args(task=FRANKA_TASK), recipe, "r")
    assert "training.num_envs=1" in one
    assert not any(o.startswith("training.rollout_count_by") for o in one)
    eight = cli.train_overrides(_args(task=FRANKA_TASK, num_envs=8), recipe, "r")
    assert "training.num_envs=8" in eight
    # Eight scenes are one batched simulation, which plays whole episodes.
    assert "training.rollout_count_by=episode" in eight
    play = cli.play_overrides(_args(task=FRANKA_TASK), recipe, "p")
    assert "training.num_envs=1" in play
    with pytest.raises(SystemExit, match="one scene"):
        cli.play_overrides(_args(task=FRANKA_TASK, num_envs=8), recipe, "p")


def test_sac_edit_collects_from_several_scenes():
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    overrides = cli.train_overrides(
        _args(task=FRANKA_TASK, algorithm="sac_edit", num_envs=8), recipe, "r"
    )
    assert "training.num_envs=8" in overrides
    assert "training.rollout_count_by=episode" in overrides


def test_skip_first_eval_flag():
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    assert "eval.eval_on_first_step=true" in cli.train_overrides(
        _args(task=FRANKA_TASK), recipe, "r"
    )
    assert "eval.eval_on_first_step=false" in cli.train_overrides(
        _args(task=FRANKA_TASK, skip_first_eval=True), recipe, "r"
    )


# --- --algorithm ppo -------------------------------------------------------------------------


def _ppo_args(**kw):
    base = {
        "task": FRANKA_TASK,
        "model_path": "/ckpt/RLinf-Gr00t-SFT-Stack-cube",
        "algorithm": "ppo",
        "num_steps": None,
        "eval_freq": None,
        "save_interval": None,
    }
    base.update(kw)
    return _args(**base)


def test_ppo_train_overrides_default_to_one_gpu_and_compose():
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    got = cli.train_overrides(_ppo_args(num_envs=16, eval_episodes=20), recipe, "ppo1")
    for needle in [
        "algorithm=ppo",
        "runtime=native",
        "placement.learner.ranks=1",
        "placement.envs.workers=1",
        "training.num_envs=16",
        "ppo.eval_episodes_per_environment=2",
        f"env.env_name=isaaclab/{FRANKA_TASK}",
        "env.max_episode_steps=448",
        "base_model.ckpt=/ckpt/RLinf-Gr00t-SFT-Stack-cube",
        "logging.wandb_name=ppo1",
    ]:
        assert needle in got, needle
    assert not any(
        o.split("=")[0].split(".")[0] in {"dsrl", "rfs", "eval"} for o in got
    ), got
    assert not any(o.startswith("ppo.iterations") for o in got)
    cfg = _compose("train", got)
    assert (cfg.runtime.name, cfg.placement.learner.ranks, cfg.training.num_envs) == (
        "native",
        1,
        16,
    )
    assert cfg.ppo.global_batch_size == 512 and cfg.base_model.denoising_steps == 4
    plan = build_plan(cfg)
    assert (plan.envs_per_worker, plan.num_rollout_workers, plan.max_episode_steps) == (
        16,
        1,
        448,
    )


def test_dsrl_ppo_is_an_algorithm_choice_with_the_steering_geometry():
    """``--algorithm dsrl_ppo``: DSRL's latent geometry, PPO's placement and evaluation, no eval.* keys."""
    cli = _cli()
    assert "dsrl_ppo" in cli.ALGORITHM_CHOICES
    recipe = cli.resolve_recipe(FRANKA_TASK)
    args = _ppo_args(algorithm="dsrl_ppo", num_envs=8, eval_episodes=20)
    got = cli.train_overrides(args, recipe, "dsrl_ppo1")
    for needle in [
        "algorithm=dsrl_ppo",
        "base_model=gr00t",
        "dsrl.noise_dim=32",
        "dsrl.action_exec_len=16",
        "training.chunk_size=16",
        "runtime=native",
        "placement.learner.ranks=1",
        "placement.envs.workers=1",
        "training.num_envs=8",
        "ppo.eval_episodes_per_environment=3",
    ]:
        assert needle in got, needle
    assert not any(o.split("=")[0].split(".")[0] in {"eval", "rfs"} for o in got), got
    cfg = _compose("train", got)
    assert (
        cfg.algorithm.name,
        cfg.training.chunk_size,
        cfg.placement.learner.ranks,
    ) == (
        "dsrl_ppo",
        16,
        1,
    )
    played = cli.play_overrides(
        _ppo_args(algorithm="dsrl_ppo", num_envs=8, checkpoint="/runs/x/iteration_2"),
        recipe,
        "play1",
    )
    assert "ppo.eval_only=true" in played and "placement.learner.ranks=1" in played
    with pytest.raises(SystemExit, match="--checkpoint"):
        cli.play_overrides(_ppo_args(algorithm="dsrl_ppo", num_envs=8), recipe, "play2")


def test_ppo_rlinf_placement_and_iterations_compose():
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    args = _ppo_args(
        runtime="rlinf",
        learner_gpus=4,
        rollout_gpus=4,
        num_envs=64,
        iterations=50,
        denoising_steps=8,
    )
    got = cli.train_overrides(args, recipe, "ppo8")
    assert {
        "runtime=rlinf",
        "placement.learner.ranks=4",
        "placement.envs.workers=4",
        "training.num_envs=64",
        "ppo.iterations=50",
        "base_model.denoising_steps=8",
        "ppo.eval_episodes_per_environment=1",
    } <= set(got)
    plan = build_plan(_compose("train", got))
    assert (
        plan.envs_per_worker,
        plan.num_learner_ranks,
        plan.ppo.iterations,
        plan.denoising_steps,
    ) == (16, 4, 50, 8)
    assert plan.runtime.rlinf.first_accelerator == 4, (
        "rollout workers start after the learner ranks"
    )


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"num_steps": 1000}, "steering trainer"),
        ({"eval_freq": 100}, "steering trainer"),
        ({"skip_first_eval": True}, "steering algorithms"),
        ({"rollout_gpus": 2}, "--runtime rlinf"),
        ({"learner_gpus": 2}, "--runtime rlinf"),
        (
            {"runtime": "rlinf", "rollout_gpus": 4, "num_envs": 30},
            "multiple of --rollout_gpus",
        ),
        ({"learner_gpus": 0}, "--learner_gpus >= 1"),
    ],
)
def test_ppo_rejects_incoherent_flags_before_launch(kw, match):
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    with pytest.raises(SystemExit, match=match):
        cli.train_overrides(_ppo_args(**kw), recipe, "r")


@pytest.mark.parametrize(
    "kw",
    [
        {"runtime": "rlinf"},
        {"learner_gpus": 2},
        {"rollout_gpus": 2},
        {"iterations": 5},
    ],
)
def test_steering_algorithms_reject_ppo_flags(kw):
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    with pytest.raises(SystemExit, match="apply to --algorithm ppo"):
        cli.train_overrides(_args(task=FRANKA_TASK, **kw), recipe, "r")
    with pytest.raises(SystemExit, match="apply to --algorithm ppo"):
        cli.play_overrides(_args(task=FRANKA_TASK, **kw), recipe, "p")


def test_ppo_play_scores_a_snapshot_without_a_learner(tmp_path):
    cli = _cli()
    recipe = cli.resolve_recipe(FRANKA_TASK)
    base = cli.play_overrides(_ppo_args(num_envs=8, eval_episodes=20), recipe, "p")
    assert {
        "ppo.eval_only=true",
        "placement.learner.ranks=0",
        "training.num_envs=8",
        "ppo.eval_episodes_per_environment=3",
    } <= set(base)
    assert not any(o.startswith("ppo.resume_from") for o in base)
    assert build_plan(_compose("train", base)).learnerless
    checkpoint = tmp_path / "checkpoints" / "iteration_40"
    trained = cli.play_overrides(_ppo_args(checkpoint=str(checkpoint)), recipe, "p")
    assert f"ppo.resume_from={checkpoint.resolve()}" in trained
    plan = build_plan(_compose("train", trained))
    assert plan.learnerless and plan.ppo.resume_from == str(checkpoint.resolve())
    with pytest.raises(SystemExit, match="without a learner"):
        cli.play_overrides(_ppo_args(learner_gpus=1), recipe, "p")


def test_task_table_lists_every_recipe():
    cli = _cli()
    table = cli.format_task_table()
    for task_id, recipe in cli.TASK_RECIPES.items():
        assert (
            task_id in table
            and recipe.checkpoint in table
            and recipe.instruction in table
        )
        assert recipe.data_config in table and str(recipe.max_episode_steps) in table
