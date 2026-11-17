# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T flow-SDE PPO (``algorithm=ppo``): RLinf recipe pins, plan validation, and a dry run free of Isaac Sim and GR00T."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.configs.normalize import normalize_train_config
from isaac_rl_posttraining.policies.trainable import FlowSDEConfig, TrainablePolicySpec
from isaac_rl_posttraining.runners.build import TrainPlan, algorithm_family
from isaac_rl_posttraining.runners.flow_sde_ppo_launch import build_plan, trainer_config
from isaac_rl_posttraining.runners.ppo_launch import (
    on_policy_config_from,
    ppo_config_from,
    rank_minibatch_size,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
TRAIN_ENTRY = REPO_ROOT / "scripts" / "train.py"
STACK_CUBE = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
#: Packages a dry run must never import (the plan is printed before any model or simulator loads).
HEAVY_PACKAGES = {
    "gr00t",
    "isaacsim",
    "isaaclab",
    "omni",
    "carb",
    "pxr",
    "rlinf",
    "ray",
}

# RLinf's Isaac Lab stack-cube recipe as algorithm=ppo composes it. Changing a value is a
# deliberate, reviewed decision: update the pin in the same commit and say why.
RLINF_RECIPE = {
    "ppo.iterations": 1000,
    "ppo.episodes_per_environment": 2,
    "ppo.eval_episodes_per_environment": 1,
    "ppo.eval_every": 10,
    "ppo.checkpoint_every": 20,
    "ppo.checkpoint_retention": "best_last_final",
    "placement.learner.ranks": 4,
    "ppo.learning_rate": 1e-5,
    "ppo.value_learning_rate": 2e-5,
    "ppo.optimizer": "adamw",
    "ppo.adam_betas": [0.9, 0.95],
    "ppo.optimizer_eps": 1e-8,
    "ppo.weight_decay": 0.01,
    "ppo.clip_ratio": 0.2,
    "ppo.value_clip": 0.2,
    "ppo.value_loss": "huber",
    "ppo.huber_delta": 10.0,
    "ppo.value_loss_coef": 1.0,
    "ppo.entropy_coef": 0.0,
    "ppo.update_epochs": 3,
    "ppo.global_batch_size": 512,
    "ppo.micro_batch_size": 32,
    "ppo.max_grad_norm": 1.0,
    "ppo.target_kl": None,
    "ppo.gamma": 0.99,
    "ppo.gae_lambda": 0.95,
    "ppo.discount_per": "decision",
    "ppo.bootstrap_at_time_limit": False,
    "ppo.eval_seed": 100_000,
    "ppo.noise_level": 0.5,
    "ppo.frozen_submodules": None,
    "ppo.cache_features": False,
    "schedule.kind": "sync",
    "ppo.isolate_evaluation_workers": False,
    "ppo.sample_evaluation_actions": False,
    "base_model.denoising_steps": 4,
    "base_model.action_horizon": 16,
    "base_model.embodiment_tag": "isaaclab_franka",
    "base_model.data_config": "simulators.isaaclab.gr00t_data_configs:IsaacLabFrankaDataConfig",
    "env.env_name": f"isaaclab/{STACK_CUBE}",
    "env.max_episode_steps": 448,
    "training.num_envs": 64,
    "training.seed": 0,
    "runtime.name": "rlinf",
    "placement.envs.workers": 4,
    "runtime.rlinf.learner_mixed_precision": "bf16",
    "runtime.rlinf.first_accelerator": 4,
    "runtime.rlinf.learner_first_accelerator": 0,
    "placement.policy.mode": "local",
    "runtime.rlinf.shutdown_ray_on_close": True,
    "runtime.rlinf.serialize_specs_by_value": False,
    "logging.wandb_entity": "nvidia-isaac",
}


def _compose(
    *overrides: str, checkpoint: str | None = "/ckpt/RLinf-Gr00t-SFT-Stack-cube"
):
    register_configs()
    base = [] if checkpoint is None else [f"base_model.ckpt={checkpoint}"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train", overrides=["algorithm=ppo", *base, *overrides]
        )
    # The launchers read the normalized config (older keys mapped onto their canonical names).
    return normalize_train_config(cfg)


def test_rlinf_recipe_is_pinned():
    cfg = _compose()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for key, expected in RLINF_RECIPE.items():
        value = OmegaConf.select(OmegaConf.create(resolved), key)
        assert value == expected, f"{key}: expected {expected!r}, composed {value!r}"


def test_algorithm_ppo_pins_its_simulator_runtime_and_base_model():
    cfg = _compose()
    assert (cfg.algorithm.name, algorithm_family(cfg)) == ("ppo", "on_policy")
    assert (cfg.simulator.provider, cfg.runtime.name, cfg.base_model.name) == (
        "isaaclab",
        "rlinf",
        "gr00t_n15",
    )
    # A launch override still wins over the pins.
    assert _compose("runtime=native").runtime.name == "native"


def test_plan_reproduces_the_launcher_geometry():
    plan = build_plan(_compose())
    assert (
        plan.task_id,
        plan.num_rollout_workers,
        plan.envs_per_worker,
        plan.num_envs,
    ) == (STACK_CUBE, 4, 16, 64)
    assert (
        plan.max_episode_steps,
        plan.image_resolution,
        plan.action_dim,
        plan.denoising_steps,
    ) == (448, 256, 7, 4)
    assert (plan.action_window, plan.chunks_per_episode) == (16, 28)
    assert plan.frozen_submodules == ("model.backbone",)
    assert plan.flow_sde.noise_level == pytest.approx(0.5)
    assert (plan.num_learner_ranks, plan.learner_device) == (4, "auto")
    learner = ppo_config_from(
        plan.ppo, num_learner_ranks=plan.num_learner_ranks, seed=plan.seed
    )
    assert (learner.minibatch_size, learner.micro_batch_size, learner.epochs) == (
        128,
        32,
        3,
    )
    assert (learner.optimizer, learner.adam_betas, learner.weight_decay) == (
        "adamw",
        (0.9, 0.95),
        0.01,
    )
    assert (learner.value_loss, learner.huber_delta, learner.value_learning_rate) == (
        "huber",
        10.0,
        2e-5,
    )
    assert (learner.discount_per, learner.bootstrap_at_time_limit) == (
        "decision",
        False,
    )
    controller = on_policy_config_from(plan.ppo, seed=plan.seed)
    assert (
        plan.ppo.checkpoint_retention,
        controller.checkpoint_every,
        controller.episodes_per_environment,
    ) == (
        "best_last_final",
        plan.ppo.checkpoint_every,
        2,
    )


def test_n15_selection_is_unchanged():
    """``algorithm=ppo`` still trains the N1.5 checkpoint exactly as before the base model became selectable."""
    plan = build_plan(_compose())
    assert plan.trainable == TrainablePolicySpec(
        base_model="gr00t_n15",
        checkpoint="/ckpt/RLinf-Gr00t-SFT-Stack-cube",
        flow_sde=FlowSDEConfig(noise_level=0.5, denoising_steps=4),
        value_head_seed=0,
        options={
            "data_config": "simulators.isaaclab.gr00t_data_configs:IsaacLabFrankaDataConfig",
            "embodiment_tag": "isaaclab_franka",
        },
    )
    assert plan.frozen_submodules == ("model.backbone",)
    assert plan.trainable.policy_class.__name__ == "Gr00tFlowSDEActorCritic"
    # base_model=gr00t (the frozen family name) needs a local checkpoint to tell the generations apart.
    with pytest.raises(
        ValueError, match="base_model=gr00t_n15 or base_model=gr00t_n17"
    ):
        build_plan(_compose("base_model=gr00t"))


def test_other_base_models_swap_in_through_the_base_model_group():
    plan = build_plan(
        _compose(
            "base_model=gr00t_n17",
            "base_model.embodiment_tag=gr1",
            "base_model.denoising_steps=4",
            "base_model.action_horizon=16",
            "ppo.cache_features=true",
        )
    )
    assert plan.trainable.base_model == "gr00t_n17" and plan.frozen_submodules == (
        "model.backbone",
    )
    assert plan.trainable.options == {
        "embodiment_tag": "gr1",
        "action_window": 16,
        "backbone_model_name_or_path": None,
    }
    # N1.7 decisions travel as cached backbone features only.
    with pytest.raises(ValueError, match="cache_features=true"):
        build_plan(
            _compose(
                "base_model=gr00t_n17",
                "base_model.embodiment_tag=gr1",
                "base_model.denoising_steps=4",
                "base_model.action_horizon=16",
            )
        )
    # A list of frozen submodules overrides the adapter's own.
    plan = build_plan(
        _compose("ppo.frozen_submodules=[model.backbone,model.action_head.vlln]")
    )
    assert plan.frozen_submodules == ("model.backbone", "model.action_head.vlln")
    with pytest.raises(ValueError, match="data_config is required"):
        build_plan(_compose("base_model.data_config=null"))


def test_single_gpu_placement_keeps_the_global_batch():
    plan = build_plan(
        _compose(
            "runtime=native",
            "placement.learner.ranks=1",
            "placement.envs.workers=1",
            "training.num_envs=16",
        )
    )
    assert plan.runtime.name == "native" and plan.envs_per_worker == 16
    assert rank_minibatch_size(plan.ppo, plan.num_learner_ranks) == 512, (
        "one rank trains the whole RLinf global batch"
    )


def test_native_runs_the_recipes_ranks_and_workers_as_ray_actors():
    """The recipe asks for four learner ranks, so runtime=native places them (and its workers) on Ray."""
    plan = build_plan(_compose("runtime=native"))
    assert (plan.num_learner_ranks, plan.num_rollout_workers) == (4, 4)


def test_one_process_hosts_several_rollout_workers():
    """Each Isaac Lab worker simulates in a child process of its own, so one process hosts several."""
    plan = build_plan(
        _compose(
            "runtime=native",
            "placement.learner.ranks=1",
            "runtime.native.processes=single",
            "placement.envs.workers=2",
            "training.num_envs=16",
        )
    )
    assert (plan.num_learner_ranks, plan.num_rollout_workers, plan.envs_per_worker) == (
        1,
        2,
        8,
    )


def test_learnerless_evaluation_plan():
    plan = build_plan(
        _compose(
            "ppo.eval_only=true",
            "placement.learner.ranks=0",
            "ppo.resume_from=/runs/x/checkpoints/iteration_40",
        )
    )
    assert (
        plan.learnerless and plan.ppo.resume_from == "/runs/x/checkpoints/iteration_40"
    )
    # No learner, nothing to checkpoint: the trainer must not demand a checkpoint store.
    assert trainer_config(plan).checkpoint_every == 0
    training_plan = build_plan(_compose())
    assert (
        trainer_config(training_plan).checkpoint_every
        == training_plan.ppo.checkpoint_every
        == 20
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        (("placement.learner.ranks=0",), "only valid with ppo.eval_only"),
        (
            ("runtime=native", "runtime.native.processes=single"),
            "one in-process learner rank",
        ),
        (("training.num_envs=30",), "multiple of"),
        (("ppo.global_batch_size=500",), "multiple of"),
        (("ppo.resume_from=relative/iteration_3",), "absolute"),
        (("ppo.isolate_evaluation_workers=true",), "dsrl_ppo"),
        (("ppo.noise_level=null",), "noise_level"),
        (("base_model.action_horizon=null",), "action_horizon is required"),
        (("env.max_episode_steps=450",), "whole number"),
        (
            ("ppo.eval_only=true", "ppo.eval_episodes_per_environment=0"),
            "eval_episodes_per_environment",
        ),
        (
            ("env.env_name=isaaclab/Isaac-Assemble-Trocar-G129-Dex3-v0",),
            "not a task of the composed simulator group",
        ),
    ],
)
def test_invalid_plans_fail_before_launch(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_plan(_compose(*overrides))


def test_checkpoint_is_required():
    with pytest.raises(ValueError, match="base_model.ckpt is required"):
        build_plan(_compose(checkpoint=None))


def dry_run(
    overrides: list[str], cwd: Path
) -> tuple[subprocess.CompletedProcess, dict[str, str], set[str]]:
    """Run scripts/train.py's dry run in a fresh interpreter and report its plan lines and imported packages."""
    script = (
        "import runpy, sys\n"
        f"sys.argv = [{str(TRAIN_ENTRY)!r}, *{overrides!r}]\n"
        f"runpy.run_path({str(TRAIN_ENTRY)!r}, run_name='__main__')\n"
        "print('IMPORTED ' + ' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    cwd.mkdir(parents=True, exist_ok=True)
    env = {
        k: v for k, v in os.environ.items() if k != "ISAAC_RL_POSTTRAINING_OUTPUT_DIR"
    }
    env.update({"ISAAC_RL_POSTTRAINING_DRYRUN": "1", "PYTHONPATH": str(REPO_ROOT)})
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=600,
        check=False,
    )
    plan: dict[str, str] = {}
    imported: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith("plan."):
            key, _, value = line[len("plan.") :].partition("=")
            plan[key] = value
        elif line.startswith("IMPORTED "):
            imported = set(line.split()[1:])
    return result, plan, imported


@pytest.fixture(scope="module")
def ppo_dry_run(tmp_path_factory):
    """One ``algorithm=ppo`` dry run shared by the tests below (each subprocess imports torch and Hydra)."""
    return dry_run(
        ["algorithm=ppo", "base_model.ckpt=/ckpt/RLinf-Gr00t-SFT-Stack-cube"],
        tmp_path_factory.mktemp("ppo"),
    )


def test_dry_run_prints_the_plan_without_isaac_sim_or_gr00t(ppo_dry_run):
    result, plan, imported = ppo_dry_run
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (plan["algorithm"], plan["family"], plan["policy_family"]) == (
        "ppo",
        "on_policy",
        "flow",
    )
    assert (plan["simulator"], plan["runtime"]) == ("isaaclab", "rlinf")
    assert plan["placement.policy"] == "local (a copy in every rollout worker)"
    assert plan["layout"] == "4 rollout worker(s) x 16 scene(s), 448-step episodes"
    assert plan["placement"].startswith(
        "4 learner rank(s) on accelerator(s) 0-3; policy servers and rollout workers from accelerator 4"
    )
    assert "global batch 512 (rank minibatch 128, micro 32)" in plan["schedule"]
    assert "learner_mixed_precision: bf16" in result.stdout, (
        "the resolved config follows the plan"
    )
    assert result.stdout.index("plan.simulator=") < result.stdout.index(
        "\nalgorithm:\n"
    )
    assert "DRY RUN" in result.stderr + result.stdout
    assert imported and not (imported & HEAVY_PACKAGES), imported & HEAVY_PACKAGES


def test_every_algorithm_shares_the_plan_format(tmp_path, ppo_dry_run):
    runs = {
        "dsrl_sac": dry_run(["algorithm=dsrl_sac", "base_model=mock"], tmp_path / "d"),
        "dsrl_ppo": dry_run(["algorithm=dsrl_ppo", "base_model=mock"], tmp_path / "s"),
        "ppo": ppo_dry_run,
    }
    plan_keys = set(TrainPlan.line_fields())
    for algorithm, (result, plan, imported) in runs.items():
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        assert {key.split(".")[0] for key in plan} == plan_keys, algorithm
        assert plan["algorithm"] == algorithm
        assert not (imported & HEAVY_PACKAGES), (algorithm, imported & HEAVY_PACKAGES)
    assert (runs["dsrl_sac"][1]["family"], runs["dsrl_ppo"][1]["family"]) == (
        "steering",
        "on_policy",
    )
    assert (
        runs["dsrl_sac"][1]["policy_family"],
        runs["dsrl_ppo"][1]["policy_family"],
    ) == ("steering", "steering")
