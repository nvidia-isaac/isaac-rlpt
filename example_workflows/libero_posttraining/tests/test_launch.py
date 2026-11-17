# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LIBERO launchers dry-run from any directory, and the plans they print compose on the library config."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.runners.build import train_plan

WORKFLOW = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKFLOW.parent.parent
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
PLACEHOLDER_CKPT = "/placeholder/GR00T-N1.7-LIBERO/libero_10"
PI0_TASK = "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray"

#: Launcher knobs a developer's shell may carry; cleared so every test sees the defaults.
LAUNCHER_VARIABLES = (
    "ALGO",
    "BASE_MODEL",
    "GPU",
    "GPUS",
    "LEARNER_PYTHON",
    "LOG_DIR",
    "NUM_ENVS",
    "NUM_ROLLOUTS",
    "POSTTRAIN_BACKBONE_MODEL",
    "POSTTRAIN_BASE_CKPT",
    "POSTTRAIN_DATASET_DIR",
    "PPO_LEARNER_RANKS",
    "RECIPE",
    "RUNTIME",
    "SIMULATOR",
    "SUITE",
    "TRAINER",
    "TRAIN_MAXSTEPS",
    "WANDB_NAME",
)


def dry_run(
    script: str, *args: str, cwd: Path, **variables: str
) -> subprocess.CompletedProcess[str]:
    """Run a launcher in dry-run mode from ``cwd`` with the given launcher variables."""
    env = {
        key: value for key, value in os.environ.items() if key not in LAUNCHER_VARIABLES
    }
    env["ISAAC_RL_POSTTRAINING_DRYRUN"] = "1"
    env.update(variables)
    return subprocess.run(
        ["bash", str(WORKFLOW / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=120,
        check=False,
    )


def plan_lines(completed: subprocess.CompletedProcess[str]) -> set[str]:
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return set(completed.stdout.splitlines())


def hydra_overrides(completed: subprocess.CompletedProcess[str]) -> list[str]:
    """The overrides a launcher's dry run passes to the entry point, one per ``launch.override=`` line."""
    prefix = "launch.override="
    return [
        line.removeprefix(prefix)
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]


def composed(*overrides: str) -> DictConfig:
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


def test_steering_launcher_plans_the_gr00t_recipe(tmp_path: Path) -> None:
    completed = dry_run("train.sh", "8", cwd=tmp_path)
    lines = plan_lines(completed)
    assert {
        "plan.algorithm=dsrl_sac",
        "plan.simulator=libero",
        "plan.runtime=native",
    } <= lines
    overrides = hydra_overrides(completed)
    assert {"base_model=gr00t", "training.num_rollouts=100000"} <= set(overrides)
    assert {
        "simulator=libero_10",
        "env.env_name=libero/libero_10/8",
        "env.max_episode_steps=${simulator.gr00t_eval.max_episode_steps}",
        "base_model.embodiment_tag=libero_sim",
        "dsrl.noise_dim=7",
        "dsrl.action_exec_len=8",
        "training.chunk_size=40",
        "algorithm=dsrl_sac",
    } <= set(overrides)
    assert overrides.index("base_model=gr00t") < overrides.index("algorithm=dsrl_sac")


def test_sac_edit_launcher_plans_one_mlp_over_the_whole_chunk(tmp_path: Path) -> None:
    completed = dry_run("train.sh", "8", cwd=tmp_path, ALGO="sac_edit")
    assert "plan.algorithm=sac_edit" in plan_lines(completed)
    assert "training.chunk_size=null" in hydra_overrides(completed)


def test_ppo_launcher_selects_dsrl_ppo(tmp_path: Path) -> None:
    completed = dry_run("train_ppo.sh", "8", cwd=tmp_path)
    assert "plan.algorithm=dsrl_ppo" in plan_lines(completed)
    # The rest of the recipe (16-step chunks, the actor's initialisation) is dsrl_ppo.yaml's.
    assert {"algorithm=dsrl_ppo", "placement.learner.ranks=1"} <= set(
        hydra_overrides(completed)
    )
    # PPO evaluates through ppo.eval_*: no off-policy evaluator key reaches it, so nothing is ignored.
    assert not [o for o in hydra_overrides(completed) if o.startswith("eval.")]
    assert "ignores eval." not in completed.stdout + completed.stderr


def test_pi0_launcher_plans_the_paper_recipe(tmp_path: Path) -> None:
    completed = dry_run(
        "train.sh",
        PI0_TASK,
        f"env.env_name=libero_pi0/libero_90/{PI0_TASK}",
        cwd=tmp_path,
        BASE_MODEL="pi0",
        TRAIN_MAXSTEPS="400",
    )
    assert "plan.algorithm=dsrl_sac" in plan_lines(completed)
    overrides = hydra_overrides(completed)
    assert {
        "base_model=pi0",
        "env.max_episode_steps=400",
        "dsrl.noise_dim=32",
        "dsrl.action_exec_len=20",
        "training.chunk_size=null",
        "algorithm.target_entropy=-16.0",
        "algorithm.critic_reduction=min",
    } <= set(overrides)
    assert overrides[-1] == f"env.env_name=libero_pi0/libero_90/{PI0_TASK}", (
        "extra overrides come last so they win"
    )


def test_episode_cap_override_and_dreamzero_default(tmp_path: Path) -> None:
    capped = dry_run("train.sh", "8", cwd=tmp_path, TRAIN_MAXSTEPS="300")
    assert "env.max_episode_steps=300" in hydra_overrides(capped)
    dreamzero = dry_run("train.sh", "7", cwd=tmp_path, BASE_MODEL="dreamzero_local")
    assert {
        "env.env_name=libero_dreamzero/libero_10/7",
        "env.max_episode_steps=240",
        "dsrl.action_exec_len=16",
    } <= set(hydra_overrides(dreamzero))


@pytest.mark.parametrize(
    "variables,message",
    [
        ({"BASE_MODEL": "nope"}, "BASE_MODEL must be gr00t, pi0 or dreamzero_local"),
        (
            {"BACKEND": "pi0"},
            "BACKEND was renamed to BASE_MODEL; set BASE_MODEL=pi0 instead",
        ),
        ({"SUITE": "libero_90"}, "no simulator config"),
        ({"BASE_MODEL": "pi0", "ALGO": "dsrl_ppo"}, "supports BASE_MODEL=gr00t"),
        ({"BASE_MODEL": "dreamzero_local", "RECIPE": "cnn"}, "unknown RECIPE"),
        (
            {"BASE_MODEL": "dreamzero_local", "ALGO": "sft"},
            "ALGO=sft trains BASE_MODEL=gr00t or pi0",
        ),
    ],
)
def test_launcher_rejects_unknown_modes(
    tmp_path: Path, variables: dict[str, str], message: str
) -> None:
    completed = dry_run("train.sh", "8", cwd=tmp_path, **variables)
    assert completed.returncode == 2 and message in completed.stderr


def test_gr00t_recipe_composes_and_plans(tmp_path: Path) -> None:
    completed = dry_run(
        "train.sh", "8", cwd=tmp_path, POSTTRAIN_BASE_CKPT=PLACEHOLDER_CKPT
    )
    cfg = composed(*hydra_overrides(completed))
    assert (cfg.simulator.provider, cfg.base_model.name) == ("libero", "gr00t")
    assert int(cfg.env.max_episode_steps) == 720, (
        "the cap comes from the simulator config"
    )
    assert cfg.base_model.ckpt == PLACEHOLDER_CKPT
    plan = train_plan(cfg)
    assert (plan.simulator, plan.algorithm, plan.family) == (
        "libero",
        "dsrl_sac",
        "steering",
    )
    assert plan.entry == "simulators.libero.runtime.Libero(name='libero')"
    assert plan.evaluation_mode == "fresh_workers"


@pytest.mark.parametrize(
    "runtime", ["runtime.native.processes=ray", "runtime=rlinf"], ids=["ray", "rlinf"]
)
def test_rollout_workers_keep_a_gpu_to_render_when_the_policy_is_served(
    tmp_path: Path, runtime: str
) -> None:
    """LIBERO renders its cameras through EGL, so its workers need a GPU even as thin clients."""
    completed = dry_run(
        "train.sh", "8", cwd=tmp_path, POSTTRAIN_BASE_CKPT=PLACEHOLDER_CKPT
    )
    cfg = composed(
        *hydra_overrides(completed),
        runtime,
        "placement.policy.mode=served",
        "placement.gpus_per_node=2",
    )
    assert train_plan(cfg).placement_plan.envs.gpus


def test_ppo_recipe_composes_on_the_mock_base_model(tmp_path: Path) -> None:
    completed = dry_run("train_ppo.sh", "3", cwd=tmp_path, SUITE="libero_spatial")
    # The mock stands in with LIBERO's 7-wide action, as the plan checks it against the simulator.
    cfg = composed(
        *hydra_overrides(completed),
        "base_model=mock",
        "base_model.action_dim=7",
        "dsrl.noise_dim=7",
    )
    assert cfg.env.env_name == "libero/libero_spatial/3"
    assert (int(cfg.base_model.action_horizon), int(cfg.training.chunk_size)) == (
        16,
        16,
    ), "one PPO decision is one decoded chunk of the checkpoint's action window"
    plan = train_plan(cfg)
    assert (plan.simulator, plan.algorithm, plan.family) == (
        "libero",
        "dsrl_ppo",
        "on_policy",
    )


def test_pi0_recipe_composes_and_plans(tmp_path: Path) -> None:
    completed = dry_run(
        "train.sh",
        PI0_TASK,
        f"env.env_name=libero_pi0/libero_90/{PI0_TASK}",
        cwd=tmp_path,
        BASE_MODEL="pi0",
        TRAIN_MAXSTEPS="400",
    )
    plan = train_plan(composed(*hydra_overrides(completed)))
    assert plan.algorithm == "dsrl_sac"


def guide_pi0_overrides() -> list[str]:
    """The Hydra overrides of the guide's pi0 command, read from the guide itself.

    The guide is command-first: the block a reader copies has to compose. Reading
    it here rather than restating it means the two cannot drift apart.
    """
    lines = (REPO_ROOT / "docs/pages/workflows/libero.md").read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if "BASE_MODEL=pi0 " in line]
    assert len(starts) == 1, "expected exactly one pi0 command in the LIBERO guide"
    block: list[str] = []
    for line in lines[starts[0] :]:
        if line.startswith("```"):
            break
        block.append(line.rstrip("\\").strip())
    tokens = " ".join(block).split()
    # Everything after the task argument is a Hydra override; the task itself and
    # the env_name the test already passes are dropped.
    return [
        t.strip('"')
        for t in tokens
        if "=" in t
        and not t.startswith(
            (
                "POSTTRAIN_",
                "GPU=",
                "BASE_MODEL=",
                "TRAIN_MAXSTEPS=",
                "TASK=",
                "env.env_name=",
            )
        )
    ]


def test_the_guide_pi0_recipe_composes(tmp_path: Path) -> None:
    """The overrides printed in the guide are accepted as written.

    A `+key=value` for a key the schema already declares is rejected by Hydra, so
    a stray plus in the guide makes the documented command fail before it starts.
    """
    overrides = guide_pi0_overrides()
    assert overrides, "no overrides parsed out of the guide's pi0 command"
    completed = dry_run(
        "train.sh",
        PI0_TASK,
        f"env.env_name=libero_pi0/libero_90/{PI0_TASK}",
        *overrides,
        cwd=tmp_path,
        BASE_MODEL="pi0",
        TRAIN_MAXSTEPS="400",
    )
    plan = train_plan(composed(*hydra_overrides(completed)))
    assert plan.algorithm == "dsrl_sac"
