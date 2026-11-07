# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Isaac Lab train/play modules dry-run end to end (compose + plan) for every algorithm, without Isaac Sim.

Isaac Lab's ``common`` module (launcher arguments, module import) is stubbed;
everything below it is the real hook and the real library entry points.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parents[1] / "isaac_rlpt"
FRANKA_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"


def _import_local_module(module_name: str, module_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isaaclab_modules(monkeypatch, tmp_path):
    """The train/play modules loaded against a stub of Isaac Lab's ``common``."""
    common = types.ModuleType("common")

    def add_isaaclab_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--enable_cameras", action="store_true", default=False)

    common.add_isaaclab_launcher_args = add_isaaclab_launcher_args
    common.import_local_module = _import_local_module
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.delitem(sys.modules, "_compat", raising=False)  # re-bind to this stub
    monkeypatch.chdir(tmp_path)
    # launch() exports both; setenv first so monkeypatch removes them again after the test.
    for name in ("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", "ISAAC_RL_POSTTRAINING_DRYRUN"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    train = _import_local_module(
        "isaaclab_rl_train_isaac_rlpt_under_test", SCRIPTS / "train_isaac_rlpt.py"
    )
    play = _import_local_module(
        "isaaclab_rl_play_isaac_rlpt_under_test", SCRIPTS / "play_isaac_rlpt.py"
    )
    return train, play


@pytest.mark.parametrize(
    "algorithm", ["dsrl_sac", "rfs", "rfs_pc", "sac_edit", "dsrl_ppo", "ppo"]
)
def test_train_dry_run_composes_and_plans(
    isaaclab_modules, capsys, tmp_path, algorithm
):
    train, _ = isaaclab_modules
    # The direct-action sac_edit runs on one scene; every other algorithm collects from four.
    scenes = "1" if algorithm == "sac_edit" else "4"
    train.run(
        [
            "--task",
            FRANKA_TASK,
            "--model_path",
            "/ckpt/stack",
            "--algorithm",
            algorithm,
            "--num_envs",
            scenes,
            "--dry_run",
            "--headless",
        ]
    )
    out = capsys.readouterr().out
    assert f"launch.output_root={tmp_path / 'logs' / 'isaac_rlpt'}" in out
    assert "\nalgorithm:\n" in out and "\nbase_model:\n" in out, (
        "the entry printed its resolved config"
    )
    assert f"plan.algorithm={ {'rfs_pc': 'rfs'}.get(algorithm, algorithm) }" in out
    assert "plan.simulator=isaaclab" in out and "plan.runtime=native" in out
    if algorithm == "dsrl_ppo":
        assert "plan.family=on_policy" in out and "plan.policy_family=steering" in out
        assert "plan.layout=1 rollout worker(s) x 4 environment(s), batched" in out
        assert "ignores eval." not in out, (
            "no off-policy evaluation keys reach DSRL-PPO"
        )
    elif algorithm == "ppo":
        assert "plan.family=on_policy" in out and "plan.policy_family=flow" in out
        assert "plan.layout=1 rollout worker(s) x 4 scene(s), 448-step episodes" in out
        assert "global batch 512 (rank minibatch 512, micro 32)" in out
    else:
        assert f"plan.family={'edit' if algorithm == 'sac_edit' else 'steering'}" in out
    assert not (tmp_path / "logs").exists(), "a dry run writes nothing"
    assert "isaaclab_tasks" not in sys.modules, "a dry run never starts Isaac Sim"


def test_play_dry_run_of_a_ppo_snapshot(isaaclab_modules, capsys, tmp_path):
    _, play = isaaclab_modules
    checkpoint = tmp_path / "run" / "checkpoints" / "iteration_40"
    play.run(
        [
            "--task",
            FRANKA_TASK,
            "--model_path",
            "/ckpt/stack",
            "--algorithm",
            "ppo",
            "--checkpoint",
            str(checkpoint),
            "--num_envs",
            "8",
            "--eval_episodes",
            "40",
            "--dry_run",
        ]
    )
    out = capsys.readouterr().out
    assert "plan.algorithm=ppo" in out and "plan.placement=0 learner rank(s)" in out
    assert (
        f"plan.schedule=evaluation only: 5 episode(s) per environment of {checkpoint}"
        in out
    )


def test_play_dry_run_of_the_base_policy(isaaclab_modules, capsys):
    _, play = isaaclab_modules
    play.run(
        [
            "--task",
            FRANKA_TASK,
            "--model_path",
            "/ckpt/stack",
            "--dry_run",
            "--headless",
        ]
    )
    out = capsys.readouterr().out
    assert (
        "eval.eval_base=true" in out
        and "plan.algorithm=dsrl_sac" in out
        and "eval_base: true" in out
    )


def test_list_tasks_needs_no_task_or_checkpoint(isaaclab_modules, capsys):
    train, play = isaaclab_modules
    for module in (train, play):
        module.run(["--list_tasks"])
        out = capsys.readouterr().out
        assert FRANKA_TASK in out and "RLinf/RLinf-Gr00t-SFT-Stack-cube" in out
        assert "Isaac-Assemble-Trocar-G129-Dex3-v0" in out


def test_missing_task_is_loud(isaaclab_modules):
    train, _ = isaaclab_modules
    with pytest.raises(SystemExit, match="--task is required"):
        train.run(["--model_path", "/ckpt/stack", "--dry_run"])


def test_output_root_honours_the_launcher_boundary_variable(
    isaaclab_modules, monkeypatch, capsys
):
    train, _ = isaaclab_modules
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", "/shared/runs/job1")
    train.run(["--task", FRANKA_TASK, "--model_path", "/ckpt/stack", "--dry_run"])
    assert "launch.output_root=/shared/runs/job1" in capsys.readouterr().out


def test_a_plan_error_stops_the_launch_before_isaac_sim_boots(
    isaaclab_modules, monkeypatch, capsys
):
    """Four scenes cannot split over three rollout workers: the plan says so before any simulator starts."""
    booted: list[str] = []
    utils = types.ModuleType("isaaclab_tasks.utils")
    utils.launch_simulation = lambda *a, **k: booted.append("launch_simulation")
    utils.load_cfg_from_registry = lambda *a, **k: booted.append("load_cfg")
    monkeypatch.setitem(
        sys.modules, "isaaclab_tasks", types.ModuleType("isaaclab_tasks")
    )
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", utils)
    train, _ = isaaclab_modules
    with pytest.raises(SystemExit, match="do not split evenly over 3 rollout workers"):
        train.run(
            [
                "--task",
                FRANKA_TASK,
                "--model_path",
                "/ckpt/stack",
                "--num_envs",
                "4",
                "--headless",
                "placement.envs.workers=3",
            ]
        )
    assert booted == []
    assert "plan.checks=" in capsys.readouterr().out


@pytest.fixture
def isaac_sim_starts(monkeypatch):
    """Record every way this process could start Isaac Sim: Isaac Lab's ``launch_simulation`` or an ``AppLauncher``."""
    started: list[str] = []

    def launch_simulation(*args, **kwargs):
        started.append("launch_simulation")
        return contextlib.nullcontext()

    class AppLauncher:
        def __init__(self, *args, **kwargs) -> None:
            started.append("AppLauncher")

    utils = types.ModuleType("isaaclab_tasks.utils")
    utils.launch_simulation = launch_simulation
    utils.load_cfg_from_registry = lambda *a, **k: types.SimpleNamespace(
        scene=types.SimpleNamespace(num_envs=0)
    )
    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = AppLauncher
    monkeypatch.setitem(
        sys.modules, "isaaclab_tasks", types.ModuleType("isaaclab_tasks")
    )
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    # The flag that tells the library an outer app is running; the launcher must not set it.
    monkeypatch.setenv("ISAAC_RLPT_APP_LAUNCHED", "0")
    return started


@pytest.mark.parametrize(
    "mode,algorithm",
    [("train", a) for a in ("dsrl_sac", "rfs", "rfs_pc", "sac_edit", "dsrl_ppo", "ppo")]
    + [("play", "dsrl_sac")],
)
def test_the_launcher_starts_no_isaac_sim_in_its_own_process(
    isaaclab_modules, isaac_sim_starts, monkeypatch, mode, algorithm
):
    """Every algorithm's scenes run in its rollout workers' simulator processes, so the launcher boots none."""
    from isaac_rl_posttraining.runners import build

    runs: list[tuple[str, str | None]] = []
    # The entry script binds build_run at import; it is imported after this patch.
    monkeypatch.setattr(
        build,
        "build_run",
        lambda cfg: (
            lambda: runs.append(
                (str(cfg.algorithm.name), os.environ.get("ISAAC_RLPT_APP_LAUNCHED"))
            )
        ),
    )
    train, play = isaaclab_modules
    argv = [
        "--task",
        FRANKA_TASK,
        "--model_path",
        "/ckpt/stack",
        "--algorithm",
        algorithm,
        "--headless",
    ]
    if mode == "train" and algorithm != "sac_edit":
        argv += ["--num_envs", "4"]
    (train if mode == "train" else play).run(argv)
    assert runs == [({"rfs_pc": "rfs"}.get(algorithm, algorithm), "0")]
    assert isaac_sim_starts == []
