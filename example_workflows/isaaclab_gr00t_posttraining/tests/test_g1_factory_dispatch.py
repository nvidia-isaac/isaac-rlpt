# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Factory tasks under ``isaaclab train/play --rl_library isaac_rlpt`` hand off to the G1 Factory launchers.

Isaac Lab's ``common`` module is stubbed; the dry runs execute the real launchers, which end with
the library's plan and start no simulator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "isaac_rlpt"
G1_SCRIPTS = Path(__file__).resolve().parents[2] / "g1_arena_posttraining" / "scripts"


def _import_local_module(module_name: str, module_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch, tmp_path):
    """The train/play modules against a stub of Isaac Lab's ``common``, run from an empty directory."""
    common = types.ModuleType("common")

    def add_isaaclab_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--headless", action="store_true", default=False)

    common.add_isaaclab_launcher_args = add_isaaclab_launcher_args
    common.import_local_module = _import_local_module
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.chdir(tmp_path)
    # A G1 Factory checkout with IsaacLab-Arena as a submodule, as the installed packages.
    for package in (
        "g1-factory/g1_factory",
        "g1-factory/submodules/IsaacLab-Arena/isaaclab_arena",
    ):
        (tmp_path / package).mkdir(parents=True)
        (tmp_path / package / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str((tmp_path / package).parent))
    for name in (
        "ISAAC_RL_POSTTRAINING_OUTPUT_DIR",
        "ISAAC_RL_POSTTRAINING_DRYRUN",
        "NUM_ENVS",
        "ALGO",
        "LOG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    train = _import_local_module(
        "g1_dispatch_train_under_test", SCRIPTS / "train_isaac_rlpt.py"
    )
    play = _import_local_module(
        "g1_dispatch_play_under_test", SCRIPTS / "play_isaac_rlpt.py"
    )
    return train, play


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    """A stand-in GR00T checkpoint: the dry run reads only its JSON metadata."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "statistics.json").write_text(json.dumps({}))
    return ckpt


def _launched(monkeypatch, module, argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """The command and environment ``module.run(argv)`` hands to the launcher."""
    calls = []

    def fake_run(cmd, env, check):
        calls.append((cmd, env))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.G1_FACTORY.subprocess, "run", fake_run)
    module.run(argv)
    assert len(calls) == 1
    return calls[0]


def test_list_tasks_includes_g1(modules, capsys):
    train, _ = modules
    train.run(["--list_tasks"])
    out = capsys.readouterr().out
    assert "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0" in out
    for task in train.G1_FACTORY.TASKS:
        assert f"  {task}" in out


def test_g1_tasks_match_the_launcher_prompts(modules):
    prompts = (G1_SCRIPTS / "task_prompts.sh").read_text()
    for task in modules[0].G1_FACTORY.TASKS:
        assert f"    {task})" in prompts


def test_train_ppo_composes_finetune_ppo(modules, monkeypatch, checkpoint, tmp_path):
    train, _ = modules
    cmd, env = _launched(
        monkeypatch,
        train,
        [
            "--task",
            "LMDrillPnP",
            "--algorithm",
            "ppo",
            "--model_path",
            str(checkpoint),
            "--iterations",
            "5",
            "--num_envs",
            "4",
            "--seed",
            "3",
            "--run_name",
            "r1",
            "--headless",
            "ppo.eval_every=5",
        ],
    )
    assert cmd == [
        "bash",
        str(G1_SCRIPTS / "finetune_ppo.sh"),
        "LMDrillPnP",
        "ppo.iterations=5",
        "training.seed=3",
        "ppo.eval_every=5",
    ]
    assert env["NUM_ENVS"] == "4"
    assert env["SIM_PYTHON"] == env["LEARNER_PYTHON"] == sys.executable
    assert env["POSTTRAIN_BASE_CKPT"] == str(checkpoint)
    assert env["WANDB_NAME"] == "r1"
    assert env["ISAACLAB_ARENA_PATH"] == str(
        tmp_path / "g1-factory" / "submodules" / "IsaacLab-Arena"
    )
    assert env["PYTHONPATH"] == str(tmp_path / "g1-factory")
    assert env["ISAAC_RL_POSTTRAINING_OUTPUT_DIR"] == str(
        tmp_path / "logs" / "isaac_rlpt"
    )
    assert "ISAAC_RL_POSTTRAINING_DRYRUN" not in env


def test_train_steering_composes_train_sh(modules, monkeypatch, checkpoint):
    train, _ = modules
    cmd, env = _launched(
        monkeypatch,
        train,
        [
            "--task",
            "LMBoxLift",
            "--algorithm",
            "rfs",
            "--model_path",
            str(checkpoint),
            "--num_steps",
            "2000",
            "--dry_run",
        ],
    )
    assert cmd[:3] == ["bash", str(G1_SCRIPTS / "train.sh"), "LMBoxLift"]
    assert (env["ALGO"], env["NUM_ROLLOUTS"], env["ISAAC_RL_POSTTRAINING_DRYRUN"]) == (
        "rfs",
        "2000",
        "1",
    )
    assert "NUM_ENVS" not in env  # the launcher's default


def test_play_composes_eval_sh_and_ppo_eval_only(
    modules, monkeypatch, checkpoint, tmp_path
):
    _, play = modules
    cmd, env = _launched(
        monkeypatch, play, ["--task", "LMPushButton", "--model_path", str(checkpoint)]
    )
    assert cmd == [
        "bash",
        str(G1_SCRIPTS / "eval.sh"),
        "LMPushButton",
        "base",
        "20",
        "training.seed=0",
    ]
    actor = tmp_path / "actor"
    actor.mkdir()
    cmd, _ = _launched(
        monkeypatch,
        play,
        [
            "--task",
            "LMPushButton",
            "--model_path",
            str(checkpoint),
            "--algorithm",
            "sac_edit",
            "--checkpoint",
            str(actor),
            "--eval_episodes",
            "50",
        ],
    )
    assert cmd[2:6] == ["LMPushButton", "policy", "50", str(actor)]
    assert "algorithm=sac_edit" in cmd
    cmd, env = _launched(
        monkeypatch,
        play,
        [
            "--task",
            "LMPushButton",
            "--model_path",
            str(checkpoint),
            "--algorithm",
            "ppo",
            "--checkpoint",
            str(actor),
            "--eval_episodes",
            "100",
        ],
    )
    assert cmd[1] == str(G1_SCRIPTS / "finetune_ppo.sh")
    assert "NUM_ENVS" not in env  # finetune_ppo.sh's eight scenes, 13 episodes each
    assert {
        "ppo.eval_only=true",
        "placement.learner.ranks=0",
        "ppo.eval_episodes_per_environment=13",
        f"ppo.resume_from={actor}",
    } <= set(cmd)


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--algorithm", "ppo", "--num_steps", "10"], "--num_steps"),
        (["--algorithm", "dsrl_sac", "--iterations", "10"], "--iterations"),
        (["--denoising_steps", "8"], "--denoising_steps"),
        (["--algorithm", "rfs_pc"], "zoo_recipes.sh"),
        (["--bogus", "1"], "--bogus"),
    ],
)
def test_inapplicable_flags_are_rejected(modules, checkpoint, argv, message):
    train, _ = modules
    with pytest.raises(SystemExit, match=message):
        train.run(["--task", "LMDrillLift", "--model_path", str(checkpoint), *argv])


def test_viz_opens_a_viewer_on_a_display(modules, monkeypatch, checkpoint, tmp_path):
    _, play = modules
    args, extra = play._parse_args(
        ["--task", "LMDrillLift", "--model_path", str(checkpoint)]
    )
    command = play.G1_FACTORY.command
    monkeypatch.setenv("DISPLAY", ":1")
    assert "SIM_VIZ" not in command(args, extra, "play", tmp_path)[1]
    args.visualizer = ["kit"]  # what Isaac Lab's launcher parser sets for --viz kit
    assert command(args, extra, "play", tmp_path)[1]["SIM_VIZ"] == "kit"
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(SystemExit, match="needs a display"):
        command(args, extra, "play", tmp_path)


def test_launcher_failure_propagates(modules, monkeypatch, checkpoint):
    train, _ = modules
    monkeypatch.setattr(
        train.G1_FACTORY.subprocess,
        "run",
        lambda cmd, env, check: types.SimpleNamespace(returncode=3),
    )
    with pytest.raises(SystemExit) as exc:
        train.run(["--task", "LMDrillLift", "--model_path", str(checkpoint)])
    assert exc.value.code == 3


def test_model_path_local_dir_or_hugging_face_id(
    modules, monkeypatch, checkpoint, tmp_path
):
    g1 = modules[0].G1_FACTORY
    downloads = []

    def fake_download(repo_id, local_dir, allow_patterns):
        downloads.append((repo_id, Path(local_dir), allow_patterns))
        (Path(local_dir)).mkdir(parents=True)
        (Path(local_dir) / "statistics.json").write_text("{}")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert g1.resolve_checkpoint(str(checkpoint), dry_run=False) == checkpoint
    assert not downloads
    got = g1.resolve_checkpoint("nvidia/G1-Factory-GR00T-1.7", dry_run=True)
    assert got == tmp_path / "checkpoints" / "G1-Factory-GR00T-1.7"
    assert downloads == [
        (
            "nvidia/G1-Factory-GR00T-1.7",
            Path("checkpoints/G1-Factory-GR00T-1.7"),
            ["*.json"],
        )
    ]
    with pytest.raises(SystemExit, match="neither a directory"):
        g1.resolve_checkpoint("/no/such/dir", dry_run=True)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="statistics.json"):
        g1.resolve_checkpoint(str(tmp_path / "empty"), dry_run=True)

    def denied(repo_id, local_dir, allow_patterns):
        raise huggingface_hub.errors.HfHubHTTPError("401 Client Error")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied)
    with pytest.raises(SystemExit, match="export HF_TOKEN"):
        g1.resolve_checkpoint("nvidia/private-checkpoint", dry_run=True)


@pytest.mark.parametrize(
    "mode, algorithm, plan",
    [
        ("train", "ppo", "plan.algorithm=ppo"),
        ("train", "dsrl_sac", "plan.algorithm=dsrl_sac"),
        ("play", "dsrl_sac", ""),
    ],
)
def test_dry_run_reaches_the_library_plan(
    modules, monkeypatch, checkpoint, capfd, mode, algorithm, plan
):
    """The real launcher dry run: launch lines, then the library's plan, exit 0."""
    train, play = modules
    # Full PPO plans from the checkpoint's config.json; the steering plans use the mock policy, as smoke.sh does.
    monkeypatch.setenv("BASE_MODEL", "gr00t" if algorithm == "ppo" else "mock")
    module = train if mode == "train" else play
    module.run(
        [
            "--task",
            "LMDrillPnP",
            "--algorithm",
            algorithm,
            "--model_path",
            str(checkpoint),
            "--dry_run",
        ]
    )
    out = capfd.readouterr().out
    assert f"launch.paths.POSTTRAIN_BASE_CKPT={checkpoint}" in out
    assert "launch.override=training.seed=0" in out
    assert "plan." in out and plan in out
