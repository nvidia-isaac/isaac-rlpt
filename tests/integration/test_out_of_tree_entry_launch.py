# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A robot brought from outside the repository trains end to end from its entry class alone.

The owner copies the stand-in package ``simulators/_template`` (a Gymnasium
environment and a ``SimulatorEntry`` subclass with ``name`` and
``make_environment``), points ``simulator.entry`` at it, and launches; no training-env builder is registered anywhere. The launcher sizes
the learner from the entry's environment, the workers build theirs from the
same entry, and the run leaves the artifacts of any real run: checkpoints,
``evaluation_summary.json`` and ``fingerprints.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from isaac_rl_posttraining.buffers import load_buffer
from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.envs import providers

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"

#: The copyable stand-in robot; the test installs a copy outside the repository.
TEMPLATE = REPO_ROOT / "simulators" / "_template"

OVERRIDES = [
    "algorithm=dsrl_sac",
    "base_model=mock",
    "env.env_name=my_robot/pick_cup",
    "+simulator.entry=robot_under_test.runtime:MyRobot",
    "base_model.action_dim=7",
    "base_model.action_horizon=16",
    "dsrl.noise_dim=7",
    "training.chunk_size=16",
    "dsrl.action_exec_len=8",
    "env.max_episode_steps=8",
    "model.dinov2_model=null",
    # Small networks and few updates: the test is about the entry-class launch path, not the model.
    "policy.transformer.d_model=32",
    "policy.transformer.nhead=2",
    "policy.transformer.num_encoder_layers=1",
    "policy.transformer.feature_hidden_dims=[32]",
    "value_function.transformer.d_model=32",
    "value_function.transformer.nhead=2",
    "value_function.transformer.num_encoder_layers=1",
    "value_function.transformer.feature_hidden_dims=[32]",
    "algorithm.num_critics=2",
    "algorithm.num_updates_per_train_step=2",
    "training.num_rollouts=48",
    "algorithm.learning_starts=16",
    "eval.eval_freq=24",
    "eval.eval_num_episodes=1",
    "eval.eval_record_video=false",
    "training.save_interval=24",
    "training.checkpoint_retention=best_last_final",
    "logging.wandb_offline=true",
    "logging.wandb_name=robot_under_test",
]


def install_robot_package(directory: Path) -> Path:
    """Copy the stand-in package to ``directory/robot_under_test``, as an owner's own package."""
    package = directory / "robot_under_test"
    package.mkdir()
    for name in ("__init__.py", "env.py", "runtime.py"):
        (package / name).write_text((TEMPLATE / name).read_text())
    return package


@pytest.fixture
def robot_package(tmp_path, monkeypatch):
    """Write the owner's package to a fresh directory and make it importable, as `pip install -e` would."""
    package = install_robot_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield package
    # The entry the run registered must not leak into other tests.
    providers._SIMULATOR_ENTRIES.pop("my_robot", None)


def _compose():
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=OVERRIDES)


def test_entry_class_alone_trains_evaluates_and_checkpoints(
    robot_package, tmp_path, monkeypatch
):
    from scripts.train import run_from_hydra_cfg

    assert providers.get_env_builder("my_robot/pick_cup") is None, (
        "no training-env builder is registered"
    )
    outputs = tmp_path / "outputs"
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(outputs))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("ISAAC_RL_POSTTRAINING_DRYRUN", raising=False)
    monkeypatch.chdir(tmp_path)

    run_from_hydra_cfg(_compose())

    runs = sorted(outputs.glob("robot_under_test_*"))
    assert len(runs) == 1, os.listdir(outputs)
    checkpoints = runs[0] / "checkpoints"
    steps = sorted(int(p.name) for p in checkpoints.iterdir() if p.name.isdigit())
    assert steps and steps[-1] >= 24, sorted(os.listdir(checkpoints))
    assert (checkpoints / "final").is_dir()
    summary = json.loads((checkpoints / "evaluation_summary.json").read_text())
    assert summary, "the run must evaluate"
    fingerprints = [
        json.loads(line)
        for line in (checkpoints / "fingerprints.jsonl").read_text().splitlines()
    ]
    assert {event["event"] for event in fingerprints} >= {"eval", "save"}


def test_a_run_handle_drives_the_same_run_step_by_step(
    robot_package, tmp_path, monkeypatch
):
    """launch() plans the run; collect/train/evaluate/save/load drive its pieces one call at a time."""
    from isaac_rl_posttraining import launch

    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    with launch(
        "dsrl_sac",
        env="my_robot/pick_cup",
        overrides=OVERRIDES,
        persist="collected",
    ) as run:
        assert run.plan.simulator == "my_robot"
        assert run.collect(rounds=3)["total_steps"] > 0
        assert len(run.buffer) > 0
        assert "critic_loss" in run.train(steps=2)
        assert 0.0 <= run.evaluate()["success_rate"] <= 1.0
        run.save("checkpoint")
        assert (tmp_path / "checkpoint").is_dir()
        run.load("checkpoint")
        with pytest.raises(RuntimeError, match="fresh run"):
            run.loop()
    # Completed training episodes were saved under the run directory as a loadable buffer.
    saved = load_buffer(Path(run.run_dir) / "collected")
    assert saved.episodes() and all(episode.complete for episode in saved.episodes())


#: DSRL-PPO on the stand-in robot: two iterations of one episode per environment.
DSRL_PPO = [
    *(
        override
        for override in OVERRIDES
        if not override.startswith(("algorithm", "training.", "eval.eval_freq"))
    ),
    "algorithm=dsrl_ppo",
    "training.chunk_size=16",
    "ppo.iterations=2",
    "ppo.eval_every=1",
    "ppo.global_batch_size=4",
    "ppo.update_epochs=1",
]


@pytest.mark.parametrize(
    "runtime",
    [
        "native",
        pytest.param("rlinf", marks=[pytest.mark.rlinf, pytest.mark.rlinf_cluster]),
    ],
)
def test_a_run_handle_steps_an_on_policy_run(
    robot_package, tmp_path, monkeypatch, runtime
):
    """collect() plays one iteration's episodes and train() updates on them; telemetry and saving follow the data source."""
    from isaac_rl_posttraining import launch

    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)
    device = ["runtime.rlinf.device=cpu"] if runtime == "rlinf" else []

    with launch(
        "dsrl_ppo",
        env="my_robot/pick_cup",
        runtime=runtime,
        overrides=[*DSRL_PPO, *device],
        persist="collected",
    ) as run:
        assert run.collect()["train/episodes"] == 1.0
        assert "learner/policy_loss" in run.train()
        assert 0.0 <= run.evaluate()["success_rate"] <= 1.0
        with pytest.raises(ValueError, match="call collect\\(\\) and then train\\(\\)"):
            run.collect(rounds=2)
        with pytest.raises(RuntimeError, match="keeps no replay buffer"):
            run.buffer  # noqa: B018
        run.save("checkpoint")
        run.load("checkpoint")
        run.collect()
        run.train()
        sources = {row["source"] for row in run.telemetry.table("episodes").to_pylist()}
        assert sources == {"train", "eval"}
        assert len(run.telemetry.table("learner")) == 2
    assert len(load_buffer(Path(run.run_dir) / "collected").episodes()) == 2
