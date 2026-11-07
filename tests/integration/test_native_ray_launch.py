# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run asks its placement for several processes and runtime=native runs it on Ray actors.

The runs go through the entry point (``scripts/train.py``'s
``run_from_hydra_cfg``) with the mock base model and a small out-of-tree robot:
served policy replicas, two off-policy learner ranks, the sync and async
schedules, and a rollout worker that dies mid-run.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from hydra import compose, initialize_config_dir

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.runtimes import RolloutInfrastructureError

pytestmark = [pytest.mark.native_ray, pytest.mark.timeout(900)]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "isaac_rl_posttraining" / "configs"

#: A robot whose joints move toward their targets; with a kill file, environment 1 SIGKILLs
#: its worker process at step ``KILL_AT`` after writing the time to the file.
ROBOT = textwrap.dedent(
    """
    from __future__ import annotations

    import os
    import signal
    import time

    import gymnasium as gym
    import numpy as np

    from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

    KILL_FILE = {kill_file!r}
    KILL_AT = 20


    class RobotEnv(gym.Env):
        def __init__(self, index: int, max_episode_steps: int) -> None:
            super().__init__()
            self.index = index
            self.max_episode_steps = max_episode_steps
            self.observation_space = gym.spaces.Dict(
                {{
                    "state": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
                    "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
                }}
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
            self._joints = np.zeros(7, np.float32)
            self._steps = 0
            self._lifetime_steps = 0

        def _observation(self):
            return {{"state": self._joints.copy(), "image": np.full((16, 16, 3), 96, np.uint8)}}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._joints = np.zeros(7, np.float32)
            self._steps = 0
            return self._observation(), {{"success": False}}

        def step(self, action):
            self._joints += 0.1 * (np.clip(action, -1, 1).astype(np.float32) - self._joints)
            self._steps += 1
            self._lifetime_steps += 1
            if KILL_FILE and self.index == 1 and self._lifetime_steps == KILL_AT:
                with open(KILL_FILE, "w") as stream:
                    stream.write(str(time.time()))
                os.kill(os.getpid(), signal.SIGKILL)
            truncated = self._steps >= self.max_episode_steps
            return self._observation(), 0.0, False, truncated, {{"success": False}}


    class Robot(SimulatorEntry):
        name = "ray_robot"

        def make_environment(self, spec, phase, context):
            index = context.environment_indices[0] if phase == "train" else -1
            return RobotEnv(index, spec.max_episode_steps)
    """
)

OVERRIDES = [
    "algorithm=dsrl_sac",
    "base_model=mock",
    "env.env_name=ray_robot/reach",
    "+simulator.entry=ray_robot.runtime:Robot",
    "base_model.action_dim=7",
    "base_model.action_horizon=16",
    "dsrl.noise_dim=7",
    "training.chunk_size=16",
    "dsrl.action_exec_len=8",
    "env.max_episode_steps=8",
    "model.dinov2_model=null",
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
    "algorithm.batch_size=8",
    "training.num_envs=4",
    "training.num_rollouts=128",
    "algorithm.learning_starts=32",
    "eval.eval_freq=64",
    "eval.eval_num_episodes=4",
    "eval.eval_record_video=false",
    "training.save_interval=64",
    "logging.wandb_name=ray_robot",
    "runtime.native.device=cpu",
    # Several processes: two policy-server replicas and two learner ranks, as Ray actors.
    "placement.policy.mode=served",
    "placement.policy.replicas=2",
    "placement.learner.ranks=2",
]


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.init(num_cpus=16, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _robot(tmp_path: Path, monkeypatch, kill_file: str = "") -> None:
    package = tmp_path / "ray_robot"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "runtime.py").write_text(ROBOT.format(kill_file=kill_file))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(tmp_path / "outputs"))
    # No experiment logging: these runs are about the runtime (and wandb-core 0.25 can panic on a
    # later offline run's artifact in the same process once these have logged theirs).
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("ISAAC_RL_POSTTRAINING_DRYRUN", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _forget_the_robot():
    yield
    providers._SIMULATOR_ENTRIES.pop("ray_robot", None)
    # Each test writes its own robot package; the next one must import it afresh.
    for name in ("ray_robot.runtime", "ray_robot"):
        sys.modules.pop(name, None)


def _compose(*extra: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=[*OVERRIDES, *extra])


def _checkpoints(tmp_path: Path) -> Path:
    (run,) = sorted((tmp_path / "outputs").glob("ray_robot_*"))
    return run / "checkpoints"


def test_the_dry_run_shows_the_ray_layout(tmp_path, monkeypatch):
    from isaac_rl_posttraining.runners.build import train_plan

    _robot(tmp_path, monkeypatch)
    lines = "\n".join(train_plan(_compose()).lines())
    assert "plan.placement=Ray actors on 1 node(s): 2 learner rank actor(s)" in lines
    assert "plan.placement.policy.base+actor=served by 2 replica(s)" in lines
    assert "plan.placement.learner=2 rank(s)" in lines


@pytest.mark.parametrize("schedule", ["sync", "async"])
def test_a_run_on_several_processes_trains_evaluates_and_checkpoints(
    schedule, tmp_path, monkeypatch
):
    from scripts.train import run_from_hydra_cfg

    _robot(tmp_path, monkeypatch)
    lag = ["schedule.max_policy_lag=2"] if schedule == "async" else []
    run_from_hydra_cfg(_compose(f"schedule.kind={schedule}", *lag))

    checkpoints = _checkpoints(tmp_path)
    assert (checkpoints / "final").is_dir()
    # Two learner ranks write a sharded checkpoint: a manifest, the shared payload, one file per rank.
    assert {"manifest.json", "learner.pt", "rank_0.pt", "rank_1.pt"} <= {
        path.name for path in (checkpoints / "final").iterdir()
    }
    summary = json.loads((checkpoints / "evaluation_summary.json").read_text())
    assert summary["history"]
    placement = json.loads(
        (checkpoints.parent / "telemetry" / "placement.json").read_text()
    )
    assert placement["learner"]["processes"] == 2
    assert placement["policy"][0]["processes"] == 2


def test_a_dead_rollout_worker_fails_the_run_within_seconds_and_saves_an_aborted_checkpoint(
    tmp_path, monkeypatch
):
    from scripts.train import run_from_hydra_cfg

    kill_file = tmp_path / "killed_at"
    _robot(tmp_path, monkeypatch, kill_file=str(kill_file))
    with pytest.raises(RolloutInfrastructureError, match="rollout worker 1 .* failed"):
        try:
            run_from_hydra_cfg(_compose())
        finally:
            failed_at = time.time()
    detection_seconds = failed_at - float(kill_file.read_text())
    assert detection_seconds < 30.0, (
        f"the failure took {detection_seconds:.1f} s to surface"
    )
    aborted = _checkpoints(tmp_path) / "aborted"
    assert (aborted / "manifest.json").is_file() and (
        aborted / "policy_state.pt"
    ).is_file()
    assert os.path.isfile(aborted / "rank_1.pt")
