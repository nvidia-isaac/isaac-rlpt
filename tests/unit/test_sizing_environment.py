# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launcher sizes the learner from the entry's environment when no training-env builder is registered."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs.simulator_entry import (
    CONVENTIONAL_STEERING_KEYS,
    SimulatorEntry,
)
from isaac_rl_posttraining.runners import off_policy_launch
from isaac_rl_posttraining.runners.simulator_entries import (
    sizing_environment,
    steering_remove_obs_keys,
)


class _Env(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "image": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": np.full(3, self._t, np.float32),
            "image": np.zeros((4, 4, 3), np.uint8),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        self._t += 1
        done = self._t >= 3
        return self._observation(), 0.0, done, False, {"success": done}


class _Entry(SimulatorEntry):
    name = "toy_sizing"

    def make_environment(self, spec, phase, context):
        assert (phase, context.environment_indices) == ("train", (0,))
        return _Env()


class _WristEnv(_Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "wrist_cam": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": np.full(3, self._t, np.float32),
            "wrist_cam": np.zeros((4, 4, 3), np.uint8),
        }


class _WristEntry(SimulatorEntry):
    name = "toy_wrist"

    def make_environment(self, spec, phase, context):
        return _WristEnv()

    def steering_key_mapping(self, spec):
        return {"state": "state", "language": None, "image": ["wrist_cam"]}


class _NotAnEnv(SimulatorEntry):
    name = "toy_not_env"

    def make_environment(self, spec, phase, context):
        return object()


def _cfg(**overrides: Any):
    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "toy_sizing/task",
                "max_episode_steps": 5,
                "image_keys": None,
                "use_full_state": False,
            },
            "training": {"num_envs": 1, "seed": 0, "chunk_size": 4},
            "eval": {"eval_num_envs": 1},
            "simulator": {},
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_sizing_environment_is_the_entry_environment_behind_the_training_wrappers() -> (
    None
):
    env = sizing_environment(_Entry(), _cfg(), device="cpu")

    assert env.steering_key_mapping == dict(CONVENTIONAL_STEERING_KEYS)
    assert steering_remove_obs_keys(env) == ["image"]
    observation, _ = env.reset(seed=0)
    assert observation["state"].shape == (1, 3) and observation["image"].shape == (
        1,
        4,
        4,
        3,
    )
    # One environment, behind the action-chunking wrapper of the configured size.
    assert env.num_envs == 1 and env.chunk_size == 4
    env.close()


def test_sizing_environment_can_use_the_evaluation_phase(monkeypatch) -> None:
    phases = []

    class _PhaseEntry(_Entry):
        name = "toy_phase_sizing"
        server_url_variables = ("TOY_TRAIN_URL", "TOY_EVAL_URL")

        def make_environment(self, spec, phase, context):
            phases.append((phase, spec.server_url, context.environment_indices))
            return _Env()

    monkeypatch.setenv("TOY_EVAL_URL", "tcp://127.0.0.1:9001")
    env = sizing_environment(_PhaseEntry(), _cfg(), phase="eval", device="cpu")

    assert phases == [("eval", "tcp://127.0.0.1:9001", (0,))]
    env.close()


def test_sizing_environment_takes_the_configured_image_keys_and_no_chunking() -> None:
    env = sizing_environment(
        _Entry(),
        _cfg(env={"image_keys": ["image"]}, training={"chunk_size": None}),
        device="cpu",
    )
    assert steering_remove_obs_keys(env) == ["image"]
    assert not hasattr(env, "chunk_size") and env.single_action_space.shape == (2,)
    env.close()


def test_sizing_environment_follows_the_entry_cameras_when_image_keys_are_unset() -> (
    None
):
    env = sizing_environment(_WristEntry(), _cfg(), device="cpu")
    assert steering_remove_obs_keys(env) == ["wrist_cam"]
    env.close()


def test_sizing_environment_refuses_an_image_key_the_observation_lacks() -> None:
    with pytest.raises(
        ValueError,
        match=r"toy_wrist: image keys \['does_not_exist'\] from env.image_keys are not camera entries of the "
        r"observation \(cameras: \['wrist_cam'\]\); set env.image_keys",
    ):
        sizing_environment(
            _WristEntry(), _cfg(env={"image_keys": ["does_not_exist"]}), device="cpu"
        )


def test_sizing_environment_refuses_a_language_model_without_an_instruction() -> None:
    with pytest.raises(ValueError, match="declares no task instruction"):
        sizing_environment(_Entry(), _cfg(), device="cpu", sentence_model=object())


def test_sizing_environment_refuses_a_non_gymnasium_environment() -> None:
    with pytest.raises(TypeError, match="toy_not_env.make_environment returned object"):
        sizing_environment(_NotAnEnv(), _cfg(), device="cpu")


def test_launcher_keeps_the_registered_builder_and_sizes_other_entries_from_the_entry(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class _Sized:
        steering_remove_obs_keys: ClassVar[list[str]] = ["cam"]

    def fake_make_env(**kwargs):
        calls.append("builder")
        return _Sized(), None

    def fake_sizing(entry, cfg, **kwargs):
        calls.append(f"entry:{entry.name}")
        return _Sized()

    monkeypatch.setattr(off_policy_launch, "make_env", fake_make_env)
    monkeypatch.setattr(off_policy_launch, "sizing_environment", fake_sizing)
    monkeypatch.setattr(
        off_policy_launch,
        "get_env_builder",
        lambda name: object() if name.startswith("arena/") else None,
    )

    cfg = _cfg(env={"env_name": "arena/drill_lift"})
    _, keys = off_policy_launch.setup_dsrl_env(
        cfg, None, None, None, "cpu", entry=_Entry()
    )
    assert calls == ["builder"] and keys == ["cam"]

    calls.clear()
    _, keys = off_policy_launch.setup_dsrl_env(
        _cfg(), None, None, None, "cpu", entry=_Entry()
    )
    assert calls == ["entry:toy_sizing"] and keys == ["cam"]

    calls.clear()
    # Without an entry (a plain Gymnasium id) the legacy path stays.
    off_policy_launch.setup_dsrl_env(
        _cfg(env={"env_name": "CartPole-v1"}), None, None, None, "cpu"
    )
    assert calls == ["builder"]


class _GroupedStateEnv(_Env):
    """Per-group state entries the policy reads but the learner's networks must not see."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "state.gripper": gym.spaces.Box(-1, 1, (1,), np.float32),
            }
        )
        self.steering_remove_obs_keys = ["image", "state.gripper"]


class _GroupedStateEntry(SimulatorEntry):
    name = "toy_grouped"

    def make_environment(self, spec, phase, context):
        return _GroupedStateEnv()


def test_sizing_keeps_the_keys_the_environment_removes_from_the_networks() -> None:
    env = sizing_environment(_GroupedStateEntry(), _cfg(), device="cpu")

    assert steering_remove_obs_keys(env) == ["image", "state.gripper"]
