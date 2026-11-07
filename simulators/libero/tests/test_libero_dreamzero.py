# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO DreamZero adapter conventions and launcher recipe (no groot required).

Pins:
1. The DreamZero libero_sim observation conventions (key renames, 180-degree
   training orientation, native 256x256 frames — the base model's transform
   chain owns crop/resize — and 8-dim state passthrough).
2. The finish-fast reward shift (-1 per step, 0 on success).
3. The launcher's BASE_MODEL=dreamzero override set (dry-run plan): the
   RLinf-validated recipe mapping.

The base model's latent geometry and noise handling are covered in
``tests/unit/test_dreamzero_local.py``.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

REPO_ROOT = Path(__file__).resolve().parents[3]
IMG = 32  # stub camera resolution


class _ScriptedInnerEnv(gym.Env):
    """Stub of LiberoEnv: two cameras + 8-dim eef state + prompt."""

    def __init__(self):
        image_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG, IMG, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agentview_image": image_space,
                "robot0_eye_in_hand_image": image_space,
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.last_action = None
        self._succeed_on_step = None

    def _obs(self):
        scene = np.zeros((IMG, IMG, 3), dtype=np.uint8)
        scene[0, 0] = (255, 0, 0)  # marker at top-left of the UPRIGHT frame
        wrist = np.full((IMG, IMG, 3), 7, dtype=np.uint8)
        return {
            "agentview_image": scene,
            "robot0_eye_in_hand_image": wrist,
            "state": np.arange(8, dtype=np.float32),
            "prompt": "pick up the black bowl",
        }

    def reset(self, **kwargs):
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        self.last_action = np.array(action)
        self._steps += 1
        done = (
            self._succeed_on_step is not None and self._steps >= self._succeed_on_step
        )
        reward = 1.0 if done else 0.0
        return self._obs(), reward, done, False, {"is_success": done}


def _adapter():
    from simulators.libero.policies.dreamzero import LiberoDreamZeroAdapter

    return LiberoDreamZeroAdapter(_ScriptedInnerEnv())


def test_adapter_renames_and_flips():
    env = _adapter()
    obs, _ = env.reset()
    assert set(obs) == {"main_image", "wrist_image", "state", "prompt"}
    # Upright frame's top-left marker lands at top-RIGHT after the horizontal
    # flip; no resize happens here (the base model's transform chain owns it).
    assert obs["main_image"].shape == (IMG, IMG, 3)
    assert tuple(obs["main_image"][0, -1]) == (255, 0, 0)
    assert np.array_equal(obs["state"], np.arange(8, dtype=np.float32))


def test_adapter_reward_is_finish_fast():
    env = _adapter()
    env.env._succeed_on_step = 2
    env.reset()
    _, r1, done1, _, info1 = env.step(np.zeros(7, dtype=np.float32))
    assert (r1, done1, info1["success"]) == (-1.0, False, False)
    _, r2, done2, _, info2 = env.step(np.zeros(7, dtype=np.float32))
    assert (r2, done2, info2["success"]) == (0.0, True, True)


def test_adapter_passes_actions_through_unchanged():
    env = _adapter()
    env.reset()
    action = np.linspace(-1, 1, 7).astype(np.float32)
    env.step(action)
    assert np.allclose(env.env.last_action, action)


def test_adapter_steering_contract():
    env = _adapter()
    assert env.steering_key_mapping == {
        "state": "state",
        "language": None,
        "image": ["main_image"],
    }
    assert env.steering_remove_obs_keys == ["main_image", "wrist_image"]
    assert env.action_space.shape == (7,)


def test_adapter_matches_base_model_conversion_contract():
    """The adapter's obs must convert cleanly through the base model's layout."""
    from isaac_rl_posttraining.policies.dreamzero.layouts import (
        LIBERO_SIM_SPEC,
        convert_env_obs,
    )

    env = _adapter()
    obs, _ = env.reset()
    batched = {
        "main_image": obs["main_image"][None],
        "wrist_image": obs["wrist_image"][None],
        "state": obs["state"][None],
        "prompt": [obs["prompt"]],
    }
    converted = convert_env_obs(batched, LIBERO_SIM_SPEC)
    assert converted["video.image"].shape == (1, 1, IMG, IMG, 3)
    assert converted["state.state"].shape == (1, 1, 8)


def test_env_builder_registered_for_libero_dreamzero_prefix():
    from isaac_rl_posttraining.envs.providers import get_env_builder

    builder = get_env_builder("libero_dreamzero/libero_spatial/0")
    assert builder.__module__ == "simulators.libero.policies.dreamzero"


def test_launcher_dreamzero_dryrun_plan(tmp_path):
    launcher = (
        REPO_ROOT / "example_workflows" / "libero_posttraining" / "scripts" / "train.sh"
    )
    result = subprocess.run(
        ["bash", str(launcher), "0"],
        env={
            "PATH": "/usr/bin:/bin",
            "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
            "LEARNER_PYTHON": sys.executable,
            "BASE_MODEL": "dreamzero_local",
            "SUITE": "libero_spatial",
            "POSTTRAIN_BASE_CKPT": "/ckpt/dreamzero_5b_libero_spatial",
            "LIBERO_CONFIG_PATH": str(tmp_path),
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = result.stdout
    for expected in (
        "launch.override=base_model=dreamzero_local",
        "launch.override=env.env_name=libero_dreamzero/libero_spatial/0",
        # The RLinf-validated geometry: one 32-dim latent per 16-step chunk,
        # the whole chunk executes, 240-step episodes (15 decisions).
        "launch.override=dsrl.noise_dim=32",
        "launch.override=dsrl.action_exec_len=16",
        "launch.override=training.chunk_size=null",
        "launch.override=env.max_episode_steps=240",
        "launch.override=base_model.embodiment_tag=libero_sim",
        # The 5B/LIBERO inference resolution the recipe pins.
        "launch.override=base_model.target_video_height=160",
        "launch.override=base_model.target_video_width=320",
        # Reference cadence: ~1.67 updates/decision -> 25 per episode, small
        # replay with heavy reuse, mean-aggregated 10-critic ensemble.
        "launch.override=training.train_after_episode=true",
        "launch.override=algorithm.num_updates_per_train_step=25",
        "launch.override=algorithm.batch_size=512",
        "launch.override=algorithm.gamma=0.999",
        "launch.override=algorithm.tau=0.005",
        "launch.override=algorithm.target_entropy=-16",
        "launch.override=algorithm.num_critics=10",
        "launch.override=algorithm.critic_reduction=mean",
        "launch.override=algorithm.critic_target_reduction=mean",
        "launch.override=buffer.capacity=1500",
        "launch.override=dsrl.warmup_exploration=normal",
        "launch.override=env.use_init_states=true",
        "launch.override=eval.eval_max_steps=240",
    ):
        assert expected in text, f"missing {expected}"
