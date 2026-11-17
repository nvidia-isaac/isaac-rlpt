# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO pi0 adapter and base model geometry (no openpi required).

Pins:
1. The pi0 observation conventions (key renames, 180-degree training
   orientation, 224x224 pad-resize, 8-dim state passthrough).
2. The finish-fast reward shift (-1 per step, 0 on success).
3. The pi0 base model's latent geometry: a 32-dim padded latent space over
   7-dim decoded actions, broadcast and tiling noise modes.
4. The launcher's BASE_MODEL=pi0 override set (dry-run plan).
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
            "prompt": "put both moka pots on the stove",
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


def _adapter(monkeypatch):
    from simulators.libero.policies import pi0 as libero_pi0

    # openpi_client is optional; the pad-resize on square frames is a plain
    # resize, so the stub keeps the test dependency-free.
    monkeypatch.setattr(
        libero_pi0,
        "_resize_with_pad",
        lambda image, size: np.asarray(image, dtype=np.uint8)[:size, :size],
    )
    return libero_pi0.LiberoPi0Adapter(_ScriptedInnerEnv())


def test_adapter_renames_and_flips(monkeypatch):
    env = _adapter(monkeypatch)
    obs, _ = env.reset()
    assert set(obs) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    # Upright frame's top-left marker lands at top-RIGHT after the horizontal flip.
    assert tuple(obs["observation/image"][0, -1]) == (255, 0, 0)
    assert np.array_equal(obs["observation/state"], np.arange(8, dtype=np.float32))


def test_adapter_reward_is_finish_fast(monkeypatch):
    env = _adapter(monkeypatch)
    env.env._succeed_on_step = 2
    env.reset()
    _, r1, done1, _, info1 = env.step(np.zeros(7, dtype=np.float32))
    assert (r1, done1, info1["success"]) == (-1.0, False, False)
    _, r2, done2, _, info2 = env.step(np.zeros(7, dtype=np.float32))
    assert (r2, done2, info2["success"]) == (0.0, True, True)


def test_adapter_passes_actions_through_unchanged(monkeypatch):
    env = _adapter(monkeypatch)
    env.reset()
    action = np.linspace(-1, 1, 7).astype(np.float32)
    env.step(action)
    assert np.allclose(env.env.last_action, action)


def test_adapter_key_mapping_targets_scene_camera(monkeypatch):
    env = _adapter(monkeypatch)
    assert env.steering_key_mapping == {
        "state": "observation/state",
        "language": None,
        "image": ["observation/image"],
    }


def test_language_instruction_survives_vector_wrapping(monkeypatch):
    """The trainer reads the task language through the full wrapper chain.

    The training env is VectorWrapper(SyncVectorEnv([adapter(inner)])) and
    vector wrappers forward neither attributes nor ``envs``; the helper
    must unwrap to the inner env's ``language_instruction``.
    """
    import gymnasium.vector as gym_vector

    from isaac_rl_posttraining.runners.off_policy_launch import (
        _env_language_instruction,
    )

    def _make():
        env = _adapter(monkeypatch)
        env.unwrapped.language_instruction = "put both moka pots on the stove"
        return env

    class _PassthroughVectorWrapper(gym_vector.VectorWrapper):
        pass

    env = _PassthroughVectorWrapper(gym_vector.SyncVectorEnv([_make]))
    assert _env_language_instruction(env) == "put both moka pots on the stove"
    assert _env_language_instruction(_ScriptedInnerEnv()) is None


class _StubPi0Policy:
    """Stands in for Pi0Policy: records noise, returns [B, 50, 7] actions."""

    action_horizon = 50

    def __init__(self):
        self.received = []

    def infer(self, observations, noise):
        self.received.append(noise)
        batch = 1
        return {"actions": np.zeros((batch, self.action_horizon, 7), dtype=np.float32)}


def test_base_model_latent_geometry_padded_space():
    from isaac_rl_posttraining.policies.pi0.base_model import Pi0BaseModelPolicy

    base_model = Pi0BaseModelPolicy(_StubPi0Policy(), action_dim=7, latent_dim=32)
    assert (
        base_model.action_horizon,
        base_model.action_dim,
        base_model.latent_dim,
    ) == (
        50,
        7,
        32,
    )

    out = base_model.infer(
        {"observation/state": np.zeros((1, 8))},
        noise=np.zeros((1, 32), dtype=np.float32),
    )
    assert out["actions"].shape == (1, 50, 7)
    out = base_model.infer(
        {"observation/state": np.zeros((1, 8))},
        noise=np.zeros((1, 20, 32), dtype=np.float32),
    )
    assert out["actions"].shape == (1, 50, 7)
    with pytest.raises(ValueError):
        base_model.infer(
            {}, noise=np.zeros((1, 7), dtype=np.float32)
        )  # wrong latent width
    with pytest.raises(ValueError):
        base_model.infer({}, noise=np.zeros((1, 51, 32), dtype=np.float32))  # K > H


def test_base_model_clips_injected_latents():
    from isaac_rl_posttraining.policies.pi0.base_model import Pi0BaseModelPolicy

    stub = _StubPi0Policy()
    base_model = Pi0BaseModelPolicy(stub, action_dim=7, latent_dim=32)
    base_model.infer({}, noise=np.full((1, 32), 100.0, dtype=np.float32))
    assert np.abs(stub.received[-1]).max() <= 3.0


def test_policy_task_fills_missing_prompt():
    """Vectorized envs drop the prompt string; ``task`` must fill it in."""
    torch = pytest.importorskip("torch")
    from isaac_rl_posttraining.policies.pi0.policy import Pi0Policy

    class _Recorder:
        action_horizon = 50

        def __init__(self):
            self.seen = []

        def infer(self, observations, noise=None):
            self.seen.append(dict(observations))
            return {"actions": np.zeros((1, 50, 7), dtype=np.float32)}

    policy = Pi0Policy.__new__(Pi0Policy)  # skip checkpoint loading
    policy.policy = _Recorder()
    policy.device = torch.device("cpu")
    policy.task = None

    obs = {"observation/state": np.zeros((1, 8), dtype=np.float32)}
    policy.infer(obs, None)
    assert "prompt" not in policy.policy.seen[-1]  # no task, nothing injected

    policy.task = "put both moka pots on the stove"
    policy.infer(obs, None)
    assert policy.policy.seen[-1]["prompt"] == "put both moka pots on the stove"
    assert "prompt" not in obs  # input dict not mutated

    policy.infer({**obs, "prompt": "explicit"}, None)
    assert policy.policy.seen[-1]["prompt"] == "explicit"  # never overrides


def test_base_model_exposes_policy_for_task_sync():
    from isaac_rl_posttraining.policies.pi0.base_model import Pi0BaseModelPolicy

    stub = _StubPi0Policy()
    base_model = Pi0BaseModelPolicy(stub, action_dim=7, latent_dim=32)
    assert base_model.policy is stub  # the trainer syncs task onto base_model.policy


def test_warmup_exploration_normal_is_unbounded_gaussian():
    """'normal' warmup draws N(0,1) latents (the flow's native noise), not
    box-bounded uniforms; shape and batch dim match the uniform path."""
    from isaac_rl_posttraining.envs.steering_wrappers import LatentSpaceEnv

    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)}
    )
    kwargs = {
        "action_space": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
        "base_model": None,
        "noise_dim": 32,
        "chunk_size": None,
        "action_bound": 1.0,
        "use_vlm_features": False,
    }
    normal_env = LatentSpaceEnv(obs_space, warmup_exploration="normal", **kwargs)
    samples = np.stack([normal_env.sample_action() for _ in range(200)])
    assert samples.shape == (200, 1, 32)
    assert np.abs(samples).max() > 1.0  # gaussian tails exceed the box bound
    assert abs(samples.std() - 1.0) < 0.1

    uniform_env = LatentSpaceEnv(obs_space, warmup_exploration="uniform", **kwargs)
    assert np.abs(uniform_env.sample_action()).max() <= 1.0

    with pytest.raises(ValueError):
        LatentSpaceEnv(obs_space, warmup_exploration="sobol", **kwargs)


def test_launcher_pi0_dryrun_plan(tmp_path):
    launcher = (
        REPO_ROOT / "example_workflows" / "libero_posttraining" / "scripts" / "train.sh"
    )
    result = subprocess.run(
        ["bash", str(launcher), "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"],
        env={
            "PATH": "/usr/bin:/bin",
            "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
            "LEARNER_PYTHON": sys.executable,
            "BASE_MODEL": "pi0",
            "TRAIN_MAXSTEPS": "520",
            "POSTTRAIN_BASE_CKPT": "gs://openpi-assets/checkpoints/pi0_libero",
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
        "launch.override=base_model=pi0",
        "launch.override=env.env_name=libero_pi0/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "launch.override=dsrl.noise_dim=32",
        "launch.override=dsrl.action_exec_len=20",
        "launch.override=training.chunk_size=null",
        # The official trajwise cadence: 20 full SAC updates per decision.
        "launch.override=training.train_after_episode=true",
        "launch.override=algorithm.num_updates_per_train_step=20",
        "launch.override=algorithm.num_critic_updates_per_actor_update=1",
        "launch.override=algorithm.batch_size=256",
        "launch.override=algorithm.learning_starts=10000",
        # N(0,1) warmup + REDQ pessimism (10 critics, min over 2) replace
        # the official CNN-with-augmentation regularization (see launcher).
        "launch.override=dsrl.warmup_exploration=normal",
        "launch.override=algorithm.num_critics=10",
        "launch.override=algorithm.n_critics_to_sample=2",
        "launch.override=algorithm.critic_target_reduction=min",
        # The references bootstrap through time-limit cuts (audit P1).
        "launch.override=algorithm.bootstrap_at_truncation=true",
        "launch.override=algorithm.target_entropy=-16",
        "launch.override=base_model.action_dim=7",
        # Official run protocol (nakamotoo wandb config): buffer window is
        # max_steps/20 decisions, evals every 10k decisions at N=10.
        "launch.override=buffer.capacity=25000",
        # Official episodes are bare BDDL resets, not benchmark init states
        # (train_sim.py never calls set_init_state; base reads ~20-40% there
        # vs 2% on the benchmark's fixed init states).
        "launch.override=env.use_init_states=false",
        "launch.override=eval.eval_freq=200000",
        "launch.override=eval.eval_num_episodes=10",
        "launch.override=training.save_interval=200000",
    ):
        assert expected in text, f"missing {expected}"


def test_policy_collapses_batched_prompt_to_single():
    """A multi-env observation's per-env prompts collapse to one (openpi is single-prompt)."""
    import numpy as np

    torch = pytest.importorskip("torch")
    from isaac_rl_posttraining.policies.pi0.policy import Pi0Policy

    class _Recorder:
        action_horizon = 50

        def __init__(self):
            self.seen = []

        def infer(self, observations, noise=None):
            self.seen.append(dict(observations))
            return {"actions": np.zeros((1, 50, 7), dtype=np.float32)}

    policy = Pi0Policy.__new__(Pi0Policy)
    policy.policy = _Recorder()
    policy.device = torch.device("cpu")
    policy.task = None
    task = "pick up the cream cheese and put it in the tray"
    obs = {
        "observation/state": np.zeros((8, 8), dtype=np.float32),
        "prompt": np.array([task] * 8),
    }
    policy.infer(obs, None)
    seen = policy.policy.seen[-1]["prompt"]
    assert isinstance(seen, str) or getattr(seen, "ndim", 1) == 0
    assert str(seen) == task
