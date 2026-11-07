# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the parallel vector-env backends (envs/vector_backends.py).

Covers the shared gym.vector-style contract for:
  * SubprocVectorEnv  (Mode 2 — N single-robot envs in N worker processes)
  * BatchedEnvAdapter (Mode 1 — one natively [N]-batched env)
  * build_vector_env  (the single-knob factory + its helpful errors)

Uses tiny mock envs — no LIBERO / GPU needed.
"""

import gymnasium as gym
import numpy as np
import pytest

from isaac_rl_posttraining.envs.vector_backends import (
    VECTOR_BACKENDS,
    BatchedEnvAdapter,
    SubprocVectorEnv,
    build_vector_env,
)


# ---------------------------------------------------------------------------
# mock single-robot env (dict obs incl. a uint8 image, like LIBERO)
# ---------------------------------------------------------------------------
class DictCountingEnv(gym.Env):
    """Single-robot env: terminates after ``ep_len`` steps. Deterministic per seed.

    obs = {"state": [t, seed_tag], "image": uint8[4,4,3]}.  Must be importable at
    module scope so ``spawn`` workers can reconstruct it.
    """

    def __init__(self, ep_len: int = 5):
        self.ep_len = ep_len
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
                "image": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0
        self._seed_tag = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed_tag = int(seed)
        self._t = 0
        return self._obs(), {"reset": True}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.ep_len
        return (
            self._obs(),
            float(self._t),
            terminated,
            False,
            {"t": self._t, "seed_tag": self._seed_tag},
        )

    def _obs(self):
        return {
            "state": np.array([self._t, self._seed_tag], dtype=np.float32),
            "image": np.full((4, 4, 3), self._t % 256, dtype=np.uint8),
        }


def _make_env_fn(ep_len):
    def _f():
        return DictCountingEnv(ep_len=ep_len)

    return _f


# ---------------------------------------------------------------------------
# mock natively-batched env (Mode 1), supports masked reset, no internal autoreset
# ---------------------------------------------------------------------------
class MockBatchedEnv:
    def __init__(self, num_envs=4, ep_lens=None):
        self.num_envs = num_envs
        self.ep_lens = np.asarray(ep_lens if ep_lens is not None else [3] * num_envs)
        self.single_observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)}
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = np.zeros(num_envs, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        mask = None
        if options and "reset_mask" in options:
            mask = np.asarray(options["reset_mask"], dtype=bool)
        if mask is None:
            self._t[:] = 0
        else:
            self._t[mask] = 0
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape[0] == self.num_envs, (
            "batched env must get one action per env"
        )
        self._t += 1
        term = self._t >= self.ep_lens
        rew = self._t.astype(np.float64)
        trunc = np.zeros(self.num_envs, dtype=bool)
        return self._obs(), rew, term, trunc, {}

    def _obs(self):
        return {"state": self._t.reshape(-1, 1).astype(np.float32)}


# ---------------------------------------------------------------------------
# SubprocVectorEnv (Mode 2)
# ---------------------------------------------------------------------------
def test_subproc_interface_and_seeding():
    env = SubprocVectorEnv([_make_env_fn(3) for _ in range(3)], context="spawn")
    try:
        assert env.num_envs == 3
        assert env.single_observation_space is not None
        assert env.single_action_space is not None

        obs, info = env.reset(seed=100)
        # batched obs: [N, ...]
        assert obs["state"].shape == (3, 2)
        assert obs["image"].shape == (3, 4, 4, 3)
        # worker i seeded base_seed + i
        np.testing.assert_array_equal(obs["state"][:, 1], [100, 101, 102])
        # infos is one dict per env
        assert isinstance(info, list) and len(info) == 3

        acts = np.zeros((3, 2), dtype=np.float32)
        obs, rew, term, trunc, info = env.step(acts)
        assert obs["state"].shape == (3, 2)
        assert rew.shape == (3,) and term.shape == (3,) and trunc.shape == (3,)
        assert isinstance(info, list) and len(info) == 3
    finally:
        env.close()


def test_subproc_next_step_autoreset():
    # ep_len=3 for all: step 3 -> all terminate; step 4 -> all auto-reset (NEXT_STEP)
    env = SubprocVectorEnv([_make_env_fn(3) for _ in range(3)], context="spawn")
    try:
        env.reset(seed=0)
        acts = np.zeros((3, 2), dtype=np.float32)
        for _ in range(2):
            _, _, term, _, _ = env.step(acts)
            assert not term.any()
        _, rew, term, trunc, _ = env.step(acts)  # 3rd step -> terminate
        assert term.all()
        assert np.allclose(rew, 3.0)

        # next step: NEXT_STEP auto-reset (action ignored, fresh obs, reward 0, not done)
        obs, rew, term, trunc, _ = env.step(acts)
        assert not term.any() and not trunc.any()
        assert np.allclose(rew, 0.0)
        np.testing.assert_array_equal(obs["state"][:, 0], [0, 0, 0])  # t reset to 0
    finally:
        env.close()


def test_subproc_masked_reset():
    env = SubprocVectorEnv([_make_env_fn(10) for _ in range(3)], context="spawn")
    try:
        env.reset(seed=0)
        acts = np.zeros((3, 2), dtype=np.float32)
        env.step(acts)
        obs, *_ = env.step(acts)
        np.testing.assert_array_equal(obs["state"][:, 0], [2, 2, 2])

        # reset ONLY env 1
        obs, _ = env.reset(options={"reset_mask": np.array([False, True, False])})
        np.testing.assert_array_equal(obs["state"][:, 0], [2, 0, 2])
    finally:
        env.close()


# ---------------------------------------------------------------------------
# BatchedEnvAdapter (Mode 1)
# ---------------------------------------------------------------------------
def test_batched_adapter_interface_and_next_step_autoreset():
    adapter = BatchedEnvAdapter(MockBatchedEnv(num_envs=4, ep_lens=[3, 3, 3, 3]))
    assert adapter.num_envs == 4
    assert adapter.single_observation_space is not None

    obs, _ = adapter.reset()
    assert obs["state"].shape == (4, 1)
    np.testing.assert_array_equal(obs["state"][:, 0], [0, 0, 0, 0])

    acts = np.zeros((4, 2), dtype=np.float32)
    for _ in range(2):
        _, _, term, _, _ = adapter.step(acts)
        assert not term.any()
    obs, rew, term, trunc, _ = adapter.step(acts)  # terminate
    assert term.all()
    assert np.allclose(rew, 3.0)
    # 4th step: adapter emulates NEXT_STEP auto-reset for the just-done envs
    obs, rew, term, trunc, _ = adapter.step(acts)
    assert not term.any() and not trunc.any()
    assert np.allclose(rew, 0.0)
    np.testing.assert_array_equal(obs["state"][:, 0], [0, 0, 0, 0])


def test_batched_adapter_async_boundaries():
    # staggered episode lengths -> envs finish at different steps (async boundaries)
    adapter = BatchedEnvAdapter(MockBatchedEnv(num_envs=3, ep_lens=[2, 4, 6]))
    adapter.reset()
    acts = np.zeros((3, 2), dtype=np.float32)

    _, _, term, _, _ = adapter.step(acts)  # step 1
    assert not term.any()
    _, _, term, _, _ = adapter.step(acts)  # step 2: env0 done
    assert list(term) == [True, False, False]
    # env0 auto-resets on the next step (reset-only, NOT stepped); env1/env2 keep going
    obs, rew, term, _, _ = adapter.step(acts)  # env0 -> reset t=0; env1 t=3; env2 t=3
    # env0 was auto-reset this step (action ignored)
    assert rew[0] == pytest.approx(0.0)
    assert not term.any()
    np.testing.assert_array_equal(obs["state"][:, 0], [0, 3, 3])


# ---------------------------------------------------------------------------
# factory + validation
# ---------------------------------------------------------------------------
def test_factory_sync_returns_gym_syncvectorenv():
    env = build_vector_env("sync", env_fns=[_make_env_fn(3) for _ in range(2)])
    assert isinstance(env, gym.vector.SyncVectorEnv)
    env.close()


def test_factory_async_returns_subproc():
    env = build_vector_env(
        "async", env_fns=[_make_env_fn(3) for _ in range(2)], num_envs=2
    )
    try:
        assert isinstance(env, SubprocVectorEnv)
    finally:
        env.close()


def test_factory_batched_returns_adapter():
    env = build_vector_env("batched", batched_env=MockBatchedEnv(num_envs=4))
    assert isinstance(env, BatchedEnvAdapter)


def test_factory_validation():
    """Unknown backends are named with the alternatives; each backend refuses a call without its inputs."""
    with pytest.raises(ValueError) as e:
        build_vector_env("turbo", env_fns=[_make_env_fn(3)])
    assert "turbo" in str(e.value)
    for name in VECTOR_BACKENDS:
        assert name in str(e.value)
    with pytest.raises(ValueError) as e:
        build_vector_env("batched")
    assert "batched" in str(e.value).lower()
    with pytest.raises(ValueError):
        build_vector_env("sync")
    with pytest.raises(ValueError):
        build_vector_env(
            "sync", env_fns=[_make_env_fn(3) for _ in range(3)], num_envs=5
        )
