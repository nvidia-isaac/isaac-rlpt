# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment wrappers: the steering observation contract (``SteeringObsWrapper``) and open-loop chunk
execution (``ActionChunkingWrapper``), cross-checked against the shipped canonical config
(``training.chunk_size`` / ``dsrl.action_exec_len``), plus the latent -> base-model-decode -> in-order env
execution route via the in-tree ``MockBaseModelPolicy``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs import ActionChunkingWrapper, SteeringObsWrapper
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_YAML = REPO_ROOT / "isaac_rl_posttraining" / "configs" / "train.yaml"

ACTION_DIM = 35
PROMPT = "lift the drill from the table"


class _StubArenaEnv(gym.Env):
    """Arena-shaped stub: dict obs with prompt, records executed actions."""

    def __init__(self, with_mapping: bool = True, action_dim: int = ACTION_DIM):
        self.observation_space = gym.spaces.Dict(
            {
                "ego_view_image": gym.spaces.Box(0, 255, (8, 10, 3), np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (43,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (action_dim,), np.float32)
        if with_mapping:
            self.steering_key_mapping = {"state": "state", "image": ["ego_view_image"]}
        self.received: list[np.ndarray] = []
        self._image = (
            (np.arange(8 * 10 * 3, dtype=np.int64) % 255)
            .astype(np.uint8)
            .reshape(8, 10, 3)
        )
        self._state = np.linspace(0.0, 1.0, 43).astype(np.float32)

    def _obs(self) -> dict:
        return {
            "ego_view_image": self._image.copy(),
            "state": self._state.copy(),
            "prompt": PROMPT,
        }

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        self.received.append(np.asarray(action).copy())
        return self._obs(), 0.0, False, False, {}


def _canonical_chunking() -> tuple[int, int]:
    """(chunk_size, action_exec_len) from the SHIPPED canonical yaml."""
    if not CANONICAL_YAML.is_file():
        pytest.fail(f"canonical config missing at {CANONICAL_YAML}", pytrace=False)
    cfg = OmegaConf.load(CANONICAL_YAML)
    return int(cfg.training.chunk_size), int(cfg.dsrl.action_exec_len)


# ---------------------------------------------------------------------------
# SteeringObsWrapper: observation contract
# ---------------------------------------------------------------------------


def test_steering_obs_wrapper_contract():
    """State and image pass through untouched, the prompt survives every step, language is a 384-d zero
    embedding without a sentence model, the key mapping is exposed, and an env without one is refused."""
    env = _StubArenaEnv()
    wrapper = SteeringObsWrapper(env, sentence_model=None)
    obs, info = wrapper.reset()
    assert {"ego_view_image", "state", "language", "prompt"} <= set(obs) and isinstance(
        info, dict
    )
    np.testing.assert_array_equal(obs["state"], env._state)
    np.testing.assert_array_equal(obs["ego_view_image"], env._image)
    assert obs["state"].dtype == np.float32 and obs["ego_view_image"].dtype == np.uint8
    assert obs["language"].shape == (384,) and obs["language"].dtype == np.float32
    np.testing.assert_array_equal(obs["language"], np.zeros((384,), dtype=np.float32))
    assert obs["prompt"] == PROMPT
    obs, _, _, _, _ = wrapper.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert obs["prompt"] == PROMPT
    assert wrapper.steering_key_mapping == {
        "state": "state",
        "image": ["ego_view_image"],
    }
    with pytest.raises((AssertionError, ValueError, TypeError, AttributeError)):
        SteeringObsWrapper(_StubArenaEnv(with_mapping=False))


# ---------------------------------------------------------------------------
# ActionChunkingWrapper: open-loop chunk execution
# ---------------------------------------------------------------------------


def test_chunk_consumes_exactly_exec_len_steps():
    """With the shipped chunking config (chunk 40, exec 8), one decoded chunk
    drives exactly action_exec_len env steps, in order, then re-arms."""
    chunk_size, exec_len = _canonical_chunking()
    assert (chunk_size, exec_len) == (40, 8)  # canonical pin, plan §3 file 1
    env = _StubArenaEnv()
    wrapper = ActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=exec_len)
    chunk = np.arange(chunk_size * ACTION_DIM, dtype=np.float32).reshape(
        chunk_size, ACTION_DIM
    )

    for step in range(exec_len):
        assert wrapper.is_chunk_empty == (step == 0)
        wrapper.step(chunk)
    assert wrapper.is_chunk_empty
    assert len(env.received) == exec_len
    for step in range(exec_len):
        np.testing.assert_array_equal(env.received[step], chunk[step])
    # The next step starts a fresh chunk from row 0 (never row exec_len).
    wrapper.step(chunk)
    np.testing.assert_array_equal(env.received[exec_len], chunk[0])


def test_partial_chunk_execution():
    env = _StubArenaEnv()
    wrapper = ActionChunkingWrapper(env, chunk_size=8, n_action_steps=4)
    chunk = np.arange(8 * ACTION_DIM, dtype=np.float32).reshape(8, ACTION_DIM)
    wrapper.step(chunk)
    wrapper.step(chunk)
    assert not wrapper.is_chunk_empty  # 2 of 4 executed
    wrapper.reset()
    assert wrapper.is_chunk_empty  # reset discards the partial chunk
    wrapper.step(chunk)
    _, _, after_reset = env.received
    np.testing.assert_array_equal(after_reset, chunk[0])  # restarts at row 0


def test_latent_routes_through_stub_base_model_in_order():
    """latent -> BaseModelPolicy.infer -> decoded chunk -> env steps, in order."""
    horizon, act_dim = 8, 5
    base_model = MockBaseModelPolicy(action_horizon=horizon, action_dim=act_dim, seed=3)
    latent = np.linspace(-1.0, 1.0, horizon * act_dim, dtype=np.float64).reshape(
        1, horizon, act_dim
    )
    decoded = base_model.infer({}, latent)["actions"][0]  # [H, A], deterministic

    env = _StubArenaEnv(action_dim=act_dim)
    exec_len = 4
    wrapper = ActionChunkingWrapper(env, chunk_size=horizon, n_action_steps=exec_len)
    for _ in range(exec_len):
        wrapper.step(decoded)

    assert len(env.received) == exec_len
    for step in range(exec_len):
        np.testing.assert_allclose(env.received[step], decoded[step], rtol=0, atol=1e-6)
    # Steered residuals survive the route: recover the injected latent from the
    # position-coded mock decode for the executed steps.
    position_code = (
        10.0 * np.arange(horizon, dtype=np.float64)[:, None]
        + 1000.0 * np.arange(act_dim, dtype=np.float64)[None, :]
    )
    executed = np.stack(env.received).astype(np.float64)
    np.testing.assert_allclose(
        executed - position_code[:exec_len], latent[0, :exec_len], atol=1e-3
    )
