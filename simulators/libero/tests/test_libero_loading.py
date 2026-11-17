# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO real-boot integration: env construction + contract check (marked libero).

Boots one LIBERO-Spatial task headless through :class:`LiberoEnv` (MuJoCo
CPU physics -- no GPU required; rendering uses ``MUJOCO_GL=egl`` or an
OSMesa fallback) and replays the observation-contract checks from the
verification section of ``simulators/libero/README.md``:

* ``reset`` yields an upright, non-black uint8 camera frame, a finite
  float32 state vector, and the language prompt,
* a 20-step random rollout keeps the contract shape and plumbs the success
  flag through ``info``,
* chunked actions execute sequentially,
* ``close`` is clean and idempotent.

Skips wherever the LIBERO stack is not installed or not configured (see
``simulators/libero/README.md`` for both).
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.libero

SUITE = "libero_spatial"
IMAGE_SIZE = 128
ACTION_DIM = 7


def _require_libero():
    """Skip unless the LIBERO stack is installed AND configured.

    The configuration check must come before any ``libero.libero`` import:
    on an unconfigured machine that import blocks on an interactive prompt.
    """
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    config_dir = os.environ.get("LIBERO_CONFIG_PATH") or os.path.expanduser("~/.libero")
    if not os.path.isfile(os.path.join(config_dir, "config.yaml")):
        pytest.skip(
            "LIBERO not configured (no config.yaml; see simulators/libero/README.md step 4)"
        )


@pytest.fixture(scope="module")
def libero_env():
    _require_libero()
    os.environ.setdefault("MUJOCO_GL", "egl")
    from simulators.libero.env import LiberoEnv

    env = LiberoEnv(suite=SUITE, task_id=0, image_size=IMAGE_SIZE, seed=0)
    yield env
    env.close()


def test_provider_resolves_to_libero_env(libero_env):
    from isaac_rl_posttraining.envs import get_provider

    assert get_provider("libero") is type(libero_env)
    assert (
        "libero"
        in __import__(
            "isaac_rl_posttraining.envs", fromlist=["available_providers"]
        ).available_providers()
    )


def test_reset_obs_contract(libero_env):
    obs, info = libero_env.reset(seed=0, options={"init_state_id": 0})
    assert set(obs) == {"agentview_image", "state", "prompt"}

    img = np.asarray(obs["agentview_image"])
    assert img.dtype == np.uint8 and img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert img.max() > 0 and img.std() > 1.0, (
        "camera frame is (near-)black: no usable GL backend "
        "(check MUJOCO_GL; see simulators/libero/README.md step 5)"
    )

    state = np.asarray(obs["state"])
    assert state.dtype == np.float32 and state.shape == (9,)
    assert np.all(np.isfinite(state))

    assert isinstance(obs["prompt"], str) and obs["prompt"]
    assert info["suite"] == SUITE and info["init_state_id"] == 0


def test_random_rollout_keeps_contract(libero_env):
    libero_env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(20):
        action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        obs, reward, terminated, truncated, info = libero_env.step(action)
        assert set(obs) == {"agentview_image", "state", "prompt"}
        assert obs["agentview_image"].dtype == np.uint8
        assert obs["state"].dtype == np.float32 and np.all(np.isfinite(obs["state"]))
        assert np.isfinite(float(reward))
        assert isinstance(info["success"], bool) and isinstance(
            info["is_success"], bool
        )
        assert info["num_steps"] == 1
        if terminated or truncated:
            break


def test_chunked_action_executes_sequentially(libero_env):
    libero_env.reset(seed=2)
    chunk = np.zeros((4, ACTION_DIM), dtype=np.float32)
    obs, _reward, terminated, truncated, info = libero_env.step(chunk)
    if not (terminated or truncated):
        assert info["num_steps"] == 4
    assert set(obs) == {"agentview_image", "state", "prompt"}


def test_close_is_clean_and_idempotent(libero_env):
    libero_env.close()
    libero_env.close()  # second close must be a no-op
    with pytest.raises(RuntimeError, match="closed"):
        libero_env.reset()
