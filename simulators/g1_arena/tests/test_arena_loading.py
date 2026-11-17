# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arena env-server boot + client contract (cluster integration, marked arena/gpu).

Boots ``simulators/g1_arena/env_server.py`` from the EXTERNAL IsaacLab-Arena
checkout (``ISAACLAB_ARENA_PATH``) under the arena venv's python, through a
PTY (Isaac Sim dies silently without a TTY), with the EULA env vars exported
and a generous boot timeout (cold boots take 1-3 minutes). Then pins:

* the readiness line (``listening on``) appears,
* ``reset()`` returns the obs contract: a native 480x640 uint8 ego image that
  is not black, and a finite state vector,
* one 35-dim ``step()`` returns a 5-tuple with ``success`` in info,
* SIGTERM shuts the server down cleanly.

These tests never run in CI; they skip without an arena checkout or an idle
GPU (running jobs are never disturbed).
"""

import os

import numpy as np
import pytest
from conftest import (
    boot_arena_server,
    require_arena_prereqs,
    stop_process,
)

from simulators.g1_arena.env import ArenaEnv

pytestmark = [
    pytest.mark.arena,
    pytest.mark.gpu,
    pytest.mark.timeout(0),
]  # boots Isaac Sim

PORT = int(os.environ.get("ISAAC_RL_POSTTRAINING_TEST_ARENA_PORT", "7397"))
TASK = "LMDrillLift"
NATIVE_HW = (480, 640)
ACTION_DIM = 35


@pytest.fixture(scope="module")
def arena_server(tmp_path_factory):
    arena = require_arena_prereqs()
    log_path = str(tmp_path_factory.mktemp("arena_logs") / "server.log")
    proc = boot_arena_server(arena, PORT, log_path, task=TASK, image_resolution=0)
    yield {"proc": proc, "log": log_path, "port": PORT}
    stop_process(proc)


def _connect_client(port):
    """Build the ArenaEnv client against a live server."""
    return ArenaEnv(
        server_url=f"tcp://127.0.0.1:{port}",
        image_resolution=0,
        action_dim=ACTION_DIM,
        max_episode_steps=450,
    )


@pytest.fixture(scope="module")
def arena_env(arena_server):
    env = _connect_client(arena_server["port"])
    yield env
    try:
        env.close()
    # Any failure means unavailable here.
    except Exception:  # noqa: BLE001, S110
        pass


def test_server_boots_with_readiness_line(arena_server):
    """The env server advertises readiness on stdout ('listening on')."""
    with open(arena_server["log"], "rb") as fh:
        log = fh.read().decode("utf-8", errors="replace")
    assert "listening on" in log, f"readiness line missing; log tail:\n{log[-4000:]}"


def test_reset_obs_contract(arena_env):
    """reset() yields a native, non-black ego image and a finite state vector."""
    obs, info = arena_env.reset()
    assert isinstance(obs, dict) and isinstance(info, dict)
    assert "ego_view_image" in obs and "state" in obs, f"obs keys: {sorted(obs)}"

    img = np.asarray(obs["ego_view_image"])
    assert img.dtype == np.uint8, f"ego image must be uint8, got {img.dtype}"
    assert img.shape == (*NATIVE_HW, 3), (
        f"ego image must be native {NATIVE_HW} (image-resolution 0; pre-resizing is OOD "
        f"for the policy), got {img.shape}"
    )
    assert img.max() > 0, (
        "ego image is all-black: rendering/asset loading failed (check EULA env vars, "
        "texture streaming, and asset sources)"
    )

    state = np.asarray(obs["state"], dtype=np.float32)
    assert state.ndim == 1 and state.size > 0
    assert np.all(np.isfinite(state)), "state vector contains non-finite values"


def test_step_35dim_five_tuple(arena_env):
    """One 35-dim WBC joint action steps the sim and returns the gym 5-tuple."""
    arena_env.reset()
    result = arena_env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert isinstance(result, tuple) and len(result) == 5, (
        f"step must return a 5-tuple, got {type(result)}"
    )
    obs, reward, done, truncated, info = result
    assert isinstance(obs, dict) and "ego_view_image" in obs and "state" in obs
    assert np.isfinite(float(reward))
    assert isinstance(done, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and "success" in info, (
        f"info must expose success, got {sorted(info)}"
    )


def test_sigterm_clean_shutdown(arena_server):
    """SIGTERM must shut the server down within its teardown grace period."""
    proc = arena_server["proc"]
    rc = stop_process(proc, term_timeout=120)
    assert rc is not None, "server did not exit after SIGTERM (teardown hang)"
    with open(arena_server["log"], "rb") as fh:
        log = fh.read().decode("utf-8", errors="replace")
    assert "SIGTERM" in log or rc == 0, (
        f"expected a clean SIGTERM shutdown (log note or rc 0); rc={rc}, log tail:\n{log[-2000:]}"
    )
