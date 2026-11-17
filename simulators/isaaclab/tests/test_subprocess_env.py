# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipe protocol between a rollout worker and its Isaac Lab child process."""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
from multiprocessing import spawn

import numpy as np
import pytest

from simulators.isaaclab.subprocess_env import (
    IsaacLabEnvSpec,
    _without_main_module_replay,
    serve_environment,
)


class _FakeEnv:
    def __init__(self) -> None:
        self.closed = False
        self.actions = []

    def reset(self, seed=None):
        self.seeds = getattr(self, "seeds", []) + [seed]
        return {"video.image": np.zeros((2, 4, 4, 3), dtype=np.uint8)}, {}

    def step(self, actions):
        self.actions.append(np.asarray(actions))
        if np.isnan(actions).any():
            raise ValueError("bad action")
        return (
            {"video.image": np.ones((2, 4, 4, 3), dtype=np.uint8)},
            np.zeros(2),
            np.zeros(2, bool),
            np.zeros(2, bool),
            [{}, {}],
        )

    def close(self):
        self.closed = True


def _serve_in_thread(env):
    parent, child = mp.Pipe(duplex=True)
    thread = threading.Thread(target=serve_environment, args=(child, env), daemon=True)
    thread.start()
    return parent, thread


def test_serve_environment_answers_reset_step_and_close():
    env = _FakeEnv()
    parent, thread = _serve_in_thread(env)
    parent.send(("reset", None))
    status, (obs, _info) = parent.recv()
    assert status == "ok" and obs["video.image"].shape == (2, 4, 4, 3)
    parent.send(("reset", 7))
    status, _ = parent.recv()
    assert status == "ok" and env.seeds == [None, 7]
    parent.send(("step", np.ones((2, 7), dtype=np.float32)))
    status, result = parent.recv()
    assert status == "ok" and result[0]["video.image"].max() == 1 and len(result) == 5
    parent.send(("close", None))
    assert parent.recv() == ("ok", None)
    thread.join(timeout=5)
    assert env.closed and len(env.actions) == 1


def test_serve_environment_reports_failures_with_traceback():
    env = _FakeEnv()
    parent, thread = _serve_in_thread(env)
    parent.send(("step", np.full((2, 7), np.nan, dtype=np.float32)))
    status, detail = parent.recv()
    assert status == "error" and "bad action" in detail
    thread.join(timeout=5)


def test_env_spec_round_trips_through_dataclass_fields():
    IsaacLabEnvSpec(task_id="Task-v0", num_envs=4, image_size=256, action_dim=7, seed=3)
    with pytest.raises(TypeError):
        IsaacLabEnvSpec(task_id="Task-v0")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        IsaacLabEnvSpec(
            task_id="Task-v0",
            num_envs=4,
            image_size=256,
            action_dim=7,
            seed=3,
            startup_attempts=0,
        )


def test_spawn_replays_no_main_module_in_the_child():
    """The child imports Isaac Lab only after Kit, so it must not replay the launcher's main module."""
    outside = spawn.get_preparation_data("check")
    assert "init_main_from_path" in outside or "init_main_from_name" in outside
    main_module = sys.modules["__main__"]
    with _without_main_module_replay():
        inside = spawn.get_preparation_data("check")
    assert "init_main_from_path" not in inside and "init_main_from_name" not in inside
    assert sys.modules["__main__"] is main_module
