# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side contracts of ArenaBatchedEnv and the batched builder path.

A fake transport stands in for the batched env server: the reset mask
translates to wire env_ids, payload arrays keep their [N] axis, the client
enforces the per-env episode cap (the simulator never sees those
truncations), and intermediate steps without images fall back to zero
frames. The builder wires vector_backend=batched to the BatchedEnvAdapter
and refuses the multi-client shapes that would silently share one server.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import simulators.g1_arena.env as arena_env
from simulators.g1_arena.env import DEFAULT_STATE_DIM, ArenaBatchedEnv, build_dsrl_env
from simulators.g1_arena.timers import RoundTripTimer

NUM_ENVS = 3
IMAGE_HW = (8, 12)


class _FakeTransport:
    """In-memory transport double recording reset/step requests."""

    def __init__(self, num_envs: int = NUM_ENVS) -> None:
        self.num_envs = num_envs
        self.reset_calls: list[list[int] | None] = []
        self.step_calls: list[tuple[np.ndarray, bool]] = []
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.include_images = True
        self.last_timing = {"recv_s": 0.02, "decode_s": 0.003}
        self.step_error: Exception | None = None

    def connect(self) -> None:
        pass  # no socket to open: the fake answers in-process

    def close(self) -> None:
        pass  # nothing to release

    def spec(self) -> dict[str, Any]:
        # The server's spec as sent for native frames (no server-side shrink).
        return {
            "num_envs": self.num_envs,
            "spaces": {
                "observation": {
                    "ego_view": {"dtype": "uint8", "hw": None, "shortest_edge": None}
                }
            },
        }

    def _payload(self, include_images: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": np.zeros((self.num_envs, DEFAULT_STATE_DIM), dtype=np.float32),
            "prompt": "Lift the drill.",
        }
        if include_images:
            payload["ego_view"] = np.zeros(
                (self.num_envs, *IMAGE_HW, 3), dtype=np.uint8
            )
        return payload

    def reset(
        self, seed: int | None = None, env_ids: list[int] | None = None
    ) -> dict[str, Any]:
        self.reset_calls.append(env_ids)
        payload = self._payload(include_images=True)
        payload["info"] = {}
        return payload

    def step(self, action: Any, need_obs: bool = True) -> dict[str, Any]:
        if self.step_error is not None:
            raise self.step_error
        action = np.asarray(action, dtype=np.float32)
        self.step_calls.append((action, need_obs))
        payload = self._payload(include_images=self.include_images)
        payload.update(
            reward=np.arange(self.num_envs, dtype=np.float32),
            done=self.terminated.copy(),
            truncated=np.zeros(self.num_envs, dtype=bool),
            success=self.terminated.copy(),
            num_steps=np.ones(self.num_envs, dtype=np.int64),
            info={},
            server_s=0.01,
        )
        return payload


@pytest.fixture()
def batched(monkeypatch):
    transport = _FakeTransport()
    monkeypatch.setattr(arena_env, "transport_from_uri", lambda url, **kw: transport)
    env = ArenaBatchedEnv(
        server_url="tcp://127.0.0.1:7300",
        num_envs=NUM_ENVS,
        image_hw=IMAGE_HW,
        max_episode_steps=3,
    )
    return env, transport


def test_spec_num_envs_mismatch_fails(monkeypatch):
    transport = _FakeTransport(num_envs=2)
    monkeypatch.setattr(arena_env, "transport_from_uri", lambda url, **kw: transport)
    with pytest.raises(ValueError, match="--num-envs"):
        ArenaBatchedEnv(server_url="tcp://127.0.0.1:7300", num_envs=NUM_ENVS)


def test_reset_mask_translates_to_env_ids(batched):
    env, transport = batched
    env.reset()
    mask = np.array([False, True, True])
    env.reset(options={"reset_mask": mask})
    assert transport.reset_calls == [None, [1, 2]]


def test_step_returns_batched_arrays_and_infos(batched):
    env, transport = batched
    env.reset()
    transport.terminated = np.array([False, True, False])
    obs, reward, terminated, _truncated, infos = env.step(
        np.zeros((NUM_ENVS, 35), dtype=np.float32)
    )
    assert obs["ego_view_image"].shape == (NUM_ENVS, *IMAGE_HW, 3)
    assert obs["state"].shape == (NUM_ENVS, DEFAULT_STATE_DIM)
    np.testing.assert_allclose(reward, [0.0, 1.0, 2.0])
    assert terminated.tolist() == [False, True, False]
    assert [info["is_success"] for info in infos] == [False, True, False]


def test_step_rejects_wrong_shapes(batched):
    env, _ = batched
    env.reset()
    with pytest.raises(ValueError, match="num_envs"):
        env.step(np.zeros(35, dtype=np.float32))
    with pytest.raises(ValueError, match="no chunks"):
        env.step(np.zeros((NUM_ENVS, 4, 35), dtype=np.float32))


def test_episode_cap_truncates_per_env(batched):
    env, _transport = batched
    env.reset()
    actions = np.zeros((NUM_ENVS, 35), dtype=np.float32)
    for _ in range(2):
        _, _, _, truncated, _ = env.step(actions)
        assert not truncated.any()
    _, _, _, truncated, _ = env.step(actions)  # step 3 hits the cap
    assert truncated.all()
    # A masked reset clears only the flagged env's step counter.
    mask = np.array([True, False, False])
    env.reset(options={"reset_mask": mask})
    _, _, _, truncated, _ = env.step(actions)
    assert truncated.tolist() == [False, True, True]


def test_need_obs_flag_reaches_wire_and_zero_fills(batched):
    env, transport = batched
    env.reset()
    env.need_obs = False
    transport.include_images = False
    obs, *_ = env.step(np.zeros((NUM_ENVS, 35), dtype=np.float32))
    _, need_obs = transport.step_calls[-1]
    assert need_obs is False
    assert obs["ego_view_image"].shape == (NUM_ENVS, *IMAGE_HW, 3)
    assert not obs["ego_view_image"].any()


def test_step_disconnect_is_not_recorded_as_a_task_failure(batched):
    env, transport = batched
    env.reset()
    transport.step_error = arena_env.TransportError("server closed the connection")
    with pytest.raises(RuntimeError, match="no environment transitions were recorded"):
        env.step(np.zeros((NUM_ENVS, 35), dtype=np.float32))


def test_builder_batched_wires_adapter_and_contract():
    env = build_dsrl_env(
        "arena/drill_lift",
        num_envs=2,
        vectorized=True,
        max_episode_steps=450,
        vector_backend="batched",
        image_resolution=0,
    )
    assert env.num_envs == 2
    assert env.steering_key_mapping["image"] == ["ego_view_image"]
    assert env.steering_remove_obs_keys == ["ego_view_image"]
    # Stub mode (no server URL): reset works offline through the adapter.
    obs, _ = env.reset()
    assert obs["ego_view_image"].shape[0] == 2


def test_builder_rejects_multi_env_sync():
    with pytest.raises(ValueError, match="vector_backend=batched"):
        build_dsrl_env(
            "arena/drill_lift",
            num_envs=4,
            vectorized=True,
            max_episode_steps=450,
            vector_backend="sync",
        )


def test_builder_rejects_single_env_batched():
    with pytest.raises(ValueError, match="num_envs > 1"):
        build_dsrl_env(
            "arena/drill_lift",
            num_envs=1,
            vectorized=True,
            max_episode_steps=450,
            vector_backend="batched",
        )


def test_round_trip_timer_splits_server_and_wire_time():
    timer = RoundTripTimer(num_envs=256, report_every=2)
    assert timer.record(0.9, 0.6, 0.85, 0.04, 0.1, with_images=False) is None
    summary = timer.record(1.1, 0.8, 1.05, 0.04, 0.1, with_images=True)
    assert summary is not None
    assert "over 2 steps (through step 2)" in summary
    # wire = round trip - server; transfer = time to the last byte - server; decode = unpickling
    assert (
        "round trip 1000 ms/step mean = server 700 + wire 300 (transfer 250, decode 40); format 100 ms/step"
        in summary
    )
    assert "images on 1 steps; 232.7 env-steps/s" in summary
    # The window resets; the running step count does not.
    assert timer.record(1.0, 0.5, 0.9, 0.0, 0.0, with_images=False) is None
    assert timer.total_steps == 3 and len(timer.round_trip_s) == 1
    with pytest.raises(ValueError, match="report_every"):
        RoundTripTimer(num_envs=1, report_every=0)


def test_batched_step_logs_the_round_trip_split(batched):
    from loguru import logger

    env, _transport = batched
    env._round_trip_timer = RoundTripTimer(env.num_envs, report_every=1)
    env.reset()
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        env.step(np.zeros((NUM_ENVS, 35), dtype=np.float32))
    finally:
        logger.remove(sink)
    line = next(text for text in lines if "[arena_batched_client] step timing" in text)
    # the fake transport reports recv 20 ms / decode 3 ms and the fake server echoes 10 ms
    assert "server 10 +" in line and "decode 3)" in line
