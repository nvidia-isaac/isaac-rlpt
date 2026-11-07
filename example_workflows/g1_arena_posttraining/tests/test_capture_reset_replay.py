# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seeded-reset capture client, driven against a scripted transport."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from example_workflows.g1_arena_posttraining.scripts import capture_reset_replay


class _FakeTransport:
    def __init__(self, *, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.connected = False
        self.closed = False
        self.seeds: list[int | None] = []

    def connect(self) -> None:
        self.connected = True

    def spec(self) -> dict[str, int]:
        return {"num_envs": self.num_envs}

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.seeds.append(seed)
        return {
            "state": np.zeros((self.num_envs, 31), dtype=np.float32),
            "ego_view": np.zeros((self.num_envs, 8, 8, 3), dtype=np.uint8),
        }

    def close(self) -> None:
        self.closed = True


def test_capture_requests_the_same_seed_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _FakeTransport()
    monkeypatch.setattr(
        capture_reset_replay, "transport_from_uri", lambda *args, **kwargs: transport
    )
    monkeypatch.setattr(
        sys, "argv", ["capture_reset_replay.py", "--num-envs", "2", "--seed", "123"]
    )

    assert capture_reset_replay.main() == 0
    assert transport.connected is True
    assert transport.closed is True
    assert transport.seeds == [123, 123]
    assert "reset 1: seed=123 num_envs=2" in capsys.readouterr().out


def test_capture_rejects_a_server_env_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport(num_envs=1)
    monkeypatch.setattr(
        capture_reset_replay, "transport_from_uri", lambda *args, **kwargs: transport
    )
    monkeypatch.setattr(sys, "argv", ["capture_reset_replay.py", "--num-envs", "2"])

    with pytest.raises(RuntimeError, match="server reports num_envs=1"):
        capture_reset_replay.main()
    assert transport.closed is True
