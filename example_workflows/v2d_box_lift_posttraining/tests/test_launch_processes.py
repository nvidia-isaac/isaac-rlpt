# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launcher stops a simulator together with every process it started, and keeps its venv to itself."""

from __future__ import annotations

import os
import time
from pathlib import Path

from example_workflows.v2d_box_lift_posttraining.launch import (
    _terminate,
    simulator_environment,
    start_simulator,
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/status").read_text().find("State:\tZ") < 0


def test_terminate_stops_the_simulator_and_its_grandchild(tmp_path: Path) -> None:
    # Like a python.sh wrapper starting Kit: signalling only the wrapper leaves the child running.
    pid_file = tmp_path / "grandchild.pid"
    script = f"sleep 300 & echo $! > {pid_file}; wait"
    with (tmp_path / "server.log").open("w") as log:
        wrapper = start_simulator(["bash", "-c", script], dict(os.environ), log)
        deadline = time.monotonic() + 10
        while not pid_file.exists() or not pid_file.read_text().strip():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        grandchild = int(pid_file.read_text())
        assert _alive(grandchild)

        _terminate([wrapper], grace_seconds=5)

    assert wrapper.returncode is not None
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_simulator_environment_drops_the_launchers_virtual_environment() -> None:
    launcher = {
        "VIRTUAL_ENV": "/repo/.venv",
        "UV": "/usr/bin/uv",
        "UV_RUN_RECURSION_DEPTH": "1",
        "PATH": os.pathsep.join(["/repo/.venv/bin", "/usr/bin"]),
        "HOME": "/home/user",
    }

    result = simulator_environment(launcher, "2")

    assert result == {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "CUDA_VISIBLE_DEVICES": "2",
    }
    assert launcher["VIRTUAL_ENV"] == "/repo/.venv", (
        "the launcher's own environment is untouched"
    )
