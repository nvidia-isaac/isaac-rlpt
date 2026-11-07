# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ARENA_SIM_DEVICE=cpu boot regression (cluster integration, marked arena/gpu).

CPU physics is what lets many simulator lanes share one GPU: the sim renders
on the GPU but steps PhysX on the CPU. The knob is the ``ARENA_SIM_DEVICE``
environment variable consumed by ``simulators/g1_arena/env_server.py`` at the
process boundary. The regression pinned here (from the HP-0/evalcpu campaign):
a ``cpu`` boot must actually put the Isaac environment on the CPU, which
IsaacLab logs as ``Environment device : cpu``. A silently-ignored knob boots
on ``cuda`` and oversubscribes the GPU.
"""

import os
import re

import pytest
from conftest import (
    boot_arena_server,
    require_arena_prereqs,
    stop_process,
    wait_for_log_line,
)

pytestmark = [
    pytest.mark.arena,
    pytest.mark.gpu,
    pytest.mark.timeout(0),
]  # boots Isaac Sim

PORT = int(os.environ.get("ISAAC_RL_POSTTRAINING_TEST_ARENA_CPU_PORT", "7399"))
DEVICE_LINE = re.compile(r"Environment device\s*:\s*cpu", re.IGNORECASE)


def test_cpu_physics_boot_logs_environment_device_cpu(tmp_path):
    """Booting with ARENA_SIM_DEVICE=cpu must log 'Environment device : cpu'."""
    arena = require_arena_prereqs()
    log_path = str(tmp_path / "server_cpu.log")
    proc = boot_arena_server(
        arena,
        PORT,
        log_path,
        task="LMDrillLift",
        sim_device="cpu",
        image_resolution=0,
    )
    try:
        # The device line is printed while the env builds, i.e. at or before
        # readiness; allow a short grace period for buffered output.
        log = wait_for_log_line(
            log_path, pattern=r"listening on", timeout=60, proc=proc
        )
        if not DEVICE_LINE.search(log):
            log = wait_for_log_line(
                log_path, pattern=DEVICE_LINE.pattern, timeout=60, proc=proc
            )
        assert DEVICE_LINE.search(log), (
            "ARENA_SIM_DEVICE=cpu was ignored: the Isaac env did not report "
            f"'Environment device : cpu'. Log tail:\n{log[-4000:]}"
        )
        assert not re.search(r"Environment device\s*:\s*cuda", log), (
            "the environment reported a cuda device despite ARENA_SIM_DEVICE=cpu"
        )
    finally:
        stop_process(proc)
