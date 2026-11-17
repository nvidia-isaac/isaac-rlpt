# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-container training driver fails fast when a simulator does not boot.

``scripts/station/run_container_train.sh`` starts a utilization sampler and two simulators
before the learner. When a simulator dies at boot the driver must exit non-zero and take every
background process with it: a sampler left behind keeps the task's output pipe open, and the
job then hangs as RUNNING instead of failing (seen on the cluster on 2026-09-17).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW.parent.parent
DRIVER = WORKFLOW / "scripts" / "station" / "run_container_train.sh"

#: Stands in for /isaac-sim/python.sh: answers the driver's config read and import preflight,
#: and refuses to boot the simulator.
_STUB_PYTHON = """#!/bin/bash
case "$1" in
  -) echo 450 ;;
  -c) echo "[train] learner imports OK (stub)" ;;
  *env_server.py) echo "Error: stub simulator refused to boot"; exit 1 ;;
  *) echo "stub: unexpected invocation: $*" >&2; exit 99 ;;
esac
"""


def _driver_env(tmp_path: Path) -> dict[str, str]:
    stub = tmp_path / "python.sh"
    stub.write_text(_STUB_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "posttrain").symlink_to(REPO_ROOT)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data" / "LMPushButton" / "lerobot").mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if key not in ("N", "TASK")}
    env.update(
        {
            "CONTAINER_PYTHON": str(stub),
            "WORKDIR": str(workdir),
            "SIM_LOG_DIR": str(tmp_path / "sim_logs"),
            "ISAAC_RL_POSTTRAINING_OUTPUT_DIR": str(tmp_path / "out"),
            "POSTTRAIN_BASE_CKPT": str(checkpoint),
            "POSTTRAIN_DATA_ROOT": str(tmp_path / "data"),
            "TASK": "PushButton",
            "N": "2",
            "WANDB_NAME": "driver-test",
        }
    )
    return env


def _session_is_empty(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.5)
    return False


@pytest.mark.parametrize("parallel_boot", ["0", "1"])
def test_driver_exits_and_reaps_its_background_processes_when_a_simulator_fails(
    tmp_path: Path, parallel_boot: str
) -> None:
    process = subprocess.Popen(
        ["bash", str(DRIVER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_driver_env(tmp_path) | {"PARALLEL_SIM_BOOT": parallel_boot},
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        pytest.fail(
            "the driver's output pipe stayed open after the simulator failed: "
            "a background process outlived the driver"
        )
    assert process.returncode == 1, output
    assert "train-sim 0 (N=2, port 7300) FAILED" in output, output
    eval_log = tmp_path / "sim_logs" / "evalsim.7301.log"
    assert eval_log.exists() == (parallel_boot == "1"), output
    if not _session_is_empty(process.pid, timeout=20):
        os.killpg(process.pid, signal.SIGKILL)
        pytest.fail("background processes survived the driver's exit")


@pytest.mark.parametrize(("sims_only", "host"), [("0", "127.0.0.1"), ("1", "0.0.0.0")])
def test_simulators_accept_other_hosts_only_when_serving_a_remote_learner(
    tmp_path: Path, sims_only: str, host: str
) -> None:
    env = _driver_env(tmp_path) | {
        "SIMS_ONLY": sims_only,
        "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
    }
    result = subprocess.run(
        ["bash", str(DRIVER)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert f"launch.sim_host={host}\n" in result.stdout, result.stdout + result.stderr
