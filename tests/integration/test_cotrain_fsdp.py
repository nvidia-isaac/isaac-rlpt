# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Co-training under a real sharded learner.

The unit tests use plain modules, where every route to a supervised loss
looks equivalent. Under FSDP they are not: the parameter all-gather is
installed on the module's forward, so a loss computed by a separate method
runs against the rank's flat shard. This suite is the only place that
difference is observable, which is exactly why it exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

WORKER = Path(__file__).with_name("_cotrain_fsdp_worker.py")
REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="needs two CUDA devices to shard across",
    ),
]


@pytest.fixture(scope="module")
def worker_output() -> str:
    """Run the two-rank worker once and return its combined output."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(WORKER),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if completed.returncode != 0:
        pytest.fail(f"FSDP worker failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout + completed.stderr


def test_supervised_batch_through_the_model_call_sees_full_parameters(worker_output):
    """The contract route yields a [B] loss and real gradients on sharded params."""
    assert "CALL_ROUTE PASS" in worker_output, worker_output


def test_a_separate_method_cannot_score_demonstrations_under_fsdp(worker_output):
    """Pins the reason for the contract: the method route is broken, not slow.

    If this ever stops raising, the capability check in
    :mod:`isaac_rl_posttraining.training.cotrain` can be relaxed -- until then
    it is load-bearing.
    """
    assert "METHOD_ROUTE RAISES_AS_EXPECTED" in worker_output, worker_output
