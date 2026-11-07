# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the extended infrastructure benchmark modes.

Each extended plane of ``scripts/bench_infra.py`` must parse its flags and
run end-to-end at trivial sizes (``--smoke``: 5 calls/steps) in whichever
venv executes the suite:

* planes whose dependencies are present must produce a non-empty result in
  the ``--json`` dump,
* planes gated on an absent optional dependency (``rlinf`` in the learner
  venv) must skip with a printed message and still exit 0, so
  ``--plane all-extended`` is runnable everywhere.

These are plumbing tests (the modes run and report), not performance
assertions -- the measured numbers live in ``docs/infrastructure.md``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

BENCH_SCRIPT = os.path.join(REPO_ROOT, "scripts", "bench_infra.py")

#: plane flag -> key under which the plane reports in the --json dump.
PLANE_JSON_KEYS = {
    "rollout-scaling": "rollout_scaling",
    "rollout-stepcost": "rollout_stepcost",
    "transport-sweep": "transport_sweep",
    "transport-tails": "transport_tails",
    "buffer-trace": "buffer_trace",
    "ray-footprint": "ray_footprint",
}


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_infra_under_test", BENCH_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_bench(args, timeout=600):
    """Run the benchmark script in a subprocess (its own real entry point)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, BENCH_SCRIPT] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("plane", sorted(PLANE_JSON_KEYS))
def test_extended_plane_smoke_runs(plane, tmp_path):
    """Every extended mode parses and completes at --smoke sizes (exit 0).

    Dependency-gated planes may skip, but only with an explicit printed
    message -- a silent empty result is a failure.
    """
    json_path = tmp_path / "results.json"
    proc = _run_bench(
        ["--plane", plane, "--smoke", "--trials", "1", "--json", json_path]
    )
    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    assert proc.returncode == 0, f"--plane {plane} --smoke failed:\n{tail}"
    results = json.loads(json_path.read_text())
    key = PLANE_JSON_KEYS[plane]
    if results.get(key):
        return  # plane ran and reported
    assert "skipped" in proc.stdout, (
        f"--plane {plane} produced no result and no skip message:\n{tail}"
    )


def test_all_extended_flag_is_exposed():
    """--plane all-extended and --smoke parse; unknown planes are rejected."""
    bench = _load_bench_module()
    parser = bench.build_parser()
    args = parser.parse_args(["--plane", "all-extended", "--smoke"])
    assert args.plane == "all-extended" and args.smoke
    for plane in PLANE_JSON_KEYS:
        assert parser.parse_args(["--plane", plane]).plane == plane
    with pytest.raises(SystemExit):
        parser.parse_args(["--plane", "bogus-plane"])


def test_original_planes_unchanged():
    """The pre-existing plane names and defaults survive the extension."""
    bench = _load_bench_module()
    parser = bench.build_parser()
    assert parser.parse_args([]).plane == "all"
    assert parser.parse_args([]).trials == 7
    for plane in ("serialization", "transport", "buffer", "rollout", "all"):
        assert parser.parse_args(["--plane", plane]).plane == plane


def test_buffer_plane_fills_past_its_transition_pool():
    """The pool is cycled with fresh episode ids, so a fill longer than the pool succeeds."""
    from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler

    bench = _load_bench_module()
    buffer = ReplayBuffer(1_000)
    result = bench._bench_one_buffer(
        buffer, UniformSampler(buffer), size=600, sample_iters=1
    )

    assert len(buffer) == 600 and result["adds_per_s"] > 0
