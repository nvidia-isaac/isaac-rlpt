# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FSDP benchmark records its revision outside a git checkout and writes results before anything else."""

import json
import subprocess

import pytest

from scripts import bench_rlinf_ppo_fsdp as bench


def _no_git(*args):
    raise subprocess.CalledProcessError(128, ["git", *args])


def test_the_revision_falls_back_to_the_environment_then_a_revision_file(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(bench, "_git_output", _no_git)
    monkeypatch.delenv(bench.REVISION_ENV, raising=False)
    assert bench._revision(tmp_path) == {
        "git_commit": None,
        "git_dirty": None,
        "revision_source": None,
    }
    (tmp_path / "REVISION").write_text("a6dab392\n")
    assert bench._revision(tmp_path)["git_commit"] == "a6dab392"
    monkeypatch.setenv(bench.REVISION_ENV, "deadbeef")
    assert bench._revision(tmp_path) == {
        "git_commit": "deadbeef",
        "git_dirty": None,
        "revision_source": bench.REVISION_ENV,
    }


def test_results_are_written_before_they_are_printed(monkeypatch, tmp_path):
    output = tmp_path / "out" / "result.json"
    monkeypatch.setattr(bench, "_run_torchrun", lambda args: {"status": "ok"})
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench",
            "--launcher",
            "torchrun",
            "--num-learner-ranks",
            "1",
            "--output",
            str(output),
        ],
    )

    def broken_print(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", broken_print)
    with pytest.raises(BrokenPipeError):
        bench.main()
    assert json.loads(output.read_text()) == {"status": "ok"}
