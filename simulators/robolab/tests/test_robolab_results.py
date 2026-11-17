# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab result ingestion stays lossless without installing RoboLab."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulators.robolab import ingest_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_ingest_results_preserves_native_row_and_links_episode_artifacts(
    tmp_path: Path,
):
    source = tmp_path / "native"
    task_dir = source / "BananaInBowlTask"
    task_dir.mkdir(parents=True)
    for path in (
        task_dir / "run_0.hdf5",
        task_dir / "env_cfg.json",
        task_dir / "log_0_env2.json",
        task_dir / "put_banana_0_env2.mp4",
        task_dir / "put_banana_0_env2_viewport.mp4",
    ):
        path.touch()
    native = {
        "env_name": "BananaInBowlTask",
        "episode": 2,
        "env_id": 2,
        "run": 0,
        "policy": "gr00t",
        "success": True,
        "events": {"TARGET_OBJECT_DROPPED": 0},
    }
    _write_jsonl(source / "episode_results.jsonl", [native])

    output = tmp_path / "trace" / "episodes.jsonl"
    summary = ingest_results(source, output)

    assert summary.result_files == 1
    assert summary.episodes == 1
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["schema_version"] == 1
    assert trace["robolab_episode"] == native
    assert trace["source"] == {
        "line": 1,
        "robolab_result_file": "episode_results.jsonl",
    }
    assert trace["artifacts"]["hdf5"] == ["BananaInBowlTask/run_0.hdf5"]
    assert trace["artifacts"]["subtask_log"] == ["BananaInBowlTask/log_0_env2.json"]
    assert trace["artifacts"]["video"] == [
        "BananaInBowlTask/put_banana_0_env2.mp4",
        "BananaInBowlTask/put_banana_0_env2_viewport.mp4",
    ]


def test_ingest_results_rejects_missing_or_malformed_native_results(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No episode_results.jsonl"):
        ingest_results(tmp_path, tmp_path / "out.jsonl")

    source = tmp_path / "native"
    source.mkdir()
    (source / "episode_results.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        ingest_results(source, tmp_path / "out.jsonl")
