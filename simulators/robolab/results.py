# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless ingestion of RoboLab episode results into harness traces.

RoboLab owns trajectory HDF5 files, videos, subtask logs, and the canonical
``episode_results.jsonl`` summary.  The harness must retain those native
artifacts rather than copy observations or reinterpret policy outcomes during
baseline reproduction.  This module writes one normalized, append-safe trace
row per native episode and links the files that belong to that episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["IngestionSummary", "ingest_results"]

_TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IngestionSummary:
    """Counts and destination produced by one RoboLab ingestion run."""

    source_root: Path
    output_path: Path
    result_files: int
    episodes: int


def ingest_results(source_root: Path | str, output_path: Path | str) -> IngestionSummary:
    """Normalize all RoboLab ``episode_results.jsonl`` files below a run root.

    The emitted rows deliberately preserve the complete native result object
    under ``robolab_episode``.  Derived fields are stable source references,
    not a replacement success metric.  A temporary file followed by
    :meth:`Path.replace` prevents a partial trace from masquerading as a
    completed ingestion after interruption.

    Args:
        source_root: RoboLab output folder containing one or more native
            ``episode_results.jsonl`` files.
        output_path: Destination JSONL path for normalized harness rows.

    Returns:
        Counts and paths describing the completed ingestion.

    Raises:
        FileNotFoundError: If ``source_root`` does not exist or has no native
            result file.
        ValueError: If a source line is not a JSON object.
    """
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboLab output directory does not exist: {root}")

    result_files = sorted(root.rglob("episode_results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No episode_results.jsonl found below: {root}")

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    episodes = 0

    with temporary.open("w", encoding="utf-8") as out:
        for result_file in result_files:
            for line_number, episode in _read_episode_results(result_file):
                trace = {
                    "schema_version": _TRACE_SCHEMA_VERSION,
                    "source": {
                        "robolab_result_file": str(result_file.relative_to(root)),
                        "line": line_number,
                    },
                    "robolab_episode": episode,
                    "artifacts": _episode_artifacts(root, result_file, episode),
                }
                out.write(json.dumps(trace, sort_keys=True, separators=(",", ":")))
                out.write("\n")
                episodes += 1

    temporary.replace(destination)
    return IngestionSummary(
        source_root=root,
        output_path=destination,
        result_files=len(result_files),
        episodes=episodes,
    )


def _read_episode_results(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield non-empty native JSONL rows while retaining source line numbers."""
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}:{line_number}; got {type(value).__name__}"
                )
            yield line_number, value


def _episode_artifacts(root: Path, result_file: Path, episode: dict[str, Any]) -> dict[str, list[str]]:
    """Return stable references to native files belonging to one episode.

    Missing optional fields and artifacts are normal for smoke runs without
    subtask tracking or video recording, so this function reports only files
    that actually exist.
    """
    env_name = episode.get("env_name")
    if not isinstance(env_name, str) or not env_name:
        return {}

    task_dir = result_file.parent / env_name
    if not task_dir.is_dir():
        return {}

    artifacts: dict[str, list[str]] = {}
    run_index = _nonnegative_int(episode.get("run"))
    env_id = _nonnegative_int(episode.get("env_id"))

    if run_index is not None:
        _add_if_file(artifacts, "hdf5", root, task_dir / f"run_{run_index}.hdf5")
    _add_if_file(artifacts, "environment_config", root, task_dir / "env_cfg.json")

    if run_index is not None and env_id is not None:
        _add_if_file(artifacts, "subtask_log", root, task_dir / f"log_{run_index}_env{env_id}.json")
        for video_path in sorted(task_dir.glob(f"*_{run_index}_env{env_id}*.mp4")):
            _add_if_file(artifacts, "video", root, video_path)

    return artifacts


def _nonnegative_int(value: Any) -> int | None:
    """Convert a JSON numeric field to an index without accepting booleans."""
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _add_if_file(artifacts: dict[str, list[str]], name: str, root: Path, path: Path) -> None:
    """Add one existing file as a path relative to the native run root."""
    if path.is_file():
        artifacts.setdefault(name, []).append(str(path.relative_to(root)))
