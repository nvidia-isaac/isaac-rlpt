# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a run's telemetry tables and reduce them to the series the dashboard plots."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac_rl_posttraining.training.telemetry import TELEMETRY_DIR, read_telemetry

__all__ = ["RunSummary", "load_run"]

#: Collection modes that are training data (evaluation is reported separately).
TRAIN_MODES = ("warmup", "train")

Series = list[tuple[float, float]]


@dataclass
class RunSummary:
    """Everything the dashboard draws for one run."""

    name: str
    throughput: Series = field(default_factory=list)
    train_success: Series = field(default_factory=list)
    train_return: Series = field(default_factory=list)
    eval_success: Series = field(default_factory=list)
    eval_return: Series = field(default_factory=list)
    policy_lags: list[int] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    workers: list[dict[str, Any]] = field(default_factory=list)
    learner: dict[str, Series] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)


def _telemetry_dir(run_dir: str) -> str:
    if os.path.basename(os.path.normpath(run_dir)) == TELEMETRY_DIR:
        return run_dir
    direct = os.path.join(run_dir, TELEMETRY_DIR)
    if os.path.isdir(direct):
        return direct
    for root, dirs, _ in os.walk(run_dir):
        if TELEMETRY_DIR in dirs:
            return os.path.join(root, TELEMETRY_DIR)
    raise FileNotFoundError(f"{run_dir} holds no {TELEMETRY_DIR}/ directory")


def _rows(directory: str, table: str) -> list[dict[str, Any]]:
    return read_telemetry(directory, table).to_pylist()


def _rolling(points: Sequence[tuple[float, float]], window: int) -> Series:
    """Trailing mean of the y values over ``window`` points, x kept."""
    if not points:
        return []
    ys = np.asarray([y for _, y in points], dtype=float)
    sums = np.cumsum(np.concatenate([[0.0], ys]))
    out = []
    for index, (x, _) in enumerate(points):
        start = max(0, index + 1 - window)
        out.append((x, float((sums[index + 1] - sums[start]) / (index + 1 - start))))
    return out


def load_run(run_dir: str, *, name: str | None = None, window: int = 20) -> RunSummary:
    """Summarize the telemetry of the run at ``run_dir`` (the run directory or its ``telemetry/``)."""
    directory = _telemetry_dir(run_dir)
    transitions = _rows(directory, "transitions")
    episodes = _rows(directory, "episodes")
    rounds = _rows(directory, "rounds")
    learner = _rows(directory, "learner")
    run_name = name or os.path.basename(os.path.normpath(os.path.dirname(directory)))
    summary = RunSummary(run_name)

    # Run env steps at the end of each episode, from its last decision.
    episode_steps: dict[tuple[str, int], int] = {}
    for row in transitions:
        key = (row["source"], row["episode_id"])
        episode_steps[key] = max(episode_steps.get(key, 0), row["env_steps"])

    train_rows = [row for row in transitions if row["mode"] in TRAIN_MODES]
    eval_sources = {row["source"] for row in transitions if row["mode"] == "eval"}

    # Throughput per training round: steps over the round's wall time.
    run_steps = 0
    for row in rounds:
        if row["mode"] not in TRAIN_MODES:
            continue
        run_steps += row["env_steps"]
        seconds = row["wall_end"] - row["wall_start"]
        if seconds > 0:
            summary.throughput.append((float(run_steps), row["env_steps"] / seconds))

    train_episodes = sorted(
        (episode_steps.get((row["source"], row["episode_id"]), 0), row)
        for row in episodes
        if row["source"] not in eval_sources
    )
    successes = [(float(x), float(row["success"])) for x, row in train_episodes]
    returns = [(float(x), float(row["return"])) for x, row in train_episodes]
    summary.train_success = _rolling(successes, window)
    summary.train_return = _rolling(returns, window)
    summary.episode_lengths = [row["length_env_steps"] for _, row in train_episodes]

    # One evaluation point per evaluation round: the mean over its episodes.
    by_eval: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if row["source"] in eval_sources:
            by_eval[float(episode_steps.get((row["source"], row["episode_id"]), 0))].append(row)
    for x in sorted(by_eval):
        group = by_eval[x]
        summary.eval_success.append((x, float(np.mean([row["success"] for row in group]))))
        summary.eval_return.append((x, float(np.mean([row["return"] for row in group]))))

    summary.policy_lags = [row["policy_lag"] for row in train_rows]

    inference = sum(row["inference_seconds"] or 0.0 for row in train_rows)
    environment = sum(row["env_seconds"] or 0.0 for row in train_rows)
    learner_seconds = sum(row["metrics"].get("system/train_wall_s", 0.0) for row in _with_metric_dicts(learner))
    times = [row["wall_start"] for row in rounds] + [row["wall_end"] for row in rounds]
    times += [row["wall_time"] for row in learner]
    wall = (max(times) - min(times)) if times else 0.0
    summary.timing = {
        "policy inference": inference,
        "simulator": environment,
        "learner": learner_seconds,
        "other": max(wall - inference - environment - learner_seconds, 0.0),
    }

    per_worker: dict[int, dict[str, Any]] = {}
    for row in train_rows:
        worker = per_worker.setdefault(
            row["worker_index"],
            {"worker": row["worker_index"], "decisions": 0, "env_steps": 0, "inference": 0.0, "env": 0.0, "last": 0.0},
        )
        worker["decisions"] += 1
        worker["env_steps"] += row["steps"]
        worker["inference"] += row["inference_seconds"] or 0.0
        worker["env"] += row["env_seconds"] or 0.0
        worker["last"] = max(worker["last"], row["t_received"])
    summary.workers = [per_worker[index] for index in sorted(per_worker)]

    for row in _with_metric_dicts(learner):
        for metric, value in row["metrics"].items():
            if metric.startswith("system/"):
                continue
            summary.learner.setdefault(metric, []).append((float(row["env_steps"]), float(value)))

    summary.totals = {
        "env steps": float(run_steps),
        "train episodes": float(len(train_episodes)),
        "learner rows": float(len(learner)),
        "wall seconds": wall,
    }
    return summary


def _with_metric_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Learner rows with ``metrics`` as a dict (Arrow returns maps as key/value pairs)."""
    return [{**row, "metrics": dict(row["metrics"] or ())} for row in rows]
