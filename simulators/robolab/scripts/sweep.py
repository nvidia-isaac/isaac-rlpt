#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover RoboLab tasks, validate launch names, and report sweep candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simulators.robolab.sweep import (
    ResultFileStatus,
    SweepReport,
    SweepTaskResult,
    discover_tasks,
    read_sweep_results,
    select_dsrl_candidates,
    validate_task_names,
)


def parse_args() -> argparse.Namespace:
    """Parse one explicit, non-simulator sweep-support command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="List concrete task classes without importing RoboLab.")
    discover.add_argument("--task-dir", type=Path, required=True)
    discover.add_argument("--json", action="store_true", help="Print JSON for launcher automation.")

    validate = subparsers.add_parser("validate", help="Fail before launch when a task name is invalid or ambiguous.")
    validate.add_argument("--task-dir", type=Path, required=True)
    validate.add_argument("--task", action="append", required=True, help="Task class name; repeat for each task.")
    validate.add_argument("--json", action="store_true", help="Print JSON for launcher automation.")

    report = subparsers.add_parser("report", help="Summarize native/ingested JSONL sweep results.")
    report.add_argument("--result", action="append", type=Path, required=True, help="Result JSONL path; repeat as needed.")
    report.add_argument("--min-success-rate", type=float, default=0.25)
    report.add_argument("--max-success-rate", type=float, default=0.80)
    report.add_argument("--min-episodes", type=int, default=1)
    report.add_argument("--json", action="store_true", help="Print JSON for automation.")
    return parser.parse_args()


def main() -> None:
    """Execute the selected discovery, validation, or reporting operation."""
    args = parse_args()
    if args.command == "discover":
        tasks = discover_tasks(args.task_dir)
        _print_tasks(tasks, as_json=args.json)
    elif args.command == "validate":
        tasks = validate_task_names(args.task_dir, args.task)
        _print_tasks(tasks, as_json=args.json)
    elif args.command == "report":
        report = read_sweep_results(args.result)
        candidates = select_dsrl_candidates(
            report,
            min_success_rate=args.min_success_rate,
            max_success_rate=args.max_success_rate,
            min_episodes=args.min_episodes,
        )
        _print_report(report, candidates, as_json=args.json)
    else:  # pragma: no cover - argparse guarantees the command choices.
        raise AssertionError(f"Unhandled command: {args.command}")


def _print_tasks(tasks: tuple[Any, ...], *, as_json: bool) -> None:
    payload = [{"task": task.name, "path": str(task.path)} for task in tasks]
    if as_json:
        print(json.dumps(payload, sort_keys=True))
        return
    for task in payload:
        print(f"{task['task']}\t{task['path']}")


def _print_report(
    report: SweepReport,
    candidates: tuple[SweepTaskResult, ...],
    *,
    as_json: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "sources": [_source_dict(source) for source in report.sources],
                    "tasks": [_task_dict(task) for task in report.tasks],
                    "dsrl_candidates": [_task_dict(task) for task in candidates],
                    "note": "Candidates use observed success rates only; this is not a statistical significance claim.",
                },
                sort_keys=True,
            )
        )
        return

    print("sources:")
    for source in report.sources:
        message = f" ({source.message})" if source.message else ""
        print(f"  {source.status.value}: episodes={source.episodes} path={source.path}{message}")
    print("completed task/config groups:")
    for task in report.tasks:
        print(
            f"  {task.task_name}: {task.successes}/{task.episodes} "
            f"({task.success_rate:.1%}) paths={','.join(str(path) for path in task.result_paths)} "
            f"config={json.dumps(task.config_metadata, sort_keys=True)}"
        )
    print("DSRL candidates (observed rates only; not a statistical significance claim):")
    for task in candidates:
        print(f"  {task.task_name}: {task.successes}/{task.episodes} ({task.success_rate:.1%})")


def _source_dict(source: ResultFileStatus) -> dict[str, Any]:
    return {
        "episodes": source.episodes,
        "message": source.message,
        "path": str(source.path),
        "status": source.status,
    }


def _task_dict(task: SweepTaskResult) -> dict[str, Any]:
    return {
        "config_metadata": task.config_metadata,
        "episodes": task.episodes,
        "paths": [str(path) for path in task.result_paths],
        "success_rate": task.success_rate,
        "successes": task.successes,
        "task": task.task_name,
    }


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:  # includes the task discovery/validation errors
        raise SystemExit(f"error: {error}") from error
