# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free discovery and reporting helpers for RoboLab task sweeps.

The simulator resolves task classes at runtime, which makes a typo in a task
name expensive: it can consume an allocated GPU worker before failing.  This
module instead inspects Python syntax only.  It deliberately does not import
RoboLab, Isaac Sim, or task modules, so it is safe to run on a launcher host.

The result helpers are descriptive.  They identify completed result files and
tasks whose *observed* success rate is in a configurable range.  They do not
compute confidence intervals or claim that a task is statistically suitable
for DSRL from one sweep alone.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "RoboLabTask",
    "ResultFileStatus",
    "ResultStatus",
    "SweepReport",
    "SweepTaskResult",
    "TaskDiscoveryError",
    "TaskValidationError",
    "discover_tasks",
    "read_sweep_results",
    "select_dsrl_candidates",
    "validate_task_names",
]


class TaskDiscoveryError(ValueError):
    """Raised when a task directory cannot be inspected safely."""


class TaskValidationError(ValueError):
    """Raised when one or more requested task names are not unambiguous."""


class ResultStatus(str, Enum):
    """The state of one requested JSONL result file."""

    COMPLETED = "completed"
    EMPTY = "empty"
    INVALID = "invalid"
    MISSING = "missing"


@dataclass(frozen=True, order=True)
class RoboLabTask:
    """One statically-discovered concrete task class."""

    name: str
    path: Path


@dataclass(frozen=True)
class ResultFileStatus:
    """Completion state and diagnostic for one requested result file."""

    path: Path
    status: ResultStatus
    episodes: int
    message: str | None = None


@dataclass(frozen=True)
class SweepTaskResult:
    """A completed task/configuration group from native RoboLab JSONL rows."""

    task_name: str
    result_paths: tuple[Path, ...]
    episodes: int
    successes: int
    success_rate: float
    config_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SweepReport:
    """Valid completed task groups plus every requested source-file state."""

    sources: tuple[ResultFileStatus, ...]
    tasks: tuple[SweepTaskResult, ...]


_CONFIG_KEYS = (
    "config",
    "evaluation_config",
    "run_config",
    "metadata",
    "model",
    "model_id",
    "checkpoint",
    "policy",
    "open_loop_horizon",
    "num_envs",
    "num_runs",
    "seed",
    "server_revision",
    "robolab_revision",
)


def discover_tasks(task_directory: Path | str) -> tuple[RoboLabTask, ...]:
    """Discover concrete subclasses of ``Task`` below ``task_directory``.

    Discovery is syntax-only.  The pinned RoboLab task layout declares each
    concrete task directly in its task file, so this helper recognizes direct
    ``Task`` bases and imported aliases only.  It intentionally does not try
    to resolve arbitrary Python inheritance across files.  A directly-declared
    abstract class is excluded.

    Args:
        task_directory: RoboLab task directory, usually ``robolab/tasks`` or
            a narrower benchmark-task directory.

    Returns:
        Concrete task classes, ordered by class name then source path.

    Raises:
        TaskDiscoveryError: If the directory is missing, contains no Python
            files, or one of those files cannot be parsed.
    """
    root = Path(task_directory).resolve()
    if not root.is_dir():
        raise TaskDiscoveryError(f"RoboLab task directory does not exist: {root}")

    paths = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not paths:
        raise TaskDiscoveryError(f"No Python task files found below: {root}")

    tasks: list[RoboLabTask] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise TaskDiscoveryError(f"Could not parse task file {path}: {error}") from error
        task_bases = _task_base_aliases(tree)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            direct_bases = {_expression_name(base) for base in node.bases}
            if direct_bases.isdisjoint(task_bases) or _contains_abstract_method(node):
                continue
            tasks.append(RoboLabTask(name=node.name, path=path))
    return tuple(sorted(tasks, key=lambda task: (task.name, str(task.path))))


def validate_task_names(
    task_directory: Path | str,
    requested_names: Sequence[str],
) -> tuple[RoboLabTask, ...]:
    """Resolve requested task names before a remote evaluation is launched.

    A duplicate class name is rejected rather than resolving based on filename
    or directory order.  That makes a launch command reproducible and avoids
    silently selecting a different task after a RoboLab update.
    """
    requested = tuple(requested_names)
    if not requested:
        raise TaskValidationError("At least one RoboLab task name is required.")

    by_name: dict[str, list[RoboLabTask]] = defaultdict(list)
    for task in discover_tasks(task_directory):
        by_name[task.name].append(task)

    selected: list[RoboLabTask] = []
    problems: list[str] = []
    for name in requested:
        matches = by_name.get(name, [])
        if not matches:
            available = ", ".join(sorted(by_name)) or "(none)"
            problems.append(f"Unknown task {name!r}. Available tasks: {available}")
        elif len(matches) > 1:
            paths = ", ".join(str(task.path) for task in matches)
            problems.append(f"Ambiguous task {name!r}; found in: {paths}")
        else:
            selected.append(matches[0])

    if problems:
        raise TaskValidationError("\n".join(problems))
    return tuple(selected)


def read_sweep_results(result_paths: Iterable[Path | str]) -> SweepReport:
    """Read native or ingested RoboLab JSONL output without reinterpreting it.

    Missing and malformed files are represented in ``sources`` rather than
    being folded into a zero-success result.  A valid row with ``success:
    false`` is a completed episode and therefore contributes to a real 0%.
    Files with no non-empty rows have status ``empty`` and are not candidates.
    """
    sources: list[ResultFileStatus] = []
    groups: dict[tuple[str, str], _ResultAccumulator] = {}

    for supplied_path in result_paths:
        path = Path(supplied_path).expanduser().resolve()
        if not path.is_file():
            sources.append(
                ResultFileStatus(
                    path=path,
                    status=ResultStatus.MISSING,
                    episodes=0,
                    message="Result JSONL file does not exist.",
                )
            )
            continue

        try:
            rows = tuple(_read_result_rows(path))
            parsed_rows = tuple(_parse_result_row(path, line, row) for line, row in rows)
        except ValueError as error:
            sources.append(
                ResultFileStatus(
                    path=path,
                    status=ResultStatus.INVALID,
                    episodes=0,
                    message=str(error),
                )
            )
            continue

        if not parsed_rows:
            sources.append(
                ResultFileStatus(
                    path=path,
                    status=ResultStatus.EMPTY,
                    episodes=0,
                    message="Result JSONL contains no episode rows.",
                )
            )
            continue

        sources.append(ResultFileStatus(path=path, status=ResultStatus.COMPLETED, episodes=len(parsed_rows)))
        for row in parsed_rows:
            key = (row.task_name, _canonical_json(row.config_metadata))
            accumulator = groups.setdefault(
                key,
                _ResultAccumulator(task_name=row.task_name, config_metadata=row.config_metadata),
            )
            accumulator.add(path, row.success)

    tasks = tuple(
        sorted(
            (accumulator.as_result() for accumulator in groups.values()),
            key=lambda result: (result.task_name, _canonical_json(result.config_metadata)),
        )
    )
    return SweepReport(sources=tuple(sources), tasks=tasks)


def select_dsrl_candidates(
    report: SweepReport,
    *,
    min_success_rate: float = 0.25,
    max_success_rate: float = 0.80,
    min_episodes: int = 1,
) -> tuple[SweepTaskResult, ...]:
    """Return completed task/config groups in an observed-success-rate band.

    The returned list is a practical queue for follow-up evaluation, not a
    statistical conclusion.  Users should freeze a protocol and collect more
    episodes before comparing steering methods on selected tasks.
    """
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be between 0 and 1.")
    if not 0.0 <= max_success_rate <= 1.0:
        raise ValueError("max_success_rate must be between 0 and 1.")
    if min_success_rate > max_success_rate:
        raise ValueError("min_success_rate cannot exceed max_success_rate.")
    if min_episodes < 1:
        raise ValueError("min_episodes must be at least 1.")

    return tuple(
        result
        for result in report.tasks
        if result.episodes >= min_episodes
        and min_success_rate <= result.success_rate <= max_success_rate
    )


def _task_base_aliases(tree: ast.Module) -> set[str]:
    """Return direct ``Task`` base spellings used by one pinned task file."""
    names = {"Task"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names |= _imported_task_names(node)
        elif isinstance(node, ast.Import):
            names.update(f"{alias.asname or alias.name}.Task" for alias in node.names)
    return names


def _imported_task_names(node: ast.ImportFrom) -> set[str]:
    """Return the ``Task`` spellings one ``from ... import`` statement binds."""
    names = set()
    for alias in node.names:
        bound = alias.asname or alias.name
        if alias.name == "Task":
            names.add(bound)
        elif alias.name != "*":
            names.add(f"{bound}.Task")
    return names


def _expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _contains_abstract_method(node: ast.ClassDef) -> bool:
    return any(
        isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            (_expression_name(decorator) or "").split(".")[-1] == "abstractmethod"
            for decorator in member.decorator_list
        )
        for member in node.body
    )


@dataclass(frozen=True)
class _ParsedResultRow:
    task_name: str
    success: bool
    config_metadata: Mapping[str, Any]


@dataclass
class _ResultAccumulator:
    task_name: str
    config_metadata: Mapping[str, Any]
    result_paths: set[Path] = field(default_factory=set)
    episodes: int = 0
    successes: int = 0

    def add(self, path: Path, success: bool) -> None:
        self.result_paths.add(path)
        self.episodes += 1
        self.successes += int(success)

    def as_result(self) -> SweepTaskResult:
        return SweepTaskResult(
            task_name=self.task_name,
            result_paths=tuple(sorted(self.result_paths)),
            episodes=self.episodes,
            successes=self.successes,
            success_rate=self.successes / self.episodes,
            config_metadata=self.config_metadata,
        )


def _read_result_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read result file {path}: {error}") from error
    with source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            yield line_number, row


def _parse_result_row(path: Path, line: int, row: Mapping[str, Any]) -> _ParsedResultRow:
    nested_episode = row.get("robolab_episode")
    if nested_episode is None:
        episode = row
        trace = {}
    else:
        episode = nested_episode
        trace = row
    if not isinstance(episode, Mapping):
        raise ValueError(f"Expected robolab_episode object at {path}:{line}.")

    task_name = next(
        (
            episode[key]
            for key in ("env_name", "task_name", "task")
            if isinstance(episode.get(key), str) and episode[key]
        ),
        None,
    )
    if task_name is None:
        raise ValueError(f"Missing task name (env_name, task_name, or task) at {path}:{line}.")

    success = episode.get("success")
    if not isinstance(success, bool):
        raise ValueError(f"Missing boolean success field at {path}:{line}.")
    return _ParsedResultRow(
        task_name=task_name,
        success=success,
        config_metadata=_config_metadata(trace, episode),
    )


def _config_metadata(row: Mapping[str, Any], episode: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for prefix, source in (("trace", row), ("episode", episode)):
        values = {key: source[key] for key in _CONFIG_KEYS if key in source}
        if values:
            metadata[prefix] = values
    return metadata


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Result configuration metadata is not JSON serializable: {error}") from error
