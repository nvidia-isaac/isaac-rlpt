# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run telemetry schema (four tables), the sink interface, and the Parquet writer.

Every run records, cheaply and always, one row per transition, per finished
episode, per collection round and per learner step. The column specs below are
the contract between writers (the data source, the trainers) and readers (the
visualizer); :class:`ParquetTelemetrySink` stores them under
``<run_dir>/telemetry/<table>/part-NNNNN.parquet``, next to ``config.yaml`` and
``placement.json``. Columns use Arrow type names; ``map<string,float64>`` holds
open-ended scalars (reward channels, learner metrics)::

    sink = open_run_telemetry(run_dir, config=cfg)
    sink.write_learner([{"run_id": "r", "learner_step": 1, ...}])
    sink.close()
    rows = read_telemetry(run_dir, "learner")   # a pyarrow.Table
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "EPISODES",
    "LEARNER",
    "ROUNDS",
    "SCHEMA_METADATA_KEY",
    "SCHEMA_VERSION",
    "TABLES",
    "TRANSITIONS",
    "Column",
    "ColumnType",
    "NullTelemetrySink",
    "ParquetTelemetrySink",
    "TableSpec",
    "TelemetrySink",
    "arrow_schema",
    "open_run_telemetry",
    "read_telemetry",
]

#: Bumped whenever a column is added, removed, renamed or retyped.
SCHEMA_VERSION = 1

ColumnType = Literal["string", "int64", "float64", "bool", "map<string,float64>"]
_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "int64": (int,),
    "float64": (float, int),
    "bool": (bool,),
    "map<string,float64>": (Mapping,),
}

Row = Mapping[str, Any]


@dataclass(frozen=True)
class Column:
    """One telemetry column: name, Arrow type, whether it may be missing, and its meaning."""

    name: str
    dtype: ColumnType
    description: str
    nullable: bool = False


@dataclass(frozen=True)
class TableSpec:
    """A telemetry table: its name and ordered columns."""

    name: str
    columns: tuple[Column, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def validate(self, row: Row) -> None:
        """Raise if ``row`` has unknown columns, misses a required one or has a wrong type."""
        unknown = sorted(set(row) - set(self.names))
        if unknown:
            raise ValueError(f"{self.name} telemetry has no columns {unknown}")
        for column in self.columns:
            value = row.get(column.name)
            if value is None:
                if not column.nullable:
                    raise ValueError(f"{self.name}.{column.name} is required")
                continue
            expected = _PYTHON_TYPES[column.dtype]
            numeric_bool = isinstance(value, bool) and column.dtype in (
                "int64",
                "float64",
            )
            if numeric_bool or not isinstance(value, expected):
                raise TypeError(
                    f"{self.name}.{column.name} must be {column.dtype}, got {type(value).__name__}"
                )


def _table(name: str, *columns: Column) -> TableSpec:
    return TableSpec(name, columns)


_RUN = Column("run_id", "string", "the run this row belongs to")
_SOURCE = Column(
    "source",
    "string",
    "the data source that produced it (train, eval, demonstrations, ...)",
)

#: One row per decision (one action chunk) ingested from a rollout worker.
TRANSITIONS = _table(
    "transitions",
    _RUN,
    _SOURCE,
    Column(
        "origin",
        "string",
        "where the dynamics came from: real (simulator or robot) or model",
    ),
    Column("worker_index", "int64", "rollout worker that collected it"),
    Column("environment_index", "int64", "global environment index"),
    Column("episode_id", "int64", "episode number within the source"),
    Column("decision_index", "int64", "decision number within the episode"),
    Column("steps", "int64", "environment steps this decision executed"),
    Column("env_steps", "int64", "run total of environment steps after ingesting it"),
    Column("policy_version", "int64", "weight version that chose the action"),
    Column("policy_lag", "int64", "published versions between acting and ingesting"),
    Column("mode", "string", "warmup, train or eval"),
    Column("reward", "float64", "decision reward the replay buffer stores"),
    Column(
        "reward_env", "float64", "sum of the environment's own reward over the chunk"
    ),
    Column(
        "reward_channels",
        "map<string,float64>",
        "per-channel reward sums",
        nullable=True,
    ),
    Column("terminated", "bool", "the episode reached a terminal state"),
    Column("truncated", "bool", "the episode was cut by a time limit"),
    Column("success", "bool", "the task succeeded during this decision"),
    Column("t_obs", "float64", "wall time (s) the worker recorded the decision"),
    Column("t_received", "float64", "wall time (s) the trainer received it"),
    Column(
        "inference_seconds",
        "float64",
        "policy time for this decision (the worker's round total shared over its decisions)",
        nullable=True,
    ),
    Column(
        "env_seconds",
        "float64",
        "simulator time for this decision (the worker's round total shared over its decisions)",
        nullable=True,
    ),
    Column("network_seconds", "float64", "transport time, when known", nullable=True),
)

#: One row per finished episode.
EPISODES = _table(
    "episodes",
    _RUN,
    _SOURCE,
    Column("episode_id", "int64", "episode number within the source"),
    Column("environment_index", "int64", "global environment index"),
    Column("length_decisions", "int64", "decisions in the episode"),
    Column("length_env_steps", "int64", "environment steps in the episode"),
    Column("return", "float64", "undiscounted sum of decision rewards"),
    Column("success", "bool", "the task succeeded at some point"),
    Column("terminal_reason", "string", "terminated, truncated or decision_cap"),
    Column("policy_version_first", "int64", "weight version of the first decision"),
    Column("policy_version_last", "int64", "weight version of the last decision"),
    Column("wall_start", "float64", "wall time (s) of the first observation"),
    Column("wall_end", "float64", "wall time (s) the episode ended"),
)

#: One row per collection round (a collect call answered by every rollout worker).
ROUNDS = _table(
    "rounds",
    _RUN,
    Column("round_index", "int64", "round number within the run"),
    Column("mode", "string", "warmup, train or eval"),
    Column("wall_start", "float64", "wall time (s) the round was submitted"),
    Column("wall_end", "float64", "wall time (s) its results were ingested"),
    Column("env_steps", "int64", "environment steps collected in the round"),
    Column("decisions", "int64", "decisions collected in the round"),
    Column("episodes", "int64", "episodes finished in the round"),
    Column("publish_version", "int64", "latest weight version published when ingested"),
    Column("learner_step", "int64", "learner updates done when ingested"),
    Column("policy_lag_mean", "float64", "mean policy lag of the round's decisions"),
    Column("policy_lag_max", "int64", "largest policy lag of the round's decisions"),
    Column("queue_depth", "int64", "rounds in flight when ingested", nullable=True),
    Column("dropped_stale", "int64", "decisions dropped for a policy lag above max_policy_lag", nullable=True),
)

#: One row per learner update.
LEARNER = _table(
    "learner",
    _RUN,
    Column("learner_step", "int64", "update number"),
    Column("env_steps", "int64", "run total of environment steps at the update"),
    Column("wall_time", "float64", "wall time (s) the update finished"),
    Column(
        "publish_version",
        "int64",
        "weight version published after it, if any",
        nullable=True,
    ),
    Column("metrics", "map<string,float64>", "losses and scalars the learner returned"),
)

TABLES: dict[str, TableSpec] = {
    table.name: table for table in (TRANSITIONS, EPISODES, ROUNDS, LEARNER)
}


@runtime_checkable
class TelemetrySink(Protocol):
    """Where telemetry rows go; each ``write_*`` takes rows of the matching table."""

    def write_transitions(self, rows: Sequence[Row]) -> None: ...

    def write_episodes(self, rows: Sequence[Row]) -> None: ...

    def write_rounds(self, rows: Sequence[Row]) -> None: ...

    def write_learner(self, rows: Sequence[Row]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class NullTelemetrySink:
    """A sink that drops every row (telemetry disabled, or tests)."""

    def write_transitions(self, rows: Sequence[Row]) -> None:
        del rows

    def write_episodes(self, rows: Sequence[Row]) -> None:
        del rows

    def write_rounds(self, rows: Sequence[Row]) -> None:
        del rows

    def write_learner(self, rows: Sequence[Row]) -> None:
        del rows

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


#: Parquet file metadata key holding the schema version (and table name) a part was written with.
SCHEMA_METADATA_KEY = b"rlpt.telemetry"
TELEMETRY_DIR = "telemetry"
_PART_PREFIX = "part-"


def arrow_schema(table: TableSpec) -> pa.Schema:
    """The Arrow schema of ``table``, tagged with :data:`SCHEMA_VERSION` in its metadata."""
    import pyarrow as pa  # base dependency; imported here to keep package import light

    types = {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "map<string,float64>": pa.map_(pa.string(), pa.float64()),
    }
    fields = [pa.field(column.name, types[column.dtype], nullable=column.nullable) for column in table.columns]
    tag = json.dumps({"schema_version": SCHEMA_VERSION, "table": table.name})
    return pa.schema(fields, metadata={SCHEMA_METADATA_KEY: tag.encode()})


class ParquetTelemetrySink:
    """Write telemetry rows as rolling Parquet part files under ``directory/<table>/``.

    Rows are buffered per table and written as a new part when a table holds
    ``flush_rows`` rows or ``flush_seconds`` have passed since its last part.
    Each part is written to a hidden file and renamed into place, so every
    ``part-*.parquet`` is complete even if the process dies; at most the rows
    buffered since the last part are lost. The first row of every table is
    checked against the schema; Arrow checks the column types of every row.
    """

    def __init__(self, directory: str | os.PathLike[str], *, flush_rows: int = 10_000, flush_seconds: float = 30.0):
        if isinstance(flush_rows, bool) or not isinstance(flush_rows, int) or flush_rows < 1:
            raise ValueError(f"flush_rows must be a positive integer, got {flush_rows!r}")
        if flush_seconds < 0:
            raise ValueError(f"flush_seconds must be non-negative, got {flush_seconds!r}")
        self.directory = os.fspath(directory)
        #: The run's id: the name of the run directory the telemetry directory sits in.
        self.run_id = os.path.basename(os.path.dirname(os.path.abspath(self.directory)))
        self.flush_rows = flush_rows
        self.flush_seconds = float(flush_seconds)
        self._rows: dict[str, list[Row]] = {name: [] for name in TABLES}
        self._parts: dict[str, int] = dict.fromkeys(TABLES, 0)
        self._last_flush: dict[str, float] = dict.fromkeys(TABLES, time.monotonic())
        self._checked: set[str] = set()
        self._closed = False

    def write_transitions(self, rows: Sequence[Row]) -> None:
        self._write(TRANSITIONS, rows)

    def write_episodes(self, rows: Sequence[Row]) -> None:
        self._write(EPISODES, rows)

    def write_rounds(self, rows: Sequence[Row]) -> None:
        self._write(ROUNDS, rows)

    def write_learner(self, rows: Sequence[Row]) -> None:
        self._write(LEARNER, rows)

    def _write(self, table: TableSpec, rows: Sequence[Row]) -> None:
        if self._closed:
            raise RuntimeError("telemetry sink is closed")
        if not rows:
            return
        if table.name not in self._checked:
            table.validate(rows[0])
            self._checked.add(table.name)
        pending = self._rows[table.name]
        pending.extend(rows)
        due = time.monotonic() - self._last_flush[table.name] >= self.flush_seconds
        if len(pending) >= self.flush_rows or due:
            self._flush_table(table)

    def _flush_table(self, table: TableSpec) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows, self._rows[table.name] = self._rows[table.name], []
        self._last_flush[table.name] = time.monotonic()
        if not rows:
            return
        directory = os.path.join(self.directory, table.name)
        os.makedirs(directory, exist_ok=True)
        index = self._parts[table.name]
        self._parts[table.name] += 1
        arrow = pa.Table.from_pylist([_complete(table, row) for row in rows], schema=arrow_schema(table))
        final = os.path.join(directory, f"{_PART_PREFIX}{index:05d}.parquet")
        staging = os.path.join(directory, f".{_PART_PREFIX}{index:05d}.parquet.tmp")
        pq.write_table(arrow, staging)
        os.replace(staging, final)

    def flush(self) -> None:
        """Write every buffered row now."""
        for table in TABLES.values():
            self._flush_table(table)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True


def _complete(table: TableSpec, row: Row) -> dict[str, Any]:
    """``row`` with every column present (missing nullable columns as ``None``)."""
    return {name: row.get(name) for name in table.names}


def open_run_telemetry(
    run_dir: str | os.PathLike[str],
    *,
    config: Any = None,
    placement: Any = None,
    flush_rows: int = 10_000,
    flush_seconds: float = 30.0,
) -> ParquetTelemetrySink:
    """A Parquet sink under ``<run_dir>/telemetry/``, with ``config.yaml`` and ``placement.json`` beside it.

    ``config`` is an OmegaConf config or a plain mapping; ``placement`` a
    dataclass (a placement plan) or a mapping. Either is skipped when ``None``.
    """
    directory = os.path.join(os.fspath(run_dir), TELEMETRY_DIR)
    os.makedirs(directory, exist_ok=True)
    if config is not None:
        with open(os.path.join(directory, "config.yaml"), "w", encoding="utf-8") as stream:
            stream.write(_config_yaml(config))
    if placement is not None:
        with open(os.path.join(directory, "placement.json"), "w", encoding="utf-8") as stream:
            json.dump(_plain(placement), stream, indent=2, sort_keys=True, default=repr)
    return ParquetTelemetrySink(directory, flush_rows=flush_rows, flush_seconds=flush_seconds)


def _config_yaml(config: Any) -> str:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        return OmegaConf.to_yaml(config, resolve=False)
    return OmegaConf.to_yaml(OmegaConf.create(_plain(config)))


def _plain(value: Any) -> Any:
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(entry) for entry in value]
    return value


def read_telemetry(run_dir: str | os.PathLike[str], table: str) -> pa.Table:
    """Every part of one table of a run as one Arrow table (empty when the run wrote none).

    ``run_dir`` is the run directory or its ``telemetry`` directory. Parts
    written with a newer schema version than this reader knows are refused.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if table not in TABLES:
        raise ValueError(f"unknown telemetry table {table!r}; known: {sorted(TABLES)}")
    root = os.fspath(run_dir)
    if os.path.basename(os.path.normpath(root)) != TELEMETRY_DIR:
        root = os.path.join(root, TELEMETRY_DIR)
    directory = os.path.join(root, table)
    schema = arrow_schema(TABLES[table])
    names = sorted(
        name
        for name in (os.listdir(directory) if os.path.isdir(directory) else ())
        if name.startswith(_PART_PREFIX) and name.endswith(".parquet")
    )
    parts = []
    for name in names:
        part = pq.read_table(os.path.join(directory, name))
        tag = json.loads((part.schema.metadata or {}).get(SCHEMA_METADATA_KEY, b"{}"))
        version = tag.get("schema_version")
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise ValueError(f"{directory}/{name} has telemetry schema version {version!r}; this reader knows 1..{SCHEMA_VERSION}")
        parts.append(part.cast(schema))
    if not parts:
        return schema.empty_table()
    return pa.concat_tables(parts)
