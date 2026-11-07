# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a workflow's ``results/results.json`` into its README results table, and check that the two agree.

python -m example_workflows._common.results_table <results.json>                 # print the table
python -m example_workflows._common.results_table <results.json> --write README  # replace the table in the README
python -m example_workflows._common.results_table <results.json> --check README  # exit 1 when they differ
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def format_rate(successes: int, episodes: int) -> str:
    """``86.5% n=200``: the rate and the episode count it was measured over."""
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if not 0 <= successes <= episodes:
        raise ValueError(f"successes must lie in [0, {episodes}], got {successes}")
    return f"{100 * successes / episodes:.1f}% n={episodes}"


START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"
_ROW_KEYS = {"task", "method", "base", "peak", "final"}
_OPTIONAL_ROW_KEYS = {"note", "date"}
_COUNT_KEYS = {"successes", "episodes"}


@dataclass(frozen=True)
class Count:
    successes: int
    episodes: int

    @property
    def rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class Row:
    task: str
    method: str
    base: Count
    peak: Count | None
    final: Count | None
    note: str
    date: str = ""

    @property
    def provenance(self) -> str:
        """The note with the row's own date appended, when it has one."""
        if not self.date:
            return self.note
        return f"{self.note}; {self.date}" if self.note else self.date


@dataclass(frozen=True)
class Results:
    workflow: str
    evaluator: str
    date: str
    rows: tuple[Row, ...]


def _count(owner: str, value: object) -> Count:
    if not isinstance(value, dict) or set(value) != _COUNT_KEYS:
        raise ValueError(
            f"{owner} must be an object with integer keys {sorted(_COUNT_KEYS)}, got {value!r}"
        )
    successes, episodes = value["successes"], value["episodes"]
    if (
        not isinstance(successes, int)
        or not isinstance(episodes, int)
        or isinstance(successes, bool)
    ):
        raise TypeError(
            f"{owner}: successes and episodes must be integers, got {value!r}"
        )
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError(
            f"{owner}: need 0 <= successes <= episodes and episodes > 0, got {value!r}"
        )
    return Count(successes, episodes)


def _optional_count(owner: str, value: object) -> Count | None:
    """``None`` for a declared-null measurement; otherwise a validated count.

    The key must be present either way, so a row states what it does not have
    instead of leaving the reader to guess.
    """
    return None if value is None else _count(owner, value)


def load_results(path: Path) -> Results:
    """Parse and validate a ``results.json``; every problem names the offending field."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("workflow", "evaluator", "date"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{path}: '{key}' must be a non-empty string")
    if not isinstance(data.get("rows"), list):
        raise TypeError(f"{path}: 'rows' must be a list")
    rows = []
    for index, row in enumerate(data["rows"]):
        owner = f"{path}: rows[{index}]"
        if (
            not isinstance(row, dict)
            or not _ROW_KEYS <= set(row)
            or not set(row) <= _ROW_KEYS | _OPTIONAL_ROW_KEYS
        ):
            raise ValueError(
                f"{owner} must have keys {sorted(_ROW_KEYS)} and optionally {sorted(_OPTIONAL_ROW_KEYS)}, got {sorted(row)}"
            )
        note, date = row.get("note", ""), row.get("date", "")
        if not all(
            isinstance(value, str) for value in (row["task"], row["method"], note, date)
        ):
            raise TypeError(f"{owner}: task, method, note and date must be strings")
        rows.append(
            Row(
                row["task"],
                row["method"],
                _count(f"{owner}.base", row["base"]),
                _optional_count(f"{owner}.peak", row["peak"]),
                _optional_count(f"{owner}.final", row["final"]),
                note,
                date,
            )
        )
    return Results(data["workflow"], data["evaluator"], data["date"], tuple(rows))


def _percent(count: Count) -> str:
    return f"{100 * count.rate:.0f}%"


def render_summary(results: Results) -> str:
    """One row per task, one column per method: the base rate, then the best checkpoints' rate pooled over seeds.

    Rows that share a task and a method are seeds of one setting; they must share their base measurement.
    """
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in results.rows:
        groups.setdefault((row.task, row.method), []).append(row)
    tasks = list(dict.fromkeys(row.task for row in results.rows))
    methods = list(dict.fromkeys(row.method for row in results.rows))
    episodes = {row.base.episodes for row in results.rows} | {
        row.peak.episodes for row in results.rows if row.peak is not None
    }
    if len(episodes) != 1:
        raise ValueError(
            f"a summary table states one episode count; got {sorted(episodes)}"
        )
    seeds = {len(group) for group in groups.values()}
    lines = [
        "| Task | " + " | ".join(methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for task in tasks:
        cells = []
        for method in methods:
            group = groups.get((task, method), [])
            if len({row.base for row in group}) > 1:
                raise ValueError(f"{task}, {method}: the seeds have different bases")
            bests = [row.peak for row in group if row.peak is not None]
            if not group:
                cells.append("—")
                continue
            after = "—"
            if bests:
                pooled = Count(
                    sum(c.successes for c in bests), sum(c.episodes for c in bests)
                )
                after = _percent(pooled)
            cells.append(f"{_percent(group[0].base)} → {after}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    (n,) = episodes
    lines += [
        "",
        "Success rate of the frozen base policy → after post-training.",
        (
            "Each training run contributes its best checkpoint, chosen by its in-training evaluations "
            f"and re-scored on {n} held-out episodes; the {max(seeds)} training seeds are pooled."
        ),
        "The last checkpoints, each seed and the evaluator are in `results/results.json`.",
    ]
    return "\n".join(lines)


def render(results: Results) -> str:
    """The markdown block that goes between the README markers; :func:`render_summary` when seeds repeat a setting."""
    if not results.rows:
        return "No validated evaluation results have been published yet."
    if len({(row.task, row.method) for row in results.rows}) < len(results.rows):
        return render_summary(results)
    lines = [
        "| Task | Method | Base | Peak | Final | Note |",
        "|---|---|---|---|---|---|",
    ]
    for row in results.rows:
        cells = [
            format_rate(c.successes, c.episodes) if c is not None else "—"
            for c in (row.base, row.peak, row.final)
        ]
        lines.append(
            f"| {row.task} | {row.method} | {cells[0]} | {cells[1]} | {cells[2]} | {row.provenance} |"
        )
    lines.append("")
    dated_rows = any(row.date for row in results.rows)
    lines.append(
        f"Success rates; n is the episode count. "
        f"Evaluator: {results.evaluator}. Measured {results.date}"
        + (", except where a row names its own." if dated_rows else ".")
    )
    return "\n".join(lines)


def _split(readme_text: str, readme: Path) -> tuple[str, str, str]:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            f"{readme} needs a '{START_MARKER}' ... '{END_MARKER}' block in its Results section"
        )
    head = readme_text[: start + len(START_MARKER)]
    return head, readme_text[start + len(START_MARKER) : end], readme_text[end:]


def write_table(results: Results, readme: Path) -> None:
    head, _, tail = _split(readme.read_text(encoding="utf-8"), readme)
    text = f"{head}\n{render(results)}\n{tail}"
    readme.write_text(text, encoding="utf-8")  # NOSONAR - user-named CLI path


def table_matches(results: Results, readme: Path) -> bool:
    _, current, _ = _split(readme.read_text(encoding="utf-8"), readme)
    return current.strip() == render(results).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results", type=Path, help="path to a workflow's results/results.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        type=Path,
        metavar="README",
        help="replace the table between the README markers",
    )
    group.add_argument(
        "--check",
        type=Path,
        metavar="README",
        help="exit 1 when the README table differs",
    )
    args = parser.parse_args(argv)
    results = load_results(args.results)
    if args.write is not None:
        write_table(results, args.write)
        return 0
    if args.check is not None:
        if table_matches(results, args.check):
            return 0
        print(
            f"{args.check}: results table is out of date; run with --write {args.check}",
            file=sys.stderr,
        )
        return 1
    print(render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
