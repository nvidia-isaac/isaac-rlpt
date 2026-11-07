# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The throughput results of this workflow: ``results/throughput.json``, its checks, and the README tables.

Every number a layout comparison rests on lives in the JSON with its provenance: the W&B run
it was read from, the recipe, and the runtime it was measured on. This module validates the
file, renders the README tables between the ``<!-- throughput:start -->`` and
``<!-- throughput:end -->`` markers, and checks that they still agree.

    python -m example_workflows.dgx_example.throughput results/throughput.json             # print the tables
    python -m example_workflows.dgx_example.throughput results/throughput.json --write README.md
    python -m example_workflows.dgx_example.throughput results/throughput.json --check README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from example_workflows.dgx_example.readme_blocks import block_matches, write_block

THROUGHPUT_BLOCK = "throughput"
STATUSES = ("finished", "out_of_memory")
#: Where a layout's rollout worker runs: inside the learner process (``native``) or as a Ray
#: actor placed by the RLinf runtime (``rlinf``); see docs/runtimes.md.
ROLLOUT_RUNTIMES = ("native", "rlinf")
#: When a layout's learner collects and trains (``schedule.kind``): one after the other
#: (``sync``) or the next round already collecting during the burst (``overlapped``).
SCHEDULES = ("sync", "overlapped")
RECIPE_KEYS = (
    "task",
    "env_steps_per_env",
    "min_env_steps",
    "action_exec_len",
    "max_updates_per_burst",
    "sac_batch_size",
    "image_shortest_edge",
    "decode_whole_batch",
    "evaluation",
)
NO_VALUE = "—"


@dataclass(frozen=True)
class Gpu:
    key: str
    label: str
    host: str


@dataclass(frozen=True)
class Layout:
    key: str
    label: str
    shape: str
    simulator_gpu: str
    learner_gpu: str
    color: str
    rollout_runtime: str
    schedule: str
    #: How many simulator servers host the training environments (one batched worker each).
    simulator_servers: int

    def same_placement(self, other: Layout) -> bool:
        """The same cards in the same roles: what a runtime or schedule comparison holds fixed."""
        return (
            self.shape == other.shape
            and self.simulator_gpu == other.simulator_gpu
            and self.learner_gpu == other.learner_gpu
        )


@dataclass(frozen=True)
class Rate:
    """Environment steps per second over the run's iterations (collection cycles), mean ± std."""

    mean: float
    std: float
    cycles: int


@dataclass(frozen=True)
class Row:
    """One layout at one environment count.

    ``rate`` is the collection rate (steps over a round's own wall time, what a
    placement changes); ``iteration_rate`` is the run's rate (steps over the whole
    iteration: round, burst and publish, what a schedule changes), with the
    per-iteration medians of those three walls next to it. Runs from before the
    trainer logged them carry none.
    """

    simulator: str
    layout: str
    num_envs: int
    status: str
    rate: Rate | None
    env_step_s: float | None
    decode_s: float | None
    sac_update_s: float | None
    peak_gib_learner: float | None
    peak_gib_simulator: float | None
    wire_ms: float | None
    wandb_run: str | None
    note: str
    iteration_rate: Rate | None = None
    collect_s: float | None = None
    train_s: float | None = None
    publish_s: float | None = None


@dataclass(frozen=True)
class Throughput:
    workflow: str
    date: str
    runtime: str
    recipe: Mapping[str, Any]
    gpus: tuple[Gpu, ...]
    layouts: tuple[Layout, ...]
    rows: tuple[Row, ...]

    def layout(self, key: str) -> Layout:
        return next(layout for layout in self.layouts if layout.key == key)

    def series(self, layout: str) -> list[Row]:
        """The rows of one layout, by environment count."""
        return sorted(
            (row for row in self.rows if row.layout == layout),
            key=lambda row: row.num_envs,
        )

    def row(self, layout: str, num_envs: int) -> Row | None:
        for candidate in self.rows:
            if candidate.layout == layout and candidate.num_envs == num_envs:
                return candidate
        return None

    @property
    def env_counts(self) -> list[int]:
        return sorted({row.num_envs for row in self.rows})

    @property
    def split_layouts(self) -> list[Layout]:
        return [layout for layout in self.layouts if layout.shape == "split"]

    @property
    def runtime_layouts(self) -> list[Layout]:
        """The layouts whose rollout worker a runtime other than the native one places."""
        return [layout for layout in self.layouts if layout.rollout_runtime != "native"]

    @property
    def schedule_layouts(self) -> list[Layout]:
        """The layouts whose learner runs a schedule other than the sync one."""
        return [layout for layout in self.layouts if layout.schedule != "sync"]

    @property
    def server_layouts(self) -> list[Layout]:
        """The layouts whose training environments several simulator servers host."""
        return [layout for layout in self.layouts if layout.simulator_servers > 1]

    def native_counterpart(self, layout: Layout) -> Layout | None:
        """The native-runtime layout on the same cards, schedule and servers, the baseline of a runtime comparison."""
        for candidate in self.layouts:
            if (
                candidate.rollout_runtime == "native"
                and candidate.schedule == layout.schedule
                and candidate.simulator_servers == layout.simulator_servers
                and candidate.same_placement(layout)
            ):
                return candidate
        return None

    def sync_counterpart(self, layout: Layout) -> Layout | None:
        """The sync-schedule layout on the same cards, runtime and servers, the baseline of a schedule comparison."""
        for candidate in self.layouts:
            if (
                candidate.schedule == "sync"
                and candidate.rollout_runtime == layout.rollout_runtime
                and candidate.simulator_servers == layout.simulator_servers
                and candidate.same_placement(layout)
            ):
                return candidate
        return None

    def one_server_counterpart(self, layout: Layout) -> Layout | None:
        """The one-server layout on the same cards, runtime and schedule, the baseline of a server-count comparison."""
        for candidate in self.layouts:
            if (
                candidate.simulator_servers == 1
                and candidate.rollout_runtime == layout.rollout_runtime
                and candidate.schedule == layout.schedule
                and candidate.same_placement(layout)
            ):
                return candidate
        return None

    def single_layouts_for(self, split: Layout) -> list[Layout]:
        """The single-card layouts a split's speed-up is measured against: one card, the same runtime, schedule and servers."""
        return [
            layout
            for layout in self.layouts
            if layout.shape == "single"
            and layout.rollout_runtime == split.rollout_runtime
            and layout.schedule == split.schedule
            and layout.simulator_servers == split.simulator_servers
        ]


def _string(owner: str, data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}: '{key}' must be a non-empty string")
    return value


def _optional_number(owner: str, data: Mapping[str, Any], key: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{owner}: '{key}' must be a non-negative number or null")
    return float(value)


def _rate(owner: str, data: Mapping[str, Any]) -> Rate:
    for key in ("mean", "std", "cycles"):
        if key not in data:
            raise ValueError(f"{owner}: env_steps_per_s needs '{key}'")
    mean, std, cycles = data["mean"], data["std"], data["cycles"]
    if not isinstance(mean, int | float) or mean <= 0:
        raise ValueError(f"{owner}: env_steps_per_s.mean must be positive")
    if not isinstance(std, int | float) or std < 0:
        raise ValueError(f"{owner}: env_steps_per_s.std must be non-negative")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError(f"{owner}: env_steps_per_s.cycles must be a positive integer")
    return Rate(float(mean), float(std), cycles)


def _row_rate(owner: str, data: Mapping[str, Any], status: str) -> Rate | None:
    """A finished row's collection rate; other rows carry none."""
    if status == "finished":
        if "env_steps_per_s" not in data:
            raise ValueError(f"{owner}: a finished row needs 'env_steps_per_s'")
        return _rate(owner, data["env_steps_per_s"])
    if "env_steps_per_s" in data:
        raise ValueError(f"{owner}: a {status} row carries no throughput")
    return None


def _row_iteration_rate(
    owner: str, data: Mapping[str, Any], status: str
) -> Rate | None:
    if data.get("iteration_env_steps_per_s") is None:
        return None
    if status != "finished":
        raise ValueError(f"{owner}: a {status} row carries no throughput")
    return _rate(owner, data["iteration_env_steps_per_s"])


def _row(owner: str, data: Mapping[str, Any], layouts: Mapping[str, Layout]) -> Row:
    simulator = _string(owner, data, "simulator")
    layout = _string(owner, data, "layout")
    if layout not in layouts:
        raise ValueError(
            f"{owner}: unknown layout {layout!r}; layouts are {sorted(layouts)}"
        )
    num_envs = data.get("num_envs")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError(f"{owner}: 'num_envs' must be a positive integer")
    status = _string(owner, data, "status")
    if status not in STATUSES:
        raise ValueError(f"{owner}: 'status' must be one of {STATUSES}")
    rate = _row_rate(owner, data, status)
    peak = data.get("peak_gib", {})
    if not isinstance(peak, Mapping):
        raise TypeError(
            f"{owner}: 'peak_gib' must be a mapping with 'learner' and 'simulator'"
        )
    wandb_run = data.get("wandb_run")
    if wandb_run is not None and (not isinstance(wandb_run, str) or not wandb_run):
        raise ValueError(f"{owner}: 'wandb_run' must be a run id or null")
    note = data.get("note", "")
    if not isinstance(note, str):
        raise TypeError(f"{owner}: 'note' must be a string")
    iteration_rate = _row_iteration_rate(owner, data, status)
    return Row(
        simulator=simulator,
        layout=layout,
        num_envs=num_envs,
        status=status,
        rate=rate,
        env_step_s=_optional_number(owner, data, "env_step_s"),
        decode_s=_optional_number(owner, data, "decode_s"),
        sac_update_s=_optional_number(owner, data, "sac_update_s"),
        peak_gib_learner=_optional_number(f"{owner}.peak_gib", peak, "learner"),
        peak_gib_simulator=_optional_number(f"{owner}.peak_gib", peak, "simulator"),
        wire_ms=_optional_number(owner, data, "wire_ms"),
        wandb_run=wandb_run,
        note=note,
        iteration_rate=iteration_rate,
        collect_s=_optional_number(owner, data, "collect_s"),
        train_s=_optional_number(owner, data, "train_s"),
        publish_s=_optional_number(owner, data, "publish_s"),
    )


def _recipe(owner: str, data: Mapping[str, Any]) -> Mapping[str, Any]:
    recipe = data.get("recipe")
    if not isinstance(recipe, Mapping):
        raise TypeError(f"{owner}: 'recipe' must be a mapping")
    missing = [key for key in RECIPE_KEYS if key not in recipe]
    if missing:
        raise ValueError(f"{owner}: recipe lacks {missing}")
    return recipe


def _layout(
    layout_owner: str,
    key: str,
    layout: Mapping[str, Any],
    gpus_data: Mapping[str, Any],
) -> Layout:
    shape = _string(layout_owner, layout, "shape")
    if shape not in ("single", "split"):
        raise ValueError(f"{layout_owner}: shape must be single or split")
    simulator_gpu = _string(layout_owner, layout, "simulator_gpu")
    learner_gpu = _string(layout_owner, layout, "learner_gpu")
    for gpu in (simulator_gpu, learner_gpu):
        if gpu not in gpus_data:
            raise ValueError(f"{layout_owner}: unknown gpu {gpu!r}")
    if (shape == "split") != (simulator_gpu != learner_gpu):
        raise ValueError(f"{layout_owner}: a split uses two cards, a single layout one")
    rollout_runtime = _string(layout_owner, layout, "rollout_runtime")
    if rollout_runtime not in ROLLOUT_RUNTIMES:
        raise ValueError(
            f"{layout_owner}: rollout_runtime must be one of {ROLLOUT_RUNTIMES}"
        )
    schedule = _string(layout_owner, layout, "schedule")
    if schedule not in SCHEDULES:
        raise ValueError(f"{layout_owner}: schedule must be one of {SCHEDULES}")
    servers = layout.get("simulator_servers")
    if isinstance(servers, bool) or not isinstance(servers, int) or servers < 1:
        raise ValueError(
            f"{layout_owner}: 'simulator_servers' must be a positive integer"
        )
    return Layout(
        key,
        _string(layout_owner, layout, "label"),
        shape,
        simulator_gpu,
        learner_gpu,
        _string(layout_owner, layout, "color"),
        rollout_runtime,
        schedule,
        servers,
    )


def _check_unique_rows(owner: str, rows: tuple[Row, ...]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        identity = (row.simulator, row.layout, row.num_envs)
        if identity in seen:
            raise ValueError(f"{owner}: duplicate row for {identity}")
        seen.add(identity)


def load_throughput(path: Path) -> Throughput:
    """Parse and validate a ``throughput.json``; every problem names the offending field."""
    data = json.loads(path.read_text(encoding="utf-8"))
    owner = str(path)
    for key in ("workflow", "date", "runtime"):
        _string(owner, data, key)
    recipe = _recipe(owner, data)
    gpus_data = data.get("gpus")
    if not isinstance(gpus_data, Mapping) or not gpus_data:
        raise ValueError(f"{owner}: 'gpus' must be a non-empty mapping")
    gpus = tuple(
        Gpu(
            key,
            _string(f"{owner}: gpus.{key}", gpu, "label"),
            _string(f"{owner}: gpus.{key}", gpu, "host"),
        )
        for key, gpu in gpus_data.items()
    )
    layouts_data = data.get("layouts")
    if not isinstance(layouts_data, Mapping) or not layouts_data:
        raise ValueError(f"{owner}: 'layouts' must be a non-empty mapping")
    layouts = {
        key: _layout(f"{owner}: layouts.{key}", key, layout, gpus_data)
        for key, layout in layouts_data.items()
    }
    rows_data = data.get("rows")
    if not isinstance(rows_data, list) or not rows_data:
        raise ValueError(f"{owner}: 'rows' must be a non-empty list")
    rows = tuple(
        _row(f"{owner}: rows[{index}]", row, layouts)
        for index, row in enumerate(rows_data)
    )
    _check_unique_rows(owner, rows)
    return Throughput(
        data["workflow"],
        data["date"],
        data["runtime"],
        recipe,
        gpus,
        tuple(layouts.values()),
        rows,
    )


def _rate_cell(row: Row | None) -> str:
    if row is None:
        return NO_VALUE
    if row.rate is None:
        return row.status.replace("_", " ")
    return f"{row.rate.mean:.0f} ± {row.rate.std:.0f}"


def _number(value: float | None, digits: int) -> str:
    return NO_VALUE if value is None else f"{value:.{digits}f}"


def _memory_cell(row: Row) -> str:
    if row.peak_gib_learner is None:
        return NO_VALUE
    if row.peak_gib_simulator is None:
        return f"{row.peak_gib_learner:.0f}"
    return (
        f"{row.peak_gib_simulator:.0f} simulator + {row.peak_gib_learner:.0f} learner"
    )


def render_throughput_table(results: Throughput) -> str:
    lines = [
        "| N envs | " + " | ".join(layout.label for layout in results.layouts) + " |",
        "|---|" + "---|" * len(results.layouts),
    ]
    for num_envs in results.env_counts:
        cells = [
            _rate_cell(results.row(layout.key, num_envs)) for layout in results.layouts
        ]
        lines.append(f"| {num_envs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _speedup_cell(numerator: Rate, other: Row | None) -> str:
    if other is None or other.rate is None:
        return NO_VALUE
    return f"{numerator.mean / other.rate.mean:.2f}×"


def render_speedup_table(results: Throughput) -> str:
    """Each split layout's throughput divided by every single-card layout's on the same runtime, per N."""
    blocks = []
    for split in results.split_layouts:
        others = results.single_layouts_for(split)
        if not others:
            continue
        lines = [
            "| N envs | "
            + " | ".join(f"vs {layout.label}" for layout in others)
            + " |",
            "|---|" + "---|" * len(others),
        ]
        for num_envs in results.env_counts:
            numerator = results.row(split.key, num_envs)
            if numerator is None or numerator.rate is None:
                continue
            cells = [
                _speedup_cell(numerator.rate, results.row(layout.key, num_envs))
                for layout in others
            ]
            lines.append(f"| {num_envs} | " + " | ".join(cells) + " |")
        blocks.append(
            f"**{split.label}** over each single card:\n\n" + "\n".join(lines)
        )
    return "\n\n".join(blocks)


def render_rollout_runtime_table(results: Throughput) -> str:
    """A runtime-placed layout against the native layout on the same cards, per environment count.

    Only sync-schedule layouts: a runtime comparison rests on the collection rate, and under
    the overlapped schedule the round shares the learner's card with the burst, so its rate is
    not the placement's alone.
    """
    lines = [
        (
            "| Placement | N envs | env-steps/s, native | env-steps/s, RLinf | RLinf ÷ native "
            "| simulator step (s), native / RLinf | decode (s), native / RLinf | SAC update (s), native / RLinf "
            "| peak GPU memory (GiB), native / RLinf |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for layout in results.runtime_layouts:
        native = results.native_counterpart(layout)
        if native is None or layout.schedule != "sync":
            continue
        for row in results.series(layout.key):
            baseline = results.row(native.key, row.num_envs)
            if row.rate is None or baseline is None or baseline.rate is None:
                continue
            lines.append(
                f"| {native.label} | {row.num_envs} | {baseline.rate.mean:.0f} ± {baseline.rate.std:.0f} "
                f"| {row.rate.mean:.0f} ± {row.rate.std:.0f} | {row.rate.mean / baseline.rate.mean:.2f}× "
                f"| {_number(baseline.env_step_s, 2)} / {_number(row.env_step_s, 2)} "
                f"| {_number(baseline.decode_s, 2)} / {_number(row.decode_s, 2)} "
                f"| {_number(baseline.sac_update_s, 2)} / {_number(row.sac_update_s, 2)} "
                f"| {_memory_cell(baseline)} / {_memory_cell(row)} |"
            )
    return "\n".join(lines)


def _walls_cell(row: Row) -> str:
    return f"{_number(row.collect_s, 0)} / {_number(row.train_s, 0)} / {_number(row.publish_s, 0)}"


def render_schedule_table(results: Throughput) -> str:
    """A layout on a non-sync schedule against the sync layout on the same cards and runtime, per N.

    The compared number is the run's rate over whole iterations. The iteration's three walls
    (round, burst, publish) show where the difference comes from, and the per-decision decode
    and per-update costs show what the overlap costs each side. The round wall is the round's
    own end, stamped where the round ran, so under the overlapped schedule it is shorter than the iteration
    whenever the burst outlasts it.
    """
    lines = [
        (
            "| Placement | N envs | env-steps/s per iteration, sync | env-steps/s per iteration, overlapped "
            "| overlapped ÷ sync | round / burst / publish (s), sync | round / burst / publish (s), overlapped "
            "| decode (s), sync / overlapped | SAC update (s), sync / overlapped |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for layout in results.schedule_layouts:
        sync = results.sync_counterpart(layout)
        if sync is None:
            continue
        for row in results.series(layout.key):
            baseline = results.row(sync.key, row.num_envs)
            if (
                row.iteration_rate is None
                or row.rate is None
                or baseline is None
                or baseline.iteration_rate is None
                or baseline.rate is None
            ):
                continue
            lines.append(
                f"| {sync.label} | {row.num_envs} "
                f"| {baseline.iteration_rate.mean:.0f} ± {baseline.iteration_rate.std:.0f} "
                f"| {row.iteration_rate.mean:.0f} ± {row.iteration_rate.std:.0f} "
                f"| {row.iteration_rate.mean / baseline.iteration_rate.mean:.2f}× "
                f"| {_walls_cell(baseline)} | {_walls_cell(row)} "
                f"| {_number(baseline.decode_s, 2)} / {_number(row.decode_s, 2)} "
                f"| {_number(baseline.sac_update_s, 2)} / {_number(row.sac_update_s, 2)} |"
            )
    return "\n".join(lines)


def render_servers_table(results: Throughput) -> str:
    """A layout whose environments several servers host against the one-server layout on the same cards,
    runtime and schedule, per N: the collection rate, and the per-step and per-decision costs that show
    whether the servers stepped while the other's frames decoded."""
    lines = [
        (
            "| Placement | N envs | servers | env-steps/s, one server | env-steps/s, several | several ÷ one "
            "| env-steps/s per iteration, one / several | simulator step (s), one / several "
            "| decode (s), one / several | peak GPU memory (GiB), one / several |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for layout in results.server_layouts:
        one = results.one_server_counterpart(layout)
        if one is None:
            continue
        for row in results.series(layout.key):
            baseline = results.row(one.key, row.num_envs)
            if row.rate is None or baseline is None or baseline.rate is None:
                continue
            lines.append(
                f"| {one.label} | {row.num_envs} | {layout.simulator_servers} "
                f"| {baseline.rate.mean:.0f} ± {baseline.rate.std:.0f} | {row.rate.mean:.0f} ± {row.rate.std:.0f} "
                f"| {row.rate.mean / baseline.rate.mean:.2f}× "
                f"| {_iteration_rate_cell(baseline)} / {_iteration_rate_cell(row)} "
                f"| {_number(baseline.env_step_s, 2)} / {_number(row.env_step_s, 2)} "
                f"| {_number(baseline.decode_s, 2)} / {_number(row.decode_s, 2)} "
                f"| {_memory_cell(baseline)} / {_memory_cell(row)} |"
            )
    return "\n".join(lines)


def _iteration_rate_cell(row: Row) -> str:
    if row.iteration_rate is None:
        return NO_VALUE
    return f"{row.iteration_rate.mean:.0f}"


def render_breakdown_table(results: Throughput) -> str:
    lines = [
        "| Layout | N envs | simulator step (s) | policy decode (s) | SAC update (s) | wire (ms/step) | peak GPU memory (GiB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for layout in results.layouts:
        for row in results.series(layout.key):
            if row.status != "finished":
                continue
            lines.append(
                f"| {layout.label} | {row.num_envs} | {_number(row.env_step_s, 2)} | {_number(row.decode_s, 2)} "
                f"| {_number(row.sac_update_s, 2)} | {_number(row.wire_ms, 1)} | {_memory_cell(row)} |"
            )
    return "\n".join(lines)


def render(results: Throughput) -> str:
    """The markdown block that goes between the README markers."""
    recipe = results.recipe
    if recipe["decode_whole_batch"]:
        decoding = "all N environments in one forward"
    else:
        decoding = "in the simulator entry's micro-batches"
    parts = [
        "Collection throughput in environment steps per second, mean ± std over the run's collection cycles:",
        render_throughput_table(results),
    ]
    if results.split_layouts:
        parts.append(render_speedup_table(results))
    if results.runtime_layouts:
        parts.append(
            "The same cards and environment count with the rollout worker placed by the RLinf runtime "
            "(`runtime=rlinf`: the worker is a Ray actor in its own process) against the native runtime "
            "(the worker inside the learner process):"
        )
        parts.append(render_rollout_runtime_table(results))
    if results.schedule_layouts:
        parts.append(
            "The same cards and environment count with the learner on the overlapped schedule "
            "(`schedule.kind=overlapped`: the next round collects during the burst) against the sync "
            "schedule, in environment steps per second over whole iterations (round, burst and publish). "
            "Under the overlapped schedule the round wall is the round's own end (stamped where the round ran), so it is "
            "shorter than the iteration whenever the burst outlasts it; the per-decision decode is the round's "
            "own cost:"
        )
        parts.append(render_schedule_table(results))
    if results.server_layouts:
        parts.append(
            "The same cards, runtime and schedule with the training environments spread over several "
            "simulator servers, one batched rollout worker each, so one server steps while another's frames "
            "decode, against one server hosting them all:"
        )
        parts.append(render_servers_table(results))
    parts.extend(
        [
            "Where one collection cycle's time goes (medians over the run), and what the cards hold:",
            render_breakdown_table(results),
            (
                f"Recipe: task {recipe['task']}, {recipe['env_steps_per_env']} environment steps per environment "
                f"(at least {recipe['min_env_steps']}), {recipe['action_exec_len']} simulator steps per policy "
                f"decision, SAC bursts capped at {recipe['max_updates_per_burst']} updates of batch "
                f"{recipe['sac_batch_size']}, frames shrunk to a {recipe['image_shortest_edge']} px shortest edge, "
                f"the policy decoding {decoding}, "
                f"{'with' if recipe['evaluation'] else 'no'} evaluation simulator. Runtime: {results.runtime}. "
                f"Measured {results.date}."
            ),
        ]
    )
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("throughput", type=Path, help="path of throughput.json")
    parser.add_argument(
        "--write",
        type=Path,
        metavar="README",
        help="replace the README's throughput block",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="README",
        help="verify the README's throughput block",
    )
    parser.add_argument(
        "--block",
        default=THROUGHPUT_BLOCK,
        help="the README block the tables go between (default: throughput)",
    )
    args = parser.parse_args(argv)
    body = render(load_throughput(args.throughput))
    if args.write is not None:
        write_block(args.write, args.block, body)
        return 0
    if args.check is not None:
        if not block_matches(args.check, args.block, body):
            print(
                f"{args.check}: the throughput tables are out of date; run this command with --write",
                file=sys.stderr,
            )
            return 1
        print(f"{args.check}: throughput tables are current")
        return 0
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
