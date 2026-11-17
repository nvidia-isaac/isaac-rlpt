# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fill ``results/throughput.json`` from the W&B runs its rows name, plus the launcher's telemetry and logs.

Every finished row carries the id of the run that produced it. For each such row this reads the
run's history and replaces the row's measured fields:

- ``env_steps_per_s`` from ``system/env_steps_per_s``, the trainer's per-iteration collection rate
  (the first iteration, which includes the simulator's initial reset, is dropped);
- ``iteration_env_steps_per_s`` and the medians ``collect_s``, ``train_s``, ``publish_s`` from the
  trainer's whole-iteration rate and its three walls (``system/iteration_env_steps_per_s``,
  ``system/collect_wall_s``, ``system/train_wall_s``, ``system/publish_wall_s``; first iteration
  dropped), when the run logged them;
- ``env_step_s`` and ``decode_s`` from the medians of ``train/collector/env_step_s_per_step`` and
  ``train/collector/decode_s_per_decision``;
- ``sac_update_s`` from the median of the learner's update time;
- ``peak_gib`` from the launcher's 30 s utilization CSV named after the launch (``--util-dir``,
  ``probe_<layout>_n<N>.csv``; the split's simulator container writes ``..._sim.csv``);
- ``wire_ms`` from the client's round-trip lines in the learner log of the same name (``--log-dir``).

    python -m example_workflows.dgx_example.fetch_wandb results/throughput.json \\
        --wandb-path <entity>/<project> [--util-dir DIR] [--log-dir DIR] [--only dgx_station:64 ...]

Needs the ``wandb`` package and a logged-in API key. A row whose run has not finished is left as
it was and reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from example_workflows.dgx_example.throughput import load_throughput

RATE_KEY = "system/env_steps_per_s"
ITERATION_RATE_KEY = "system/iteration_env_steps_per_s"
ITERATION_WALL_KEYS = (
    ("collect_s", "system/collect_wall_s"),
    ("train_s", "system/train_wall_s"),
    ("publish_s", "system/publish_wall_s"),
)
ENV_STEP_KEY = "train/collector/env_step_s_per_step"
DECODE_KEY = "train/collector/decode_s_per_decision"
SAC_UPDATE_KEY = "learner/train_step_total_time_s"
#: The batched client's periodic summary: ``round trip 771 ms/step mean = server 768 + wire 4 (...)``.
ROUND_TRIP = re.compile(r"round trip (\d+) ms/step mean = server (\d+) \+ wire (\d+)")
#: The trainer appends a timestamp to the W&B run name; the launcher's files carry the name it was given.
RUN_NAME_TIMESTAMP = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def launch_name(run_name: str) -> str:
    """The name the launcher was given (``probe_gb300_only_n8``) from the W&B run name."""
    return RUN_NAME_TIMESTAMP.sub("", run_name)


def peak_mib(util_csv: Path) -> int:
    """The largest per-GPU memory sample in a launcher utilization CSV (``index:util%:MiB:W`` per GPU, ``;``-joined)."""
    peak = 0
    with util_csv.open(encoding="utf-8") as stream:
        for sample in csv.DictReader(stream):
            for gpu in sample["gpus"].split(";"):
                fields = gpu.split(":")
                if len(fields) >= 3:
                    peak = max(peak, int(float(fields[2])))
    return peak


def wire_ms(log: Path) -> float | None:
    """The median wire cost per step over the client's round-trip summaries, or None when the log has none."""
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    wires = [int(match.group(3)) for match in ROUND_TRIP.finditer(text)]
    if not wires:
        return None
    return float(statistics.median(wires))


def history_rows(run: Any) -> list[dict[str, Any]]:
    """Every logged row of a run, in step order.

    Scanned once without a key filter: a key-filtered history query returns
    nothing for the ``system/`` namespace, which W&B reserves in its UI.
    """
    return [dict(record) for record in run.scan_history()]


def series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Every logged value of one metric, in step order."""
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _gib(mib: int) -> float:
    return round(mib / 1024, 1)


def _rate_summary(rates: Sequence[float]) -> dict[str, float | int]:
    return {
        "mean": round(statistics.fmean(rates), 1),
        "std": round(statistics.pstdev(rates), 1),
        "cycles": len(rates),
    }


def refresh_row(
    row: dict[str, Any], run: Any, util_dir: Path | None, log_dir: Path | None
) -> None:
    """Replace ``row``'s measured fields with what ``run`` and the launcher's files say."""
    rows = history_rows(run)
    rates = series(rows, RATE_KEY)[1:]
    if not rates:
        raise ValueError(
            f"run {run.id} ({run.name}) logged no {RATE_KEY} after its first iteration"
        )
    row["env_steps_per_s"] = _rate_summary(rates)
    # The whole-iteration rate and walls exist only in runs of trainers that log them.
    iteration_rates = series(rows, ITERATION_RATE_KEY)[1:]
    if iteration_rates:
        row["iteration_env_steps_per_s"] = _rate_summary(iteration_rates)
        for field, key in ITERATION_WALL_KEYS:
            row[field] = round(statistics.median(series(rows, key)[1:]), 1)
    for field, key in (
        ("env_step_s", ENV_STEP_KEY),
        ("decode_s", DECODE_KEY),
        ("sac_update_s", SAC_UPDATE_KEY),
    ):
        values = series(rows, key)
        row[field] = round(statistics.median(values), 3) if values else None
    name = launch_name(run.name)
    if util_dir is not None:
        learner_csv = util_dir / f"{name}.csv"
        simulator_csv = util_dir / f"{name}_sim.csv"
        row["peak_gib"] = {
            "learner": _gib(peak_mib(learner_csv)) if learner_csv.exists() else None,
            "simulator": _gib(peak_mib(simulator_csv))
            if simulator_csv.exists()
            else None,
        }
    if log_dir is not None:
        log = log_dir / f"{name}.log"
        row["wire_ms"] = wire_ms(log) if log.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("throughput", type=Path)
    parser.add_argument("--wandb-path", required=True, metavar="ENTITY/PROJECT")
    parser.add_argument("--util-dir", type=Path, help="the launcher's util/ directory")
    parser.add_argument("--log-dir", type=Path, help="the sweep's log directory")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="LAYOUT:N",
        help="refresh only these rows",
    )
    args = parser.parse_args(argv)
    import wandb

    api = wandb.Api(timeout=120)
    data = json.loads(args.throughput.read_text(encoding="utf-8"))
    selected = {tuple(item.split(":", 1)) for item in args.only}
    refreshed = 0
    for row in data["rows"]:
        if row["status"] != "finished" or row.get("wandb_run") is None:
            continue
        if selected and (row["layout"], str(row["num_envs"])) not in selected:
            continue
        run = api.run(f"{args.wandb_path}/{row['wandb_run']}")
        if run.state != "finished":
            print(
                f"{row['layout']} N={row['num_envs']}: run {run.id} is {run.state}; left unchanged"
            )
            continue
        refresh_row(row, run, args.util_dir, args.log_dir)
        rate = row["env_steps_per_s"]
        print(
            f"{row['layout']} N={row['num_envs']}: {rate['mean']:.1f} ± {rate['std']:.1f} env-steps/s "
            f"over {rate['cycles']} cycles; step {row['env_step_s']} decode {row['decode_s']} "
            f"sac {row['sac_update_s']} peak {row.get('peak_gib')} wire {row.get('wire_ms')}"
        )
        refreshed += 1
    args.throughput.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    load_throughput(args.throughput)
    print(f"refreshed {refreshed} row(s) in {args.throughput}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
