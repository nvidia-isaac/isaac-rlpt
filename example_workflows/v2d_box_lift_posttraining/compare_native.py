# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare paired native and steered V2D evaluation manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from example_workflows.v2d_box_lift_posttraining.config import EVALUATOR_NAME


@dataclass(frozen=True)
class Summary:
    episodes: int
    successes: int
    lift_hold_successes: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes


def _records(path: Path) -> dict[tuple[int, int, int], dict[str, object]]:
    records: dict[tuple[int, int, int], dict[str, object]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        metrics = value["task_metrics"]
        if not isinstance(metrics, dict):
            raise TypeError(f"{path}:{line_number} task_metrics must be an object")
        if metrics["evaluator"] != EVALUATOR_NAME:
            raise ValueError(f"{path}:{line_number} uses a different evaluator")
        key = (
            int(value["reset_seed"]),
            int(value["round_index"]),
            int(value["environment_index"]),
        )
        if key in records:
            raise ValueError(f"{path} repeats evaluation case {key}")
        records[key] = value
    if not records:
        raise ValueError(f"{path} contains no evaluation episodes")
    return records


def summarize(records: dict[tuple[int, int, int], dict[str, object]]) -> Summary:
    successes = 0
    lift_hold_successes = 0
    for record in records.values():
        metrics = record["task_metrics"]
        assert isinstance(metrics, dict)
        combined = bool(metrics["combined_success"])
        if bool(record["success"]) != combined:
            raise ValueError("record success disagrees with combined_success")
        successes += combined
        lift_hold_successes += bool(metrics["lift_hold_success"])
    return Summary(len(records), successes, lift_hold_successes)


def compare(native_path: Path, steered_path: Path) -> dict[str, object]:
    native_records = _records(native_path)
    steered_records = _records(steered_path)
    if native_records.keys() != steered_records.keys():
        raise ValueError("native and steered evaluations do not use identical cases")
    for key in native_records:
        native_fingerprint = native_records[key]["initial_state_fingerprint"]
        steered_fingerprint = steered_records[key]["initial_state_fingerprint"]
        if native_fingerprint != steered_fingerprint:
            raise ValueError(f"initial-state fingerprint differs for case {key}")
    native = summarize(native_records)
    steered = summarize(steered_records)
    return {
        "episodes": native.episodes,
        "native_successes": native.successes,
        "native_success_rate": native.success_rate,
        "native_lift_hold_successes": native.lift_hold_successes,
        "steered_successes": steered.successes,
        "steered_success_rate": steered.success_rate,
        "steered_lift_hold_successes": steered.lift_hold_successes,
        "success_rate_change": steered.success_rate - native.success_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("steered", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.native, args.steered)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
