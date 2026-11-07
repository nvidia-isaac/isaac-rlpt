# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate one V2D evaluation manifest against its batching contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from example_workflows.v2d_box_lift_posttraining.compare_native import (
    _records,
    summarize,
)


def verify(path: Path, *, num_envs: int, num_episodes: int) -> dict[str, object]:
    if num_envs < 1 or num_episodes < num_envs or num_episodes % num_envs:
        raise ValueError("need positive N and B with N >= B and N % B == 0")
    records = _records(path)
    if len(records) != num_episodes:
        raise ValueError(f"expected {num_episodes} episodes, found {len(records)}")
    environments_by_reset: dict[int, set[int]] = {}
    case_seeds: set[int] = set()
    for (reset_seed, _round_index, environment_index), record in records.items():
        if not 0 <= environment_index < num_envs:
            raise ValueError(
                f"environment index {environment_index} is outside B={num_envs}"
            )
        environments_by_reset.setdefault(reset_seed, set()).add(environment_index)
        case_seed = record["case_seed"]
        if isinstance(case_seed, bool) or not isinstance(case_seed, int):
            raise TypeError("evaluation case_seed must be an integer")
        fingerprint = str(record["initial_state_fingerprint"])
        if case_seed in case_seeds:
            raise ValueError("evaluation cases must have distinct case seeds")
        if not fingerprint:
            raise ValueError("every evaluation case needs an initial-state fingerprint")
        case_seeds.add(case_seed)
    expected_environments = set(range(num_envs))
    if len(environments_by_reset) != num_episodes // num_envs:
        raise ValueError("manifest has the wrong number of evaluation rounds")
    if any(
        indices != expected_environments for indices in environments_by_reset.values()
    ):
        raise ValueError(
            "each evaluation round must score every environment exactly once"
        )
    summary = summarize(records)
    return {
        "episodes": summary.episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "lift_hold_successes": summary.lift_hold_successes,
        "num_envs": num_envs,
        "rounds": num_episodes // num_envs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    args = parser.parse_args(argv)
    result = verify(
        args.manifest,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
