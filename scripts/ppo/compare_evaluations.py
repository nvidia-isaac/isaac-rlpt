#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two retained PPO evaluation records episode by episode."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-iteration", type=int)
    parser.add_argument("--right-iteration", type=int)
    parser.add_argument(
        "--assert-identical",
        action="store_true",
        help="exit nonzero unless policy, reset, trajectory, and outcome fields match",
    )
    parser.add_argument(
        "--assert-same-setup",
        action="store_true",
        help=(
            "exit nonzero unless the policy, the episodes, their reset seeds and the initial robot "
            "state match; the reproducible part of two evaluations in separate processes"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the comparison as JSON",
    )
    return parser.parse_args()


def _load_record(path: Path, iteration: int | None) -> dict[str, Any]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"{path} has no evaluation records")
    if iteration is None:
        return records[-1]
    matches = [record for record in records if int(record["iteration"]) == iteration]
    if len(matches) != 1:
        raise ValueError(
            f"expected one iteration {iteration} record in {path}, found {len(matches)}"
        )
    return matches[0]


def _episodes_by_key(
    record: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    episodes = {}
    for episode in record["episodes"]:
        key = (int(episode["environment_index"]), int(episode["episode_index"]))
        if key in episodes:
            raise ValueError(f"duplicate episode key {key}")
        episodes[key] = episode
    return episodes


def _match_count(
    left: Mapping[tuple[int, int], Mapping[str, Any]],
    right: Mapping[tuple[int, int], Mapping[str, Any]],
    keys: set[tuple[int, int]],
    field: str,
) -> int:
    return sum(left[key].get(field) == right[key].get(field) for key in keys)


def _success_rate(
    episodes: Mapping[tuple[int, int], Mapping[str, Any]],
) -> float | None:
    if not episodes:
        return None
    return sum(bool(episode.get("success")) for episode in episodes.values()) / len(
        episodes
    )


def compare_records(
    left_record: Mapping[str, Any], right_record: Mapping[str, Any]
) -> dict[str, Any]:
    left = _episodes_by_key(left_record)
    right = _episodes_by_key(right_record)
    left_keys = set(left)
    right_keys = set(right)
    common_keys = left_keys & right_keys
    fields: tuple[str, ...] = (
        "reset_seed",
        "reset_seed_scope",
        "initial_observation_sha256",
        "first_execution_action_sha256",
        "execution_action_sha256",
        "observation_sha256_by_decision",
        "execution_action_sha256_by_decision",
        "success",
        "env_steps",
        "return",
    )
    optional_fields = ("initial_raw_observation_sha256",)
    fields += tuple(
        field
        for field in optional_fields
        if any(field in episode for episode in (*left.values(), *right.values()))
    )
    match_counts = {
        field: _match_count(left, right, common_keys, field) for field in fields
    }
    observation_keys = sorted(
        {
            key
            for episode_key in common_keys
            for episode in (left[episode_key], right[episode_key])
            for key in episode.get("initial_observation_sha256_by_key", {})
        }
    )
    observation_key_match_counts = {
        observation_key: sum(
            left[key].get("initial_observation_sha256_by_key", {}).get(observation_key)
            == right[key]
            .get("initial_observation_sha256_by_key", {})
            .get(observation_key)
            for key in common_keys
        )
        for observation_key in observation_keys
    }
    raw_observation_keys = sorted(
        {
            key
            for episode_key in common_keys
            for episode in (left[episode_key], right[episode_key])
            for key in episode.get("initial_raw_observation_sha256_by_key", {})
        }
    )
    raw_observation_key_match_counts = {
        observation_key: sum(
            left[key]
            .get("initial_raw_observation_sha256_by_key", {})
            .get(observation_key)
            == right[key]
            .get("initial_raw_observation_sha256_by_key", {})
            .get(observation_key)
            for key in common_keys
        )
        for observation_key in raw_observation_keys
    }
    matches = {
        **match_counts,
        "initial_observation_sha256_by_key": observation_key_match_counts,
        "initial_raw_observation_sha256_by_key": (raw_observation_key_match_counts),
    }
    report = {
        "schema_version": 1,
        "left_iteration": int(left_record["iteration"]),
        "right_iteration": int(right_record["iteration"]),
        "left_policy_sha256": left_record.get("policy_sha256"),
        "right_policy_sha256": right_record.get("policy_sha256"),
        # Unknown digests (the workers' loaded weights, nothing published) never match.
        "policy_sha256_matches": left_record.get("policy_sha256") is not None
        and left_record.get("policy_sha256") == right_record.get("policy_sha256"),
        "left_episodes": len(left),
        "right_episodes": len(right),
        "common_episodes": len(common_keys),
        "episode_keys_match": left_keys == right_keys,
        "matches": matches,
    }
    state_keys = [
        key
        for key in raw_observation_keys
        if key == "state" or key.startswith("state.")
    ]
    report["same_setup"] = bool(
        report["policy_sha256_matches"]
        and report["episode_keys_match"]
        and match_counts["reset_seed"] == len(common_keys)
        and match_counts["reset_seed_scope"] == len(common_keys)
        and all(
            raw_observation_key_match_counts[key] == len(common_keys)
            for key in state_keys
        )
    )
    report["left_success_rate"] = _success_rate(left)
    report["right_success_rate"] = _success_rate(right)
    report["identical"] = bool(
        report["policy_sha256_matches"]
        and report["episode_keys_match"]
        and all(match_counts[field] == len(common_keys) for field in fields)
        and all(
            count == len(common_keys) for count in observation_key_match_counts.values()
        )
        and all(
            count == len(common_keys)
            for count in raw_observation_key_match_counts.values()
        )
    )
    return report


def main() -> int:
    args = parse_args()
    report = compare_records(
        _load_record(args.left, args.left_iteration),
        _load_record(args.right, args.right_iteration),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = rendered + "\n"
        args.output.write_text(text, encoding="utf-8")  # NOSONAR - user-named CLI path
    if args.assert_identical and not report["identical"]:
        return 1
    return int(args.assert_same_setup and not report["same_setup"])


if __name__ == "__main__":
    raise SystemExit(main())
