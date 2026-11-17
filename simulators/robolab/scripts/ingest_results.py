#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest a RoboLab output directory into a normalized harness trace index."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulators.robolab import ingest_results


def parse_args() -> argparse.Namespace:
    """Parse explicit source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="RoboLab output directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination harness trace JSONL")
    return parser.parse_args()


def main() -> None:
    """Run ingestion and print a concise, machine-readable summary."""
    args = parse_args()
    summary = ingest_results(args.source, args.output)
    print(
        "ingested "
        f"episodes={summary.episodes} "
        f"result_files={summary.result_files} "
        f"output={summary.output_path}"
    )


if __name__ == "__main__":
    main()
