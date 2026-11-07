# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line: write one HTML dashboard from the telemetry of one or more runs."""

from __future__ import annotations

import argparse

from isaac_rl_posttraining.viz import write_dashboard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m isaac_rl_posttraining.viz",
        description="Write a self-contained HTML dashboard from run telemetry (several runs are overlaid).",
    )
    parser.add_argument("run_dirs", nargs="+", help="run directories (or their telemetry/ directories)")
    parser.add_argument("-o", "--output", default="dashboard.html", help="HTML file to write")
    parser.add_argument("--name", action="append", dest="names", help="legend name per run, in order")
    args = parser.parse_args(argv)
    path = write_dashboard(args.run_dirs, args.output, names=args.names)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
