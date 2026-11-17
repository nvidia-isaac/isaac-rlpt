# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run dashboards from telemetry: ``python -m isaac_rl_posttraining.viz <run_dir> ... -o dashboard.html``."""

from isaac_rl_posttraining.viz.dashboard import render_dashboard
from isaac_rl_posttraining.viz.data import RunSummary, load_run

__all__ = ["RunSummary", "load_run", "render_dashboard", "write_dashboard"]


def write_dashboard(run_dirs: list[str], output: str, *, names: list[str] | None = None) -> str:
    """Summarize ``run_dirs`` and write one self-contained HTML dashboard to ``output``; returns its path."""
    if names is not None and len(names) != len(run_dirs):
        raise ValueError("give one name per run directory")
    runs = [load_run(path, name=None if names is None else names[index]) for index, path in enumerate(run_dirs)]
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(render_dashboard(runs))
    return output
