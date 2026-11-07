#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a RoboLab task after Isaac Sim is initialized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="RoboLab task class name to resolve.")
parser.add_argument(
    "--task-file",
    required=True,
    type=Path,
    help="Absolute path to the expected RoboLab task Python file.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

try:
    from robolab.constants import TASK_DIR
    from robolab.core.task.task_utils import load_task_from_file, resolve_task_path

    direct_task = load_task_from_file(str(args.task_file), allow_multiple=False)
    resolved_path, resolved_task = resolve_task_path(args.task, TASK_DIR)
    print(
        json.dumps(
            {
                "direct_task": direct_task.__name__,
                "resolved_path": resolved_path,
                "resolved_task": resolved_task,
            },
            sort_keys=True,
        )
    )
finally:
    simulation_app.close()
