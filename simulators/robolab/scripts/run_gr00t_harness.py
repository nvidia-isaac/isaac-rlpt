#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run GR00T N1.7 DROID through the no-op policy harness in RoboLab.

This mirrors RoboLab's ``policies/gr00t/run.py`` entry point.  It constructs
the stock GR00T client, wraps it with :class:`HarnessInferenceClient`, and
keeps all evaluated task and simulator behavior inside RoboLab's normal
``run_evaluation`` path.
"""

from __future__ import annotations

import argparse
import sys
import traceback

# RoboLab requires this before isaaclab.
import cv2  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate GR00T through the no-op policy harness.")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=5555)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None)
parser.add_argument(
    "--harness-trace",
    type=str,
    required=True,
    help="JSONL path for harness decisions; keep it outside the RoboLab output directory.",
)
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.gr00t.client import GR00TDroidJointposClient  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

from simulators.robolab.policies.gr00t import Gr00tRoboLabAdapter  # noqa: E402
from simulators.robolab.policies.harness_client import (  # noqa: E402
    HarnessInferenceClient,
)
from simulators.robolab.runtime.trace import TraceRecorder  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> HarnessInferenceClient:
    """Build the upstream GR00T client, then add only no-op harness behavior."""
    kwargs = {
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "open_loop_horizon": args.open_loop_horizon,
    }
    native_client = GR00TDroidJointposClient(
        **{name: value for name, value in kwargs.items() if value is not None}
    )
    return HarnessInferenceClient(
        Gr00tRoboLabAdapter(native_client),
        trace_recorder=TraceRecorder(args.harness_trace),
    )


def main() -> None:
    """Run the normal RoboLab evaluator with the harness client factory."""
    run_evaluation(args_cli, policy="gr00t_harness", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
