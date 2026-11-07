# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve container-owned paths and download the frozen base policy before launch."""

import os
import sys
from pathlib import Path

from example_workflows.v2d_box_lift_posttraining.launch import parser, resolve_plan

CHECKPOINT_REPO = "nvidia/video-to-data-GR00T-N1.7-finetuned-tissue-box-lift"
CHECKPOINT_REVISION = "e3ad7eb8dcde2e667a8b5ef9c46d1c66aa0dc9aa"
V2D_ROOT = Path("/workspace/video_to_data")
MOTION_FILE = V2D_ROOT / (
    "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
    "human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/"
    "robot_name=vega_sharpa/data.parquet"
)
CHECKPOINT = Path("/cache/checkpoints") / CHECKPOINT_REVISION


def launch_arguments(arguments: list[str], *, gpu_count: int) -> list[str]:
    if gpu_count < 1:
        raise ValueError("No CUDA GPU is visible; launch Docker with --gpus")
    owned = {
        "--v2d-root",
        "--sim-python",
        "--motion-file",
        "--checkpoint",
        "--output-dir",
    }
    for argument in arguments:
        if argument.split("=", 1)[0] in owned:
            raise ValueError(f"The container supplies {argument.split('=', 1)[0]}")
    return [
        "--v2d-root",
        str(V2D_ROOT),
        "--sim-python",
        "/usr/local/bin/python",
        "--motion-file",
        str(MOTION_FILE),
        "--checkpoint",
        str(CHECKPOINT),
        "--output-dir",
        "/output/run",
        "--num-train-envs",
        "1",
        "--eval-num-envs",
        "1",
        "--train-device",
        "cuda:0" if gpu_count == 1 else "cuda:1",
        "--eval-device",
        "cuda:2" if gpu_count >= 3 else "cuda:0",
        *arguments,
    ]


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download

    gpu_count = torch.cuda.device_count()
    arguments = launch_arguments(sys.argv[1:], gpu_count=gpu_count)
    argument_parser = parser()
    argument_parser.allow_abbrev = False
    args = argument_parser.parse_args(arguments)
    # Resolve the recipe before a download or either expensive model/simulator startup.
    resolve_plan(args)
    if not args.dry_run:
        snapshot_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            local_dir=CHECKPOINT,
        )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "example_workflows.v2d_box_lift_posttraining.launch",
            *arguments,
        ],
    )


if __name__ == "__main__":
    main()
