#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${V2D_ROOT:?set V2D_ROOT to the pinned video-to-data checkout}"
: "${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT to the 58-D joint checkpoint}"
: "${V2D_MOTION_FILE:?set V2D_MOTION_FILE to the tissue-box motion parquet}"
: "${V2D_SIM_PYTHON:?set V2D_SIM_PYTHON to the Isaac Sim Python executable}"
: "${POSTTRAIN_OUTPUT_DIR:?set POSTTRAIN_OUTPUT_DIR under ~/data_drive}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 native [launcher args...] | steered STEERING_CHECKPOINT [launcher args...]" >&2
  exit 2
fi
mode="$1"
shift
policy_args=()
if [[ "$mode" == "native" ]]; then
  policy_args=(--native)
elif [[ "$mode" == "steered" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "steered evaluation requires a steering checkpoint" >&2
    exit 2
  fi
  policy_args=(--steering-checkpoint "$1")
  shift
else
  echo "evaluation mode must be 'native' or 'steered'" >&2
  exit 2
fi

python "$ROOT/example_workflows/v2d_box_lift_posttraining/launch.py" \
  --v2d-root "$V2D_ROOT" \
  --checkpoint "$POSTTRAIN_BASE_CKPT" \
  --motion-file "$V2D_MOTION_FILE" \
  --sim-python "$V2D_SIM_PYTHON" \
  --output-dir "$POSTTRAIN_OUTPUT_DIR" \
  --evaluation-mode enabled \
  --training-steps 0 \
  "${policy_args[@]}" \
  "$@"
