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

python "$ROOT/example_workflows/v2d_box_lift_posttraining/launch.py" \
  --v2d-root "$V2D_ROOT" \
  --checkpoint "$POSTTRAIN_BASE_CKPT" \
  --motion-file "$V2D_MOTION_FILE" \
  --sim-python "$V2D_SIM_PYTHON" \
  --output-dir "$POSTTRAIN_OUTPUT_DIR" \
  "$@"
