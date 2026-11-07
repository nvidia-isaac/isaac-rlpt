#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v python >/dev/null || {
  echo "smoke check needs 'python' on PATH (run it as 'uv run bash <this script>', or activate the environment first)" >&2
  exit 1
}
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python -m example_workflows.v2d_box_lift_posttraining.launch \
  --v2d-root /dry-run/v2d \
  --checkpoint /dry-run/checkpoint \
  --motion-file /dry-run/tissue_box.parquet \
  --sim-python /dry-run/isaac-python \
  --output-dir "$WORK/output" \
  --dry-run >"$WORK/plan.json" 2>"$WORK/plan.err" || { cat "$WORK/plan.err"; exit 1; }
python -m example_workflows.v2d_box_lift_posttraining.verify_smoke "$WORK/plan.json"
grep -q "^plan.algorithm=dsrl_sac$" "$WORK/plan.json"
if grep -q "is deprecated" "$WORK/plan.json" "$WORK/plan.err"; then echo "smoke: the learner gets a deprecated key" >&2; exit 1; fi
python -m example_workflows._common.golden v2d_box_lift_posttraining

python -m pytest \
  example_workflows/v2d_box_lift_posttraining/tests \
  simulators/v2d/tests -m "not gpu and not arena and not libero and not slow" -q

python -m example_workflows._common.results_table \
  example_workflows/v2d_box_lift_posttraining/results/results.json \
  --check example_workflows/v2d_box_lift_posttraining/README.md

echo "V2D box-lift workflow smoke check passed"
