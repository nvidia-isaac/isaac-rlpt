#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke check: under two minutes, no GPU, no external repository. Fails on the first broken step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v python > /dev/null || { echo "smoke check needs 'python' on PATH (run it as 'uv run bash <this script>', or activate the environment first)"; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1. The launch plan resolves with no errors. The dry run stops before any worker or model is built;
#    the stand-in robot (simulators/_template) stands in for your simulator and the mock base model
#    for the base policy. Replace the entry, env_name and geometry with your simulator's.
ISAAC_RL_POSTTRAINING_DRYRUN=1 python scripts/train.py algorithm=dsrl_sac base_model=mock \
    +simulator.entry=simulators._template.runtime:MyRobot env.env_name=my_robot/reach \
    env.max_episode_steps=50 base_model.action_dim=7 base_model.action_horizon=16 \
    dsrl.noise_dim=7 training.chunk_size=16 \
    hydra.run.dir="$WORK/dryrun" > "$WORK/plan.txt" 2>&1
grep -q "DRY RUN" "$WORK/plan.txt"
grep -q "plan.checks=.* 0 error(s)" "$WORK/plan.txt"

# 2. Every recipe resolves to the config pinned in tests/golden_config.json (golden-config parity).
python -m example_workflows._common.golden _template

# 3. The reported numbers are well-formed and the README table matches them.
python -m example_workflows._common.results_table example_workflows/_template/results/results.json \
    --check example_workflows/_template/README.md

echo "smoke check passed"
