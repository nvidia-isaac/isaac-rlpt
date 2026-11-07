#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke check: under two minutes, no GPU, no LIBERO installation. Fails on the first broken step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SCRIPTS=example_workflows/libero_posttraining/scripts
export ISAAC_RL_POSTTRAINING_DRYRUN=1
# A dry run never opens the checkpoint, so a placeholder path stands in for the frozen policy.
export POSTTRAIN_BASE_CKPT=/placeholder/GR00T-N1.7-LIBERO/libero_10

# 1. The launchers resolve their plans from the repository root: frozen-GR00T steering (DSRL),
#    DSRL PPO, offline SFT and IQL, and the pi0 recipe of the DSRL paper. A dry run prints the launcher's overrides, then
#    the library's dry run of them (plan.* lines; nothing loads), and exits 0 before anything starts.
bash "$SCRIPTS/train.sh" 8 > "$WORK/launch_dsrl.txt" 2>&1
grep -q "^plan.algorithm=dsrl_sac$" "$WORK/launch_dsrl.txt"
grep -q "^launch.override=base_model=gr00t$" "$WORK/launch_dsrl.txt"
bash "$SCRIPTS/train_ppo.sh" 8 > "$WORK/launch_ppo.txt" 2>&1
grep -q "^plan.algorithm=dsrl_ppo$" "$WORK/launch_ppo.txt"
for algo in sft iql; do
  for base_model in gr00t pi0; do
    ALGO=$algo BASE_MODEL=$base_model bash "$SCRIPTS/train.sh" 8 > "$WORK/launch_${algo}_$base_model.txt" 2>&1
    grep -q "^plan.algorithm=$algo$" "$WORK/launch_${algo}_$base_model.txt"
    grep -q "^plan.evaluation_mode=infer$" "$WORK/launch_${algo}_$base_model.txt"
  done
done
TASK=LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray
BASE_MODEL=pi0 TRAIN_MAXSTEPS=400 bash "$SCRIPTS/train.sh" "$TASK" "env.env_name=libero_pi0/libero_90/$TASK" \
    > "$WORK/launch_pi0.txt" 2>&1
grep -q "^launch.override=base_model=pi0$" "$WORK/launch_pi0.txt"
grep -q "DRY RUN" "$WORK/launch_pi0.txt"
# The Python form of the GR00T steering recipe (run.py) plans the same run.
python -m example_workflows.libero_posttraining.run 8 --dry-run > "$WORK/python_dsrl.txt" 2>&1
grep -q "^plan.algorithm=dsrl_sac$" "$WORK/python_dsrl.txt"
# No shipped recipe uses a deprecated key spelling.
if grep -h "is deprecated" "$WORK"/*.txt; then echo "smoke: a launcher passes a deprecated key" >&2; exit 1; fi

# 2. Every recipe resolves to the config pinned in tests/golden_config.json (golden-config parity).
python -m example_workflows._common.golden libero_posttraining

# 3. The reported numbers are well-formed and the README table matches them.
python -m example_workflows._common.results_table example_workflows/libero_posttraining/results/results.json \
    --check example_workflows/libero_posttraining/README.md

echo "smoke check passed"
