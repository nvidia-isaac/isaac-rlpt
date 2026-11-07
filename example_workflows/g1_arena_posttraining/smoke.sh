#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke check: under two minutes, no GPU, no external repository. Fails on the first broken step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SCRIPTS=example_workflows/g1_arena_posttraining/scripts

# Every dry run stops before a simulator, policy or learner is built, so the simulator addresses
# are placeholders and the mock base model stands in for the frozen policy.
export ISAAC_RL_POSTTRAINING_DRYRUN=1 BASE_MODEL=mock
export ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301
export LEARNER_PYTHON="${LEARNER_PYTHON:-$(command -v python)}"

plan () {  # $1=name, the rest is the command; its output lands in $WORK/<name>.txt
  local name=$1
  shift
  "$@" > "$WORK/$name.txt" 2>&1 || { cat "$WORK/$name.txt"; echo "smoke: $name failed" >&2; exit 1; }
}
expect () {  # a plan name and a line the plan must contain
  local name=$1 line=$2
  grep -qF -- "$line" "$WORK/$name.txt" || { cat "$WORK/$name.txt"; echo "smoke: $name plan lacks '$line'" >&2; exit 1; }
}

# 1. Every launcher resolves its plan from the repository root: its own launch.* lines and overrides,
#    then the library's dry run of those overrides (plan.* lines, data specs and placement checked).
plan train bash $SCRIPTS/train.sh DrillLift
expect train "launch.task=LMDrillLift"
expect train "launch.override=env.max_episode_steps=450"
expect train "plan.algorithm=dsrl_sac"
expect train "plan.placement.learner="
plan eval_base bash $SCRIPTS/eval.sh DrillLift base 50
expect eval_base "launch.override=eval.eval_base=true"
expect eval_base "launch.override=eval.eval_num_episodes=50"
expect eval_base "DRY RUN"
mkdir -p "$WORK/checkpoint"
plan eval_policy bash $SCRIPTS/eval.sh DrillLift policy 50 "$WORK/checkpoint"
expect eval_policy "launch.override=training.load_dir=$WORK/checkpoint"
plan eval_parallel env EVAL_NUM_ENVS=10 bash $SCRIPTS/eval.sh PushButton base 100
expect eval_parallel "launch.override=eval.eval_num_envs=10"
expect eval_parallel "--num-envs 10"
plan ppo bash $SCRIPTS/train_ppo.sh DrillLift
expect ppo "plan.algorithm=dsrl_ppo"
# Full fine-tuning trains the base policy itself, so the stand-in is a GR00T checkpoint directory (the dry
# run reads only its config.json) rather than the mock policy.
mkdir -p "$WORK/gr00t_n17" && echo "{}" > "$WORK/gr00t_n17/config.json"
plan finetune_ppo env BASE_MODEL=gr00t POSTTRAIN_BASE_CKPT="$WORK/gr00t_n17" bash $SCRIPTS/finetune_ppo.sh PushButton
expect finetune_ppo "plan.algorithm=ppo"
expect finetune_ppo "launch.override=base_model.action_horizon=10"
# Offline SFT and IQL train the N1.7 adapter from demonstrations; eval.sh scores their checkpoints.
for algo in sft iql; do
  plan "$algo" env ALGO=$algo BASE_MODEL=gr00t bash $SCRIPTS/train.sh DrillLift
  expect "$algo" "plan.algorithm=$algo"
  expect "$algo" "launch.override=base_model=gr00t_n17"
  expect "$algo" "launch.cmd.train_sim=<none: $algo trains offline>"
  plan "${algo}_eval" env ALGO=$algo BASE_MODEL=gr00t bash $SCRIPTS/eval.sh DrillLift policy 50 "$WORK/checkpoint"
  expect "${algo}_eval" "plan.evaluation_mode=infer"
  expect "${algo}_eval" "launch.override=training.load_dir=$WORK/checkpoint"
done
for mode in rfs rfs_pc rfs_nb2 sac_edit; do
  plan "zoo_$mode" bash $SCRIPTS/zoo_recipes.sh "$mode" DrillPnP 0
done
expect zoo_rfs "plan.algorithm=rfs"
expect zoo_rfs_nb2 "launch.override=dsrl.noise_action_bound=2.0"
expect zoo_sac_edit "plan.algorithm=sac_edit"
plan repro_train bash $SCRIPTS/repro_sep2026.sh train DrillPnP 1
expect repro_train "launch.override=training.seed=1"
plan repro_base bash $SCRIPTS/repro_sep2026.sh base DrillPnP
expect repro_base "launch.override=eval.eval_num_episodes=100"
plan repro_rescore bash $SCRIPTS/repro_sep2026.sh rescore DrillPnP "$WORK/checkpoint"
expect repro_rescore "launch.override=training.load_dir=$WORK/checkpoint"
# The Python form of the DSRL recipe (run.py) plans the same run.
POSTTRAIN_BASE_CKPT="$WORK/gr00t_n17" POSTTRAIN_DATA_ROOT=/data/g1 \
  plan python_dsrl python -m example_workflows.g1_arena_posttraining.run DrillLift --dry-run
expect python_dsrl "plan.algorithm=dsrl_sac"
# No shipped recipe uses a deprecated key spelling.
if grep -h "is deprecated" "$WORK"/*.txt; then echo "smoke: a launcher passes a deprecated key" >&2; exit 1; fi

# 2. Every recipe resolves to the config pinned in tests/golden_config.json (golden-config parity).
python -m example_workflows._common.golden g1_arena_posttraining

# 3. The reported numbers are well-formed and the README table matches them.
python -m example_workflows._common.results_table example_workflows/g1_arena_posttraining/results/results.json \
    --check example_workflows/g1_arena_posttraining/README.md

echo "smoke check passed"
