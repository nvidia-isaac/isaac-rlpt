#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reproduce the September-2026 G1 Factory DSRL runs behind the reference tables
# (simulators/g1_arena/README.md, "Reference results") with their exact launch protocol.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh train   TASK [SEED]        # 40k-step DSRL run
#   example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh base    TASK               # frozen base, N=100
#   example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh rescore TASK CKPT_DIR      # trained ckpt, N=100
#
#   TASK      task name with or without the LM prefix (DrillPnP or LMDrillPnP)
#   SEED      training seed (default 0)
#   CKPT_DIR  a checkpoint step directory, e.g. <run>/checkpoints/16088
#
# Optional: VARIANT picks a variant of the recipe.
#   b2       dsrl.noise_action_bound=2.0
#   b2_succ  b2 plus the +1-on-success reward (env.success_reward=1.0 env.failure_reward=0.0), the
#            settings of the released PushButton steering policy
#
# Optional: EVAL_SEED=<int> seeds the base/rescore evaluation episodes: episode i resets the
# scene and the policy's own noise from seed EVAL_SEED+i-1, so the frozen policy and every
# re-scored checkpoint meet the same 100 initial scenes (<run>/evaluation_episodes.jsonl
# records each episode's seed and outcome). The GPU simulation is not deterministic, so
# seeded runs share scenes, not outcomes (about 60-70% of episodes agree between two seeded
# runs of the same policy); the published protocol stays unseeded (EVAL_SEED unset).
#
# Everything else is the branch default (the recipe pinned by
# tests/unit/test_recipe_pins.py): noise bound 1.0, synchronous learner, 4-member
# critic with pooled features and mean target reduction, actor lr 5e-5, critic lr
# 1e-4, batch 64, gamma 0.995, auto-tuned entropy. Environment variables
# (ISAACLAB_ARENA_PATH, POSTTRAIN_BASE_CKPT, ...) are the ones train.sh documents.
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 prints the resolved plan and exits.
set -uo pipefail

MODE=${1:?usage: repro_sep2026.sh train|base|rescore TASK [SEED|CKPT_DIR]}
TASK=${2:?TASK}
shift 2
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
HERE=$(cd "$(dirname "$0")" && pwd)

# Task prompt (task_prompts.sh); Hydra-quoted below because one prompt has a comma.
source "$HERE/task_prompts.sh"
INSTR=$(task_prompt "$TASK") || { echo "[repro] unknown task $TASK" >&2; exit 2; }

COMMON=(
  training.chunk_size=40                 # steering latents per flow step
  eval.action_selection=sample           # stochastic actor, as measured
  base_model.dataset_dir=null               # baked-embodiment path; no dataset needed
  "base_model.task_instruction='$INSTR'"
)
VOPTS=()
case "${VARIANT:-}" in
  b2)      VOPTS+=(dsrl.noise_action_bound=2.0);;
  b2_succ) VOPTS+=(dsrl.noise_action_bound=2.0 env.success_reward=1.0 env.failure_reward=0.0);;
  "")      ;;
  *)       echo "[repro] unknown VARIANT ${VARIANT}; use b2 or b2_succ" >&2; exit 2;;
esac
SCORE=(eval.eval_on_first_step=true eval.eval_num_episodes=100 training.num_rollouts=0 training.seed=0)
if [[ -n "${EVAL_SEED:-}" ]]; then SCORE+=("eval.reset_seed=$EVAL_SEED"); fi

case "$MODE" in
  train)
    SEED=${1:-0}
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" "${VOPTS[@]}" \
      "training.seed=$SEED" \
      training.num_rollouts=40000 \
      eval.eval_num_episodes=50 eval.eval_freq=5000 \
      training.save_interval=5000 training.checkpoint_retention=best_last_final
    ;;
  base)
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" eval.eval_base=true "${SCORE[@]}"
    ;;
  rescore)
    CKPT=${1:?rescore needs CKPT_DIR (a checkpoints/<step> directory)}
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" "${VOPTS[@]}" "training.load_dir=$CKPT" "${SCORE[@]}"
    ;;
  *) echo "[repro] MODE must be train, base, or rescore" >&2; exit 2;;
esac
