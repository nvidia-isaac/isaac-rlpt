#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reproduce the October-2026 G1 Factory DSRL and PPO runs with their exact launch protocol.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh dsrl        TASK [SEED]      # 80k-step DSRL run
#   example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh ppo         TASK [SEED]      # 100-iteration PPO fine-tuning
#   example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh base        TASK             # frozen base, N=50
#   example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh rescore     TASK CKPT_DIR    # DSRL checkpoint, N=50
#   example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh rescore_ppo TASK ITER_DIR    # PPO checkpoint, N=50
#
#   TASK      task name with or without the LM prefix (DrillPnP or LMDrillPnP)
#   SEED      training seed (default 0)
#   CKPT_DIR  a DSRL checkpoint step directory, e.g. <run>/checkpoints/40000
#   ITER_DIR  a PPO checkpoint directory named iteration_<n>
#
# Every score uses the same 50 held-out episodes: episode i resets the scene and the policy's own
# noise from seed 5000+i-1. In-training evaluations use seed 7000, so they never see those scenes.
# DSRL: noise bound 2.0, actor mean initialized at zero with log std -0.46 (the base policy's own
# noise moments), Gaussian warmup, +1 on success and 0 otherwise, fixed entropy coefficient 0.05.
# PPO: full flow-SDE fine-tuning of the action head with finetune_ppo.sh's defaults (actor lr 1e-5),
# 8 scenes per simulator; its evaluator decodes deterministically over 2 batched scenes.
# Environment variables (ISAACLAB_ARENA_PATH, POSTTRAIN_BASE_CKPT, ...) are the ones train.sh documents.
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 prints the resolved plan and exits.
set -uo pipefail

MODE=${1:?usage: reproduce_results.sh dsrl|ppo|base|rescore|rescore_ppo TASK [SEED|CKPT_DIR]}
TASK=${2:?TASK}
shift 2
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
HERE=$(cd "$(dirname "$0")" && pwd)

source "$HERE/task_prompts.sh"
INSTR=$(task_prompt "$TASK") || { echo "[repro] unknown task $TASK" >&2; exit 2; }

COMMON=(
  algorithm=dsrl_sac
  training.chunk_size=40
  eval.action_selection=sample
  base_model.dataset_dir=null
  "base_model.task_instruction='$INSTR'"
)
RECIPE=(
  dsrl.noise_action_bound=2.0 policy.log_std_init=-0.46 policy.official_policy_init=true
  dsrl.warmup_exploration=normal env.success_reward=1.0 env.failure_reward=0.0 algorithm.ent_coef=0.05
)
SCORE=(eval.eval_on_first_step=true eval.eval_num_episodes=50 eval.reset_seed=5000 training.num_rollouts=0 training.seed=5000)

case "$MODE" in
  dsrl)
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" "${RECIPE[@]}" \
      "training.seed=${1:-0}" training.num_rollouts=80000 \
      eval.eval_num_episodes=20 eval.eval_freq=5000 eval.reset_seed=7000 eval.eval_on_first_step=true \
      training.save_interval=2500 training.checkpoint_retention=all
    ;;
  ppo)
    NUM_ENVS=8 exec bash "$HERE/finetune_ppo.sh" "$TASK" \
      "training.seed=${1:-0}" ppo.iterations=100 \
      ppo.eval_every=5 ppo.eval_episodes_per_environment=4 \
      ppo.checkpoint_every=5 ppo.checkpoint_retention=best_last_final
    ;;
  base)
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" eval.eval_base=true "${SCORE[@]}"
    ;;
  rescore)
    CKPT=${1:?rescore needs CKPT_DIR (a checkpoints/<step> directory)}
    exec bash "$HERE/train.sh" "$TASK" "${COMMON[@]}" "${RECIPE[@]}" "training.load_dir=$CKPT" "${SCORE[@]}"
    ;;
  rescore_ppo)
    CKPT=${1:?rescore_ppo needs ITER_DIR (an iteration_<n> directory)}
    NUM_ENVS=2 exec bash "$HERE/finetune_ppo.sh" "$TASK" \
      ppo.eval_only=true placement.learner.ranks=0 "ppo.resume_from=$CKPT" \
      ppo.eval_episodes_per_environment=25 ppo.eval_seed=5000 training.seed=5000
    ;;
  *) echo "[repro] MODE must be dsrl, ppo, base, rescore or rescore_ppo" >&2; exit 2;;
esac
