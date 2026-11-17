#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the algorithm-zoo conditions that reached or beat the frozen base on the nine
# G1 Factory tasks (example_workflows/g1_arena_posttraining/README.md, "Method") under the
# same 40k-step protocol as repro_sep2026.sh.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh MODE TASK [SEED]
#
#   MODE  rfs      Residual Flow Steering, branch defaults
#         rfs_pc   RFS with the residual conditioned on the frozen policy's proposal
#         rfs_nb2  RFS with the flow-noise bound widened to 2.0 (the PushButton condition)
#         sac_edit      SAC-edit: direct 35-dim chunk edits (fp32, 0.02 edit, learning from 4k)
#   TASK  task name with or without the LM prefix (DrillPnP or LMDrillPnP)
#   SEED  training seed (default 0)
#
# Everything else is the branch default (pinned by tests/unit/test_sac_edit_guards.py).
# Environment variables are the ones train.sh documents.
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 prints the resolved plan and exits.
set -uo pipefail

MODE=${1:?usage: zoo_recipes.sh rfs|rfs_pc|rfs_nb2|sac_edit TASK [SEED] [hydra overrides...]}
TASK=${2:?TASK}
SEED=${3:-0}
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
HERE=$(cd "$(dirname "$0")" && pwd)

# Task prompt (task_prompts.sh); Hydra-quoted below because one prompt has a comma.
source "$HERE/task_prompts.sh"
INSTR=$(task_prompt "$TASK") || { echo "[zoo] unknown task $TASK" >&2; exit 2; }

# ALGO selects the algorithm config group in train.sh; ALG carries the mode's overrides.
case "$MODE" in
  rfs)     ALGO=rfs; ALG=(training.chunk_size=40);;
  rfs_pc)  ALGO=rfs; ALG=(training.chunk_size=40 rfs.proposal_conditioned_residual=true rfs.reference_action_conditioning=true);;
  rfs_nb2) ALGO=rfs; ALG=(training.chunk_size=40 dsrl.noise_action_bound=2.0);;
  sac_edit) ALGO=sac_edit; ALG=();;
  *) echo "[zoo] MODE must be rfs, rfs_pc, rfs_nb2, or sac_edit" >&2; exit 2;;
esac

ALGO=$ALGO exec bash "$HERE/train.sh" "$TASK" "${ALG[@]}" \
  eval.action_selection=sample \
  base_model.dataset_dir=null "base_model.task_instruction='$INSTR'" \
  "training.seed=$SEED" training.num_rollouts=40000 \
  eval.eval_num_episodes=50 eval.eval_freq=5000 \
  training.save_interval=5000 training.checkpoint_retention=best_last_final \
  "${@:4}"
