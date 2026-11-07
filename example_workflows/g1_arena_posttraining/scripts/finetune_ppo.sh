#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Full PPO fine-tuning of GR00T N1.7's action head on one G1 Factory task (algorithm=ppo); the
# vision-language backbone stays frozen and NUM_ENVS (default 8) scenes share one batched simulator.
# Usage: finetune_ppo.sh TASK [Hydra overrides...]; the knobs are train.sh's (RUNTIME, NUM_SERVERS, ...).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TASK=${1:?usage: finetune_ppo.sh TASK [hydra overrides...]}
shift
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
source "$HERE/task_prompts.sh"
INSTR=$(task_prompt "$TASK")

export ALGO=ppo NUM_ENVS=${NUM_ENVS:-8}
# One rollout worker per training simulator (NUM_SERVERS), not the Isaac Lab recipe's four.
# The G1 server chooses the task, so env.env_name only labels the run. Ten of the head's 40 predicted
# steps run per decision, a whole number of chunks for every task's episode cap. The checkpoint's
# baked embodiment supplies the modality and statistics, so no dataset is read.
exec "$HERE/train.sh" "$TASK" \
  simulator=g1_arena placement.envs.workers=null placement.learner.ranks=1 \
  "env.env_name=arena/$TASK" "base_model.task_instruction='$INSTR'" base_model.dataset_dir=null \
  base_model.embodiment_tag=new_embodiment base_model.denoising_steps=4 base_model.action_horizon=10 \
  ppo.cache_features=true ppo.global_batch_size=64 ppo.micro_batch_size=8 \
  "$@"
