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
WORKFLOW=example_workflows/dgx_example

# 1. Placement follows the simulator entry: G1 Factory runs single and split, an in-process simulator
#    refuses the split, and the README's capability table is what the registered entries say.
python -m example_workflows.dgx_example.layouts capabilities --check $WORKFLOW/README.md
python -m example_workflows.dgx_example.layouts plan --simulator arena --num-envs 64 \
    --simulator-gpu 0 --learner-gpu 1 --format shell > "$WORK/plan.sh"
grep -q "^DGX_SHAPE=split" "$WORK/plan.sh"
grep -q "rollout_count_by=episode" "$WORK/plan.sh"
if python -m example_workflows.dgx_example.layouts plan --simulator isaaclab --num-envs 8 \
    --simulator-gpu 0 --learner-gpu 1 2> "$WORK/refused.txt"; then
  echo "an in-process simulator must refuse the split layout"; exit 1
fi
grep -q "same GPU" "$WORK/refused.txt"

# 2. The G1 Factory launcher's dry run agrees with the plan: batched collection counted by episode,
#    frames shrunk at the source. The dry run needs only the shape of the external trees.
mkdir -p "$WORK/arena/.venv/bin" "$WORK/ckpt" "$WORK/data/LMDrillLift/lerobot" "$WORK/logs" "$WORK/g1f"
printf '#!/bin/sh\nexit 0\n' > "$WORK/arena/.venv/bin/python"
chmod +x "$WORK/arena/.venv/bin/python"
ln -s "$ROOT" "$WORK/g1f/posttrain"
export ISAAC_RL_POSTTRAINING_DRYRUN=1 ISAACLAB_ARENA_PATH="$WORK/arena" POSTTRAIN_BASE_CKPT="$WORK/ckpt" \
    POSTTRAIN_DATA_ROOT="$WORK/data" LOG_DIR="$WORK/logs" HF_TOKEN=
NUM_ENVS=64 IMAGE_SHORTEST_EDGE=256 bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift > "$WORK/train_plan.txt" 2>&1
grep -q "^launch.override=training.num_envs=64$" "$WORK/train_plan.txt"
grep -q "^launch.override=training.rollout_count_by=episode$" "$WORK/train_plan.txt"
grep -q -- "--image-shortest-edge 256" "$WORK/train_plan.txt"
grep -q "^plan.layout=1 rollout worker(s) x 64 environment(s)" "$WORK/train_plan.txt"
# The container launcher's dry run, the learner's side of every station run, ends in the library's dry run:
# the probe's overrides compose, and the RLinf runtime's settings follow RUNTIME and ROLLOUT_GPU.
WORKDIR="$WORK/g1f" PY=python TASK=DrillLift N=64 EVAL_SIM=0 RUNTIME=rlinf WANDB_NAME=smoke \
    bash example_workflows/g1_arena_posttraining/scripts/station/run_container_train.sh \
    training.max_updates_per_burst=20 simulator.decode_batch_size=64 > "$WORK/container_plan.txt" 2>&1
grep -q "^launch.runtime=rlinf" "$WORK/container_plan.txt"
grep -q "^launch.override=runtime.rlinf.first_accelerator=0$" "$WORK/container_plan.txt"
grep -q "^launch.override=runtime.rlinf.exclusive_accelerators=false$" "$WORK/container_plan.txt"
grep -q "^plan.runtime=rlinf" "$WORK/container_plan.txt"
grep -q "decode_batch_size: 64" "$WORK/container_plan.txt"

# 3. The layout sweep resolves every launch on a fake station tree: one plan per single layout and N,
#    two per split (the simulator's container, then the learner's), each pinned to its card.
G1F="$WORK/g1f" HF_CACHE="$WORK/hf" LAYOUTS="one_card:0:0 two_cards:1:0" NS="8 64" \
    bash $WORKFLOW/scripts/layout_sweep.sh > "$WORK/sweep.txt" 2>&1
[[ "$(grep -c 'plan layout=' "$WORK/sweep.txt")" = 6 ]]
[[ "$(grep -c '^launch.docker.gpus=device=' "$WORK/sweep.txt")" = 6 ]]
[[ "$(grep -c '^launch.docker.privileged=none' "$WORK/sweep.txt")" = 6 ]]
grep -q "SIMS_ONLY=1" "$WORK/sweep.txt"
grep -q "TRAIN_SIM_URL=tcp://127.0.0.1:7300" "$WORK/sweep.txt"
grep -q "training.max_updates_per_burst=20" "$WORK/sweep.txt"
grep -q "IMAGE_SHORTEST_EDGE=256" "$WORK/sweep.txt"
# RUNTIME=rlinf reaches the learner (its own image, the runtime in the run's name); the simulator's container stays native.
RUNTIME=rlinf G1F="$WORK/g1f" HF_CACHE="$WORK/hf" LAYOUTS="two_cards:1:0" NS="8" \
    bash $WORKFLOW/scripts/layout_sweep.sh > "$WORK/sweep_rlinf.txt" 2>&1
grep -q "WANDB_NAME=probe_two_cards_rlinf_n8 " "$WORK/sweep_rlinf.txt"
[[ "$(grep -c '^launch.image=g1_factory_learner_rlinf:' "$WORK/sweep_rlinf.txt")" = 1 ]]
[[ "$(grep -c '^launch.image=g1_factory_learner:' "$WORK/sweep_rlinf.txt")" = 1 ]]
# SCHEDULE=overlapped and its burst cap reach the learner; the schedule follows the runtime in the run's name.
SCHEDULE=overlapped MAX_UPDATES_PER_BURST=300 G1F="$WORK/g1f" HF_CACHE="$WORK/hf" LAYOUTS="two_cards:1:0" NS="8" \
    bash $WORKFLOW/scripts/layout_sweep.sh > "$WORK/sweep_overlapped.txt" 2>&1
grep -q "WANDB_NAME=probe_two_cards_overlapped_n8 " "$WORK/sweep_overlapped.txt"
grep -q "schedule.kind=overlapped" "$WORK/sweep_overlapped.txt"
grep -q "training.max_updates_per_burst=300" "$WORK/sweep_overlapped.txt"
# NUM_SERVERS=2 reaches both containers of the split: two servers on consecutive ports, the learner on both.
NUM_SERVERS=2 RUNTIME=rlinf G1F="$WORK/g1f" HF_CACHE="$WORK/hf" LAYOUTS="two_cards:1:0" NS="8" \
    bash $WORKFLOW/scripts/layout_sweep.sh > "$WORK/sweep_servers.txt" 2>&1
grep -q "WANDB_NAME=probe_two_cards_rlinf_2srv_n8 " "$WORK/sweep_servers.txt"
grep -q "TRAIN_SIM_URL=tcp://127.0.0.1:7300,tcp://127.0.0.1:7301" "$WORK/sweep_servers.txt"
[[ "$(grep -c 'NUM_SERVERS=2' "$WORK/sweep_servers.txt")" -ge 2 ]]
# The full recipe on several servers: the sized burst and the cadence reach the learner, the evaluation
# width every container, and the run carries the servers in its name.
RUNTIME=rlinf SCHEDULE=overlapped NUM_SERVERS=2 NUM_ENVS=256 MAX_UPDATES_PER_BURST=310 EVAL_FREQ=400000 EVAL_NUM_ENVS=25 \
    G1F="$WORK/g1f" HF_CACHE="$WORK/hf" SIMULATOR_GPU=1 LEARNER_GPU=0 OUT_DIR="$WORK/out" \
    bash $WORKFLOW/scripts/train_eval.sh DrillLift > "$WORK/train_eval.txt" 2>&1
grep -q "WANDB_NAME=dsrl_LMDrillLift_split_rlinf_overlapped_2srv_n256 " "$WORK/train_eval.txt"
grep -q "training.max_updates_per_burst=310" "$WORK/train_eval.txt"
grep -q "eval.eval_freq=400000" "$WORK/train_eval.txt"
[[ "$(grep -c 'EVAL_NUM_ENVS=25' "$WORK/train_eval.txt")" -ge 4 ]]
unset ISAAC_RL_POSTTRAINING_DRYRUN
if grep -h "is deprecated" "$WORK"/*.txt; then echo "smoke: a recipe passes a deprecated key" >&2; exit 1; fi

# 4. Every station recipe's learner resolves to the config pinned in tests/golden_config.json.
python -m example_workflows._common.golden dgx_example

# 5. The reported numbers are well-formed and the README tables match them.
python -m example_workflows._common.results_table $WORKFLOW/results/results.json --check $WORKFLOW/README.md
python -m example_workflows.dgx_example.throughput $WORKFLOW/results/throughput.json --check $WORKFLOW/README.md
python -m example_workflows.dgx_example.throughput $WORKFLOW/results/throughput_dgx_station.json \
    --check $WORKFLOW/README.md --block throughput_dgx_station
python -m example_workflows.dgx_example.throughput $WORKFLOW/results/throughput_schedules.json \
    --check $WORKFLOW/README.md --block throughput_schedules

echo "smoke check passed"
