#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The layout sweep on a DGX Station: the whole stack on one card, the whole stack on the other, then
# the split (simulator on one card, learner on the other), each at several environment counts, with
# the probe settings of configs/probe.env. Runs go one after another; each is named
# probe_<layout>_n<N> in its log and its W&B run.
#
# Usage, on the station with the G1 Factory tree deployed (docs/pages/workflows/dgx_example.md):
#   GB300_GPU=<uuid> RTX_GPU=<uuid> bash example_workflows/dgx_example/scripts/layout_sweep.sh
# Knobs:
#   LAYOUTS    name:simulator_gpu:learner_gpu ... (default: the three station layouts built from GB300_GPU
#              and RTX_GPU; the same card twice is the single layout, two different cards the split)
#   NS         environment counts (default "8 64 128 256 512")
#   SIMULATOR  environment-name segment the run is placed for (default arena); TASK (default from probe.env)
#   RUNTIME    the learner's rollout runtime, native or rlinf (default from probe.env); rlinf runs are named
#              probe_<layout>_rlinf_n<N>, and the simulator's own container stays on the native image
#   SCHEDULE   the learner's schedule (schedule.kind), sync or overlapped (default from probe.env);
#              overlapped runs carry _overlapped in their names after the runtime
#              (probe_<layout>[_rlinf]_overlapped_n<N>)
#   MAX_UPDATES_PER_BURST  SAC updates per collection round (default from probe.env); a cap sized so a
#              burst takes about as long as a round is what shows a schedule's gain
#   RUN_PREFIX the first part of every run's name (default probe); a sweep with its own recipe, such as
#              a burst cap other than the probe's, takes another prefix so its logs and W&B runs are its own
#   NUM_SERVERS simulator servers hosting the training environments (default 1), on consecutive ports from
#              the probe's, each with N/NUM_SERVERS environments behind its own rollout worker (RUNTIME=rlinf);
#              runs with several carry _<K>srv in their names after the schedule
#   LOG_DIR    where the run logs go (default $HOME/dgx_example_logs)
#   COOLDOWN_S seconds between runs after the containers are stopped (default 15)
#   PYTHON     the interpreter for the placement module (default python; on a container-only host,
#              example_workflows/g1_arena_posttraining/scripts/station/station_python.sh runs it in the learner image)
#   ISAAC_RL_POSTTRAINING_DRYRUN=1  print every launch plan and start nothing
#
# The placement (which card, which layout, batched or not) comes from layouts.py, generic over the
# simulator entries; the launch itself goes through the simulator's own station launcher, which only
# G1 Factory ships (example_workflows/g1_arena_posttraining/scripts/station/station_train.sh), so the environment it is given here
# is that launcher's: GPUS pins the container to a card, SIMS_ONLY starts the simulator alone,
# TRAIN_SIM_URL points a learner at a simulator started elsewhere.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORKFLOW=$ROOT/example_workflows/dgx_example
# shellcheck source=../configs/probe.env
source "$WORKFLOW/configs/probe.env"
# shellcheck source=station.sh
source "$WORKFLOW/scripts/station.sh"
DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}
SIMULATOR=${SIMULATOR:-arena}
TASK=${TASK:-$PROBE_TASK}
RUNTIME=${RUNTIME:-$PROBE_RUNTIME}
SCHEDULE=${SCHEDULE:-$PROBE_SCHEDULE}
schedule_kind "$SCHEDULE" > /dev/null || exit 1
MAX_UPDATES_PER_BURST=${MAX_UPDATES_PER_BURST:-$PROBE_MAX_UPDATES_PER_BURST}
RUN_PREFIX=${RUN_PREFIX:-probe}
NUM_SERVERS=${NUM_SERVERS:-1}
NS=${NS:-"8 64 128 256 512"}
LOG_DIR=${LOG_DIR:-$HOME/dgx_example_logs}
PYTHON=${PYTHON:-python}
COOLDOWN_S=${COOLDOWN_S:-15}
if [[ -z "${LAYOUTS:-}" ]]; then
  : "${GB300_GPU:?set GB300_GPU and RTX_GPU (nvidia-smi UUIDs or indices), or LAYOUTS=name:simulator_gpu:learner_gpu ...}"
  : "${RTX_GPU:?set GB300_GPU and RTX_GPU (nvidia-smi UUIDs or indices), or LAYOUTS=name:simulator_gpu:learner_gpu ...}"
  LAYOUTS="gb300_only:$GB300_GPU:$GB300_GPU rtx_pro_6000_only:$RTX_GPU:$RTX_GPU dgx_station:$RTX_GPU:$GB300_GPU"
fi
mkdir -p "$LOG_DIR"
cd "$ROOT"

say () { echo "[layout_sweep] $(date +%H:%M:%S) $*" | tee -a "$LOG_DIR/layout_sweep.log"; }
budget () {  # environment steps for N environments
  local n_envs=$1
  local steps=$((PROBE_ENV_STEPS_PER_ENV * n_envs))
  [[ "$steps" -ge "$PROBE_MIN_ENV_STEPS" ]] || steps=$PROBE_MIN_ENV_STEPS
  echo "$steps"
}
launch () {  # $1 log file, then the command; a dry run prints the plan instead of logging it
  local log=$1; shift
  if [[ "$DRYRUN" = 1 ]]; then "$@"; else "$@" > "$log" 2>&1; fi
}
kill_ours () {  # this stack's containers only, through the simulator's own stop script
  bash "$STATION/station_stop.sh" > /dev/null 2>&1 || true
}
run_name () {  # the run's log and W&B name; a non-native runtime, a non-sync schedule and several servers are part of it
  local layout=$1 n_envs=$2
  local runtime="" schedule="" servers=""
  [[ "$RUNTIME" = native ]] || runtime="_$RUNTIME"
  schedule=$(schedule_tag "$SCHEDULE")
  [[ "$NUM_SERVERS" = 1 ]] || servers="_${NUM_SERVERS}srv"
  echo "${RUN_PREFIX}_${layout}${runtime}${schedule}${servers}_n${n_envs}"
}
finish () {  # one line on how the run ended, then a clean slate for the next one
  local layout=$1 n_envs=$2
  local verdict
  # The pipelines below may match nothing; that must not end the sweep (set -e, pipefail).
  verdict=$(tr '\r' '\n' < "$LOG_DIR/$(run_name "$layout" "$n_envs").log" 2> /dev/null \
    | grep -aE 'wandb: Synced|Traceback|Error|FATAL|OutOfMemory|out of memory' | tail -1 | cut -c1-120 || true)
  say "layout=$layout N=$n_envs ended: ${verdict:-no verdict line in the log}"
  kill_ours
  sleep "$COOLDOWN_S"
}
probe_env () {  # the launcher environment every probe run shares; both containers of a split boot or connect to the same servers
  echo "EVAL_SIM=$PROBE_EVAL_SIM" "IMAGE_SHORTEST_EDGE=$PROBE_IMAGE_SHORTEST_EDGE" "TRAIN_PORT=$PROBE_PORT" "NUM_SERVERS=$NUM_SERVERS"
}
learner_env () {  # what only the learner's container gets: its rollout runtime
  echo "RUNTIME=$RUNTIME"
}
simulator_env () {  # a simulator-only container runs no learner, so it stays on the native image whatever RUNTIME says
  echo "RUNTIME=native"
}
learner_overrides () {  # the config overrides every probe learner gets
  local n_envs=$1
  echo "training.max_updates_per_burst=$MAX_UPDATES_PER_BURST" "schedule.kind=$(schedule_kind "$SCHEDULE")"
  [[ "$PROBE_DECODE_WHOLE_BATCH" = 1 ]] && echo "simulator.decode_batch_size=$n_envs"
  return 0
}

run_single () {  # the whole stack in one container pinned to the card
  local layout=$1 n_envs=$2 gpu=$3
  local name
  name=$(run_name "$layout" "$n_envs")
  say "plan layout=$layout N=$n_envs role=single gpu=$gpu runtime=$RUNTIME schedule=$SCHEDULE servers=$NUM_SERVERS env_steps=$(budget "$n_envs")"
  launch "$LOG_DIR/$name.log" env $(probe_env) $(learner_env) "NUM_ROLLOUTS=$(budget "$n_envs")" "GPUS=$gpu" \
    "WANDB_NAME=$name" bash "$LAUNCHER" "$TASK" "$n_envs" $(learner_overrides "$n_envs")
}

run_split () {  # the simulator on one card, the learner on another, connected over the simulator address
  local layout=$1 n_envs=$2 simulator_gpu=$3 learner_gpu=$4 simulator_url=$5
  say "plan layout=$layout N=$n_envs role=simulator gpu=$simulator_gpu"
  local sim_pid=""
  local name
  name=$(run_name "$layout" "$n_envs")
  if [[ "$DRYRUN" = 1 ]]; then
    env $(probe_env) $(simulator_env) SIMS_ONLY=1 "GPUS=$simulator_gpu" "WANDB_NAME=${name}_sim" bash "$LAUNCHER" "$TASK" "$n_envs"
  else
    env $(probe_env) $(simulator_env) SIMS_ONLY=1 "GPUS=$simulator_gpu" "WANDB_NAME=${name}_sim" bash "$LAUNCHER" "$TASK" "$n_envs" \
      > "$LOG_DIR/${name}_sim.log" 2>&1 &
    sim_pid=$!
    # Every server must listen before the learner connects; they boot one after another.
    for port in $(seq "$PROBE_PORT" $((PROBE_PORT + NUM_SERVERS - 1))); do
      until timeout 5 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2> /dev/null; do
        kill -0 "$sim_pid" 2> /dev/null || { say "layout=$layout N=$n_envs: the simulator exited before listening (${name}_sim.log)"; return 1; }
        sleep 15
      done
    done
  fi
  say "plan layout=$layout N=$n_envs role=learner gpu=$learner_gpu runtime=$RUNTIME schedule=$SCHEDULE servers=$NUM_SERVERS env_steps=$(budget "$n_envs") simulator=$simulator_url"
  launch "$LOG_DIR/$name.log" env $(probe_env) $(learner_env) "NUM_ROLLOUTS=$(budget "$n_envs")" "GPUS=$learner_gpu" "TRAIN_SIM_URL=$simulator_url" \
    "WANDB_NAME=$name" bash "$LAUNCHER" "$TASK" "$n_envs" $(learner_overrides "$n_envs")
  if [[ -n "$sim_pid" ]]; then
    kill_ours
    for _ in $(seq 1 24); do kill -0 "$sim_pid" 2> /dev/null || break; sleep 5; done   # the simulator's launcher exits with its container
    kill "$sim_pid" 2> /dev/null || true
  fi
}

for N in $NS; do
  for spec in $LAYOUTS; do
    IFS=: read -r name simulator_gpu learner_gpu <<< "$spec"
    plan=$("$PYTHON" -m example_workflows.dgx_example.layouts plan --simulator "$SIMULATOR" --num-envs "$N" \
      --simulator-gpu "$simulator_gpu" --learner-gpu "$learner_gpu" --port "$PROBE_PORT" --num-servers "$NUM_SERVERS" \
      --format shell) \
      || { echo "FATAL: layout $name cannot be placed for $SIMULATOR" >&2; exit 1; }
    eval "$plan"
    STATION=$(station_dir "$DGX_ENTRY") || exit 1
    LAUNCHER=$STATION/station_train.sh
    # A run that fails (an out-of-memory card, a crashed simulator) is reported and the sweep goes on.
    if [[ "$DGX_SHAPE" = single ]]; then
      run_single "$name" "$N" "$DGX_SIMULATOR_GPU" || say "layout=$name N=$N: the launcher exited with status $?"
    else
      run_split "$name" "$N" "$DGX_SIMULATOR_GPU" "$DGX_LEARNER_GPU" "$DGX_TRAIN_SERVER_URL" \
        || say "layout=$name N=$N: the launcher exited with status $?"
    fi
    [[ "$DRYRUN" = 1 ]] || finish "$name" "$N"
  done
done
say "done"
