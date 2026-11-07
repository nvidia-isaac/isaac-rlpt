#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The success-rate recipe on a DGX Station: DSRL training in the split layout (the training and the
# evaluation simulator on one card, the frozen policy and the learner on the other), then the frozen
# base policy and the retained checkpoint scored on EVAL_EPISODES episodes with the overrides of
# example_workflows/g1_arena_posttraining/scripts/eval.sh, through the same station launcher.
#
# Usage, on the station with the G1 Factory tree deployed (docs/pages/workflows/dgx_example.md):
#   SIMULATOR_GPU=<uuid> LEARNER_GPU=<uuid> bash example_workflows/dgx_example/scripts/train_eval.sh [TASK]
# Knobs:
#   NUM_ENVS (64)             environments of the training simulator
#   TRAIN_ENV_STEPS (100000)  environment steps to train for
#   EVAL_EPISODES (100)       episodes per score
#   CKPT_DIR                  skip training and score this checkpoint (the base is scored too)
#   OUT_DIR (/data/dsrl_out_g1arena)  the launcher's output root, where checkpoints and summaries land
#   RUNTIME (from probe.env)  the learner's rollout runtime, native or rlinf; rlinf puts _rlinf in RUN_NAME
#   SCHEDULE (from probe.env) the learner's schedule (schedule.kind): sync or overlapped;
#                             overlapped puts _overlapped in RUN_NAME
#   NUM_SERVERS (1)           Arena servers hosting the training environments, N/NUM_SERVERS each behind
#                             its own rollout worker (needs RUNTIME=rlinf); puts _<K>srv in RUN_NAME
#   MAX_UPDATES_PER_BURST     SAC updates per training burst; unset keeps the reference rule of one update
#                             per collected decision, which at N in the hundreds makes a burst of hours
#   EVAL_FREQ                 environment steps between evaluations and checkpoints; unset keeps the
#                             config's 5000, less than one round at N in the hundreds
#   EVAL_NUM_ENVS (1)         environments of the evaluation simulator, scored in parallel; EVAL_EPISODES
#                             and the config's eval_num_episodes must be multiples of it
#   TRAIN_PORT (from probe.env) first simulator port; the evaluation server takes TRAIN_PORT + NUM_SERVERS
#   TRAIN_OVERRIDES            further Hydra overrides for the training run only, space-separated (for
#                             instance algorithm.batch_size=256); the scoring runs never see them
#   SIMULATOR (arena)  RUN_NAME (dsrl_<task>_<layout>[_rlinf][_overlapped][_<K>srv]_n<N>)  LOG_DIR ($HOME/dgx_example_logs)
#   PYTHON                    the interpreter for the placement module (default python; on a container-only
#                             host, example_workflows/g1_arena_posttraining/scripts/station/station_python.sh runs it in the learner image)
#   ISAAC_RL_POSTTRAINING_DRYRUN=1  print every launch plan and start nothing
#
# The same card for both roles runs everything in one container (the single layout).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORKFLOW=$ROOT/example_workflows/dgx_example
# shellcheck source=../configs/probe.env
source "$WORKFLOW/configs/probe.env"
# shellcheck source=station.sh
source "$WORKFLOW/scripts/station.sh"
# shellcheck source=../../../example_workflows/g1_arena_posttraining/scripts/task_prompts.sh
source "$ROOT/example_workflows/g1_arena_posttraining/scripts/task_prompts.sh"
DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}
# A run's directory: <name>_<YYYY-MM-DD_HH-MM-SS>; a plain <name>_* would also match <name>_eval_base_*.
RUN_STAMP='_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
TASK=${1:-DrillLift}
[[ "$TASK" == LM* ]] || TASK="LM$TASK"
SIMULATOR=${SIMULATOR:-arena}
RUNTIME=${RUNTIME:-$PROBE_RUNTIME}
SCHEDULE=${SCHEDULE:-$PROBE_SCHEDULE}
NUM_SERVERS=${NUM_SERVERS:-1}
MAX_UPDATES_PER_BURST=${MAX_UPDATES_PER_BURST:-}
EVAL_FREQ=${EVAL_FREQ:-}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-1}
EXTRA_TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}
NUM_ENVS=${NUM_ENVS:-64}
TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
OUT_DIR=${OUT_DIR:-/data/dsrl_out_g1arena}
LOG_DIR=${LOG_DIR:-$HOME/dgx_example_logs}
PYTHON=${PYTHON:-python}
: "${SIMULATOR_GPU:?set SIMULATOR_GPU and LEARNER_GPU (nvidia-smi UUIDs or indices; the same card for both is the single layout)}"
: "${LEARNER_GPU:?set SIMULATOR_GPU and LEARNER_GPU (nvidia-smi UUIDs or indices; the same card for both is the single layout)}"
TRAIN_PORT=${TRAIN_PORT:-$PROBE_PORT}
EVAL_PORT=$((TRAIN_PORT + NUM_SERVERS))
if [[ "$NUM_SERVERS" != 1 ]] && [[ "$RUNTIME" != rlinf ]]; then
  echo "FATAL: NUM_SERVERS=$NUM_SERVERS needs RUNTIME=rlinf: the native runtime steps its rollout workers one after another" >&2; exit 1
fi
mkdir -p "$LOG_DIR"
cd "$ROOT"

plan=$("$PYTHON" -m example_workflows.dgx_example.layouts plan --simulator "$SIMULATOR" --num-envs "$NUM_ENVS" \
  --simulator-gpu "$SIMULATOR_GPU" --learner-gpu "$LEARNER_GPU" --port "$TRAIN_PORT" --num-servers "$NUM_SERVERS" --format shell) \
  || { echo "FATAL: this run cannot be placed for $SIMULATOR" >&2; exit 1; }
eval "$plan"
# From here the plan's DGX_* values name the cards. The recipe's own two variables must not reach the
# container launcher, whose LEARNER_GPU is the learner's CUDA index inside its pinned container (and the
# RLInf actor's accelerator index); a card UUID there is refused at the learner's start.
unset SIMULATOR_GPU LEARNER_GPU
STATION=$(station_dir "$DGX_ENTRY") || exit 1
LAUNCHER=$STATION/station_train.sh
SCHEDULE_KIND=$(schedule_kind "$SCHEDULE") || exit 1
RUNTIME_TAG=""
[[ "$RUNTIME" = native ]] || RUNTIME_TAG="_$RUNTIME"
SCHEDULE_TAG=$(schedule_tag "$SCHEDULE")
SERVERS_TAG=""
[[ "$NUM_SERVERS" = 1 ]] || SERVERS_TAG="_${NUM_SERVERS}srv"
RUN_NAME=${RUN_NAME:-dsrl_${TASK}_${DGX_SHAPE}${RUNTIME_TAG}${SCHEDULE_TAG}${SERVERS_TAG}_n${NUM_ENVS}}
PROMPT=$(task_prompt "$TASK") || { echo "FATAL: no language prompt for $TASK in example_workflows/g1_arena_posttraining/scripts/task_prompts.sh" >&2; exit 1; }
SHARED_ENV=(EVAL_SIM=1 "EVAL_NUM_ENVS=$EVAL_NUM_ENVS" "IMAGE_SHORTEST_EDGE=$PROBE_IMAGE_SHORTEST_EDGE" "TRAIN_PORT=$TRAIN_PORT" "EVAL_PORT=$EVAL_PORT")
# The training environments' servers; a scoring run has one environment on one server.
TRAIN_ENV=("NUM_SERVERS=$NUM_SERVERS")
# The learner's container alone gets the rollout runtime; a simulator-only container runs no learner and
# stays on the native image whatever RUNTIME says.
LEARNER_ENV=("RUNTIME=$RUNTIME")
SIMULATOR_ENV=(RUNTIME=native)
TRAIN_OVERRIDES=("schedule.kind=$SCHEDULE_KIND")
[[ "$PROBE_DECODE_WHOLE_BATCH" != 1 ]] || TRAIN_OVERRIDES+=("simulator.decode_batch_size=$NUM_ENVS")
[[ -z "$MAX_UPDATES_PER_BURST" ]] || TRAIN_OVERRIDES+=("training.max_updates_per_burst=$MAX_UPDATES_PER_BURST")
# A saved checkpoint stays an evaluated one: the checkpoints follow the evaluation cadence.
[[ -z "$EVAL_FREQ" ]] || TRAIN_OVERRIDES+=("eval.eval_freq=$EVAL_FREQ" "training.save_interval=$EVAL_FREQ")
# shellcheck disable=SC2206  # the knob is a space-separated list of key=value overrides
[[ -z "$EXTRA_TRAIN_OVERRIDES" ]] || TRAIN_OVERRIDES+=($EXTRA_TRAIN_OVERRIDES)

say () { echo "[train_eval] $(date +%H:%M:%S) $*" | tee -a "$LOG_DIR/train_eval.log"; }
launch () {  # $1 log file, then the command; a dry run prints the plan instead of logging it
  local log=$1; shift
  if [[ "$DRYRUN" = 1 ]]; then "$@"; else "$@" > "$log" 2>&1; fi
}
kill_ours () {  # this stack's containers only, through the simulator's own stop script
  bash "$STATION/station_stop.sh" > /dev/null 2>&1 || true
}
wait_port () {  # a port and the pid of the launcher serving it
  local port=$1 launcher_pid=$2
  until timeout 5 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2> /dev/null; do
    kill -0 "$launcher_pid" 2> /dev/null || return 1
    sleep 15
  done
}

train () {
  local sim_pid="" learner_env=() status=0
  if [[ "$DGX_SHAPE" = split ]]; then
    say "plan run=$RUN_NAME role=simulator gpu=$DGX_SIMULATOR_GPU N=$NUM_ENVS servers=$NUM_SERVERS eval_envs=$EVAL_NUM_ENVS (training and evaluation simulators)"
    if [[ "$DRYRUN" = 1 ]]; then
      env "${SHARED_ENV[@]}" "${TRAIN_ENV[@]}" "${SIMULATOR_ENV[@]}" SIMS_ONLY=1 "GPUS=$DGX_SIMULATOR_GPU" "WANDB_NAME=${RUN_NAME}_sim" \
        bash "$LAUNCHER" "$TASK" "$NUM_ENVS"
    else
      env "${SHARED_ENV[@]}" "${TRAIN_ENV[@]}" "${SIMULATOR_ENV[@]}" SIMS_ONLY=1 "GPUS=$DGX_SIMULATOR_GPU" "WANDB_NAME=${RUN_NAME}_sim" \
        bash "$LAUNCHER" "$TASK" "$NUM_ENVS" > "$LOG_DIR/${RUN_NAME}_sim.log" 2>&1 &
      sim_pid=$!
      for port in $(seq "$TRAIN_PORT" "$EVAL_PORT"); do
        wait_port "$port" "$sim_pid" || { say "the simulators exited before listening (${RUN_NAME}_sim.log)"; return 1; }
      done
    fi
    learner_env=("TRAIN_SIM_URL=$DGX_TRAIN_SERVER_URL" "EVAL_SIM_URL=tcp://127.0.0.1:$EVAL_PORT")
  fi
  say "plan run=$RUN_NAME role=learner gpu=$DGX_LEARNER_GPU env_steps=$TRAIN_ENV_STEPS servers=$NUM_SERVERS"
  launch "$LOG_DIR/$RUN_NAME.log" env "${SHARED_ENV[@]}" "${TRAIN_ENV[@]}" "${LEARNER_ENV[@]}" "NUM_ROLLOUTS=$TRAIN_ENV_STEPS" "GPUS=$DGX_LEARNER_GPU" \
    "WANDB_NAME=$RUN_NAME" ${learner_env[@]+"${learner_env[@]}"} bash "$LAUNCHER" "$TASK" "$NUM_ENVS" ${TRAIN_OVERRIDES[@]+"${TRAIN_OVERRIDES[@]}"} \
    || status=$?
  if [[ -n "$sim_pid" ]]; then
    kill_ours
    for _ in $(seq 1 24); do kill -0 "$sim_pid" 2> /dev/null || break; sleep 5; done
    kill "$sim_pid" 2> /dev/null || true
  fi
  [[ "$status" = 0 ]] || say "training failed (exit $status); see $LOG_DIR/$RUN_NAME.log"
  return "$status"
}

score () {  # base|policy [checkpoint dir]: one scoring run on the learner card, as eval.sh scores
  local mode=$1 checkpoint=${2:-}
  local name=${RUN_NAME}_eval_$mode extra=() status=0
  case "$mode" in
    base) extra=(eval.eval_base=true) ;;
    policy) extra=("training.load_dir=$checkpoint") ;;
    *) ;;
  esac
  say "plan run=$name role=score mode=$mode gpu=$DGX_LEARNER_GPU episodes=$EVAL_EPISODES${checkpoint:+ checkpoint=$checkpoint}"
  # The launcher puts its own dataset and budget values first on the learner's command line;
  # these later ones win, which turns the run into the evaluation eval.sh performs. A scoring run
  # has one environment on one server whatever the training run used (NUM_SERVERS is inherited).
  launch "$LOG_DIR/$name.log" env "${SHARED_ENV[@]}" NUM_SERVERS=1 "${LEARNER_ENV[@]}" "GPUS=$DGX_LEARNER_GPU" "WANDB_NAME=$name" \
    bash "$LAUNCHER" "$TASK" 1 base_model.dataset_dir=null training.num_rollouts=0 eval.eval_on_first_step=true \
      "eval.eval_num_episodes=$EVAL_EPISODES" eval.eval_record_video=false "\"base_model.task_instruction='$PROMPT'\"" "${extra[@]}" \
    || status=$?
  [[ "$DRYRUN" = 1 ]] && return "$status"
  kill_ours
  [[ "$status" = 0 ]] || { say "score $mode: failed (exit $status); see $LOG_DIR/$name.log"; return 1; }
  local summary
  # shellcheck disable=SC2086  # RUN_STAMP is a glob
  summary=$(ls -t "$OUT_DIR/${name}"$RUN_STAMP/checkpoints/evaluation_summary.json 2> /dev/null | head -1 || true)
  [[ -n "$summary" ]] || { say "score $mode: no evaluation_summary.json under $OUT_DIR/${name}_<timestamp>; see $LOG_DIR/$name.log"; return 1; }
  # Read on the host: PYTHON may run inside the learner image, where OUT_DIR is not mounted. The
  # summary's first success_rate is the "latest" block's.
  say "score $mode: success rate $(awk -F': *' '/"success_rate"/ { printf "%.1f%%", 100 * $2; exit }' "$summary") over $EVAL_EPISODES episodes ($summary)"
}

if [[ -z "${CKPT_DIR:-}" ]]; then
  train || exit 1
  if [[ "$DRYRUN" = 1 ]]; then
    CKPT_DIR="$OUT_DIR/${RUN_NAME}_<timestamp>/checkpoints/final"
  else
    # shellcheck disable=SC2086  # RUN_STAMP is a glob
    CKPT_DIR=$(ls -d "$OUT_DIR/${RUN_NAME}"$RUN_STAMP/checkpoints/final 2> /dev/null | tail -1 || true)
    [[ -n "$CKPT_DIR" ]] || { say "no final checkpoint under $OUT_DIR/${RUN_NAME}_<timestamp>; see $LOG_DIR/$RUN_NAME.log"; exit 1; }
  fi
fi
score base || true
score policy "$CKPT_DIR" || true
say "done"
