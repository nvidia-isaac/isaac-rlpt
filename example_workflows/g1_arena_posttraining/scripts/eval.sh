#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Evaluation launcher: success rate of the base policy or a trained checkpoint
# on one G1 Factory task.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/eval.sh TASK MODE N [CKPT_DIR] [extra Hydra overrides...]
#     TASK   task name, with or without the LM prefix (DrillLift or LMDrillLift)
#     MODE   base   = the native GR00T policy, no latent actor
#            policy = a trained checkpoint (CKPT_DIR), from any training method
#                     (offline SFT/IQL checkpoints also need ALGO=sft|iql)
#            replay = open-loop replay of a dataset episode through the evaluation simulator
#     N      number of evaluation episodes
#
# Required environment (not needed for a dry run):
#   ISAACLAB_ARENA_PATH  IsaacLab-Arena checkout (with its .venv)
#   POSTTRAIN_BASE_CKPT       GR00T checkpoint directory (HF-style)
#   POSTTRAIN_BC_CKPT         optional separate BC checkpoint (defaults to POSTTRAIN_BASE_CKPT)
#   POSTTRAIN_DATA_ROOT       datasets root (replay mode only); expects $POSTTRAIN_DATA_ROOT/<LMTask>/lerobot
# Optional:
#   BASE_MODEL      base model (default gr00t; mock|gr00t|pi0|dreamzero,
#                   see docs/base_models.md); becomes the Hydra base_model group override
#   SIMULATOR (default g1_arena)        simulator config the per-task episode
#                                       caps are read from; this launcher
#                                       supports only g1_arena (see
#                                       simulators/README.md)
#   TRAIN_SIM_DEVICE / EVAL_SIM_DEVICE  physics device per simulator, cpu|gpu
#                                       (default gpu); becomes ARENA_SIM_DEVICE
#   LEARNER_PYTHON, SIM_PYTHON (default: the workflow environment's .venv/bin/python once `uv sync` built it,
#     else python3 and $ISAACLAB_ARENA_PATH/.venv/bin/python)
#   GR00T_REPO      Isaac-GR00T checkout for PYTHONPATH (if gr00t isn't installed)
#   POSTTRAIN_BACKBONE_MODEL  optional local/HF override for GR00T's VLM backbone
#   GPU (default 0)  PORT_BASE (default 7300)  LOG_DIR (default ./dsrl_eval_logs)
#   RUNTIME (default native)  native | rlinf; rlinf shares the one GPU between its groups
#   ALGO            sft | iql: score GR00T N1.7 (BASE_MODEL=gr00t trains gr00t_n17) with its
#                   deterministic decode; the run reads the task's demonstrations under
#                   POSTTRAIN_DATA_ROOT to register the G1 layout of a checkpoint that has none.
#                   EMBODIMENT_TAG, ACTION_HORIZON, DENOISING_STEPS as in train.sh
#   SIM_BOOT_TIMEOUT  seconds to wait for each simulator to listen (default 600, plus 120 per extra eval env)
#   EVAL_NUM_ENVS   evaluation environments hosted by one batched eval simulator and scored in
#                   parallel (default 1; >1 needs the per-env WBC patch, see README, and N a multiple of it)
#   SIM_VIZ         kit opens Isaac Sim's viewer on the simulator that steps the scored episodes
#   EVAL_MAXSTEPS   override the per-task episode cap
#   WANDB_NAME      wandb run name (default eval_<task>_<mode>_<timestamp>)
#   ISAAC_RL_POSTTRAINING_OUTPUT_DIR  checkpoint/log root used by the trainer
#   SIM_STARTUP_LOCK    lock file serializing simulator cold-starts on one node
# Secrets are read from the environment only and are never printed:
#   HF_TOKEN, WANDB_API_KEY
#
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 (or a --dry-run argument) prints this launcher's
# launch.* lines and every override it passes, then the library's own dry run (plan.* lines and
# the resolved config), and exits without starting anything.
set -uo pipefail

TASK=${1:?usage: example_workflows/g1_arena_posttraining/scripts/eval.sh TASK base|policy|replay N [CKPT_DIR] [hydra overrides...]}
MODE=${2:?MODE: base | policy | replay}
NEPS=${3:?N episodes}
shift 3
CKPT_DIR=""
if [ "$MODE" = "policy" ]; then
  CKPT_DIR=${1:?MODE=policy needs CKPT_DIR}
  shift
  [ -d "$CKPT_DIR" ] || { echo "[dsrl_eval] checkpoint directory $CKPT_DIR not found" >&2; exit 2; }
fi

# A --dry-run (or --print-config) argument anywhere is equivalent to ISAAC_RL_POSTTRAINING_DRYRUN=1.
DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}
EXTRA_OVERRIDES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|--print-config) DRYRUN=1 ;;
    *) EXTRA_OVERRIDES+=("$arg") ;;
  esac
done

[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
if [ "$MODE" != "base" ] && [ "$MODE" != "policy" ] && [ "$MODE" != "replay" ]; then
  echo "[dsrl_eval] unknown MODE '$MODE' (expected base, policy, or replay)" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../../_common/launch.sh
source "$REPO_ROOT/example_workflows/_common/launch.sh"
# The workflow's own environment (scripts/fetch_externals.sh && uv sync) runs both the learner and the simulator.
WORKFLOW_PY=$(workflow_python "$REPO_ROOT/example_workflows/g1_arena_posttraining")
LEARNER_PY=${LEARNER_PYTHON:-${WORKFLOW_PY:-python3}}
refuse_renamed_knob BACKEND BASE_MODEL

# Simulator selection (see simulators/README.md). This launcher boots the
# g1_arena env servers; the knob exists so every launcher resolves its
# simulator config the same way.
SIMULATOR=${SIMULATOR:-g1_arena}
if [ "$SIMULATOR" != "g1_arena" ]; then
  echo "[dsrl_eval] SIMULATOR=$SIMULATOR is not supported by this launcher (it drives the g1_arena simulator); see simulators/README.md" >&2
  exit 2
fi
SIMULATOR_CONFIG="$REPO_ROOT/simulators/g1_arena/configs/simulator/$SIMULATOR.yaml"

# Per-task episode caps (longest demonstration plus margin), read from the
# simulator config -- the single source of truth
# (tests/unit/test_simulators.py pins launcher/config agreement). A cap that
# is too small truncates episodes that would have succeeded and distorts the
# success rate.
declare -A TASK_MAXSTEPS=()
while read -r task steps; do
  TASK_MAXSTEPS[$task]=$steps
done < <("$LEARNER_PY" - "$SIMULATOR_CONFIG" <<'PYREAD'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for task, steps in (cfg.get("max_episode_steps") or {}).items():
    print(task, steps)
PYREAD
)
if [ ${#TASK_MAXSTEPS[@]} -eq 0 ]; then
  echo "[dsrl_eval] could not read the max_episode_steps map from $SIMULATOR_CONFIG" >&2
  exit 1
fi
MAXSTEPS=${EVAL_MAXSTEPS:-${TASK_MAXSTEPS[$TASK]:-450}}

# Drill tasks use the default drill object; the others define their own
# object and reject --object.
OBJ="--object none"
case "$TASK" in LMDrillLift|LMDrillLiftObs|LMDrillPnP|LMPickDrillFromHolder) OBJ="";; esac

GPU=${GPU:-0}
TRAIN_PORT=$(first_port "$GPU" 2) || exit 2; EVAL_PORT=$((TRAIN_PORT + 1))
TRAIN_SIM_DEVICE=${TRAIN_SIM_DEVICE:-gpu}
EVAL_SIM_DEVICE=${EVAL_SIM_DEVICE:-gpu}
BASE_MODEL=${BASE_MODEL:-gr00t}
ALGO=${ALGO:-}
case "$ALGO" in
  "") ;;
  sft|iql) if [ "$BASE_MODEL" = "gr00t" ]; then BASE_MODEL=gr00t_n17; fi ;;
  *) echo "[dsrl_eval] ALGO must be sft or iql (unset scores DSRL), got $ALGO" >&2; exit 2 ;;
esac
CKPT_KIND=""
if [ -n "$CKPT_DIR" ]; then
  CKPT_KIND=$("$LEARNER_PY" "$HERE/checkpoint_kind.py" "$CKPT_DIR") || { echo "[dsrl_eval] could not read the checkpoint type of $CKPT_DIR" >&2; exit 2; }
  case "$CKPT_KIND/${ALGO:-unset}" in
    ppo/unset) ;;
    ppo/*) echo "[dsrl_eval] $CKPT_DIR is a full-PPO checkpoint; unset ALGO (it is $ALGO) to score it" >&2; exit 2 ;;
    offline/unset|sft/unset|iql/unset) echo "[dsrl_eval] $CKPT_DIR is an offline (SFT/IQL) checkpoint; set ALGO=sft or ALGO=iql to score it" >&2; exit 2 ;;
    *) ;;
  esac
fi
RUNTIME=${RUNTIME:-native}
NAME=${WANDB_NAME:-eval_${TASK}_${MODE}_$(date +%m%d_%H%M%S)}
LOGD=${LOG_DIR:-$PWD/dsrl_eval_logs}
# Isaac Sim boots slowly on a loaded node; train.sh honors the same variable.
EVAL_NUM_ENVS_SET=${EVAL_NUM_ENVS:-}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-1}
SIM_BOOT_TIMEOUT=${SIM_BOOT_TIMEOUT:-$((600 + 120*(EVAL_NUM_ENVS - 1)))}
OUTPUT_ROOT=${ISAAC_RL_POSTTRAINING_OUTPUT_DIR:-$HOME/isaac_rl_outputs}

report_result () {  # $1=learner log; the trainer writes evaluation_summary.json next to the run's checkpoints
  # (<output root>/<wandb name>_<YYYY-MM-DD_HH-MM-SS>/checkpoints/); the log line is only the fallback.
  local summary sr
  summary=$(ls -t "$OUTPUT_ROOT"/"$NAME"_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/checkpoints/evaluation_summary.json 2>/dev/null | head -1)
  if [ -n "$summary" ]; then
    sr=$("$LEARNER_PY" -c 'import json, sys; rate = json.load(open(sys.argv[1]))["latest"]["success_rate"]; print(f"{100 * rate:.1f}")' "$summary")
    echo "[dsrl_eval] DONE $NAME SR=${sr}%  (summary: $summary; log: $1)"
  else
    sr=$(grep -aoE "Success Rate: [0-9.]+%" "$1" | tail -1 | grep -oE "[0-9.]+")
    echo "[dsrl_eval] DONE $NAME SR=${sr:-PARSE_FAIL}%  (no evaluation_summary.json; log: $1)"
  fi
  [ -n "$sr" ]
}

if [ "$CKPT_KIND" = "ppo" ]; then
  # As `isaaclab play --algorithm ppo`: finetune_ppo.sh scores the policy snapshot without a learner.
  PPO_ENVS=${EVAL_NUM_ENVS_SET:-$((NEPS < 8 ? NEPS : 8))}
  PPO_ARGS=(
    "$TASK" "ppo.eval_only=true" "placement.learner.ranks=0"
    "ppo.eval_episodes_per_environment=$(( (NEPS + PPO_ENVS - 1) / PPO_ENVS ))"
    "ppo.resume_from=$(cd "$CKPT_DIR" && pwd)" "training.seed=0"
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
  )
  echo "[dsrl_eval] $CKPT_DIR is a full-PPO checkpoint: $PPO_ENVS scenes x $(( (NEPS + PPO_ENVS - 1) / PPO_ENVS )) episodes through finetune_ppo.sh"
  export NUM_ENVS=$PPO_ENVS WANDB_NAME=$NAME LOG_DIR=$LOGD ISAAC_RL_POSTTRAINING_DRYRUN=$DRYRUN
  [ "$DRYRUN" = "1" ] && exec bash "$HERE/finetune_ppo.sh" "${PPO_ARGS[@]}"
  bash "$HERE/finetune_ppo.sh" "${PPO_ARGS[@]}" || { echo "[dsrl_eval] FAILED $NAME (exit $?; log: $LOGD/$NAME.train.log)"; exit 1; }
  report_result "$LOGD/$NAME.train.log"
  exit
fi

# Learner-side Hydra overrides (the dry run prints exactly this list as launch.override= lines).
# "base_model=$BASE_MODEL" selects the frozen-policy config group.
HYDRA_OVERRIDES=(
  "runtime=$RUNTIME"
  "base_model=$BASE_MODEL"
  "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
  "base_model.bc_ckpt=${POSTTRAIN_BC_CKPT:-${POSTTRAIN_BASE_CKPT:-}}"
  # Evaluation never fills the steering buffer (num_rollouts=0), so no dataset
  # is loaded: the checkpoint's own baked modality (correct state dim) is
  # authoritative for both the native policy and a restored DSRL actor. A
  # dataset here would register the dataset's modality instead, and the GR00T
  # state normalizer then crashes on the first evaluation.
  "base_model.dataset_dir=null"
  "training.num_rollouts=0"
  "eval.eval_on_first_step=true"
  "eval.eval_num_episodes=$NEPS"
  "eval.eval_num_envs=$EVAL_NUM_ENVS"
  "eval.eval_record_video=false"
  "env.max_episode_steps=$MAXSTEPS"
  "logging.wandb_offline=true"
  "logging.wandb_name=$NAME"
)
if [ "$RUNTIME" = "rlinf" ]; then
  # The evaluator sees one GPU; RLInf's rollout and policy-server groups share it.
  HYDRA_OVERRIDES+=("runtime.rlinf.first_accelerator=0" "runtime.rlinf.exclusive_accelerators=false")
fi
if [ -n "${POSTTRAIN_BACKBONE_MODEL:-}" ]; then
  HYDRA_OVERRIDES+=("base_model.backbone_model_name_or_path=$POSTTRAIN_BACKBONE_MODEL")
fi
# The GR00T policy is language-conditioned. With no dataset the task language
# is unset, so supply the task's prompt (task_prompts.sh) as the baked-embodiment default.
source "$HERE/task_prompts.sh"
TASK_LANG=$(task_prompt "$TASK") || TASK_LANG=""
# Hydra-quote the value: an unquoted comma (LMBoxLiftFloor) parses as a list and crashes.
[ -n "$TASK_LANG" ] && HYDRA_OVERRIDES+=("base_model.task_instruction='$TASK_LANG'")
if [ -n "$ALGO" ]; then
  # The offline policy decodes with infer; it reads the dataset only to register the G1 layout.
  HYDRA_OVERRIDES+=(
    "algorithm=$ALGO"
    "base_model.dataset_dir=${POSTTRAIN_DATA_ROOT:-}/$TASK/lerobot"
    "base_model.embodiment_tag=${EMBODIMENT_TAG:-new_embodiment}"
    "base_model.dataset_adapter=g1_wbc"
    "base_model.action_horizon=${ACTION_HORIZON:-16}"
    "base_model.denoising_steps=${DENOISING_STEPS:-4}"
    "eval.eval_freq=0"
    "logging.wandb_prefix=$ALGO"
  )
elif [ "$MODE" = "base" ]; then
  HYDRA_OVERRIDES+=("eval.eval_base=true")
fi
if [ "$MODE" = "policy" ]; then
  HYDRA_OVERRIDES+=("training.load_dir=$CKPT_DIR")
fi
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
  HYDRA_OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

SIM_ARGS="--embodiment g1_factory_wbc_joint --action-scale 1.0 --image-resolution 0 --disable-texture-streaming"

if [ "$DRYRUN" = "1" ]; then
  echo "[dsrl_eval] DRY RUN: the launch plan, then the library's plan; nothing will be launched"
  echo "launch.task=$TASK"
  echo "launch.mode=$MODE"
  echo "launch.episodes=$NEPS"
  echo "launch.gpu=$GPU"
  echo "launch.ports.train_sim=$TRAIN_PORT"
  echo "launch.ports.eval_sim=$EVAL_PORT"
  echo "launch.devices.train_sim=$TRAIN_SIM_DEVICE"
  echo "launch.devices.eval_sim=$EVAL_SIM_DEVICE"
  echo "launch.env.learner.ARENA_SERVER_URL=tcp://127.0.0.1:$TRAIN_PORT"
  echo "launch.env.learner.ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:$EVAL_PORT"
  echo "launch.env.learner.ARENA_IMAGE_RESOLUTION=0"
  echo "launch.env.learner.ARENA_ACTION_DIM=35"
  echo "launch.paths.ISAACLAB_ARENA_PATH=${ISAACLAB_ARENA_PATH:-<unset>}"
  echo "launch.paths.POSTTRAIN_BASE_CKPT=${POSTTRAIN_BASE_CKPT:-<unset>}"
  echo "launch.paths.POSTTRAIN_DATA_ROOT=${POSTTRAIN_DATA_ROOT:-<unset>}"
  echo "launch.secrets.HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
  echo "launch.secrets.WANDB_API_KEY=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo unset)"
  if [ -z "$ALGO" ]; then
    echo "launch.cmd.train_sim=simulators/g1_arena/env_server.py --task $TASK --port $TRAIN_PORT $SIM_ARGS $OBJ"
  fi
  echo "launch.cmd.eval_sim=simulators/g1_arena/env_server.py --task $TASK --port $EVAL_PORT $SIM_ARGS $OBJ --num-envs $EVAL_NUM_ENVS"
  echo "launch.cmd.learner=scripts/train.py"
  print_overrides "${HYDRA_OVERRIDES[@]}"
  # Plan with the GPU the learner will see, so the dry run matches the real placement.
  CUDA_VISIBLE_DEVICES=$GPU \
  ARENA_SERVER_URL=tcp://127.0.0.1:$TRAIN_PORT ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:$EVAL_PORT \
  PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-} \
    library_dry_run "$LEARNER_PY" "$REPO_ROOT" "${HYDRA_OVERRIDES[@]}"
  exit $?
fi

# ---------------------------------------------------------------------------
# Real launch
# ---------------------------------------------------------------------------
ARENA=${ISAACLAB_ARENA_PATH:?set ISAACLAB_ARENA_PATH}
: "${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT}"
{ [ "$MODE" = "replay" ] || [ -n "$ALGO" ]; } && : "${POSTTRAIN_DATA_ROOT:?set POSTTRAIN_DATA_ROOT (replay and ALGO=sft|iql read the dataset)}"
refuse_viz_without_display dsrl_eval
SIM_PY=${SIM_PYTHON:-${WORKFLOW_PY:-$ARENA/.venv/bin/python}}
mkdir -p "$LOGD"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE

start_sim () {  # $1=port $2=sim device (cpu|gpu) $3=log file $4=extra server args
  ( export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
           ISAACLAB_ARENA_PATH=$ARENA CUDA_VISIBLE_DEVICES=$GPU \
           ARENA_SIM_DEVICE=$2
    cd "$ARENA"
    export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
    exec setsid "$SIM_PY" "$REPO_ROOT/simulators/g1_arena/env_server.py" \
      --task "$TASK" --port "$1" $SIM_ARGS $OBJ ${4:-} ) > "$3" 2>&1 < /dev/null &
  echo $!
}

sim_alive () {  # the simulator (by pid, or its group once setsid ran) has a live, not zombie, process; a dead child of a non-reaping init stays a zombie
  ps -eo pid=,pgid=,stat= | awk -v g="$1" '($1 == g || $2 == g) && $3 !~ /^Z/ { f = 1 } END { exit !f }'
}

wait_listen () {  # $1=log file $2=timeout in seconds $3=simulator pid
  local t=0
  while [ "$t" -lt "$2" ]; do
    grep -aq "listening on" "$1" 2>/dev/null && return 0
    grep -aqE "Traceback|ModuleNotFound|Error:" "$1" 2>/dev/null && return 1
    sim_alive "$3" || { echo "[dsrl_eval] the simulator exited before listening"; return 1; }
    sleep 5; t=$((t+5))
  done
  echo "[dsrl_eval] the simulator was not listening after ${2}s"
  return 1
}

stop_sims () {  # TERM, grace period, then KILL (simulator teardown can hang)
  # Each simulator leads its own process group (setsid in start_sim), so a launcher that wraps
  # the interpreter (Isaac Sim's python.sh) goes down with every child holding the port.
  local pid
  for pid in "$@"; do [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null; done
  for _ in 1 2 3 4 5; do
    sleep 3
    local live=0
    for pid in "$@"; do [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null && live=1; done
    [ "$live" = "1" ] || return 0
  done
  for pid in "$@"; do [ -n "$pid" ] && kill -9 -- "-$pid" 2>/dev/null; done
}

echo "[dsrl_eval] $NAME task=$TASK mode=$MODE eps=$NEPS maxsteps=$MAXSTEPS gpu=$GPU ports=$TRAIN_PORT/$EVAL_PORT devices=$TRAIN_SIM_DEVICE/$EVAL_SIM_DEVICE"
# SIM_STARTUP_LOCK serializes simulator cold-starts across concurrent jobs on
# one node (parallel kit-cache builds race).
if [ -n "${SIM_STARTUP_LOCK:-}" ]; then exec 200>"$SIM_STARTUP_LOCK"; flock 200; fi
refuse_busy_ports "$LEARNER_PY" "$TRAIN_PORT" "$EVAL_PORT"
TRAIN_SIM_PID=""; EVAL_SIM_PID=""
cleanup () { stop_sims "$TRAIN_SIM_PID" "$EVAL_SIM_PID"; }
trap cleanup EXIT  # before booting: an interrupted boot must not leave simulators behind
if [ "$MODE" != "replay" ] && [ -z "$ALGO" ]; then  # replay and offline evaluation drive the eval sim only
  TRAIN_SIM_PID=$(SIM_VIZ= start_sim "$TRAIN_PORT" "$TRAIN_SIM_DEVICE" "$LOGD/$NAME.trainsim.log")
  wait_listen "$LOGD/$NAME.trainsim.log" "$SIM_BOOT_TIMEOUT" "$TRAIN_SIM_PID" || { echo "[dsrl_eval] train sim failed (see $LOGD/$NAME.trainsim.log)"; exit 1; }
fi
EVAL_SIM_PID=$(start_sim "$EVAL_PORT" "$EVAL_SIM_DEVICE" "$LOGD/$NAME.evalsim.log" "--num-envs $EVAL_NUM_ENVS")
wait_listen "$LOGD/$NAME.evalsim.log" "$SIM_BOOT_TIMEOUT" "$EVAL_SIM_PID" || { explain_viz_failure dsrl_eval "$LOGD/$NAME.evalsim.log"; echo "[dsrl_eval] eval sim failed (see $LOGD/$NAME.evalsim.log)"; exit 1; }
if [ -n "${SIM_STARTUP_LOCK:-}" ]; then flock -u 200; exec 200>&-; fi

if [ "$MODE" = "replay" ]; then  # open-loop dataset replay through the eval sim
  ( export CUDA_VISIBLE_DEVICES=$GPU ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:$EVAL_PORT
    export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
    exec "$LEARNER_PY" -u "$HERE/replay_dataset.py" \
      --dataset-dir "${POSTTRAIN_DATA_ROOT:-}/$TASK/lerobot" \
      --episode "${REPLAY_EPISODE:-0}" \
      --server-url "tcp://127.0.0.1:$EVAL_PORT" \
      --out "$LOGD/$NAME.replay.mp4" --max-steps "$MAXSTEPS" ) > "$LOGD/$NAME.eval.log" 2>&1 \
    || { echo "[dsrl_eval] REPLAY FAILED $NAME (exit $?; log: $LOGD/$NAME.eval.log)"; exit 1; }
  echo "[dsrl_eval] REPLAY DONE $NAME -> $LOGD/$NAME.replay.mp4  (log: $LOGD/$NAME.eval.log)"
  exit 0
fi

( export CUDA_VISIBLE_DEVICES=$GPU ARENA_IMAGE_RESOLUTION=0 ARENA_ACTION_DIM=35
  export ARENA_SERVER_URL=tcp://127.0.0.1:$TRAIN_PORT ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:$EVAL_PORT
  export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
  exec "$LEARNER_PY" -u "$REPO_ROOT/scripts/train.py" \
    "${HYDRA_OVERRIDES[@]}" \
) > "$LOGD/$NAME.eval.log" 2>&1 || { echo "[dsrl_eval] FAILED $NAME (exit $?; log: $LOGD/$NAME.eval.log)"; exit 1; }

report_result "$LOGD/$NAME.eval.log"
