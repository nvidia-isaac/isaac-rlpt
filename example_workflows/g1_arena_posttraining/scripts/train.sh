#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Training launcher for one G1 Factory task. DSRL and PPO use separate train and
# evaluation simulators so fixed evaluation cannot perturb training state.
# ALGO=sft|iql trains GR00T N1.7 offline from the task's demonstrations: no
# simulator starts, and eval.sh scores a saved checkpoint afterwards.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/train.sh TASK [extra Hydra overrides...]
#     TASK   task name, with or without the LM prefix (DrillLift or LMDrillLift)
#
# Required environment (not needed for a dry run):
#   ISAACLAB_ARENA_PATH  IsaacLab-Arena checkout (with its .venv)
#   POSTTRAIN_BASE_CKPT       GR00T checkpoint directory (HF-style)
#   POSTTRAIN_BC_CKPT         optional separate BC checkpoint (defaults to POSTTRAIN_BASE_CKPT)
#   POSTTRAIN_DATA_ROOT       datasets root for ALGO=sft|iql; expects $POSTTRAIN_DATA_ROOT/<LMTask>/lerobot
#                             (online algorithms read the checkpoint's baked embodiment instead)
# Optional:
#   BASE_MODEL      base model (default gr00t; mock|gr00t|pi0|dreamzero,
#                   see docs/base_models.md); becomes the Hydra base_model group override
#   SIMULATOR (default g1_arena)        simulator config the per-task episode
#                                       caps are read from; this launcher
#                                       supports only g1_arena (see
#                                       simulators/README.md)
#   NUM_ENVS (default 1)  parallel envs hosted by ONE batched train simulator
#                         (env_server.py --num-envs; Isaac Lab's native scene
#                         cloning). >1 also sets training.num_envs for the
#                         learner; DSRL's eval simulator stays single-env. PPO uses the
#                         same env count for both simulators. Requires the WBC patch in
#                         simulators/g1_arena/patches/ (see README).
#   NUM_SERVERS (default 1)  training simulators; NUM_ENVS is split equally over them
#                         (at least two each when >1), one rollout worker per server
#                         on consecutive ports from the train port, the evaluation
#                         server(s) on the ports after them (DSRL-PPO: as many, shaped
#                         like the training ones). With runtime.native.processes=ray
#                         the workers collect in parallel.
#   ENV_SPACING           meters between cloned envs (server --env-spacing;
#                         unset: the server's default, 200 m for NUM_ENVS>1)
#   RENDER_ONCE_PRIMS     server --render-once-prims override (unset: the
#                         server renders the factory's dome light once
#                         for NUM_ENVS>1; "none" disables)
#   TRAIN_SIM_DEVICE / EVAL_SIM_DEVICE  physics device per simulator, cpu|gpu
#                                       (default gpu); becomes ARENA_SIM_DEVICE
#   LEARNER_PYTHON, SIM_PYTHON (default: the workflow environment's .venv/bin/python once `uv sync` built it,
#     else python3 and $ISAACLAB_ARENA_PATH/.venv/bin/python)
#   GR00T_REPO      Isaac-GR00T checkout for PYTHONPATH (if gr00t isn't installed)
#   POSTTRAIN_BACKBONE_MODEL  optional local/HF override for GR00T's VLM backbone
#   GPU (default 0)  PORT_BASE (default 7300)  LOG_DIR (default ./dsrl_train_logs)
#   NUM_ROLLOUTS    env-step budget (unset: the config's training.num_rollouts)
#   TRAIN_MAXSTEPS  override the per-task episode cap
#   ALGO (default dsrl_sac)  algorithm config group: dsrl_sac | rfs | sac_edit | dsrl_ppo | ppo | sft | iql
#   sft and iql only: BATCH_SIZE (default 8; a 3B N1.7 forward fits 48 GB), SAVE_INTERVAL (default 5000),
#     EMBODIMENT_TAG (default new_embodiment), ACTION_HORIZON (default 16), DENOISING_STEPS (default 4);
#     NUM_ROLLOUTS counts updates (default 40000), and BASE_MODEL=gr00t trains the N1.7 adapter (gr00t_n17)
#   TRAINER         dsrl | ppo: one-release alias for ALGO=dsrl_sac | ALGO=dsrl_ppo
#   RUNTIME (default native)  native | rlinf; rlinf with one learner rank on one
#                             LEARNER_GPUS device shares it with the rollout and policy-server groups
#   PPO_LEARNER_RANKS (default 1)  learner ranks of dsrl_ppo (placement.learner.ranks); RLInf uses FSDP above 1
#   LEARNER_GPUS (default GPU)  CUDA devices visible to the learner/runtime
#   WANDB_NAME      wandb run name (default dsrl_<task>_<timestamp>)
#   ISAAC_RL_POSTTRAINING_OUTPUT_DIR  checkpoint/log root used by the trainer
#   EVAL_RESET_STATE_DIR  optional directory for recording and replaying full
#                         Isaac Lab evaluation scene states across checkpoints
#   EVAL_RESET_RERENDERS  RTX sensor refreshes forced for each recorded or
#                         restored full evaluation reset (default 1)
#   EVAL_RESET_OBSERVATION_DIR  optional diagnostic directory for compressed
#                               camera and state arrays from full seeded resets
#   SIM_ANTIALIASING_MODE       optional common renderer mode for both sims:
#                               off, taa, fxaa, dlss, or rtxaa
#   SIM_STRICT_REPRODUCIBILITY  set to 1 to enable deterministic RTX, PhysX,
#                               and torch settings in both PPO simulators
#   SIM_STARTUP_LOCK    lock file serializing simulator cold-starts on one node
# Secrets are read from the environment only and are never printed:
#   HF_TOKEN, WANDB_API_KEY
#
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 (or a --dry-run argument) prints this launcher's
# launch.* lines and every override it passes, then the library's own dry run (plan.* lines and
# the resolved config), and exits without starting anything.
set -uo pipefail

TASK=${1:?usage: example_workflows/g1_arena_posttraining/scripts/train.sh TASK [hydra overrides...]}
shift || true

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

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# shellcheck source=../../_common/launch.sh
source "$REPO_ROOT/example_workflows/_common/launch.sh"
refuse_renamed_knob BACKEND BASE_MODEL
# The workflow's own environment (scripts/fetch_externals.sh && uv sync) runs both the learner and the simulator.
WORKFLOW_PY=$(workflow_python "$REPO_ROOT/example_workflows/g1_arena_posttraining")
LEARNER_PY=${LEARNER_PYTHON:-${WORKFLOW_PY:-python3}}

# Simulator selection (see simulators/README.md). This launcher boots the
# g1_arena env servers; the knob exists so every launcher resolves its
# simulator config the same way.
SIMULATOR=${SIMULATOR:-g1_arena}
if [[ "$SIMULATOR" != "g1_arena" ]]; then
  echo "[dsrl_train] SIMULATOR=$SIMULATOR is not supported by this launcher (it drives the g1_arena simulator); see simulators/README.md" >&2
  exit 2
fi
SIMULATOR_CONFIG="$REPO_ROOT/simulators/g1_arena/configs/simulator/$SIMULATOR.yaml"

# Per-task episode caps (longest demonstration plus margin), read from the
# simulator config -- the single source of truth
# (tests/unit/test_simulators.py pins launcher/config agreement). A cap that
# is too small truncates episodes that would have succeeded.
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
if [[ ${#TASK_MAXSTEPS[@]} -eq 0 ]]; then
  echo "[dsrl_train] could not read the max_episode_steps map from $SIMULATOR_CONFIG" >&2
  exit 1
fi
MAXSTEPS=${TRAIN_MAXSTEPS:-${TASK_MAXSTEPS[$TASK]:-450}}

# Drill tasks use the default drill object; the others define their own
# object and reject --object.
OBJ="--object none"
case "$TASK" in LMDrillLift|LMDrillLiftObs|LMDrillPnP|LMPickDrillFromHolder) OBJ="";; *) ;; esac

GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-1}
NUM_SERVERS=${NUM_SERVERS:-1}
case "$NUM_SERVERS" in
  ''|*[!0-9]*|0) echo "[dsrl_train] NUM_SERVERS must be a positive integer, got '$NUM_SERVERS'" >&2; exit 2 ;;
  *) ;;
esac
if [[ "$NUM_SERVERS" -gt 1 ]] && { [[ $((NUM_ENVS % NUM_SERVERS)) -ne 0 ]] || [[ $((NUM_ENVS / NUM_SERVERS)) -lt 2 ]]; }; then
  echo "[dsrl_train] NUM_SERVERS=$NUM_SERVERS must split NUM_ENVS=$NUM_ENVS environments equally, at least two per server" >&2
  exit 2
fi
ENVS_PER_SERVER=$((NUM_ENVS / NUM_SERVERS))
TRAIN_SIM_DEVICE=${TRAIN_SIM_DEVICE:-gpu}
EVAL_SIM_DEVICE=${EVAL_SIM_DEVICE:-gpu}
EVAL_RESET_STATE_DIR=${EVAL_RESET_STATE_DIR:-}
EVAL_RESET_OBSERVATION_DIR=${EVAL_RESET_OBSERVATION_DIR:-}
SIM_ANTIALIASING_MODE=${SIM_ANTIALIASING_MODE:-}
SIM_STRICT_REPRODUCIBILITY=${SIM_STRICT_REPRODUCIBILITY:-0}
BASE_MODEL=${BASE_MODEL:-gr00t}
# ALGO selects the algorithm config group; TRAINER=dsrl|ppo stays an alias for one release.
TRAINER=${TRAINER:-}
case "$TRAINER" in
  ""|dsrl) ALGO=${ALGO:-dsrl_sac} ;;
  ppo)     ALGO=${ALGO:-dsrl_ppo} ;;
  *) echo "[dsrl_train] TRAINER must be dsrl or ppo, got $TRAINER" >&2; exit 2 ;;
esac
ON_POLICY=0
if [[ "$ALGO" = "dsrl_ppo" ]]; then ON_POLICY=1; fi
OFFLINE=0
if [[ "$ALGO" = "sft" ]] || [[ "$ALGO" = "iql" ]]; then
  OFFLINE=1
  if [[ "$BASE_MODEL" = "gr00t" ]]; then BASE_MODEL=gr00t_n17; fi
fi
RUNTIME=${RUNTIME:-native}
PPO_LEARNER_RANKS=${PPO_LEARNER_RANKS:-1}
LEARNER_GPUS=${LEARNER_GPUS:-$GPU}
PPO_EVAL_ONLY=0
VIZ_SIM_KIND=eval  # the simulator that steps the rollouts being watched: SIM_VIZ=kit opens its viewer
for override in "${EXTRA_OVERRIDES[@]}"; do
  case "$override" in
    ppo.eval_only=true|ppo.eval_only=True)
      # dsrl_ppo scores on the evaluation simulators; full PPO scores its rollouts on the training ones.
      if [[ "$ON_POLICY" = "1" ]]; then PPO_EVAL_ONLY=1; else VIZ_SIM_KIND=train; fi ;;
    *) ;;
  esac
done
# DSRL-PPO evaluates on servers shaped like the training ones (its trainer swaps equal rollout
# groups); DSRL scores one episode at a time on a single evaluation server.
EVAL_SERVERS=1
if [[ "$ON_POLICY" = "1" ]]; then EVAL_SERVERS=$NUM_SERVERS; fi
# The training servers' ports, then the evaluation servers'.
TRAIN_PORT=$(first_port "$GPU" $((NUM_SERVERS + EVAL_SERVERS))) || exit 2
TRAIN_PORTS=(); EVAL_PORTS=()
for ((i = 0; i < NUM_SERVERS; i++)); do TRAIN_PORTS+=($((TRAIN_PORT + i))); done
for ((i = 0; i < EVAL_SERVERS; i++)); do EVAL_PORTS+=($((TRAIN_PORT + NUM_SERVERS + i))); done
server_urls () { local urls; urls=$(printf 'tcp://127.0.0.1:%s,' "$@"); echo "${urls%,}"; }
TRAIN_URLS=$(server_urls "${TRAIN_PORTS[@]}"); EVAL_URLS=$(server_urls "${EVAL_PORTS[@]}")
LEARNER_SERVER_URLS=$TRAIN_URLS
if [[ "$PPO_EVAL_ONLY" = "1" ]]; then LEARNER_SERVER_URLS=$EVAL_URLS; fi
TRAIN_ENTRYPOINT="scripts/train.py"
NUM_ROLLOUTS=${NUM_ROLLOUTS:-}
if [[ "$OFFLINE" = "1" ]]; then NUM_ROLLOUTS=${NUM_ROLLOUTS:-40000}; fi
NAME=${WANDB_NAME:-${ALGO}_${TASK}_$(date +%m%d_%H%M%S)}
LOGD=${LOG_DIR:-$PWD/dsrl_train_logs}

# Learner-side Hydra overrides (the dry run prints exactly this list as launch.override= lines).
# "base_model=$BASE_MODEL" selects the frozen-policy config group.
HYDRA_OVERRIDES=(
  "algorithm=$ALGO"
  "runtime=$RUNTIME"
  "base_model=$BASE_MODEL"
  "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
  "base_model.bc_ckpt=${POSTTRAIN_BC_CKPT:-${POSTTRAIN_BASE_CKPT:-}}"
  "env.max_episode_steps=$MAXSTEPS"
  "logging.wandb_name=$NAME"
)
if [[ -n "$NUM_ROLLOUTS" ]]; then
  HYDRA_OVERRIDES+=("training.num_rollouts=$NUM_ROLLOUTS")
fi
if [[ -n "${POSTTRAIN_BACKBONE_MODEL:-}" ]]; then
  HYDRA_OVERRIDES+=("base_model.backbone_model_name_or_path=$POSTTRAIN_BACKBONE_MODEL")
fi
if [[ "$ON_POLICY" = "1" ]]; then
  # The rest of the dsrl_ppo recipe (one decision = one whole 40-step chunk, the actor's
  # initialisation) lives in configs/algorithm/dsrl_ppo.yaml.
  HYDRA_OVERRIDES+=("placement.learner.ranks=$PPO_LEARNER_RANKS")
fi
if [[ "$OFFLINE" != "1" ]]; then
  # Online training needs no dataset: the checkpoint's baked embodiment is authoritative (as in eval.sh),
  # and the task prompt replaces the dataset's task text. Hydra-quoted: one prompt has a comma.
  source "$REPO_ROOT/example_workflows/g1_arena_posttraining/scripts/task_prompts.sh"
  TASK_LANG=$(task_prompt "$TASK") || { echo "[dsrl_train] no task prompt for $TASK (task_prompts.sh)" >&2; exit 2; }
  HYDRA_OVERRIDES+=("base_model.dataset_dir=null" "base_model.task_instruction='$TASK_LANG'")
fi
if [[ "$OFFLINE" = "1" ]]; then
  # Offline training only reads the demonstrations; eval.sh scores the saved checkpoint.
  HYDRA_OVERRIDES+=(
    "base_model.dataset_dir=${POSTTRAIN_DATA_ROOT:-}/$TASK/lerobot"
    "base_model.embodiment_tag=${EMBODIMENT_TAG:-new_embodiment}"
    "base_model.dataset_adapter=g1_wbc"
    "base_model.action_horizon=${ACTION_HORIZON:-16}"
    "base_model.denoising_steps=${DENOISING_STEPS:-4}"
    "algorithm.batch_size=${BATCH_SIZE:-8}"
    "training.save_interval=${SAVE_INTERVAL:-5000}"
    "training.checkpoint_retention=best_last_final"
    "eval.eval_freq=0"
    "eval.eval_on_first_step=false"
    "logging.wandb_prefix=$ALGO"
  )
fi
if [[ "$RUNTIME" = "rlinf" ]] && [[ "$LEARNER_GPUS" != *,* ]] && [[ "$PPO_LEARNER_RANKS" = 1 ]]; then
  # RLInf puts its rollout groups after the learner's GPU by default; with one GPU and one rank they share it.
  HYDRA_OVERRIDES+=("runtime.rlinf.first_accelerator=0" "runtime.rlinf.exclusive_accelerators=false")
fi
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  HYDRA_OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi
# NUM_ENVS>1 boots natively batched train and PPO evaluation simulators.
# DSRL's evaluator stays single-env: it scores episodes on its own dedicated server.
if [[ "$NUM_ENVS" -gt 1 ]]; then
  # One batched simulator hosts every environment and hands back whole episodes, so collection is counted by episode.
  HYDRA_OVERRIDES+=("training.num_envs=$NUM_ENVS" "training.rollout_count_by=episode")
fi

# IMAGE_SHORTEST_EDGE shrinks frames on the simulator to the policy processor's input edge (GR00T-N1.7: 256),
# the same first transform the policy applies; unset sends native frames.
SIM_ARGS="--embodiment g1_factory_wbc_joint --action-scale 1.0 --image-resolution 0 --disable-texture-streaming${IMAGE_SHORTEST_EDGE:+ --image-shortest-edge $IMAGE_SHORTEST_EDGE}"
# Batched-scene arguments; each training server (and each of PPO's evaluation servers) hosts ENVS_PER_SERVER environments.
SCENE_ARGS="${ENV_SPACING:+ --env-spacing $ENV_SPACING}${RENDER_ONCE_PRIMS:+ --render-once-prims $RENDER_ONCE_PRIMS}"
if [[ -n "$SIM_ANTIALIASING_MODE" ]]; then
  SCENE_ARGS="$SCENE_ARGS --renderer-antialiasing $SIM_ANTIALIASING_MODE"
fi
if [[ "$SIM_STRICT_REPRODUCIBILITY" = "1" ]]; then
  SCENE_ARGS="$SCENE_ARGS --strict-reproducibility"
elif [[ "$SIM_STRICT_REPRODUCIBILITY" != "0" ]]; then
  echo "[dsrl_train] SIM_STRICT_REPRODUCIBILITY must be 0 or 1" >&2
  exit 2
fi
TRAIN_SIM_ARGS="--num-envs $ENVS_PER_SERVER$SCENE_ARGS"
EVAL_SIM_ARGS=
if [[ "$ON_POLICY" = "1" ]]; then
  EVAL_SIM_ARGS="--num-envs $ENVS_PER_SERVER$SCENE_ARGS"
  if [[ -n "$EVAL_RESET_STATE_DIR" ]]; then
    EVAL_SIM_ARGS="$EVAL_SIM_ARGS --reset-state-dir $EVAL_RESET_STATE_DIR --reset-rerenders ${EVAL_RESET_RERENDERS:-1}"
  fi
  if [[ -n "$EVAL_RESET_OBSERVATION_DIR" ]]; then
    EVAL_SIM_ARGS="$EVAL_SIM_ARGS --reset-observation-dir $EVAL_RESET_OBSERVATION_DIR"
  fi
fi

if [[ "$DRYRUN" = "1" ]]; then
  echo "[dsrl_train] DRY RUN: the launch plan, then the library's plan; nothing will be launched"
  echo "launch.task=$TASK"
  echo "launch.num_envs=$NUM_ENVS"
  echo "launch.num_servers=$NUM_SERVERS"
  echo "launch.ppo_eval_only=$PPO_EVAL_ONLY"
  echo "launch.viz_sim=$VIZ_SIM_KIND"
  echo "launch.gpu=$GPU"
  echo "launch.learner_gpus=$LEARNER_GPUS"
  echo "launch.ports.train_sim=$(IFS=,; echo "${TRAIN_PORTS[*]}")"
  echo "launch.ports.eval_sim=$(IFS=,; echo "${EVAL_PORTS[*]}")"
  echo "launch.devices.train_sim=$TRAIN_SIM_DEVICE"
  echo "launch.devices.eval_sim=$EVAL_SIM_DEVICE"
  echo "launch.env.learner.ARENA_SERVER_URL=$LEARNER_SERVER_URLS"
  echo "launch.env.learner.ARENA_EVAL_SERVER_URL=$EVAL_URLS"
  echo "launch.env.learner.ARENA_IMAGE_RESOLUTION=0"
  echo "launch.env.learner.ARENA_ACTION_DIM=35"
  echo "launch.paths.ISAACLAB_ARENA_PATH=${ISAACLAB_ARENA_PATH:-<unset>}"
  echo "launch.paths.POSTTRAIN_BASE_CKPT=${POSTTRAIN_BASE_CKPT:-<unset>}"
  echo "launch.paths.POSTTRAIN_DATA_ROOT=${POSTTRAIN_DATA_ROOT:-<unset>}"
  echo "launch.paths.eval_reset_state_dir=${EVAL_RESET_STATE_DIR:-<disabled>}"
  echo "launch.paths.eval_reset_observation_dir=${EVAL_RESET_OBSERVATION_DIR:-<disabled>}"
  echo "launch.sim.antialiasing_mode=${SIM_ANTIALIASING_MODE:-<default>}"
  echo "launch.sim.strict_reproducibility=$SIM_STRICT_REPRODUCIBILITY"
  echo "launch.secrets.HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo unset)"
  echo "launch.secrets.WANDB_API_KEY=$([[ -n "${WANDB_API_KEY:-}" ]] && echo set || echo unset)"
  if [[ "$OFFLINE" = "1" ]]; then
    echo "launch.cmd.train_sim=<none: $ALGO trains offline>"
  elif [[ "$PPO_EVAL_ONLY" = "1" ]]; then
    echo "launch.cmd.train_sim=<disabled for PPO evaluation-only>"
  else
    for port in "${TRAIN_PORTS[@]}"; do
      echo "launch.cmd.train_sim=simulators/g1_arena/env_server.py --task $TASK --port $port $SIM_ARGS $TRAIN_SIM_ARGS $OBJ"
    done
  fi
  if [[ "$OFFLINE" != "1" ]]; then
    for port in "${EVAL_PORTS[@]}"; do
      echo "launch.cmd.eval_sim=simulators/g1_arena/env_server.py --task $TASK --port $port $SIM_ARGS $EVAL_SIM_ARGS $OBJ"
    done
  fi
  echo "launch.cmd.learner=$TRAIN_ENTRYPOINT"
  print_overrides "${HYDRA_OVERRIDES[@]}"
  # Plan with the GPUs the learner will see, so the dry run matches the real placement.
  CUDA_VISIBLE_DEVICES=$LEARNER_GPUS \
  ARENA_SERVER_URL=$LEARNER_SERVER_URLS ARENA_EVAL_SERVER_URL=$EVAL_URLS \
  PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-} \
    library_dry_run "$LEARNER_PY" "$REPO_ROOT" "${HYDRA_OVERRIDES[@]}"
  exit $?
fi

# ---------------------------------------------------------------------------
# Real launch
# ---------------------------------------------------------------------------
if [[ "$OFFLINE" != "1" ]]; then ARENA=${ISAACLAB_ARENA_PATH:?set ISAACLAB_ARENA_PATH}; fi
: "${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT}"
if [[ "$OFFLINE" = "1" ]]; then : "${POSTTRAIN_DATA_ROOT:?set POSTTRAIN_DATA_ROOT}"; else refuse_viz_without_display dsrl_train; fi
SIM_PY=${SIM_PYTHON:-${WORKFLOW_PY:-${ARENA:-}/.venv/bin/python}}
mkdir -p "$LOGD"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE

start_sim () {  # port, sim device (cpu|gpu), log file, extra server args
  local port=$1 sim_device=$2 log=$3 extra_args=${4:-}
  ( export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
           ISAACLAB_ARENA_PATH=$ARENA CUDA_VISIBLE_DEVICES=$GPU \
           ARENA_SIM_DEVICE=$sim_device
    cd "$ARENA"
    export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
    exec setsid "$SIM_PY" "$REPO_ROOT/simulators/g1_arena/env_server.py" \
      --task "$TASK" --port "$port" $SIM_ARGS $extra_args $OBJ ) > "$log" 2>&1 < /dev/null &
  echo $!
  return 0
}

sim_alive () {  # the simulator (by pid, or its group once setsid ran) has a live, not zombie, process; a dead child of a non-reaping init stays a zombie
  ps -eo pid=,pgid=,stat= | awk -v g="$1" '($1 == g || $2 == g) && $3 !~ /^Z/ { f = 1 } END { exit !f }'
}

wait_listen () {  # log file, timeout in seconds, simulator pid
  local log=$1 timeout_s=$2 pid=$3
  local t=0
  while [[ "$t" -lt "$timeout_s" ]]; do
    grep -aq "listening on" "$log" 2>/dev/null && return 0
    grep -aqE "Traceback|ModuleNotFound|Error:" "$log" 2>/dev/null && return 1
    sim_alive "$pid" || { echo "[dsrl_train] the simulator exited before listening"; return 1; }
    sleep 5; t=$((t+5))
  done
  echo "[dsrl_train] the simulator was not listening after ${timeout_s}s"
  return 1
}

stop_sims () {  # TERM, grace period, then KILL (simulator teardown can hang)
  # Each simulator leads its own process group (setsid in start_sim), so a launcher that wraps
  # the interpreter (Isaac Sim's python.sh) goes down with every child holding the port.
  local pid
  for pid in "$@"; do [[ -n "$pid" ]] && kill -- "-$pid" 2>/dev/null; done
  for _ in 1 2 3 4 5; do
    sleep 3
    local live=0
    for pid in "$@"; do [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null && live=1; done
    [[ "$live" = "1" ]] || return 0
  done
  for pid in "$@"; do [[ -n "$pid" ]] && kill -9 -- "-$pid" 2>/dev/null; done
}

echo "[dsrl_train] $NAME task=$TASK rollouts=${NUM_ROLLOUTS:-config-default} maxsteps=$MAXSTEPS num_envs=$NUM_ENVS servers=$NUM_SERVERS gpu=$GPU ports=$(IFS=,; echo "${TRAIN_PORTS[*]}")/$(IFS=,; echo "${EVAL_PORTS[*]}") devices=$TRAIN_SIM_DEVICE/$EVAL_SIM_DEVICE"
# SIM_STARTUP_LOCK serializes simulator cold-starts across concurrent jobs on
# one node (parallel kit-cache builds race).
if [[ -n "${SIM_STARTUP_LOCK:-}" ]]; then exec 200>"$SIM_STARTUP_LOCK"; flock 200; fi
if [[ "$OFFLINE" != "1" ]]; then refuse_busy_ports "$LEARNER_PY" "${TRAIN_PORTS[@]}" "${EVAL_PORTS[@]}"; fi
# Scene cloning is per-env USD work (Arena pins replicate_physics=False), so
# the batched simulator's cold start grows with NUM_ENVS — measured at 1-2
# minutes per env on a GB300 station, plus the app boot itself.
SIM_BOOT_TIMEOUT=${SIM_BOOT_TIMEOUT:-$((600 + 120*ENVS_PER_SERVER))}
SIM_PIDS=()
cleanup () {
  stop_sims ${SIM_PIDS[@]+"${SIM_PIDS[@]}"}
  return 0
}
trap cleanup EXIT  # before booting: an interrupted boot must not leave simulators behind
boot_sims () {  # kind (train|eval), sim device, server args, ports...: one after another, logs <name>.<kind>sim[<i>].log
  local kind=$1 device=$2 args=$3 i=0 port log viz=
  shift 3
  if [[ "$kind" = "$VIZ_SIM_KIND" ]]; then viz=${SIM_VIZ:-}; fi  # SIM_VIZ=kit opens a viewer on that one simulator
  for port in "$@"; do
    log=$LOGD/$NAME.${kind}sim.log
    if [[ "$i" -gt 0 ]]; then log=$LOGD/$NAME.${kind}sim$i.log; fi
    SIM_PIDS+=("$(SIM_VIZ=$viz start_sim "$port" "$device" "$log" "$args")")
    wait_listen "$log" "$SIM_BOOT_TIMEOUT" "${SIM_PIDS[-1]}" || { SIM_VIZ=$viz explain_viz_failure dsrl_train "$log"; echo "[dsrl_train] $kind sim failed (see $log)"; exit 1; }
    i=$((i + 1))
  done
}
if [[ "$PPO_EVAL_ONLY" != "1" ]] && [[ "$OFFLINE" != "1" ]]; then boot_sims train "$TRAIN_SIM_DEVICE" "$TRAIN_SIM_ARGS" "${TRAIN_PORTS[@]}"; fi
if [[ "$OFFLINE" != "1" ]]; then boot_sims eval "$EVAL_SIM_DEVICE" "$EVAL_SIM_ARGS" "${EVAL_PORTS[@]}"; fi
if [[ -n "${SIM_STARTUP_LOCK:-}" ]]; then flock -u 200; exec 200>&-; fi

( export CUDA_VISIBLE_DEVICES=$LEARNER_GPUS ARENA_IMAGE_RESOLUTION=0 ARENA_ACTION_DIM=35
  export ARENA_SERVER_URL=$LEARNER_SERVER_URLS
  export ARENA_EVAL_SERVER_URL=$EVAL_URLS
  export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
  exec "$LEARNER_PY" -u "$REPO_ROOT/$TRAIN_ENTRYPOINT" \
    "${HYDRA_OVERRIDES[@]}" \
) 2>&1 | tee "$LOGD/$NAME.train.log"
