#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container DSRL training: boots the batched train env-server (--num-envs N)
# and an eval env-server of EVAL_NUM_ENVS environments (one by default), then runs the learner (scripts/train.py) as a
# client, all under /isaac-sim/python.sh in one container. Invoked by
# station_train.sh; not run directly on the host.
#
# The learner connects over TCP and never boots Isaac itself, so only the two
# env-servers need the Arena runtime (ISAACLAB_ARENA_PATH). Paths are the
# container's: our repo is bind-mounted at /workspaces/isaaclab_arena/posttrain,
# and /data/{ckpts,datasets,out} are mounted from the station /data.
set -uo pipefail

WORKDIR=${WORKDIR:-/workspaces/isaaclab_arena}
ARENA=$WORKDIR/submodules/IsaacLab-Arena
POSTTRAIN=$WORKDIR/posttrain
# The Isaac Sim interpreter; overridable so the dry run resolves its plan on a host without it.
PY=${PY:-${CONTAINER_PYTHON:-/isaac-sim/python.sh}}
# ISAAC_RL_POSTTRAINING_DRYRUN=1 prints the resolved plan (runtime, cards, learner overrides) and
# exits before anything boots.
DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}

TASK=${TASK:-LMDrillLift}
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
N=${N:-8}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-}   # env-step budget (unset: the config's training.num_rollouts)
TRAIN_PORT=${TRAIN_PORT:-7300}
# Training simulators: NUM_SERVERS Arena servers on consecutive ports from TRAIN_PORT, each hosting
# N/NUM_SERVERS environments behind its own rollout worker, so one server can step while another's
# frames decode. The eval server takes the port after them.
NUM_SERVERS=${NUM_SERVERS:-1}
case "$NUM_SERVERS" in
  ''|*[!0-9]*|0) echo "[train] NUM_SERVERS must be a positive integer, got '$NUM_SERVERS'" >&2; exit 1 ;;
  *) ;;
esac
EVAL_PORT=${EVAL_PORT:-$((TRAIN_PORT + NUM_SERVERS))}
# EVAL_SIM=0 boots no eval simulator and turns evaluation off in the learner
# (collect-only: a throughput measurement, or a collector in a fan-out).
EVAL_SIM=${EVAL_SIM:-1}
# EVAL_NUM_ENVS environments on the eval server; the learner scores that many episodes in
# parallel (eval.eval_num_episodes must be a multiple of it). One episode at a time by default.
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-1}
case "$EVAL_NUM_ENVS" in
  ''|*[!0-9]*|0) echo "[train] EVAL_NUM_ENVS must be a positive integer, got '$EVAL_NUM_ENVS'" >&2; exit 1 ;;
  *) ;;
esac
# PARALLEL_SIM_BOOT=1 cold-starts the eval simulator alongside the first train simulator (~4 min faster);
# off by default because concurrent cold-starts can race on the shared asset download and kit cache.
PARALLEL_SIM_BOOT=${PARALLEL_SIM_BOOT:-0}
# Two-host split. SIMS_ONLY=1: boot the simulators (they listen on every
# interface) and serve until killed -- no learner here. TRAIN_SIM_URL (+
# EVAL_SIM_URL unless EVAL_SIM=0): the simulators run elsewhere; connect the
# learner to them and boot nothing.
SIMS_ONLY=${SIMS_ONLY:-0}
SIM_HOST=127.0.0.1
if [[ "$SIMS_ONLY" = 1 ]]; then SIM_HOST=0.0.0.0; fi  # the learner connects from another host
TRAIN_SIM_URL=${TRAIN_SIM_URL:-}
EVAL_SIM_URL=${EVAL_SIM_URL:-}
if [[ "$SIMS_ONLY" = 1 ]] && [[ -n "$TRAIN_SIM_URL" ]]; then
  echo "[train] SIMS_ONLY=1 and TRAIN_SIM_URL are exclusive: this host either serves the simulators or connects to them" >&2; exit 1
fi
if [[ -n "$TRAIN_SIM_URL" ]] && [[ "$EVAL_SIM" = 1 ]] && [[ -z "$EVAL_SIM_URL" ]]; then
  echo "[train] TRAIN_SIM_URL needs EVAL_SIM_URL too (or EVAL_SIM=0 for a run that never evaluates)" >&2; exit 1
fi
# GPU split inside the container: both simulators on SIM_GPU, the frozen policy
# + learner on LEARNER_GPU. Same GPU by default (the station's single card).
# The simulators should stay on GPU 0: Isaac Sim renders on the first Vulkan
# device regardless of CUDA_VISIBLE_DEVICES, so a sim pinned elsewhere would
# split its physics and render across two cards.
SIM_GPU=${SIM_GPU:-0}
LEARNER_GPU=${LEARNER_GPU:-0}
# Seconds a simulator may take to start listening; raise it when several stacks boot on one GPU.
SIM_BOOT_TIMEOUT=${SIM_BOOT_TIMEOUT:-$((300 + 120*N))}
# Rollout runtime. native keeps the rollout worker (Arena client, frozen policy, actor
# sampling) inside the learner process; rlinf places it as a Ray actor on ROLLOUT_GPU
# (default: the learner's card) and needs the image from Dockerfile.learner_rlinf.
RUNTIME=${RUNTIME:-native}
ROLLOUT_GPU=${ROLLOUT_GPU:-$LEARNER_GPU}
case "$RUNTIME" in
  native|rlinf) ;;
  *) echo "[train] RUNTIME must be native or rlinf, got $RUNTIME" >&2; exit 1 ;;
esac
if [[ "$RUNTIME" = native ]] && [[ "$ROLLOUT_GPU" != "$LEARNER_GPU" ]]; then
  echo "[train] ROLLOUT_GPU applies to RUNTIME=rlinf; the native runtime runs the rollout worker on LEARNER_GPU" >&2; exit 1
fi
if [[ "$NUM_SERVERS" -gt 1 ]]; then
  if [[ $((N % NUM_SERVERS)) -ne 0 ]] || [[ $((N / NUM_SERVERS)) -lt 2 ]]; then
    echo "[train] NUM_SERVERS=$NUM_SERVERS must split N=$N environments equally, at least two per server" >&2; exit 1
  fi
  # The learner's side of several servers: one rollout worker per server, which the native runtime
  # would step one after another (the servers collecting in turn), and which RLinf places together
  # on the learner's card. A simulator-only container runs no learner and serves any runtime.
  if [[ "$SIMS_ONLY" != 1 ]] && [[ "$RUNTIME" != rlinf ]]; then
    echo "[train] NUM_SERVERS=$NUM_SERVERS needs RUNTIME=rlinf: the native runtime steps its rollout workers one after another" >&2; exit 1
  fi
  if [[ "$SIMS_ONLY" != 1 ]] && [[ "$ROLLOUT_GPU" != "$LEARNER_GPU" ]]; then
    echo "[train] with NUM_SERVERS>1 the rollout workers share the learner's card; ROLLOUT_GPU must equal LEARNER_GPU" >&2; exit 1
  fi
fi
ENVS_PER_SERVER=$((N / NUM_SERVERS))
if [[ "$SIMS_ONLY" = 1 ]]; then
  BASE_CKPT=; DATA_ROOT=   # no learner on this host
else
  BASE_CKPT=${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT (the G1-N1.7 checkpoint dir)}
  DATA_ROOT=${POSTTRAIN_DATA_ROOT:?set POSTTRAIN_DATA_ROOT (holds <LMTask>/lerobot)}
fi
WANDB_NAME=${WANDB_NAME:-dsrl_${TASK}_gb300_$(date +%m%d_%H%M%S)}
WANDB_PROJECT=${WANDB_PROJECT:-isaac_rl_posttraining}
WANDB_ENTITY=${WANDB_ENTITY:-}
OUT_DIR=${ISAAC_RL_POSTTRAINING_OUTPUT_DIR:-/data/dsrl_out_g1arena}

# Per-task episode cap, read from the simulator config (the single source of
# truth; a wrong flat cap distorts success rates).
MAXSTEPS=$($PY - "$POSTTRAIN/simulators/g1_arena/configs/simulator/g1_arena.yaml" "$TASK" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print((cfg.get("max_episode_steps") or {}).get(sys.argv[2], 450))
PY
)

OBJ="--object none"
case "$TASK" in LMDrillLift|LMDrillLiftObs|LMDrillPnP|LMPickDrillFromHolder) OBJ="";; *) ;; esac
# Frames leave the simulator already shrunk to the policy transform's first
# stage (shortest side 256, GR00T-N1.7's shortest_image_edge): identical model
# inputs, 3.5x less to copy, send and preprocess. 0 sends native frames.
IMAGE_SHORTEST_EDGE=${IMAGE_SHORTEST_EDGE:-256}
SIM_ARGS="--embodiment g1_factory_wbc_joint --action-scale 1.0 --image-resolution 0 --disable-texture-streaming --image-shortest-edge $IMAGE_SHORTEST_EDGE"

export ISAACLAB_ARENA_PATH=$ARENA
export PYTHONPATH=$POSTTRAIN:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE

# Learner-side overrides that follow from this launcher's choices. A batched simulator
# (N>1) collects whole episodes, so rollouts are counted by episode (the runtime rejects
# step-counted collection for it).
BATCH_OVERRIDES=()
if [[ "$N" -gt 1 ]]; then BATCH_OVERRIDES=(training.rollout_count_by=episode); fi
if [[ "$EVAL_SIM" = 1 ]]; then
  EVAL_OVERRIDES=("eval.eval_num_envs=$EVAL_NUM_ENVS")
else
  EVAL_OVERRIDES=(eval.eval_freq=0 eval.eval_on_first_step=false)
fi
RUNTIME_OVERRIDES=("runtime=$RUNTIME")
RAY_TMPDIR=
if [[ "$RUNTIME" = rlinf ]]; then
  # RLinf enumerates the container's cards itself and hands the rollout actor exactly the
  # card named here, whatever CUDA_VISIBLE_DEVICES the learner runs under. The dedicated
  # evaluation group lands on that same card, so the runtime's one-group-per-card rule is lifted.
  RUNTIME_OVERRIDES+=("runtime.rlinf.first_accelerator=$ROLLOUT_GPU" runtime.rlinf.exclusive_accelerators=false)
  if [[ "$NUM_SERVERS" -gt 1 ]]; then
    RUNTIME_OVERRIDES+=(runtime.rlinf.rollout_workers_share_accelerator=true)
  fi
  # Ray's session directory is per run, so a run never attaches to another Ray on this host
  # (address=auto looks for a running cluster's marker under Ray's temp dir).
  RAY_TMPDIR=/tmp/ray_$(printf '%s' "$WANDB_NAME" | md5sum | cut -c1-8)  # NOSONAR: short run-name key, not a security use
fi
# The training servers' addresses, as the learner's ARENA_SERVER_URL lists them (one per server).
TRAIN_URLS=()
for i in $(seq 0 $((NUM_SERVERS - 1))); do TRAIN_URLS+=("tcp://127.0.0.1:$((TRAIN_PORT + i))"); done
# The learner's overrides (the dry run prints exactly this list as launch.override= lines).
LEARNER_OVERRIDES=(
  base_model=gr00t
  "base_model.ckpt=$BASE_CKPT"
  "base_model.bc_ckpt=$BASE_CKPT"
  "base_model.dataset_dir=$DATA_ROOT/$TASK/lerobot"
  ${NUM_ROLLOUTS:+"training.num_rollouts=$NUM_ROLLOUTS"}
  "training.num_envs=$N"
  ${BATCH_OVERRIDES[@]+"${BATCH_OVERRIDES[@]}"}
  "${RUNTIME_OVERRIDES[@]}"
  "env.max_episode_steps=$MAXSTEPS"
  "logging.wandb_project=$WANDB_PROJECT"
  ${WANDB_ENTITY:+"logging.wandb_entity=$WANDB_ENTITY"}
  "logging.wandb_name=$WANDB_NAME"
  "${EVAL_OVERRIDES[@]}"
  "$@"
)
if [[ "$DRYRUN" = 1 ]]; then
  echo "DRY RUN (run_container_train): the launch plan, then the library's plan; nothing booted"
  echo "launch.task=$TASK"
  echo "launch.num_envs=$N"
  echo "launch.num_servers=$NUM_SERVERS"
  echo "launch.train_server_urls=$(IFS=,; echo "${TRAIN_URLS[*]}")"
  echo "launch.eval_port=$EVAL_PORT"
  echo "launch.eval_num_envs=$EVAL_NUM_ENVS"
  echo "launch.sim_host=$SIM_HOST"
  echo "launch.runtime=$RUNTIME"
  echo "launch.sim_gpu=$SIM_GPU"
  echo "launch.learner_gpu=$LEARNER_GPU"
  echo "launch.rollout_gpu=$ROLLOUT_GPU"
  echo "launch.runtime_deps=${RLPT_RUNTIME_DEPS:-none}"
  echo "launch.ray_tmpdir=${RAY_TMPDIR:-none}"
  if [[ "$SIMS_ONLY" = 1 ]]; then
    echo "launch.learner=<none: SIMS_ONLY=1 serves the simulators alone>"
    exit 0
  fi
  # shellcheck source=../../../_common/launch.sh
  source "$POSTTRAIN/example_workflows/_common/launch.sh"
  print_overrides "${LEARNER_OVERRIDES[@]}"
  ARENA_SERVER_URL=${TRAIN_SIM_URL:-$(IFS=,; echo "${TRAIN_URLS[*]}")} \
  ARENA_EVAL_SERVER_URL=${EVAL_SIM_URL:-tcp://127.0.0.1:$EVAL_PORT} \
    library_dry_run "$PY" "$POSTTRAIN" "${LEARNER_OVERRIDES[@]}"
  exit $?
fi

boot_server () {
  local port=$1 num_envs_arg=$2 log_file=$3  # num_envs_arg: "--num-envs N" or ""
  : > "$log_file"  # a stale log from an earlier run would satisfy wait_ready's grep before the child reopens it
  CUDA_VISIBLE_DEVICES=$SIM_GPU $PY "$POSTTRAIN/simulators/g1_arena/env_server.py" --task "$TASK" --port "$port" $num_envs_arg \
    --host "$SIM_HOST" $SIM_ARGS $OBJ > "$log_file" 2>&1 &
  echo $!
  return 0
}
wait_ready () {  # log file, display name, the server's pid
  local log_file=$1 name=$2 pid=$3
  local deadline=$((SECONDS + SIM_BOOT_TIMEOUT))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    grep -aq "listening on" "$log_file" && { echo "[train] $name ready (${SECONDS}s)"; return 0; }
    grep -aqE "Traceback|Error:|FATAL|SystemExit:" "$log_file" && { echo "[train] $name FAILED:"; tail -50 "$log_file"; return 1; }
    kill -0 "$pid" 2>/dev/null || { echo "[train] $name exited before listening:"; tail -50 "$log_file"; return 1; }
    sleep 5
  done
  echo "[train] $name did not become ready in time:"; tail -50 "$log_file"; return 1
}

SIM_LOG_DIR=${SIM_LOG_DIR:-/tmp/g1train_logs}
mkdir -p "$OUT_DIR/util" "$SIM_LOG_DIR"
# Named by port so several instances in one container keep separate logs.
EVAL_LOG=$SIM_LOG_DIR/evalsim.$EVAL_PORT.log

# System-utilization telemetry for the whole run (boot, eval, collection,
# learner bursts), one row every 30 s, so phase-by-phase GPU/host load can be
# read after the fact instead of catching a phase live. Multi-GPU fields are
# ';'-joined so a two-GPU split (sims / learner) still yields one row per
# sample: per GPU index:util%:MiB:W, per process bus:pid:MiB (container pids).
UTIL_CSV=$OUT_DIR/util/${WANDB_NAME}.csv
sample_util () {
  local csv=$1
  echo "time_utc,gpus,proc_gpu_mib,load1,host_mem_used_mib" > "$csv"
  while true; do
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | tr -d ' ' | tr ',' ':' | paste -sd';')
    procs=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr ',' ':' | paste -sd';')
    load=$(cut -d' ' -f1 /proc/loadavg)
    host_mem=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%d", (t-a)/1024}' /proc/meminfo)
    echo "$(date -u +%FT%TZ),$gpu,$procs,$load,$host_mem" >> "$csv"
    sleep 30
  done
  return 0
}
sample_util "$UTIL_CSV" &
UTIL_PID=$!
echo "[train] utilization telemetry -> $UTIL_CSV (30 s samples)"

# Every exit path (a simulator that fails to boot, a learner crash, normal completion) takes the
# background processes with it: a survivor holding this script's stdout keeps the job RUNNING
# instead of failing. Whole process trees are signalled because the simulators run under the
# python.sh wrapper, which does not forward signals to Isaac Sim.
children_of () {
  local parent=$1
  awk -v parent="$parent" '/^Pid:/{pid=$2} /^PPid:/{if ($2 == parent) print pid}' /proc/[0-9]*/status 2>/dev/null
  return 0
}
descendants () {
  local root=$1 child
  for child in $(children_of "$root"); do descendants "$child"; done
  echo "$root"
  return 0
}
TRAIN_PIDS=(); EVAL_PID=""
cleanup () {
  local pids="" pid alive i
  for pid in ${TRAIN_PIDS[@]+"${TRAIN_PIDS[@]}"} $EVAL_PID $UTIL_PID; do pids="$pids $(descendants "$pid")"; done
  [[ -n "${pids// /}" ]] || return 0
  kill -TERM $pids 2>/dev/null
  for ((i = 0; i < 20; i++)); do  # a 10 s grace period, then force
    alive=0; for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ "$alive" = 1 ]] || return 0
    sleep 0.5
  done
  kill -KILL $pids 2>/dev/null
  return 0
}
trap cleanup EXIT

echo "[train] task=$TASK num_envs=$N servers=$NUM_SERVERS rollouts=${NUM_ROLLOUTS:-config-default} maxsteps=$MAXSTEPS sim_gpu=$SIM_GPU learner_gpu=$LEARNER_GPU"
if [[ "$SIMS_ONLY" != 1 ]]; then
  echo "[train] ckpt=$BASE_CKPT dataset=$DATA_ROOT/$TASK/lerobot"
  # The GR00T-N1.7 checkpoint carries its normalization stats in
  # experiment_cfg/dataset_statistics.json alongside config.json + safetensors.
  [[ -f "$BASE_CKPT/config.json" ]] && [[ -d "$BASE_CKPT/experiment_cfg" ]] || {
    echo "[train] FATAL: $BASE_CKPT is not an HF GR00T checkpoint dir (need config.json + experiment_cfg/)"; exit 1; }
  [[ -d "$DATA_ROOT/$TASK/lerobot" ]] || { echo "[train] FATAL: $DATA_ROOT/$TASK/lerobot missing"; exit 1; }

  # Import the learner stack BEFORE paying two Isaac boots: a missing dependency
  # (gr00t is installed --no-deps in the base image) should fail in seconds,
  # not after the ~7 minutes the simulators take to come up.
  echo "[train] learner import preflight"
  $PY -c "
import isaac_rl_posttraining
from isaac_rl_posttraining.policies.gr00t.wrapper import SteeredGr00tWrapper
# The policy imports the GR00T model lazily at construction; import it here so
# a missing gr00t runtime dep surfaces now, not after the checkpoint loads.
import gr00t.model.gr00t_n1d7.gr00t_n1d7
import hydra, omegaconf, loguru, wandb, transformers, torch
print('[train] learner imports OK — torch', torch.__version__)
" || { echo "[train] FATAL: learner imports failed — rebuild the learner image (Dockerfile.learner)"; exit 1; }
  if [[ "$RUNTIME" = rlinf ]]; then
    # The RLinf deps live outside the interpreter's site-packages (Dockerfile.learner_rlinf).
    [[ -n "${RLPT_RUNTIME_DEPS:-}" ]] && [[ -d "$RLPT_RUNTIME_DEPS" ]] || {
      echo "[train] FATAL: RUNTIME=rlinf needs the image from Dockerfile.learner_rlinf (RLPT_RUNTIME_DEPS unset or missing); station_train.sh selects it for RUNTIME=rlinf"; exit 1; }
    PYTHONPATH=$RLPT_RUNTIME_DEPS:$PYTHONPATH $PY -c "
import ray, rlinf.scheduler
from importlib.metadata import version
print('[train] rlinf', version('rlinf'), 'ray', ray.__version__)
" || { echo "[train] FATAL: the RLinf runtime does not import — rebuild the rlinf learner image (Dockerfile.learner_rlinf)"; exit 1; }
  fi
fi

if [[ -n "$TRAIN_SIM_URL" ]]; then
  # Simulators on another host: fail here, not deep in the learner, if they
  # are not reachable (firewall, wrong address, not booted yet).
  reachable () {  # $1=tcp://host:port
    local hp=${1#tcp://}
    timeout 5 bash -c "exec 3<>/dev/tcp/${hp%:*}/${hp##*:}" 2>/dev/null
    return $?
  }
  IFS=, read -ra REMOTE_TRAIN_URLS <<< "$TRAIN_SIM_URL"
  if [[ "${#REMOTE_TRAIN_URLS[@]}" -ne "$NUM_SERVERS" ]]; then
    echo "[train] FATAL: TRAIN_SIM_URL lists ${#REMOTE_TRAIN_URLS[@]} address(es) for NUM_SERVERS=$NUM_SERVERS"; exit 1
  fi
  for url in "${REMOTE_TRAIN_URLS[@]}"; do
    reachable "$url" || { echo "[train] FATAL: train simulator not reachable at $url"; exit 1; }
    echo "[train] train simulator: $url (remote)"
  done
  if [[ "$EVAL_SIM" = 1 ]]; then
    reachable "$EVAL_SIM_URL" || { echo "[train] FATAL: eval simulator not reachable at $EVAL_SIM_URL"; exit 1; }
    echo "[train] eval simulator: $EVAL_SIM_URL (remote)"
    export ARENA_EVAL_SERVER_URL=$EVAL_SIM_URL
  fi
  export ARENA_SERVER_URL=$TRAIN_SIM_URL
else
  # RENDER_ONCE_PRIMS overrides the server's default (the factory's dome light
  # rendered once for N>1; "none" disables). The servers boot one after another:
  # they share one asset download, which a second boot would race.
  for i in $(seq 0 $((NUM_SERVERS - 1))); do
    port=$((TRAIN_PORT + i)); log=$SIM_LOG_DIR/trainsim.$port.log
    TRAIN_PIDS+=("$(boot_server "$port" "--num-envs $ENVS_PER_SERVER${RENDER_ONCE_PRIMS:+ --render-once-prims $RENDER_ONCE_PRIMS}" "$log")")
    if [[ "$i" = 0 && "$EVAL_SIM" = "1" && "$PARALLEL_SIM_BOOT" = "1" ]]; then
      EVAL_PID=$(boot_server "$EVAL_PORT" "--num-envs $EVAL_NUM_ENVS${RENDER_ONCE_PRIMS:+ --render-once-prims $RENDER_ONCE_PRIMS}" "$EVAL_LOG")
    fi
    wait_ready "$log" "train-sim $i (N=$ENVS_PER_SERVER, port $port)" "${TRAIN_PIDS[-1]}" || exit 1
  done
  if [[ "$EVAL_SIM" = "1" ]]; then
    [[ -n "$EVAL_PID" ]] || EVAL_PID=$(boot_server "$EVAL_PORT" "--num-envs $EVAL_NUM_ENVS${RENDER_ONCE_PRIMS:+ --render-once-prims $RENDER_ONCE_PRIMS}" "$EVAL_LOG")
    wait_ready "$EVAL_LOG" "eval-sim (N=$EVAL_NUM_ENVS)" "$EVAL_PID" || exit 1
    export ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:$EVAL_PORT
  fi
  export ARENA_SERVER_URL=$(IFS=,; echo "${TRAIN_URLS[*]}")
fi
if [[ "$EVAL_SIM" != 1 ]]; then
  echo "[train] EVAL_SIM=0: no eval simulator, evaluation off"
fi

servers_alive () {  # every training server (and the eval server, when booted) still runs
  local pid
  for pid in "${TRAIN_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || return 1; done
  [[ -z "$EVAL_PID" ]] || kill -0 "$EVAL_PID" 2>/dev/null
}

if [[ "$SIMS_ONLY" = 1 ]]; then
  echo "[train] SIMS_ONLY=1: serving $NUM_SERVERS train sim(s) (N=$ENVS_PER_SERVER each) on port(s) $TRAIN_PORT..$((TRAIN_PORT + NUM_SERVERS - 1))${EVAL_PID:+ and eval sim on port $EVAL_PORT} on every interface; connect a learner with"
  echo "[train]   NUM_SERVERS=$NUM_SERVERS TRAIN_SIM_URL=$(IFS=,; echo "${TRAIN_URLS[*]//127.0.0.1/<this host>}")${EVAL_PID:+ EVAL_SIM_URL=tcp://<this host>:$EVAL_PORT}"
  # boot_server starts the servers from a subshell, so they are not children
  # of this shell and `wait` cannot follow them; poll instead.
  while servers_alive; do sleep 30; done
  echo "[train] a simulator exited; stopping"
  exit 1
fi

echo "[train] launching learner (sims on GPU $SIM_GPU, learner on GPU $LEARNER_GPU, runtime $RUNTIME${RAY_TMPDIR:+, $NUM_SERVERS rollout actor(s) on GPU $ROLLOUT_GPU}; train servers $ARENA_SERVER_URL)"
export CUDA_VISIBLE_DEVICES=$LEARNER_GPU
export ARENA_IMAGE_RESOLUTION=0 ARENA_ACTION_DIM=35
export ISAAC_RL_POSTTRAINING_OUTPUT_DIR=$OUT_DIR
if [[ "$RUNTIME" = rlinf ]]; then
  # ray's console script must be on PATH for the raylet bootstrap.
  export PYTHONPATH=$RLPT_RUNTIME_DEPS:$PYTHONPATH PATH=$RLPT_RUNTIME_DEPS/bin:$PATH RAY_TMPDIR
  mkdir -p "$RAY_TMPDIR"
fi

$PY "$POSTTRAIN/scripts/train.py" "${LEARNER_OVERRIDES[@]}"
