#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Station-side DSRL training driver (run ON the GB300 host): builds the learner
# image (g1_factory:latest + our deps) if needed, then runs batched training in
# it HEADLESS. The batched env-server and the learner share the one container.
#
# Usage (on the station):
#   POSTTRAIN_BASE_CKPT=/data/ckpts/g1f_n35 \
#   POSTTRAIN_DATA_ROOT=/data/datasets/g1f \
#   [WANDB_API_KEY=...] [WANDB_ENTITY=...] \
#     bash /data/src/g1-factory/posttrain/example_workflows/g1_arena_posttraining/scripts/station/station_train.sh [TASK] [N]
#     TASK  task name (default DrillLift)   N  parallel envs (default 8)
#
# Extra Hydra overrides pass through after the two positionals.
set -euo pipefail

# ISAAC_RL_POSTTRAINING_DRYRUN=1 prints the resolved container launch as launch.* lines (image,
# GPU pin, container command) and exits without touching docker.
DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}
TASK=${1:-DrillLift}; shift || true
N=${1:-8}; [[ $# -gt 0 ]] && shift || true
EXTRA=("$@")

G1F=${G1F:-/data/src/g1-factory}
BASE_IMAGE=${BASE_IMAGE:-g1_factory:latest}
WORKDIR=/workspaces/isaaclab_arena
POSTTRAIN=$G1F/posttrain
# Rollout runtime: native, or rlinf (the rollout worker as a Ray actor, placed by
# run_container_train.sh on ROLLOUT_GPU). rlinf runs in its own image: the learner image
# plus the RLinf deps (Dockerfile.learner_rlinf).
RUNTIME=${RUNTIME:-native}
case "$RUNTIME" in
  native|rlinf) ;;
  *) echo "FATAL: RUNTIME must be native or rlinf, got $RUNTIME" >&2; exit 1 ;;
esac
# Tag the learner image by its Dockerfile's content so a dependency change
# rebuilds automatically instead of silently reusing a stale image.
LEARNER_DOCKERFILE=$POSTTRAIN/example_workflows/g1_arena_posttraining/scripts/station/Dockerfile.learner
LEARNER_IMAGE=${LEARNER_IMAGE:-g1_factory_learner:$(md5sum "$LEARNER_DOCKERFILE" | cut -c1-12)}  # NOSONAR: content tag, not a security use
RLINF_DOCKERFILE=$POSTTRAIN/example_workflows/g1_arena_posttraining/scripts/station/Dockerfile.learner_rlinf
RLINF_INSTALLER=$POSTTRAIN/example_workflows/isaaclab_gr00t_posttraining/scripts/setup_runtime.sh
if [[ "$RUNTIME" = rlinf ]]; then
  # Built on the learner image, so its tag covers both Dockerfiles and the installer they run.
  IMAGE=${IMAGE:-g1_factory_learner_rlinf:$(cat "$LEARNER_DOCKERFILE" "$RLINF_DOCKERFILE" "$RLINF_INSTALLER" | md5sum | cut -c1-12)}  # NOSONAR: content tag, not a security use
else
  IMAGE=${IMAGE:-$LEARNER_IMAGE}
fi
if [[ "${SIMS_ONLY:-0}" = 1 ]]; then
  BASE_CKPT=${POSTTRAIN_BASE_CKPT:-/nonexistent}; DATA_ROOT=${POSTTRAIN_DATA_ROOT:-/nonexistent}   # no learner on this host
else
  BASE_CKPT=${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT=/data/ckpts/... (the G1-N1.7 checkpoint dir)}
  DATA_ROOT=${POSTTRAIN_DATA_ROOT:?set POSTTRAIN_DATA_ROOT=/data/datasets/... (holds <LMTask>/lerobot)}
fi
OUT_DIR=${OUT_DIR:-/data/dsrl_out_g1arena}

# Read the W&B key from the station secret store when not already set, so the
# launch command never has to carry $(cat ...) — the station login shell is
# tcsh and can't parse bash command substitution / inline env prefixes.
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f /data/secrets/wandb_key ]]; then
  WANDB_API_KEY=$(cat /data/secrets/wandb_key); export WANDB_API_KEY
fi
# The GR00T-N1.7 backbone pulls its base VLM config (nvidia/Cosmos-Reason2-2B,
# a gated HF repo) by model id at construction, so an authenticated token is
# required even though the fine-tuned checkpoint carries the weights.
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /data/secrets/hf_token ]]; then
  HF_TOKEN=$(cat /data/secrets/hf_token); export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || echo "[station_train] WARN: no HF_TOKEN (set it or place it in /data/secrets/hf_token) — the gated base backbone fetch will 401" >&2
# Persistent HF cache so the base backbone downloads once, not per --rm run.
HF_CACHE=${HF_CACHE:-/data/hf_cache}
[[ "$DRYRUN" = 1 ]] || mkdir -p "$HF_CACHE"

[[ -f "$POSTTRAIN/example_workflows/g1_arena_posttraining/scripts/station/run_container_train.sh" ]] || { echo "FATAL: $POSTTRAIN not deployed" >&2; exit 1; }
if [[ "${SIMS_ONLY:-0}" != 1 ]]; then
  [[ -d "$BASE_CKPT" ]] || { echo "FATAL: POSTTRAIN_BASE_CKPT=$BASE_CKPT does not exist on the host" >&2; exit 1; }
  [[ -d "$DATA_ROOT" ]] || { echo "FATAL: POSTTRAIN_DATA_ROOT=$DATA_ROOT does not exist on the host" >&2; exit 1; }
fi

# Build the images once (the base must exist; docs/pages/workflows/dgx_example.md builds it). The rlinf image
# builds on the learner image and copies the Isaac Lab integration's RLinf installer, hence
# that installer's directory as its build context.
build_image () {
  local image=$1 dockerfile=$2 base_image=$3 context=$4
  echo "[station_train] building $image (on $base_image; one-time)…"
  docker build --build-arg BASE_IMAGE="$base_image" -t "$image" -f "$dockerfile" "$context"
}
require_base () {
  [[ -n "$(docker images -q "$BASE_IMAGE" 2>/dev/null)" ]] || { echo "FATAL: base $BASE_IMAGE missing — build it first, as in docs/pages/workflows/dgx_example.md" >&2; exit 1; }
}
if [[ "$DRYRUN" != 1 ]]; then
  if [[ -n "$(docker images -q "$IMAGE" 2>/dev/null)" ]]; then
    echo "[station_train] reusing learner image $IMAGE"
  elif [[ "$RUNTIME" = native ]]; then
    require_base
    build_image "$IMAGE" "$LEARNER_DOCKERFILE" "$BASE_IMAGE" "$POSTTRAIN/example_workflows/g1_arena_posttraining/scripts/station"
  else
    if [[ -z "$(docker images -q "$LEARNER_IMAGE" 2>/dev/null)" ]]; then
      require_base
      build_image "$LEARNER_IMAGE" "$LEARNER_DOCKERFILE" "$BASE_IMAGE" "$POSTTRAIN/example_workflows/g1_arena_posttraining/scripts/station"
    fi
    build_image "$IMAGE" "$RLINF_DOCKERFILE" "$LEARNER_IMAGE" "$(dirname "$RLINF_INSTALLER")"
  fi
fi

[[ "$DRYRUN" = 1 ]] || mkdir -p "$OUT_DIR"
opt_mount() {  # always succeeds: a missing dir is not an error under set -e
  local host_dir=$1 container_dir=$2
  if [[ -d "$host_dir" ]]; then printf -- '-v %s:%s ' "$host_dir" "$container_dir"; fi
}
DATA_MOUNTS="$(opt_mount "$G1F/datasets" /datasets)"

[[ "$TASK" == LM* ]] || TASK="LM${TASK}"
if [[ "${SMOKE_DECODE:-0}" = "1" ]]; then
  # Real-weights batched-decode smoke: loads the checkpoint and checks the
  # batched GR00T decode against the per-sample path (no simulator boot).
  # DECODE_SWEEP=8,64,128,256 additionally times one batched decode at each N.
  CONTAINER_CMD="/isaac-sim/python.sh $WORKDIR/posttrain/example_workflows/g1_arena_posttraining/scripts/smoke_batched_decode.py \
--ckpt $BASE_CKPT --dataset $DATA_ROOT/$TASK/lerobot --num-envs $N ${DECODE_SWEEP:+--sweep $DECODE_SWEEP}"
  echo "[station_train] SMOKE_DECODE: batched GR00T decode check at N=$N (no sim)${DECODE_SWEEP:+, sweep $DECODE_SWEEP}"
else
  CONTAINER_CMD="TASK=$TASK N=$N \
POSTTRAIN_BASE_CKPT=$BASE_CKPT POSTTRAIN_DATA_ROOT=$DATA_ROOT \
ISAAC_RL_POSTTRAINING_OUTPUT_DIR=$OUT_DIR \
${NUM_ROLLOUTS:+NUM_ROLLOUTS=$NUM_ROLLOUTS} \
${WANDB_PROJECT:+WANDB_PROJECT=$WANDB_PROJECT} ${WANDB_ENTITY:+WANDB_ENTITY=$WANDB_ENTITY} \
${WANDB_NAME:+WANDB_NAME=$WANDB_NAME} ${RENDER_ONCE_PRIMS:+RENDER_ONCE_PRIMS=$RENDER_ONCE_PRIMS} \
${EVAL_SIM:+EVAL_SIM=$EVAL_SIM} ${EVAL_NUM_ENVS:+EVAL_NUM_ENVS=$EVAL_NUM_ENVS} ${SIMS_ONLY:+SIMS_ONLY=$SIMS_ONLY} ${IMAGE_SHORTEST_EDGE:+IMAGE_SHORTEST_EDGE=$IMAGE_SHORTEST_EDGE} \
${SIM_GPU:+SIM_GPU=$SIM_GPU} ${LEARNER_GPU:+LEARNER_GPU=$LEARNER_GPU} \
RUNTIME=$RUNTIME ${ROLLOUT_GPU:+ROLLOUT_GPU=$ROLLOUT_GPU} ${NUM_SERVERS:+NUM_SERVERS=$NUM_SERVERS} \
${TRAIN_PORT:+TRAIN_PORT=$TRAIN_PORT} ${EVAL_PORT:+EVAL_PORT=$EVAL_PORT} \
${TRAIN_SIM_URL:+TRAIN_SIM_URL=$TRAIN_SIM_URL} ${EVAL_SIM_URL:+EVAL_SIM_URL=$EVAL_SIM_URL} \
bash $WORKDIR/posttrain/example_workflows/g1_arena_posttraining/scripts/station/run_container_train.sh ${EXTRA[*]:-}"
fi

echo "[station_train] task=$TASK num_envs=$N image=$IMAGE (headless)"
echo "[station_train] ckpt=$BASE_CKPT data=$DATA_ROOT out=$OUT_DIR"
# Docker CE with the container toolkit's CDI hook passes GPUs with --gpus alone;
# --runtime=nvidia only exists where the toolkit registered that runtime.
# GPUS pins the container to specific cards (nvidia-smi indices, comma-separated; default all).
# Pin at the docker level rather than with CUDA_VISIBLE_DEVICES inside: Isaac Sim's renderer
# pairs Vulkan devices with CUDA devices by enumeration and fails when they disagree.
GPU_SPEC=all
if [[ -n "${GPUS:-}" ]] && [[ "$GPUS" != all ]]; then GPU_SPEC="device=$GPUS"; fi
# --privileged would mount every host device node and expose all cards, defeating the pin.
PRIVILEGED=--privileged
if [[ "$GPU_SPEC" != all ]]; then PRIVILEGED=""; fi

if [[ "$DRYRUN" = 1 ]]; then
  echo "DRY RUN (station_train): no container started; the container's own dry run (run_container_train.sh) prints the library's plan"
  echo "launch.task=$TASK"
  echo "launch.num_envs=$N"
  echo "launch.num_servers=${NUM_SERVERS:-1}"
  echo "launch.eval_num_envs=${EVAL_NUM_ENVS:-1}"
  echo "launch.runtime=$RUNTIME"
  echo "launch.image=$IMAGE"
  echo "launch.docker.gpus=$GPU_SPEC"
  echo "launch.docker.privileged=${PRIVILEGED:-none}"
  echo "launch.container_cmd=$CONTAINER_CMD"
  exit 0
fi

NVIDIA_RUNTIME=""
if docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
  NVIDIA_RUNTIME=--runtime=nvidia
fi

exec docker run --rm \
  $PRIVILEGED --ulimit memlock=-1 --ulimit stack=-1 \
  --ipc=host --net=host $NVIDIA_RUNTIME --gpus="$GPU_SPEC" \
  -v "$G1F:$WORKDIR" \
  -v "$G1F/g1_factory:/opt/g1_factory/g1_factory" \
  $DATA_MOUNTS \
  -v /data:/data \
  -v /tmp:/tmp \
  -v "/home/$(id -un)" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DOCKER_RUN_USER_ID="$(id -u)" -e DOCKER_RUN_USER_NAME="$(id -un)" \
  -e DOCKER_RUN_GROUP_ID="$(id -g)" -e DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  -e ISAACLAB_PATH="$WORKDIR/submodules/IsaacLab-Arena/submodules/IsaacLab" \
  -e ARENA_PATH="$WORKDIR/submodules/IsaacLab-Arena" \
  -e GROOT_PATH="$WORKDIR/submodules/Isaac-GR00T" \
  -e G1_FACTORY_PATH=/opt/g1_factory \
  ${WANDB_API_KEY:+-e WANDB_API_KEY="$WANDB_API_KEY"} \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  -e HF_HOME="$HF_CACHE" \
  "$IMAGE" \
  "$CONTAINER_CMD"
