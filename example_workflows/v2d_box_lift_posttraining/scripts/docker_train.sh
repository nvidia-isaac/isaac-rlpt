#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMAGE=${POSTTRAIN_DOCKER_IMAGE:-posttrain-v2d:local}
CACHE=${POSTTRAIN_DOCKER_CACHE:-$HOME/data_drive/v2d-posttraining}
GPUS=0
OUTPUT=""
BUILD=0
DRYRUN=0
ARGS=()
while (($#)); do
    case "$1" in
        --gpus|--output-dir|--cache-dir|--image)
            (($# >= 2)) || { echo "$1 needs a value" >&2; exit 2; }
            case "$1" in
                --gpus) GPUS=$2 ;;
                --output-dir) OUTPUT=$2 ;;
                --cache-dir) CACHE=$2 ;;
                --image) IMAGE=$2 ;;
                *) ;;  # the outer case only routes these four options here
            esac
            shift 2 ;;
        --build) BUILD=1; shift ;;
        --dry-run) DRYRUN=1; shift ;;
        --help|-h)
            echo 'Usage: docker_train.sh --output-dir DIR [--gpus 0,1] [--cache-dir DIR] [--image IMAGE] [--build] [--dry-run] [-- LAUNCHER_ARGS...]'
            echo 'Downloads the private base checkpoint; pass HF_TOKEN in the environment.'
            echo 'A missing default image is built automatically. --build rebuilds changed source.'
            exit 0 ;;
        --) shift; ARGS=("$@"); break ;;
        *) echo "Unknown wrapper option: $1 (pass training options after --)" >&2; exit 2 ;;
    esac
done
[[ -n "$OUTPUT" ]] || { echo '--output-dir is required' >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo '--gpus must be comma-separated GPU indices' >&2; exit 2; }
CACHE=$(realpath -m "$CACHE")
OUTPUT=$(realpath -m "$OUTPUT")
[[ "$CACHE" != *','* && "$OUTPUT" != *','* ]] || { echo 'Mount paths must not contain commas' >&2; exit 2; }
[[ ! -e "$OUTPUT/run" && ! -e "$OUTPUT/runtime" ]] || { echo 'Use a fresh output directory' >&2; exit 2; }
COMMAND=(docker run --rm --init --gpus "\"device=$GPUS\"" --shm-size 8g
    --user "$(id -u):$(id -g)" --group-add 1234
    --mount "type=bind,source=$CACHE,target=/cache"
    --mount "type=bind,source=$OUTPUT,target=/output"
    --mount "type=bind,source=$CACHE/kit/data,target=/isaac-sim/kit/data"
    --mount "type=bind,source=$CACHE/kit/cache,target=/isaac-sim/kit/cache"
    --mount "type=bind,source=$CACHE/kit/logs,target=/isaac-sim/kit/logs"
    --env HOME=/cache/home --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES
    --env PRIVACY_CONSENT=Y --env NVIDIA_DRIVER_CAPABILITIES=all
    --env HF_TOKEN --env WANDB_API_KEY)
if ((DRYRUN)); then
    printf '%q ' "${COMMAND[@]}" "$IMAGE" "${ARGS[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$CACHE/home" "$CACHE/kit/data" "$CACHE/kit/cache" "$CACHE/kit/logs" "$OUTPUT"
if ((BUILD)); then
    POSTTRAIN_DOCKER_IMAGE="$IMAGE" POSTTRAIN_DOCKER_CACHE="$CACHE" bash "$REPO/example_workflows/v2d_box_lift_posttraining/scripts/docker_build.sh"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [[ "$IMAGE" == posttrain-v2d:local ]]; then
        POSTTRAIN_DOCKER_IMAGE="$IMAGE" POSTTRAIN_DOCKER_CACHE="$CACHE" bash "$REPO/example_workflows/v2d_box_lift_posttraining/scripts/docker_build.sh"
    else
        docker pull "$IMAGE"
    fi
fi
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
exec "${COMMAND[@]}" --env "POSTTRAIN_DOCKER_IMAGE_ID=$IMAGE_ID" "$IMAGE_ID" "${ARGS[@]}"
