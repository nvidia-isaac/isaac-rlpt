#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CACHE=${POSTTRAIN_DOCKER_CACHE:-$HOME/data_drive/v2d-posttraining}
export POSTTRAIN_IMAGE=${POSTTRAIN_DOCKER_IMAGE:-posttrain-v2d:local}
export POSTTRAIN_REVISION
POSTTRAIN_REVISION=$(git -C "$REPO" rev-parse HEAD)
V2D_REVISION=$(sed -n 's/^V2D_REVISION = "\([a-f0-9]*\)"$/\1/p' "$REPO/example_workflows/v2d_box_lift_posttraining/launch.py")
[[ ${#V2D_REVISION} == 40 ]] || { echo 'Invalid V2D revision' >&2; exit 2; }
export V2D_SOURCE="$CACHE/build/video_to_data-$V2D_REVISION"
mkdir -p "$V2D_SOURCE"
cd "$REPO"
docker buildx build -f example_workflows/v2d_box_lift_posttraining/docker/Dockerfile.source \
    --build-arg "V2D_REVISION=$V2D_REVISION" \
    --output "type=local,dest=$V2D_SOURCE" .
docker buildx bake --allow "fs.read=$V2D_SOURCE" \
    -f example_workflows/v2d_box_lift_posttraining/docker/docker-bake.hcl training --load
