#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `python` for a container-only station host (run ON the station): the learner image's interpreter
# with the G1 Factory tree mounted where the launchers expect it, so a repository module runs on a
# host that has no Python environment of its own. Point a script's PYTHON knob at it:
#   PYTHON=/data/src/g1-factory/posttrain/example_workflows/g1_arena_posttraining/scripts/station/station_python.sh
# No GPU is attached; stdout is the interpreter's alone, so a caller may evaluate it.
set -euo pipefail
G1F=${G1F:-/data/src/g1-factory}
WORKDIR=/workspaces/isaaclab_arena
LEARNER_DOCKERFILE=$G1F/posttrain/example_workflows/g1_arena_posttraining/scripts/station/Dockerfile.learner
IMAGE=${IMAGE:-g1_factory_learner:$(md5sum "$LEARNER_DOCKERFILE" | cut -c1-12)}  # NOSONAR: content tag, not a security use
# The image's entrypoint runs one command string through bash; quote every argument into it.
command="cd $WORKDIR/posttrain && PYTHONPATH=$WORKDIR/posttrain /isaac-sim/python.sh"
for argument in "$@"; do
  command+=" $(printf '%q' "$argument")"
done
# The entrypoint creates the caller's user and writes into its home; the anonymous volume at the
# home path gives it a directory to write to (the station launcher mounts the same).
exec docker run --rm \
  -v "$G1F:$WORKDIR" \
  -v "/home/$(id -un)" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DOCKER_RUN_USER_ID="$(id -u)" -e DOCKER_RUN_USER_NAME="$(id -un)" \
  -e DOCKER_RUN_GROUP_ID="$(id -g)" -e DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  "$IMAGE" "$command"
