#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage the G1 Factory stack on a network-connected DEV BOX and deploy it to a
# station without internet access (the GB300 fetches nothing itself).
#
# Run on the dev box:
#   STATION=<user@host> example_workflows/g1_arena_posttraining/scripts/station/stage_and_deploy_station.sh
#
# It clones the G1 Factory repo + submodules + task assets, applies the homie
# multi-env WBC patch (so the station needs no git), and rsyncs the result to
# $STATION:/data/src/g1-factory. Idempotent: existing clones are reused and
# the patch is applied only once. Transfers over rsync's own ssh transport.
set -euo pipefail

STATION=${STATION:?set STATION=<user@host>}
STAGE=${STAGE:-$HOME/gb300-staging}
DEST=${DEST:-/data/src/g1-factory}
G1F=$STAGE/g1-factory
G1F_URL=https://github.com/NVlabs/G1-Factory.git
ASSETS_DATASET=nvidia/G1-Factory  # Hugging Face dataset with the task assets
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PATCH=$REPO_ROOT/simulators/g1_arena/patches/arena-homie-per-env-wbc.patch
HOMIE_REL=submodules/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/g1_homie_policy.py

[[ -f "$PATCH" ]] || { echo "FATAL: $PATCH missing" >&2; exit 1; }
mkdir -p "$STAGE"

# GR00T ships model weights via git-LFS; the LM-task sim runtime does not need
# them (our own checkpoint drives the policy), so skip the LFS smudge.
export GIT_LFS_SKIP_SMUDGE=1

echo "== 1/5 G1 Factory checkout =="
if [[ ! -d "$G1F/.git" ]]; then
  git clone --depth 1 "$G1F_URL" "$G1F"
else
  echo "reusing $G1F ($(git -C "$G1F" rev-parse --short HEAD))"
fi
git -C "$G1F" submodule update --init --recursive --depth 1

echo "== 2/5 task assets =="
# Rooms, tables, objects and the G1 robot USD; files already present are not downloaded again.
uvx hf download "$ASSETS_DATASET" --repo-type dataset --local-dir "$G1F/datasets/assets"

echo "== 3/5 homie WBC multi-env patch =="
ARENA=$G1F/submodules/IsaacLab-Arena
if grep -q supports_env_ids_reset "$G1F/$HOMIE_REL"; then
  echo "already applied"
else
  git -C "$ARENA" apply "$PATCH"
  grep -q supports_env_ids_reset "$G1F/$HOMIE_REL" || { echo "FATAL: patch did not take" >&2; exit 1; }
  echo "applied"
fi

echo "== 4/5 embed this repo under the G1 Factory tree =="
# run_docker.sh mounts the G1 Factory repo at /workspaces/isaaclab_arena, so
# placing our repo here makes env_server.py and the smoke client importable in
# the container without touching run_docker.sh's fixed mounts.
rsync -a --delete \
  --exclude .git --exclude __pycache__ --exclude '*.pyc' \
  --exclude dsrl_train_logs --exclude outputs --exclude wandb \
  "$REPO_ROOT/" "$G1F/posttrain/"

echo "== 5/5 deploy to $STATION:$DEST =="
du -sh "$G1F" 2>/dev/null | awk '{print "staged size: "$1}'
# --rsync-path creates the remote dir through rsync's transport (no bare ssh).
# Exit 23 = every file transferred but some attribute was not: rsync cannot set
# the mtime of the station's _isaac_sim mount point. Anything else is a failure.
rsync -az --delete --info=stats1 --rsync-path="mkdir -p $DEST && rsync" "$G1F/" "$STATION:$DEST/" || {
  rc=$?; [[ "$rc" -eq 23 ]] || exit "$rc"
  echo "rsync: attributes not set on some paths (exit 23); file contents are deployed"
}

echo
echo "deployed -> $STATION:$DEST"
echo "next (on the station host):"
echo "  cd $DEST/posttrain   # then docs/pages/workflows/dgx_example.md, section 3.1"
