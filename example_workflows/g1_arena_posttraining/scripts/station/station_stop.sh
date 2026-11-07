#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stop this stack's containers on the station (run ON the station host): every container running
# the learner image, whatever its content-hash tag. Other people's containers are left alone.
# docker's ancestor filter needs the exact tag, so the image is matched by name.
set -euo pipefail
ids=$(docker ps --format '{{.ID}} {{.Image}}' 2> /dev/null | awk '$2 ~ /^g1_factory_learner/ {print $1}' || true)
if [[ -z "$ids" ]]; then
  echo "[station_stop] no learner containers running"
  exit 0
fi
docker kill $ids > /dev/null
echo "[station_stop] stopped: $(echo $ids | tr '\n' ' ')"
