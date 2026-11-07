#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export ALGO=dsrl_ppo  # RUNTIME=rlinf on one GPU needs runtime.rlinf.first_accelerator=0 runtime.rlinf.exclusive_accelerators=false (see train.sh)
exec "$HERE/train.sh" "$@"
