#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export ALGO=dsrl_ppo
exec "$HERE/train.sh" "$@"
