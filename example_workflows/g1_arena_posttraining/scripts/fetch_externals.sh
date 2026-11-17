#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Link the G1 Factory checkout (docs/pages/workflows/g1_arena.md, steps 3-4) into .externals/ (git-ignored),
# which the workflow's uv environment installs from; run it once before `uv sync`:
#   G1_FACTORY=/path/to/g1-factory bash scripts/fetch_externals.sh   # default: ../g1-factory next to this repository
# It modifies that checkout, idempotently: it checks its IsaacLab-Arena and Isaac Lab submodules out at the commits in
# ../externals.lock.yaml, applies the patches in simulators/g1_arena/patches/ to Arena (the per-environment
# whole-body controller; Isaac Lab's renamed Franka asset), and registers the isaac_rlpt module in Isaac Lab
# (example_workflows/isaaclab_gr00t_posttraining/scripts/install.py; --uninstall reverts).
set -euo pipefail
WORKFLOW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$WORKFLOW/../.." && pwd)"
G1_FACTORY="$(realpath -m "${G1_FACTORY:-$REPO/../g1-factory}")"
ARENA="$G1_FACTORY/submodules/IsaacLab-Arena"
ISAACLAB="$ARENA/submodules/IsaacLab"

[[ -d "$ISAACLAB/source/isaaclab" && -d "$G1_FACTORY/submodules/Isaac-GR00T/gr00t" ]] || {
  echo "[fetch_externals] no G1 Factory checkout with submodules at $G1_FACTORY; clone https://github.com/NVlabs/G1-Factory (docs/pages/workflows/g1_arena.md) or set G1_FACTORY" >&2
  exit 2
}
pin="$(awk '/name: G1 Factory/ {f = 1} f && $1 == "commit:" {print $2; exit}' "$REPO/example_workflows/dgx_example/externals.lock.yaml")"
head="$(git -C "$G1_FACTORY" rev-parse HEAD 2>/dev/null || echo unknown)"
[[ "$head" == "$pin" ]] || echo "[fetch_externals] warning: G1 Factory is at ${head:0:10}, the results used ${pin:0:10}"
[[ -d "$G1_FACTORY/datasets/assets" ]] || echo "[fetch_externals] warning: no $G1_FACTORY/datasets/assets; no scene loads without them; download nvidia/G1-Factory there (step 3 of the G1 docs)"

# G1 Factory's Arena and Isaac Lab submodules are older revisions: switch them to the ones this workflow is built for.
pin_checkout () {  # repository, upstream URL, name in externals.lock.yaml
  local pin
  pin="$(awk -v n="$3" '$3 == n {f = 1} f && $1 == "commit:" {print $2; exit}' "$WORKFLOW/externals.lock.yaml")"
  [[ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" == "$pin" ]] && return
  git -C "$1" fetch --depth 1 "$2" "$pin"
  git -C "$1" checkout --force --quiet "$pin"
  echo "[fetch_externals] $3 at ${pin:0:10}"
}
pin_checkout "$ARENA" https://github.com/isaac-sim/IsaacLab-Arena IsaacLab-Arena
pin_checkout "$ISAACLAB" https://github.com/isaac-sim/IsaacLab IsaacLab

mkdir -p "$WORKFLOW/.externals"
ln -sfn "$G1_FACTORY" "$WORKFLOW/.externals/g1-factory"
echo "[fetch_externals] .externals/g1-factory -> $G1_FACTORY"

for PATCH in "$REPO"/simulators/g1_arena/patches/*.patch; do
  if git -C "$ARENA" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[fetch_externals] $(basename "$PATCH") already applied"
  else
    git -C "$ARENA" apply "$PATCH"
    echo "[fetch_externals] $(basename "$PATCH") applied to $ARENA"
  fi
done

# install.py's closing hint names the isaaclab command; here it is `uv run isaaclab`.
uv run --no-project python "$REPO/example_workflows/isaaclab_gr00t_posttraining/scripts/install.py" --isaaclab "$ISAACLAB" \
  | grep -v -e "^Ready:" -e "^$"
echo "[fetch_externals] done; next: uv sync"
