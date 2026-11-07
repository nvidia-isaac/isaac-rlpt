#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch the sources the workflow's uv environment installs from, at the commits pinned in
# externals.lock.yaml, into .externals/ (git-ignored); run it once before `uv sync`, and again
# after the pins change:
#   Isaac Lab (editable: `uv run isaaclab train` runs the training script of this checkout),
#     with the isaac_rlpt module registered in it (scripts/install.py);
#   Isaac-GR00T at the GR00T N1.5 commit, switched to PyTorch attention (no flash-attn wheel exists
#     for this torch) with the patch Isaac Lab ships;
#   pytorch3d's rotation conversions, the one pytorch3d module GR00T imports, as a small package.
set -euo pipefail
WORKFLOW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WORKFLOW/externals.lock.yaml"
EXT="$WORKFLOW/.externals"
PYTORCH3D_TAG=v0.7.9

# pin <name>: the commit (or tag) and URL of one externals.lock.yaml entry.
pin() { awk -v n="$1" '$1 == "-" && $2 == "name:" {cur = $3} cur == n && ($1 == "commit:" || $1 == "url:") {print $2}' "$LOCK"; }

# checkout <name> <dest>: shallow, detached checkout of the pinned commit; local edits are reset.
checkout() {
  local url commit
  read -r url commit <<< "$(pin "$1" | tr '\n' ' ')"
  [[ -n "$url" && -n "$commit" ]] || { echo "[fetch_externals] no url or commit for $1 in $LOCK" >&2; exit 2; }
  [[ -d "$2/.git" ]] || { git init -q "$2"; git -C "$2" remote add origin "$url"; }
  git -C "$2" fetch -q --depth 1 origin "$commit"
  git -C "$2" checkout -q -f --detach FETCH_HEAD
  git -C "$2" clean -qfd
  echo "[fetch_externals] $1 at $commit"
}

mkdir -p "$EXT"
checkout IsaacLab "$EXT/IsaacLab"
checkout Isaac-GR00T "$EXT/Isaac-GR00T"
git -C "$EXT/Isaac-GR00T" apply "$EXT/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch"
# install.py's closing hint names ./isaaclab.sh; here the command is `uv run isaaclab`.
uv run --no-project python "$WORKFLOW/scripts/install.py" --isaaclab "$EXT/IsaacLab" | grep -v -e "^Ready:" -e "^$"

P3D="$EXT/pytorch3d"
mkdir -p "$P3D/pytorch3d/transforms"
curl --proto '=https' --tlsv1.2 -fsSL \
  "https://raw.githubusercontent.com/facebookresearch/pytorch3d/$PYTORCH3D_TAG/pytorch3d/transforms/rotation_conversions.py" \
  | sed 's/^from \.\.common\.datatypes import Device$/from typing import Union\nDevice = Union[str, torch.device]/' \
  > "$P3D/pytorch3d/transforms/__init__.py"
printf '"""pytorch3d.transforms rotation conversions only (BSD-3, Meta Platforms), %s."""\n' "$PYTORCH3D_TAG" > "$P3D/pytorch3d/__init__.py"
# The BSD-3-Clause license travels with the code it covers.
curl --proto '=https' --tlsv1.2 -fsSL "https://raw.githubusercontent.com/facebookresearch/pytorch3d/$PYTORCH3D_TAG/LICENSE" > "$P3D/pytorch3d/LICENSE"
printf '[project]\nname = "pytorch3d"\nversion = "%s"\n[tool.setuptools]\npackages = ["pytorch3d", "pytorch3d.transforms"]\n' \
  "${PYTORCH3D_TAG#v}" > "$P3D/pyproject.toml"
echo "[fetch_externals] pytorch3d.transforms from $PYTORCH3D_TAG"
echo "[fetch_externals] done; next: uv sync"
