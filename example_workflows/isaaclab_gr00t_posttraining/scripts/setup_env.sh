#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare an Isaac Lab 3.0 environment for isaac_rlpt without touching the Isaac Sim site-packages
# (in the Isaac Lab 3.0.0-beta2 container image they are read-only for the container user).
#
# What it does, mirroring Isaac Lab's "RL Post-Training for VLA Models" install steps:
#   1. GR00T N1.5 at the commit Isaac Lab pins, plus its runtime dependencies, installed with
#      `pip --target` into $DEPS_DIR (transformers 4.51.3 shadows the image's newer one via PYTHONPATH);
#   2. this library on PYTHONPATH (source checkout, no install);
#   3. the isaac_rlpt module registered in a writable copy of Isaac Lab's RL scripts when the
#      checkout itself is read-only;
#   4. `uv` and Isaac Sim's Python environment in the shell, so the README's `uv run isaaclab train ...` runs
#      unchanged on the image's own Isaac Sim (UV_NO_PROJECT and UV_PYTHON stop uv from building a second one).
# It writes $WORK/isaac_rlpt_env.sh; source it before running the train/play commands.
#
#   ISAACLAB_ROOT (default /workspace/isaaclab)   PYTHON (default /isaac-sim/python.sh, falls back to python)
#   WORK (default $HOME/isaac_rlpt)                RLPT_ROOT (default: this repository)
set -euo pipefail
ISAACLAB_ROOT=${ISAACLAB_ROOT:-/workspace/isaaclab}
WORK=${WORK:-$HOME/isaac_rlpt}
RLPT_ROOT=${RLPT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
GR00T_ROOT=${GR00T_ROOT:-$WORK/Isaac-GR00T}
DEPS_DIR=${DEPS_DIR:-$WORK/pydeps}
GR00T_COMMIT=4af2b622892f7dcb5aae5a3fb70bcb02dc217b96   # Isaac-GR00T commit Isaac Lab 3.0 pins for GR00T N1.5
if [[ -z "${PYTHON:-}" ]]; then PYTHON=/isaac-sim/python.sh; command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python; fi

mkdir -p "$WORK" "$DEPS_DIR"
echo "[setup_env] isaaclab=$ISAACLAB_ROOT python=$PYTHON ($($PYTHON --version 2>&1)) rlpt=$RLPT_ROOT work=$WORK"
[[ -f "$ISAACLAB_ROOT/scripts/reinforcement_learning/train.py" ]] || { echo "ISAACLAB_ROOT is not an Isaac Lab 3.0 checkout"; exit 2; }

# 1. GR00T N1.5 (pinned source) and the dependency set Isaac Lab installs next to it.
if [[ ! -d "$GR00T_ROOT/.git" ]]; then git clone -q https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_ROOT"; fi
git -C "$GR00T_ROOT" checkout -q "$GR00T_COMMIT"
# Resolve against the image's packages (a dry run writes nothing), then install exactly the missing
# closure into $DEPS_DIR with --no-deps. The image's torch/numpy stay; transformers 4.51.3 (GR00T
# N1.5's pin) lands in $DEPS_DIR and shadows the image's newer version through PYTHONPATH.
REQS=("transformers==4.51.3" "tokenizers>=0.21,<0.22" "av>=12.3.0" "pydantic>=2.7,<2.12" numpydantic albumentations
      decord2 dm_tree diffusers timm peft pandas einops loguru rich opencv-python-headless tqdm Pillow)
$PYTHON -m pip install -q --dry-run --report "$WORK/pip_plan.json" "${REQS[@]}" >/dev/null
WORK="$WORK" $PYTHON - > "$WORK/pip_plan.txt" <<'PLAN'
import json, os
report = json.load(open(os.path.join(os.environ["WORK"], "pip_plan.json")))
for item in report["install"]:
    meta = item["metadata"]
    print(f'{meta["name"]}=={meta["version"]}')
PLAN
echo "[setup_env] installing $(wc -l < "$WORK/pip_plan.txt") packages into $DEPS_DIR: $(tr '\n' ' ' < "$WORK/pip_plan.txt")"
$PYTHON -m pip install -q --no-deps --target "$DEPS_DIR" --upgrade -r "$WORK/pip_plan.txt"
# GR00T imports pytorch3d only for its pure-torch rotation conversions; no wheel exists for Python 3.12,
# so vendor that single BSD-licensed module as the `pytorch3d.transforms` package.
if ! PYTHONPATH="$DEPS_DIR" $PYTHON -c "import pytorch3d.transforms" >/dev/null 2>&1; then
  PYTORCH3D_TAG=v0.7.9  # the tag fetch_externals.sh pins
  mkdir -p "$DEPS_DIR/pytorch3d/transforms"
  curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/facebookresearch/pytorch3d/$PYTORCH3D_TAG/pytorch3d/transforms/rotation_conversions.py \
    | sed 's/^from \.\.common\.datatypes import Device$/from typing import Union\nDevice = Union[str, torch.device]/' \
    > "$DEPS_DIR/pytorch3d/transforms/__init__.py"
  printf '"""Vendored pytorch3d.transforms rotation conversions (BSD-3, Meta Platforms); see setup_env.sh."""\n' > "$DEPS_DIR/pytorch3d/__init__.py"
  curl --proto '=https' --tlsv1.2 -fsSL "https://raw.githubusercontent.com/facebookresearch/pytorch3d/$PYTORCH3D_TAG/LICENSE" > "$DEPS_DIR/pytorch3d/LICENSE"
  echo "[setup_env] vendored pytorch3d.transforms (rotation conversions only)"
fi
# flash-attn is optional: without it, switch GR00T to PyTorch SDPA (patch shipped by Isaac Lab).
PATCH="$ISAACLAB_ROOT/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch"
if ! $PYTHON -c "import flash_attn" >/dev/null 2>&1 && git -C "$GR00T_ROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$GR00T_ROOT" apply "$PATCH"; echo "[setup_env] applied no_flash_attn.patch"
fi


# 2. Register the isaac_rlpt module. Isaac Lab's scripts dir may be read-only: use a writable copy.
RL_SCRIPTS="$ISAACLAB_ROOT/scripts/reinforcement_learning"
if [[ ! -w "$RL_SCRIPTS" ]]; then
  cp -r "$RL_SCRIPTS" "$WORK/reinforcement_learning"; RL_SCRIPTS="$WORK/reinforcement_learning"
  echo "[setup_env] Isaac Lab scripts are read-only; using a copy at $RL_SCRIPTS"
fi
$PYTHON "$RLPT_ROOT/example_workflows/isaaclab_gr00t_posttraining/scripts/install.py" --rl_dir "$RL_SCRIPTS"

# 3. uv, and an `hf` command (the image ships huggingface_hub, but no `hf` on PATH).
mkdir -p "$WORK/bin"
# The uv version CI uses (.github/workflows/ci.yml); newer ones warn on the workflow's prerelease mode.
UV_VERSION=0.11.19
if [[ "$("$WORK/bin/uv" --version 2>/dev/null)" != "uv $UV_VERSION"* ]]; then
  curl --proto '=https' --tlsv1.2 -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | UV_UNMANAGED_INSTALL="$WORK/bin" sh
fi
if ! command -v hf >/dev/null 2>&1; then
  cat > "$WORK/bin/hf" <<EOF
#!/bin/bash
exec "$PYTHON" -c "
import sys
try:
    from huggingface_hub.cli.hf import main
except ImportError:
    from huggingface_hub.commands.huggingface_cli import main
sys.argv[0] = 'hf'
sys.exit(main())
" "\$@"
EOF
  chmod +x "$WORK/bin/hf"
fi

# A `python` on PATH for scripts run under `uv run` (no project, so uv adds none): Isaac Sim's interpreter.
PY_EXE=$($PYTHON -c 'import sys; print(sys.executable)')
ln -sfn "$PY_EXE" "$WORK/bin/python"

cat > "$WORK/isaac_rlpt_env.sh" <<EOF
export PATH="$WORK/bin:\$PATH"
export PYTHONPATH="$RLPT_ROOT:$GR00T_ROOT:$DEPS_DIR:\${PYTHONPATH:-}"
export ISAAC_RLPT_PYTHON="$PYTHON"
export UV_NO_PROJECT=1 UV_PYTHON_DOWNLOADS=never UV_PYTHON="$PY_EXE"
export ISAAC_RLPT_RL_SCRIPTS="$RL_SCRIPTS"
export OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y PRIVACY_CONSENT=Y
EOF
# Isaac Sim's python.sh sets its library paths before starting Python; `uv run` starts Python directly.
ISAAC_SIM=$(dirname "$(readlink -f "$(command -v "$PYTHON")")")
if [[ -f "$ISAAC_SIM/setup_python_env.sh" ]]; then
  cat >> "$WORK/isaac_rlpt_env.sh" <<EOF
export CARB_APP_PATH="$ISAAC_SIM/kit" ISAAC_PATH="$ISAAC_SIM" EXP_PATH="$ISAAC_SIM/apps"
source "$ISAAC_SIM/setup_python_env.sh"
EOF
fi
# shellcheck disable=SC1090
set +u; source "$WORK/isaac_rlpt_env.sh"; set -u   # Isaac Sim's setup_python_env.sh reads unset variables
$PYTHON -c "import gr00t, isaac_rl_posttraining, transformers; print('[setup_env] gr00t + isaac_rl_posttraining importable; transformers', transformers.__version__)"
echo "[setup_env] done. Next: source $WORK/isaac_rlpt_env.sh && uv run isaaclab train --rl_library isaac_rlpt --list_tasks"
