#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add a distributed runtime's dependencies on top of setup_env.sh, without touching the image's
# site-packages: everything lands in $DEPS_DIR (already on PYTHONPATH through $WORK/isaac_rlpt_env.sh).
#   RUNTIME=rlinf (default): rlinf 0.3.0 + ray 2.58.0, the versions the runtime adapter is validated
#     against. rlinf's own requirement list drags in training stacks the image already provides
#     (torch, transformers, ...) or that the runtime adapter never imports (vllm, sglang, megatron,
#     ...); those are skipped, the rest is resolved against the image and installed with --no-deps.
set -euo pipefail
WORK=${WORK:-$HOME/isaac_rlpt}
DEPS_DIR=${DEPS_DIR:-$WORK/pydeps}
if [[ -z "${PYTHON:-}" ]]; then PYTHON=/isaac-sim/python.sh; command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python; fi
RLPT_ROOT=${RLPT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
RUNTIME=${RUNTIME:-rlinf}
RLINF_SPEC=${RLINF_SPEC:-rlinf==0.3.0}
RAY_SPEC=${RAY_SPEC:-ray[default]==2.58.0}

# install_resolved <name> <requirement>...: resolve against the image, then install with --no-deps into $DEPS_DIR.
install_resolved() {
  local name=$1; shift
  $PYTHON -m pip install -q --dry-run --report "$WORK/${name}_plan.json" "$@" >/dev/null
  WORK="$WORK" NAME="$name" $PYTHON - > "$WORK/${name}_plan.txt" <<'PLAN'
import json, os
report = json.load(open(os.path.join(os.environ["WORK"], os.environ["NAME"] + "_plan.json")))
for item in report["install"]:
    meta = item["metadata"]
    print(f'{meta["name"]}=={meta["version"]}')
PLAN
  echo "[setup_runtime] installing $(wc -l < "$WORK/${name}_plan.txt") packages: $(tr '\n' ' ' < "$WORK/${name}_plan.txt")"
  $PYTHON -m pip install -q --no-deps --target "$DEPS_DIR" --upgrade -r "$WORK/${name}_plan.txt"
}

# ray's console script must be on PATH for the raylet bootstrap.
add_ray_to_path() {
  export PATH="$DEPS_DIR/bin:$PATH"
  grep -qF "$DEPS_DIR/bin" "$WORK/isaac_rlpt_env.sh" 2>/dev/null || echo "export PATH=\"$DEPS_DIR/bin:\$PATH\"" >> "$WORK/isaac_rlpt_env.sh"
}

[[ "$RUNTIME" == rlinf ]] || { echo "[setup_runtime] unknown RUNTIME $RUNTIME" >&2; exit 2; }

mkdir -p "$WORK/rlinf_runtime_wheel"
$PYTHON -m pip download -q --no-deps --ignore-requires-python -d "$WORK/rlinf_runtime_wheel" "$RLINF_SPEC"
WORK="$WORK" $PYTHON - > "$WORK/rlinf_runtime_reqs.txt" <<'PLAN'
import glob, os, re, zipfile
wheel = glob.glob(os.path.join(os.environ["WORK"], "rlinf_runtime_wheel", "*.whl"))[0]
archive = zipfile.ZipFile(wheel)
meta = next(n for n in archive.namelist() if n.endswith("METADATA"))
skip = ("torch", "transformers", "tokenizers", "flash", "pytorch3d", "pipablepytorch3d", "isaac",
        "numpy", "gymnasium", "hydra", "omegaconf", "vllm", "sglang", "megatron", "deepspeed",
        "apex", "xformers", "triton", "nvidia", "cuda", "decord", "ray", "accelerate", "datasets",
        "wandb", "peft", "diffusers", "timm")
for line in archive.read(meta).decode().splitlines():
    if not line.startswith("Requires-Dist:"):
        continue
    req = line.split(":", 1)[1].strip()
    if ";" in req:  # optional extras / environment markers
        continue
    name = re.split(r"[<>=!~\[ ]", req)[0].lower()
    if any(name.startswith(s) for s in skip):
        continue
    print(req)
PLAN
echo "[setup_runtime] rlinf requirements considered: $(tr '\n' ' ' < "$WORK/rlinf_runtime_reqs.txt")"
install_resolved rlinf_runtime "$RAY_SPEC" -r "$WORK/rlinf_runtime_reqs.txt"
$PYTHON -m pip install -q --no-deps --ignore-requires-python --target "$DEPS_DIR" --upgrade "$WORK"/rlinf_runtime_wheel/*.whl
add_ray_to_path
PYTHONPATH="$DEPS_DIR:${PYTHONPATH:-}" $PYTHON -c "import rlinf, ray, numpy; print('[setup_runtime] rlinf', getattr(rlinf, '__version__', '?'), 'ray', ray.__version__, 'numpy', numpy.__version__)"
echo "[setup_runtime] done"
