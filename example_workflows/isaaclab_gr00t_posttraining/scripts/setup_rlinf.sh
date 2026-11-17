#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add RLinf (the version Isaac Lab 3.0 pins) on top of setup_env.sh so Isaac Lab's own
# `--rl_library rlinf` baseline runs in the same container against the same checkpoint.
# rlinf installs with --no-deps into $DEPS_DIR; its declared requirements are resolved against the
# image (dry run) minus the stacks the image or GR00T already pin (torch, transformers, ...) and the
# packages without Python 3.12 wheels (pytorch3d; the pure-torch rotation module is vendored).
set -euo pipefail
WORK=${WORK:-$HOME/isaac_rlpt}
DEPS_DIR=${DEPS_DIR:-$WORK/pydeps}
if [[ -z "${PYTHON:-}" ]]; then PYTHON=/isaac-sim/python.sh; command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python; fi
RLINF_SPEC=${RLINF_SPEC:-rlinf==0.2.0dev2}   # pinned by Isaac Lab's RLinf docs

mkdir -p "$WORK/rlinf_wheel"
# rlinf 0.2.0dev2 declares Requires-Python <=3.11 but ships a pure-Python wheel; Isaac Lab installs it into its 3.12 env too.
$PYTHON -m pip download -q --no-deps --ignore-requires-python -d "$WORK/rlinf_wheel" "$RLINF_SPEC"
WORK="$WORK" $PYTHON - > "$WORK/rlinf_reqs.txt" <<'PLAN'
import glob, os, re, zipfile
wheel = glob.glob(os.path.join(os.environ["WORK"], "rlinf_wheel", "*.whl"))[0]
archive = zipfile.ZipFile(wheel)
meta = next(n for n in archive.namelist() if n.endswith("METADATA"))
skip = ("torch", "transformers", "tokenizers", "flash", "pytorch3d", "pipablepytorch3d", "isaac",
        "numpy", "gymnasium", "hydra", "omegaconf", "vllm", "sglang", "megatron", "deepspeed",
        "apex", "xformers", "triton", "nvidia", "cuda", "decord")
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
echo "[setup_rlinf] requirements considered: $(tr '\n' ' ' < "$WORK/rlinf_reqs.txt")"
$PYTHON -m pip install -q --dry-run --report "$WORK/rlinf_plan.json" -r "$WORK/rlinf_reqs.txt" >/dev/null
WORK="$WORK" $PYTHON - > "$WORK/rlinf_plan.txt" <<'PLAN'
import json, os
report = json.load(open(os.path.join(os.environ["WORK"], "rlinf_plan.json")))
for item in report["install"]:
    meta = item["metadata"]
    print(f'{meta["name"]}=={meta["version"]}')
PLAN
echo "[setup_rlinf] installing $(wc -l < "$WORK/rlinf_plan.txt") packages: $(tr '\n' ' ' < "$WORK/rlinf_plan.txt")"
$PYTHON -m pip install -q --no-deps --target "$DEPS_DIR" --upgrade -r "$WORK/rlinf_plan.txt"
$PYTHON -m pip install -q --no-deps --ignore-requires-python --target "$DEPS_DIR" --upgrade "$WORK"/rlinf_wheel/*.whl
PYTHONPATH="$DEPS_DIR:${PYTHONPATH:-}" $PYTHON -c "import rlinf; print('[setup_rlinf] rlinf importable:', getattr(rlinf, '__version__', '?'))"
echo "[setup_rlinf] done"
