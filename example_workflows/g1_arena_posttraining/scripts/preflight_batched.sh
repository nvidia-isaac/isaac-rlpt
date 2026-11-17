#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Preflight for a batched (NUM_ENVS>1) g1_arena run: verifies everything the
# simulator and learner will need BEFORE burning minutes on an Isaac Sim boot.
#
# Usage:
#   example_workflows/g1_arena_posttraining/scripts/preflight_batched.sh [TASK]
#     TASK   task name, with or without the LM prefix (default DrillLift)
#
# Environment (same knobs as train.sh):
#   ISAACLAB_ARENA_PATH  the IsaacLab-Arena checkout the sim runs from. In the
#                        G1 Factory container/repo layout this is the
#                        submodules/IsaacLab-Arena directory (the one holding
#                        isaaclab_arena_gr00t/), not the G1 Factory repo root.
#   SIM_PYTHON           simulator interpreter (default: the workflow .venv, else $ISAACLAB_ARENA_PATH/.venv/bin/python)
#   LEARNER_PYTHON       learner interpreter (default: the workflow .venv, else python3)
#   POSTTRAIN_BASE_CKPT / POSTTRAIN_DATA_ROOT   as for train.sh
#   GPU (default 0)  PORT_BASE (default 7300)
#
# Every check prints PASS or FAIL with the reason; the script exits nonzero
# if any check failed. It never boots a simulator.
set -uo pipefail

TASK=${1:-DrillLift}
[[ "$TASK" == LM* ]] || TASK="LM${TASK}"

GPU=${GPU:-0}
# shellcheck source=../../_common/launch.sh
source "$(cd "$(dirname "$0")/../../_common" && pwd)/launch.sh"
TRAIN_PORT=$(first_port "$GPU" 2) || exit 2; EVAL_PORT=$((TRAIN_PORT + 1))
WORKFLOW_PY=$(workflow_python "$(cd "$(dirname "$0")/.." && pwd)")
LEARNER_PY=${LEARNER_PYTHON:-${WORKFLOW_PY:-python3}}

FAILURES=0
pass () { local check=$1; echo "PASS  $check"; return 0; }
fail () { local check=$1; echo "FAIL  $check"; FAILURES=$((FAILURES+1)); return 0; }

# --- simulator side ---------------------------------------------------------
ARENA=${ISAACLAB_ARENA_PATH:-}
if [[ -z "$ARENA" ]]; then
  fail "ISAACLAB_ARENA_PATH is unset"
elif [[ ! -d "$ARENA" ]]; then
  fail "ISAACLAB_ARENA_PATH=$ARENA does not exist"
else
  pass "ISAACLAB_ARENA_PATH=$ARENA"
fi

SIM_PY=${SIM_PYTHON:-${WORKFLOW_PY:-${ARENA:+$ARENA/.venv/bin/python}}}
if [[ -z "$SIM_PY" ]] || [[ ! -x "$SIM_PY" ]]; then
  fail "simulator python not found (${SIM_PY:-<unset>}); set SIM_PYTHON (in the G1 Factory container this is Isaac Sim's python)"
  SIM_PY=""
else
  pass "simulator python: $SIM_PY"
fi

# The WBC 35->50 assembly reads these joint-space configs at server boot.
if [[ -n "$ARENA" ]]; then
  for yaml in gr00t_43dof_joint_space.yaml 43dof_joint_space.yaml; do
    if [[ -f "$ARENA/isaaclab_arena_gr00t/embodiments/g1/$yaml" ]]; then
      pass "WBC joint config: $yaml"
    else
      fail "$ARENA/isaaclab_arena_gr00t/embodiments/g1/$yaml missing (is ISAACLAB_ARENA_PATH the Arena checkout itself, not the G1 Factory repo root?)"
    fi
  done
fi

# find_spec never executes the package, so these probes stay Isaac-free.
if [[ -n "$SIM_PY" ]]; then
  if "$SIM_PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('g1_factory') else 1)" 2>/dev/null; then
    pass "g1_factory importable by the sim python (the LM* task registry)"
  else
    fail "g1_factory not importable by $SIM_PY — install the G1 Factory repo into the sim environment or add it to PYTHONPATH"
  fi

  G1PKG=$("$SIM_PY" -c "import importlib.util; spec=importlib.util.find_spec('isaaclab_arena_g1'); print(spec.submodule_search_locations[0] if spec else '')" 2>/dev/null)
  HOMIE="$G1PKG/g1_whole_body_controller/wbc_policy/policy/g1_homie_policy.py"
  if [[ -z "$G1PKG" ]]; then
    fail "isaaclab_arena_g1 not importable by $SIM_PY (Arena not installed in the sim environment)"
  elif grep -q "supports_env_ids_reset" "$HOMIE" 2>/dev/null; then
    pass "homie WBC multi-env patch applied ($HOMIE)"
  else
    fail "homie WBC multi-env patch NOT applied — run: cd \$ISAACLAB_ARENA_PATH && git apply $(cd "$(dirname "$0")/../../.." && pwd)/simulators/g1_arena/patches/arena-homie-per-env-wbc.patch"
  fi
fi

# --- learner side ------------------------------------------------------------
if "$LEARNER_PY" -c "import isaac_rl_posttraining" 2>/dev/null; then
  pass "learner python imports isaac_rl_posttraining ($LEARNER_PY)"
else
  fail "learner python $LEARNER_PY cannot import isaac_rl_posttraining"
fi
if "$LEARNER_PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  pass "learner sees CUDA"
else
  fail "learner python $LEARNER_PY has no CUDA (torch.cuda.is_available() false)"
fi

# Normalization statistics: experiment_cfg/metadata.json (GR00T N1.5), statistics.json (N1.7).
STATS=$(find "${POSTTRAIN_BASE_CKPT:-/nonexistent}" -maxdepth 2 \( -path '*/experiment_cfg/metadata.json' \
  -o -name statistics.json -o -name dataset_statistics.json \) -print -quit 2>/dev/null)
if [[ -n "$STATS" ]]; then
  pass "POSTTRAIN_BASE_CKPT has normalization statistics (${STATS#"$POSTTRAIN_BASE_CKPT"/})"
else
  fail "POSTTRAIN_BASE_CKPT=${POSTTRAIN_BASE_CKPT:-<unset>} has no normalization statistics (experiment_cfg/metadata.json or statistics.json)"
fi
if [[ -n "${POSTTRAIN_DATA_ROOT:-}" ]] && [[ -d "${POSTTRAIN_DATA_ROOT:-}/$TASK/lerobot" ]]; then
  pass "dataset present: \$POSTTRAIN_DATA_ROOT/$TASK/lerobot"
else
  fail "dataset missing: ${POSTTRAIN_DATA_ROOT:-<unset>}/$TASK/lerobot"
fi

# --- host --------------------------------------------------------------------
if command -v nvidia-smi >/dev/null && nvidia-smi -i "$GPU" >/dev/null 2>&1; then
  pass "GPU $GPU visible: $(nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader)"
else
  fail "GPU $GPU not visible to nvidia-smi"
fi
for port in "$TRAIN_PORT" "$EVAL_PORT"; do
  if fuser "${port}/tcp" >/dev/null 2>&1; then
    fail "port $port already in use (stale env server?)"
  else
    pass "port $port free"
  fi
done

echo
if [[ "$FAILURES" -gt 0 ]]; then
  echo "[preflight] $FAILURES check(s) FAILED — fix the items above before launching."
  exit 1
fi
echo "[preflight] all checks passed. Next:"
echo "  smoke:  \$SIM_PYTHON .../env_server.py --task $TASK --port $TRAIN_PORT --num-envs 2 ...  +  example_workflows/g1_arena_posttraining/scripts/smoke_batched.py --url tcp://127.0.0.1:$TRAIN_PORT --num-envs 2"
echo "  train:  NUM_ENVS=8 example_workflows/g1_arena_posttraining/scripts/train.sh ${TASK#LM}"
