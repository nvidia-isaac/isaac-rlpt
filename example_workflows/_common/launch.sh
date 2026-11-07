# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shell helpers the workflow launchers source: print the overrides a launch passes to the entry
# point, and run the library's own dry run on them. Source it, then call the functions:
#   source "$REPO_ROOT/example_workflows/_common/launch.sh"
#
# A launcher's dry run prints its own facts (ports, GPUs, simulator commands) as launch.* lines and
# every override as a launch.override= line, then ends with the library's dry run, which composes
# the config, checks the data specs and the placement, and prints the plan.* lines.

# print_overrides OVERRIDE...: one launch.override=<override> line each.
print_overrides () {
  local override
  for override in "$@"; do echo "launch.override=$override"; done
}

# library_dry_run PYTHON REPO_ROOT OVERRIDE...: scripts/train.py's dry run of these overrides; its exit status.
library_dry_run () {
  local python=$1 repo_root=$2 scratch status
  shift 2
  scratch=$(mktemp -d)
  # Hydra writes its run directory and log there; nothing lands in the working directory.
  ISAAC_RL_POSTTRAINING_DRYRUN=1 "$python" "$repo_root/scripts/train.py" "$@" \
    "hydra.run.dir=$scratch" hydra.output_subdir=null
  status=$?
  rm -rf "$scratch"
  return "$status"
}

# workflow_python WORKFLOW_DIR: the interpreter of the workflow's own uv environment, when `uv sync` built it there.
workflow_python () {
  if [[ -x "$1/.venv/bin/python" ]]; then echo "$1/.venv/bin/python"; fi
}

_LAUNCH_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# refuse_renamed_knob OLD NEW: stop when OLD still names one of NEW's choices (a file in the configs/<new> group);
# other values of OLD, such as BACKEND=nccl, belong to someone else.
refuse_renamed_knob () {
  if [[ -n "${!1:-}" ]] && [[ -f "$_LAUNCH_REPO_ROOT/isaac_rl_posttraining/configs/${2,,}/${!1}.yaml" ]]; then
    echo "$1 was renamed to $2; set $2=${!1} instead" >&2
    exit 2
  fi
}

# Simulator ports: a run on GPU g takes consecutive ports from PORT_BASE (default 7300) + PORTS_PER_GPU*g, so runs
# on different GPUs never share one; a second run on the same GPU needs its own PORT_BASE.
PORTS_PER_GPU=16

# first_port GPU COUNT: the first of the COUNT consecutive ports a run on GPU listens on.
first_port () {
  if [[ "$2" -gt "$PORTS_PER_GPU" ]]; then
    echo "this run needs $2 simulator ports; a run gets at most $PORTS_PER_GPU (lower NUM_SERVERS)" >&2
    return 2
  fi
  echo $((${PORT_BASE:-7300} + PORTS_PER_GPU * $1))
}

# refuse_busy_ports PYTHON PORT...: stop when another process already listens on one of the ports.
refuse_busy_ports () {
  local python=$1
  shift
  "$python" - "$@" <<'PY' || exit 2
import socket, sys

busy = []
for port in map(int, sys.argv[1:]):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the simulator binds: only a listener blocks
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            busy.append(str(port))
if busy:
    sys.exit(f"port(s) {', '.join(busy)} already in use, likely by another run's simulator; "
             "stop that run, or give this one other ports with GPU or PORT_BASE")
PY
}

# refuse_viz_without_display TAG: a viewer (SIM_VIZ) needs an X display; stop before anything boots.
refuse_viz_without_display () {
  if [[ -n "${SIM_VIZ:-}" ]] && [[ -z "${DISPLAY:-}" ]]; then
    echo "[$1] viewer requested (SIM_VIZ=$SIM_VIZ) but no usable display: DISPLAY is unset; unset SIM_VIZ for headless, or run with a display" >&2
    exit 2
  fi
}

# explain_viz_failure TAG LOG: after a failed boot, name the cause when the viewer found no usable display.
explain_viz_failure () {
  # Isaac Sim's viewer logs this when it cannot open a window on $DISPLAY.
  if [[ -n "${SIM_VIZ:-}" ]] && grep -aq "\[omni.ui.python\] AttributeError" "$2" 2>/dev/null; then
    echo "[$1] viewer requested (SIM_VIZ=$SIM_VIZ) but no usable display: DISPLAY=${DISPLAY:-} is unreachable; unset SIM_VIZ for headless, or run with a display"
  fi
}
