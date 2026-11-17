# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared by train_eval.sh and layout_sweep.sh (sourced after ROOT is set): where the station
# launchers live, and the learner's schedule knob as the library's schedule.kind.

# The station launchers (station_train.sh, station_stop.sh) of a simulator entry. G1 Factory is the
# only entry that ships them; its workflow keeps them in scripts/station/.
station_dir () {
  local entry=$1
  if [[ "$entry" != g1_arena ]]; then
    echo "FATAL: simulator entry $entry ships no station launcher; these recipes drive G1 Factory's" >&2
    return 1
  fi
  echo "$ROOT/example_workflows/g1_arena_posttraining/scripts/station"
}

# SCHEDULE as schedule.kind: sync collects a round, trains, then publishes; overlapped collects the
# next round during the training burst.
schedule_kind () {
  case "$1" in
    sync|overlapped) echo "$1" ;;
    serial) echo "FATAL: SCHEDULE=serial was renamed; use SCHEDULE=sync" >&2; return 1 ;;
    lookahead) echo "FATAL: SCHEDULE=lookahead was renamed; use SCHEDULE=overlapped" >&2; return 1 ;;
    *) echo "FATAL: SCHEDULE must be sync or overlapped, got $1" >&2; return 1 ;;
  esac
}

# The schedule's part of a run name: none for sync, _overlapped for the overlapped schedule.
schedule_tag () {
  local kind
  kind=$(schedule_kind "$1") || return 1
  [[ "$kind" = sync ]] || echo _overlapped
}
