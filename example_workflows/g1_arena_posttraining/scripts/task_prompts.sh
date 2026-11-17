#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Canonical language prompt per G1 Factory task (from the g1_factory closed-loop configs).
# Source this file, then: PROMPT=$(task_prompt LMDrillPnP) || echo "unknown task".
# The policy is language-conditioned; when no dataset is loaded the launchers pass the
# prompt as the baked-embodiment default instruction.
task_prompt() {
  local task=$1
  case "$task" in
    LMDrillLift)           echo "Walk to the table and pick up the power drill";;
    LMDrillLiftObs)        echo "Navigate around the shelf and pick up the power drill from the table";;
    LMDrillPnP)            echo "Move the power drill from one table to another";;
    LMPickDrillFromHolder) echo "Pick up the power drill from the holder";;
    LMBoxLift)             echo "Walk to the table and pick up the box";;
    LMBoxLiftFloor)        echo "Walk to the box and pick it up from the floor";;
    LMBoxTableToShelfPnP)  echo "Move the box from the table to the shelf";;
    LMPushButton)          echo "Walk to the console and press the red button";;
    LMPushShelfForward)    echo "Push the tall wheeled shelf to the marked area";;
    *) return 1;;
  esac
}
