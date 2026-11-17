# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-to-model mappings of the IsaacLab tasks that ship with this provider.

Each mapping is plain data (camera terms -> video keys, state terms and slices,
action padding and bounds, task description); :mod:`simulators.isaaclab.env`
registers them in its ``TASK_MAPPINGS`` and documents the keys.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FRANKA_STACK_GR00T_MAPPING", "FRANKA_STACK_TASKS", "TROCAR_GR00T_MAPPING"]


TROCAR_GR00T_MAPPING: dict[str, Any] = {
    # GR00T video key -> IsaacLab camera observation term (group camera_images).
    "cameras": {
        "video.room_view": "front_camera",
        "video.left_wrist_view": "left_wrist_camera",
        "video.right_wrist_view": "right_wrist_camera",
    },
    # GR00T state key -> {term (group policy), [start, stop) slice}. The
    # mapping order is the concatenation order of the flat ``state`` vector.
    "states": {
        "state.left_arm": {"term": "robot_joint_state", "slice": [15, 22]},
        "state.right_arm": {"term": "robot_joint_state", "slice": [22, 29]},
        "state.left_hand": {"term": "robot_dex3_joint_state", "slice": [0, 7]},
        "state.right_hand": {"term": "robot_dex3_joint_state", "slice": [7, 14]},
    },
    # Action term whose joint order defines the simulator action; the model's
    # 28 controlled joints sit between the zero-padded prefix and suffix.
    "action_term": "joint_pos",
    "prefix_pad": 15,
    "suffix_pad": 0,
    "task_description": "assemble trocar from tray",
    # Optional ``module:function`` (or callable) returning the per-env task stage; logged at
    # episode end and exposed as ``info["task_stage"]`` so progress is visible without success.
    "stage_fn": "isaaclab_tasks.manager_based.manipulation.assemble_trocar.mdp.terminations:get_task_stage",
}
"""Simulator-to-model mapping of the trocar task (IsaacLab's
``isaaclab_ppo_gr00t_assemble_trocar.yaml`` ``isaaclab:`` block)."""

FRANKA_STACK_TASKS: tuple[str, ...] = ("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",)
"""Franka cube stacking with cameras (the task RLinf's stack-cube SFT data comes from)."""

FRANKA_STACK_GR00T_MAPPING: dict[str, Any] = {
    # The stack task keeps its camera terms in the ``policy`` group.
    "cameras": {"video.image": "table_cam", "video.wrist_image": "wrist_cam"},
    # LIBERO-style state: end-effector position, orientation as axis-angle and the
    # two finger joints (RLinf's ``IsaaclabStackCubeEnv._wrap_obs`` order). A
    # ``transform`` is applied to the whole term before the slice is taken.
    "states": {
        "state.x": {"term": "eef_pos", "slice": [0, 1]},
        "state.y": {"term": "eef_pos", "slice": [1, 2]},
        "state.z": {"term": "eef_pos", "slice": [2, 3]},
        "state.roll": {
            "term": "eef_quat",
            "transform": "quat_wxyz_to_axis_angle",
            "slice": [0, 1],
        },
        "state.pitch": {
            "term": "eef_quat",
            "transform": "quat_wxyz_to_axis_angle",
            "slice": [1, 2],
        },
        "state.yaw": {
            "term": "eef_quat",
            "transform": "quat_wxyz_to_axis_angle",
            "slice": [2, 3],
        },
        "state.gripper": {"term": "gripper_pos", "slice": [0, 2]},
    },
    # Relative end-effector pose delta (6) + gripper command (1) go straight to
    # the IK-Rel action (no padding); the gripper is binarized like RLinf does.
    "action_term": None,
    "prefix_pad": 0,
    "suffix_pad": 0,
    "sign_dims": [6],
    # Bounds in the SFT data's action range (metadata min/max: +-0.1 for the pose
    # deltas, +-1 for the gripper); the residual/edit algorithms scale to them.
    "action_bounds": {
        "low": [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -1.0],
        "high": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0],
    },
    "task_description": "Stack the red block on the blue block, then stack the green block on the red block.",
    # The SFT data was recorded at 256x256 (IsaacLab's cameras default to 200x200).
    # The Franka USD moved to Legacy/ on the 6.0 asset server; the Isaac Lab 3.0.0
    # container still points at the old path (404), newer sources at this one.
    "env_cfg_overrides": {
        "scene.table_cam.height": 256,
        "scene.table_cam.width": 256,
        "scene.wrist_cam.height": 256,
        "scene.wrist_cam.width": 256,
        "scene.robot.spawn.usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
    },
    # Boolean sub-task terms (group ``subtask_terms``); the stage is how many hold.
    "stage_terms": ["grasp_1", "stack_1", "grasp_2"],
}
"""Simulator-to-model mapping of the Franka stack-cube task (RLinf's
``isaaclab_stack_cube`` observation converter and ``LiberoFrankaDataConfig``)."""
