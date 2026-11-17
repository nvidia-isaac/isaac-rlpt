# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 whole-body-control (WBC) embodiment knowledge for IsaacLab Arena.

The single home for everything specific to the Unitree-G1 whole-body-control
joint embodiment the ``g1_arena`` integration uses: the flat state/action
group layout, how a LeRobot teleoperation episode is assembled into the
policy's contiguous state/action arrays, and the action layout the env
server unpacks. The generic GR00T policy (``policies/gr00t/``) owns none of
this; it receives the layout and the assembler through a
:class:`DatasetAdapter` and reads everything else from the checkpoint.

Two views of the same 35-dim action share one source of truth,
:data:`G1_WBC_MODALITY`:

* the *dataset* view (:func:`assemble_wbc_arrays`) builds the array the
  offline buffer normalizes, and
* the *simulator* view (:func:`wbc_joint_action_slices`,
  :func:`wbc_base_height_slice`, :func:`wbc_navigate_slice`) tells the env
  server how to unpack a policy action into the 50-dim Arena action.

Keeping both off one constant is deliberate: the two used to be maintained
separately and silently diverged.

``numpy`` is imported only inside :func:`assemble_wbc_arrays`, so importing
this module stays dependency-free (the env server imports it under the Arena
runtime).
"""

from __future__ import annotations

from typing import Any

from isaac_rl_posttraining.envs.embodiments import DatasetAdapter

__all__ = [
    "G1_WBC_MODALITY",
    "G1_DECOUPLED_STATE_GROUPS",
    "G1_DECOUPLED_ACTION_GROUPS",
    "G1_WBC_STATE_GROUP_ORDER",
    "G1_WBC_ACTION_DIM",
    "is_wbc_dataset",
    "assemble_wbc_arrays",
    "wbc_joint_action_slices",
    "wbc_base_height_slice",
    "wbc_navigate_slice",
    "g1_wbc_adapter",
]

# Flat state/action layout of the decoupled G1 whole-body-control policy,
# in the exact group order the policy was trained on. State is the full
# 43-dim joint vector the simulator reports (legs included); action is the
# 35-dim vector the policy predicts: arm/hand/waist joint targets sliced
# from the joint action, then the teleop base-height and navigate commands.
# Everything else in this module is derived from these spans.
G1_WBC_MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
    },
    "action": {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14},
        "left_hand": {"start": 14, "end": 21},
        "right_hand": {"start": 21, "end": 28},
        "waist": {"start": 28, "end": 31},
        "base_height_command": {"start": 31, "end": 32},
        "navigate_command": {"start": 32, "end": 35},
    },
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}

#: Executed action steps in the G1 35-dim recipe (``g1_35dim_config.py``).
#: Independent of the flow head's ``action_horizon`` (40 on the G1 checkpoint).
G1_DECODE_HORIZON = 16

# The decoupled upper-body groups the policy actually predicts. A raw base
# model with no baked config gets its modality config and statistics built
# for this subset only (never the dataset's declared full-body/end-effector
# groups); a fine-tuned checkpoint bakes the authoritative config and this
# fallback is unused.
G1_DECOUPLED_STATE_GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
G1_DECOUPLED_ACTION_GROUPS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]

# Joint-group order of the policy's 31-dim proprioceptive state: the simulator
# reports interleaved joints and re-groups them into this order (arms, hands,
# waist only -- the legs are whole-body-controlled and NOT part of the policy
# state, so a leg-inclusive 43-dim vector would scramble proprioception).
G1_WBC_STATE_GROUP_ORDER = tuple(G1_DECOUPLED_STATE_GROUPS)

# Action groups mapped by joint name into the sim's 43-joint vector; the
# base-height and navigate commands ride separately (see the slice helpers).
_JOINT_ACTION_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
_BASE_HEIGHT_GROUP = "base_height_command"
_NAVIGATE_GROUP = "navigate_command"

# Width of the policy action (max end of the action spans): 35.
G1_WBC_ACTION_DIM = max(span["end"] for span in G1_WBC_MODALITY["action"].values())


def _action_slice(group: str) -> slice:
    span = G1_WBC_MODALITY["action"][group]
    return slice(int(span["start"]), int(span["end"]))


def wbc_joint_action_slices(prefix: str = "") -> dict[str, slice]:
    """Map each joint action group to its slice of the 35-dim policy action.

    Covers only the arm/hand/waist groups the env server remaps by joint
    name into the sim's 43-joint vector (the base-height and navigate
    commands are handled by :func:`wbc_base_height_slice` /
    :func:`wbc_navigate_slice`).

    Args:
        prefix: Prepended to each group name. Defaults to none: the Arena
            joint remap matches bare group names, so a prefixed key (e.g.
            ``action.<group>``) silently zeros that group's targets.

    Returns:
        ``{f"{prefix}{group}": slice}`` in the trained group order.
    """
    return {f"{prefix}{group}": _action_slice(group) for group in _JOINT_ACTION_GROUPS}


def wbc_base_height_slice() -> slice:
    """Slice of the base-height command in the 35-dim policy action."""
    return _action_slice(_BASE_HEIGHT_GROUP)


def wbc_navigate_slice() -> slice:
    """Slice of the navigate command in the 35-dim policy action."""
    return _action_slice(_NAVIGATE_GROUP)


def is_wbc_dataset(df: Any) -> bool:
    """Whether an episode dataframe uses the WBC teleop-command layout."""
    return "teleop.base_height_command" in df.columns and "teleop.navigate_command" in df.columns


def assemble_wbc_arrays(df: Any) -> tuple[Any, Any]:
    """Assemble contiguous state/action arrays from a WBC episode dataframe.

    WBC datasets spread the policy's action across several columns: the
    43-dim joint action plus separate teleop base-height and navigate
    commands. The offline buffer assumes one contiguous action vector per
    step, so this pre-assembles the 35-dim action matching
    :data:`G1_WBC_MODALITY` (arms, hands, waist sliced from the joint
    action, then the teleop commands) and passes the full 43-dim
    ``observation.state`` through.

    Args:
        df: Episode dataframe with ``action``, ``observation.state``, and
            ``teleop.*`` columns.

    Returns:
        ``(state, action)`` float32 arrays of shapes ``[T, 43]`` and
        ``[T, 35]``.
    """
    import numpy as np

    joint_action = np.stack(df["action"].to_numpy()).astype(np.float32)  # [T, 43]
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # [T, 43]
    base_height = np.asarray(df["teleop.base_height_command"].to_list(), np.float32).reshape(
        len(joint_action), -1
    )
    navigate = np.asarray(df["teleop.navigate_command"].to_list(), np.float32).reshape(
        len(joint_action), -1
    )
    # Order matches G1_WBC_MODALITY["action"]: left_arm, right_arm, left_hand,
    # right_hand, waist, base_height_command, navigate_command.
    action = np.concatenate(
        [
            joint_action[:, 15:22],
            joint_action[:, 29:36],
            joint_action[:, 22:29],
            joint_action[:, 36:43],
            joint_action[:, 12:15],
            base_height[:, 0:1],
            navigate[:, 0:3],
        ],
        axis=1,
    )  # [T, 35]
    return state, action


def g1_wbc_adapter() -> DatasetAdapter:
    """The G1 whole-body-control dataset adapter (the ``g1_arena`` default)."""
    return DatasetAdapter(
        assemble=assemble_wbc_arrays,
        modality_json=G1_WBC_MODALITY,
        fallback_state_groups=G1_DECOUPLED_STATE_GROUPS,
        fallback_action_groups=G1_DECOUPLED_ACTION_GROUPS,
        decode_horizon=G1_DECODE_HORIZON,
    )
