# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reusable 58-D Vega/Sharpa joint and camera transport contract."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from isaac_rl_posttraining.data import ActionSpec, ArraySpec, ObservationSpec

JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
Payload: TypeAlias = dict[str, JsonValue | np.ndarray]

STATE_KEYS = ("right_arm", "left_arm", "right_finger", "left_finger")
ACTION_KEYS = ("right_arm", "right_finger", "left_arm", "left_finger")
WIDTHS = {"right_arm": 7, "left_arm": 7, "right_finger": 22, "left_finger": 22}
VIDEO_KEYS = ("front", "right_wrist_view", "left_wrist_view")
IMAGE_KEYS = tuple(f"video.{key}" for key in VIDEO_KEYS)
CAMERA_TERMS = ("image", "image_right_wrist", "image_left_wrist")
JOINT_DIM = sum(WIDTHS.values())


def v2d_observation_spec() -> ObservationSpec:
    """One V2D observation; camera height and width are server-declared."""
    arrays = {"state": ArraySpec((JOINT_DIM,), "float32")}
    for key in IMAGE_KEYS:
        arrays[key] = ArraySpec((None, None, 3), "uint8")
    return ObservationSpec(arrays)


def v2d_action_spec() -> ActionSpec:
    """One absolute joint target in the checkpoint's 58-D action order."""
    return ActionSpec(width=JOINT_DIM, dtype="float32")


def batched_observation_from_record(
    record: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert V2D's record group to the checkpoint's batched observation."""
    arms = np.asarray(record["arm_joint_pos"])
    fingers = np.asarray(record["finger_joint_pos"])
    if (
        arms.ndim != 2
        or fingers.ndim != 2
        or arms.shape[0] == 0
        or arms.shape != (fingers.shape[0], 14)
        or fingers.shape[1] != 44
    ):
        raise ValueError("Expected arm_joint_pos [N,14] and finger_joint_pos [N,44]")
    state = np.concatenate((arms, fingers), axis=1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite joint observation")
    observation = {"state": state}
    for key, term in zip(IMAGE_KEYS, CAMERA_TERMS, strict=True):
        value = np.asarray(record[term])
        if value.ndim != 4 or value.shape[0] != len(state) or value.shape[-1] not in (3, 4):
            raise ValueError(f"Invalid camera shape for {term}: {value.shape}")
        rgb = value[..., :3]
        if rgb.dtype != np.uint8:
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(f"Float camera {term} must be finite RGB in [0,1]")
            rgb = (rgb * 255).astype(np.uint8)
        observation[key] = np.ascontiguousarray(rgb)
    return observation


def observation_from_record(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert one-environment V2D record to an unbatched observation."""
    batched = batched_observation_from_record(record)
    return {key: value[0] for key, value in batched.items()}


def object_height(record: dict[str, np.ndarray], object_index: int) -> np.ndarray:
    """Copy the selected object's Z coordinate for every environment."""
    value = np.asarray(record["object_position_e"])
    column = object_index * 3 + 2
    if value.ndim != 2 or value.shape[0] == 0 or not 0 <= column < value.shape[1]:
        raise ValueError("Selected object missing from object_position_e")
    height = value[:, column].astype(np.float64, copy=True)
    if not np.isfinite(height).all():
        raise ValueError("Non-finite object height")
    return height
