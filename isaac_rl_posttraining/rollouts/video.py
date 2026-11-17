# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn captured camera frames into evaluation videos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["detect_image_keys", "stack_camera_frames", "write_video"]


def detect_image_keys(keys: Sequence[str]) -> list[str]:
    """The camera keys among ``keys`` by name.

    ``critic_image`` is a learner input (a small pixel-critic frame), not a
    display stream; mixing its size into the video stack breaks frame
    stacking, so detection skips it.
    """
    return [
        key
        for key in keys
        if ("image" in key.lower() or "cam" in key.lower()) and key != "critic_image"
    ]


def stack_camera_frames(
    frames: Mapping[str, np.ndarray], image_keys: Sequence[str]
) -> np.ndarray | None:
    """Horizontally stack RGB camera frames (height-matched) into one BGR frame."""
    import cv2

    frames_to_stack: list[np.ndarray] = []
    for key in sorted(image_keys):
        if key not in frames:
            continue
        frame = frames[key]
        if frames_to_stack and frame.shape[0] != frames_to_stack[0].shape[0]:
            target_height = frames_to_stack[0].shape[0]
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(target_height * aspect_ratio), target_height))
        frames_to_stack.append(frame)
    if not frames_to_stack:
        return None
    return cv2.cvtColor(np.hstack(frames_to_stack), cv2.COLOR_RGB2BGR)


def write_video(frames: Sequence[np.ndarray], video_path: str, fps: int = 20) -> None:
    """Write BGR frames to an MP4 file."""
    import cv2

    if not frames:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    finally:
        video_writer.release()
