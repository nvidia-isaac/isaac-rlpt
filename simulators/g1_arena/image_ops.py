# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-frame downscaling shared by the g1_arena env-server and its client.

The frozen policy's image transform starts by shrinking the frame so its
shortest side is ``shortest_image_edge`` pixels (GR00T-N1.7:
``albumentations.SmallestMaxSize(256, interpolation=cv2.INTER_AREA)``), and
DINOv2's processor starts the same way. Doing that one step on the simulator
side, with the same operation and the same size rounding, leaves the model
inputs bit-identical while the frame crosses the wire at a fraction of its
native size (640x480 -> 341x256: 3.5x fewer bytes to copy, pickle and
unpickle, and 3.5x fewer pixels for the per-image CPU preprocessing).

Imported by the env-server under the simulator's minimal Python, so this
module depends on numpy only; ``cv2`` is passed in by the caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def shrunk_hw(native_hw: tuple[int, int], edge: int) -> tuple[int, int]:
    """Frame size after shrinking the shortest side of ``native_hw`` to ``edge``.

    Uses albumentations' rounding (``round(dim * edge / shortest)``) so the
    result matches ``SmallestMaxSize(edge)`` to the pixel. Frames are never
    upscaled: a native frame smaller than ``edge`` is an error.
    """
    height, width = native_hw
    shortest = min(height, width)
    if edge <= 0:
        raise ValueError(f"shortest edge must be positive, got {edge}")
    if shortest < edge:
        raise ValueError(
            f"camera frame {height}x{width} is smaller than the requested shortest edge {edge}; "
            "frames are never upscaled"
        )
    if shortest == edge:
        return height, width
    scale = edge / float(shortest)
    return round(height * scale), round(width * scale)


def shrink_shortest_edge(frames: np.ndarray, edge: int, cv2: Any) -> np.ndarray:
    """Downscale ``[H, W, 3]`` or ``[N, H, W, 3]`` uint8 frames to shortest side ``edge``.

    Exactly ``albumentations.SmallestMaxSize(edge, interpolation=cv2.INTER_AREA)``
    per frame (see :func:`shrunk_hw` for the size rule). Returns the input
    unchanged when the shortest side already equals ``edge``.
    """
    batched = frames.ndim == 4
    stack = frames if batched else frames[None]
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError(f"expected [H, W, 3] or [N, H, W, 3] frames, got shape {frames.shape}")
    height, width = stack.shape[1:3]
    new_height, new_width = shrunk_hw((height, width), edge)
    if (new_height, new_width) == (height, width):
        return frames
    out = np.empty((stack.shape[0], new_height, new_width, 3), dtype=stack.dtype)
    for i in range(stack.shape[0]):
        out[i] = cv2.resize(stack[i], (new_width, new_height), interpolation=cv2.INTER_AREA)
    return out if batched else out[0]


def letterbox_square(frames: np.ndarray, resolution: int, cv2: Any) -> np.ndarray:
    """Pad ``[H, W, 3]`` or ``[N, H, W, 3]`` uint8 frames to a square, then resize to ``resolution``.

    A direct resize of the non-square native frame distorts the aspect ratio,
    which the policy's fine manipulation is sensitive to, so the frame is
    letterboxed onto a black square first. Returns the input unchanged when it
    is already ``resolution`` square.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    batched = frames.ndim == 4
    stack = frames if batched else frames[None]
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError(f"expected [H, W, 3] or [N, H, W, 3] frames, got shape {frames.shape}")
    height, width = stack.shape[1:3]
    if (height, width) == (resolution, resolution):
        return frames
    side = max(height, width)
    y0, x0 = (side - height) // 2, (side - width) // 2
    out = np.empty((stack.shape[0], resolution, resolution, 3), dtype=stack.dtype)
    for i in range(stack.shape[0]):
        frame = stack[i]
        if height != width:
            square = np.zeros((side, side, 3), dtype=stack.dtype)
            square[y0 : y0 + height, x0 : x0 + width] = frame
            frame = square
        out[i] = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return out if batched else out[0]
