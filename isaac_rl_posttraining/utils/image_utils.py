# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image helpers shared by rollout collection and evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["resize_images"]


def resize_images(images: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Resize a batch of RGB frames to a square resolution.

    Args:
        images: Array of shape ``[B, H, W, C]``.
        target_size: Output height and width in pixels.

    Returns:
        Array of shape ``[B, target_size, target_size, C]``. The input is
        returned unchanged when it already matches the target resolution.
    """
    _, height, width, _ = images.shape
    if height == target_size and width == target_size:
        return images

    resized = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size))
        resized.append(np.array(pil_img))
    return np.array(resized)
