# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation wrappers that adapt plain Gymnasium envs to dict observations.

The learner stack expects dictionary observations. These wrappers lift flat
state vectors (and rendered frames) of standard Gymnasium environments into
that format, mainly for smoke tests and algorithm development away from the
simulator.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def center_crop(image: np.ndarray | None, size: int | None) -> np.ndarray | None:
    """Center-crop an image, or a batch of images, to a square of ``size``.

    Args:
        image: ``[H, W, C]`` or ``[N, H, W, C]`` array (returned unchanged for
            other ranks, or when ``image``/``size`` is ``None``).
        size: Side length of the crop; clamped to the image's smaller side.

    Returns:
        The cropped array.
    """
    if image is None or size is None:
        return image

    if image.ndim == 3:
        h, w = image.shape[:2]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[y : y + crop, x : x + crop, :]
    if image.ndim == 4:
        h, w = image.shape[1:3]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[:, y : y + crop, x : x + crop, :]
    return image


class FlatToDictObsWrapper(gym.ObservationWrapper):
    """Wrap a flat observation into a dict under the ``"state"`` key."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"state": env.observation_space})

    def observation(self, obs: Any) -> dict[str, Any]:
        return {"state": obs}


class ImageDictObsWrapper(gym.ObservationWrapper):
    """Expose a proprioceptive slice plus the rendered frame as a dict.

    The observation becomes ``{"state": first 4 elements of the flat
    observation, "image": center-cropped render}``. The rendered frame is
    flipped vertically before cropping (renderers return the image upside
    down relative to screen coordinates).

    Args:
        env: Environment with a ``Box`` observation and an ``rgb_array``
            render mode.
        size: Side length of the square image observation.
    """

    def __init__(self, env: gym.Env, size: int = 224) -> None:
        super().__init__(env)
        self.size = int(size)

        orig_space = self.env.observation_space
        if isinstance(orig_space, gym.spaces.Box):
            state_space: gym.spaces.Space = gym.spaces.Box(
                low=orig_space.low[:4],
                high=orig_space.high[:4],
                shape=(4,),
                dtype=orig_space.dtype,
            )
        else:
            state_space = orig_space  # non-Box spaces pass through unsliced

        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8
                ),
            }
        )

    def observation(self, obs: Any) -> dict[str, Any]:
        state = obs[:4] if isinstance(obs, np.ndarray) and obs.shape[0] >= 4 else obs
        return {"state": state, "image": self._render_image()}

    def _render_image(self) -> np.ndarray:
        """Render, flip, and center-crop one frame."""
        image = self.env.render()
        if image is None:
            raise ValueError("Failed to render image")
        image = np.flipud(image)
        image = center_crop(image, self.size)
        return image.copy()
