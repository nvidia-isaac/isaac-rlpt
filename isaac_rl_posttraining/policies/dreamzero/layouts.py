# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from RLinf (Apache-2.0) and modified; see THIRD_PARTY_NOTICES.md.
#
# SPDX-FileCopyrightText: Copyright 2026 The RLinf Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embodiment layouts for the in-process DreamZero base model (numpy-only).

Everything DreamZero knows about an embodiment lives in one
:class:`EmbodimentSpec`: how env observations map to the model's modality
keys, how multi-view frames are tiled into the single video the WAN backbone
sees, and how the task instruction is wrapped into the training-time prompt
template. The specs pin the exact preprocessing the released checkpoints
were trained with (ported from RLinf's validated DSRL recipes), so any
drift here silently degrades the base policy — change them only against a
matching checkpoint.

This module never imports the ``groot`` model stack; the groot-dependent
transform chain that consumes these specs lives in
:mod:`~isaac_rl_posttraining.policies.dreamzero.transforms`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "EmbodimentSpec",
    "convert_env_obs",
    "get_embodiment_spec",
    "normalize_instruction_text",
]

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a robot "

_LIBERO_MULTIVIEW_LAYOUT = (
    " The video is split into two horizontal views: the left view shows the "
    "exterior camera and the right view shows the wrist camera. The robot "
)

def _concat_libero_views(images: np.ndarray) -> np.ndarray:
    """Horizontal concat: exterior (left) | wrist (right)."""
    v, t, c, h, w = images.shape
    if v < 2:
        raise ValueError(
            f"libero_sim expects at least 2 video views, got v={v} with shape {images.shape}"
        )
    concat_images = np.zeros((1, t, c, h, 2 * w), dtype=images.dtype)
    concat_images[0, :, :, :, :w] = images[0]
    concat_images[0, :, :, :, w:] = images[1]
    return concat_images


@dataclass(frozen=True)
class EmbodimentSpec:
    """Everything embodiment-specific about a DreamZero checkpoint.

    Attributes:
        tag: groot ``EmbodimentTag`` value (also the ``metadata.json`` key).
        projector_id: The embodiment-projector index the checkpoint was
            trained with (``embodiment_tag_mapping`` at SFT time).
        video_fields: ``(env_key, model_key)`` pairs, in the checkpoint's
            view-concat order.
        state_fields: ``(env_key, model_key)`` pairs.
        video_keys / state_keys / action_keys / language_key: The modality
            keys of the checkpoint's dataset, in concat order.
        video_height / video_width: The per-view resize the transform chain
            applies before view concat.
        action_horizon: The checkpoint's action-chunk length.
        binarize_gripper: Threshold the decoded gripper action to ±1
            (LIBERO's native convention; DROID grippers are continuous).
        multiview_layout: The prompt fragment describing the view tiling.
        concat_views: Tiles ``[V, T, C, H, W]`` frames into the single
            ``[1, T, C, H', W']`` video the checkpoint expects.
    """

    tag: str
    projector_id: int
    video_fields: tuple[tuple[str, str], ...]
    state_fields: tuple[tuple[str, str], ...]
    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    language_key: str
    video_height: int
    video_width: int
    action_horizon: int
    binarize_gripper: bool
    multiview_layout: str
    concat_views: Callable[[np.ndarray], np.ndarray]

    @property
    def tag_mapping(self) -> dict[str, int]:
        """``embodiment_tag -> projector id`` mapping for the transform chain."""
        return {self.tag: self.projector_id}

    def format_prompt(self, instruction: str) -> str:
        """Wrap a task instruction in the checkpoint's multi-view prompt template."""
        return _TRAINING_PROMPT_PREFIX + instruction + self.multiview_layout + instruction


LIBERO_SIM_SPEC = EmbodimentSpec(
    tag="libero_sim",
    projector_id=21,
    video_fields=(
        ("main_image", "video.image"),
        ("wrist_image", "video.wrist_image"),
    ),
    state_fields=(("state", "state.state"),),
    video_keys=("video.image", "video.wrist_image"),
    state_keys=("state.state",),
    action_keys=("action.actions",),
    language_key="annotation.task",
    video_height=256,
    video_width=256,
    action_horizon=16,
    binarize_gripper=True,
    multiview_layout=_LIBERO_MULTIVIEW_LAYOUT,
    concat_views=_concat_libero_views,
)

_EMBODIMENT_SPECS: dict[str, EmbodimentSpec] = {
    LIBERO_SIM_SPEC.tag: LIBERO_SIM_SPEC,
}


def get_embodiment_spec(tag: str) -> EmbodimentSpec:
    """Resolve an :class:`EmbodimentSpec` by groot embodiment tag."""
    try:
        return _EMBODIMENT_SPECS[tag]
    except KeyError:
        raise ValueError(
            f"Unsupported DreamZero embodiment_tag {tag!r}; "
            f"known tags: {sorted(_EMBODIMENT_SPECS)}. Add an EmbodimentSpec "
            "in the shape of its siblings for a new checkpoint family."
        ) from None


def normalize_instruction_text(raw: Any) -> str:
    """Decode a dataset/env ``text`` field into a lowercase instruction string."""
    if not isinstance(raw, str):
        return str(raw).lower()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return str(parsed[0]).lower()
        return str(parsed).lower()
    except (ValueError, SyntaxError, TypeError):
        return raw.lower()


def _image_to_b1hwc_uint8(value: Any, env_key: str, batch_size: int) -> np.ndarray:
    """Normalize one camera batch to ``[B, T=1, H, W, C]`` uint8.

    ``T=1`` is load-bearing: single-frame input is what makes in-process
    inference stateless (the WAN head resets its autoregressive context
    whenever ``videos.shape[2] == 1``).
    """
    arr = np.asarray(value)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Observation {env_key!r} must be [B, H, W, 3] (single frame, "
            f"channels-last); got shape {arr.shape}."
        )
    if arr.shape[0] != batch_size:
        raise ValueError(
            f"Observation {env_key!r} has batch size {arr.shape[0]}, "
            f"expected {batch_size}."
        )
    if arr.dtype != np.uint8:
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                f"Observation {env_key!r} must be uint8 or float images; "
                f"got dtype {arr.dtype}."
            )
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr[:, None, ...]


def ensure_batched_env_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Lift a single-environment observation dict to batch size 1.

    The vectorized collector decodes one env at a time with UNBATCHED
    observation slices (``main_image`` as ``[H, W, 3]``, ``state`` as
    ``[D]``); the single-env path always delivers ``[1, ...]``. Both shapes
    are valid inputs to :func:`convert_env_obs` after this normalization.
    """
    main = np.asarray(env_obs["main_image"])
    if main.ndim != 3:
        return env_obs
    lifted = dict(env_obs)
    for key, value in env_obs.items():
        if key == "prompt" or value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1:
            lifted[key] = arr[None]
    return lifted


def convert_env_obs(env_obs: dict[str, Any], spec: EmbodimentSpec) -> dict[str, Any]:
    """Convert a batched env observation dict to DreamZero modality keys.

    Consumes the canonical steering-env keys (``main_image``, ``wrist_image``,
    ``state``, ``prompt``) and ignores everything else in ``env_obs``. Output arrays carry
    ``[B, T=1, ...]`` axes as the transform chain expects.

    The language entry is emitted as an ``object`` ndarray of strings, never
    a plain list: the transform's per-sample split (``tree.map_structure``)
    treats a list of strings as a structure of string leaves and would index
    *characters*, silently conditioning every sample on one letter of the
    prompt.
    """
    if "main_image" not in env_obs:
        raise KeyError(
            f"env_obs must contain 'main_image' (got keys {sorted(env_obs)})."
        )
    batch_size = int(np.asarray(env_obs["main_image"]).shape[0])
    converted: dict[str, Any] = {}

    for env_key, model_key in spec.video_fields:
        if env_key in env_obs and env_obs[env_key] is not None:
            converted[model_key] = _image_to_b1hwc_uint8(
                env_obs[env_key], env_key, batch_size
            )

    for env_key, model_targets in spec.state_fields:
        if env_key not in env_obs:
            raise KeyError(
                f"env_obs must contain {env_key!r} (got keys {sorted(env_obs)})."
            )
        states = np.asarray(env_obs[env_key], dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != batch_size:
            raise ValueError(
                f"Observation {env_key!r} must be [B, D]; got shape {states.shape}."
            )
        converted[model_targets] = states[:, None, :]

    prompts = env_obs.get("prompt")
    if prompts is None:
        raise KeyError(
            f"env_obs must contain 'prompt' (got keys {sorted(env_obs)}); the "
            "checkpoint's prompt template needs the task instruction."
        )
    if isinstance(prompts, str):
        prompts = [prompts] * batch_size
    prompts = [str(p) for p in prompts]
    if len(prompts) != batch_size:
        raise ValueError(
            f"'prompt' carries {len(prompts)} entries for batch size {batch_size}."
        )
    converted[spec.language_key] = np.asarray(prompts, dtype=object)

    return converted
