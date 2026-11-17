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

"""The groot transform chain for in-process DreamZero inference.

Builds the exact ``ComposedModalityTransform`` the released checkpoints were
SFT-trained with (ported from RLinf's validated recipes): per-view video
preprocessing → q99 state normalization → per-modality concat → the
DreamZero transform that tiles views into one video, wraps the instruction
in the multi-view prompt template, and tokenizes it for umt5.

:class:`InferenceDreamTransform` subclasses groot's ``DreamTransform``
because the stock class cannot reproduce the checkpoints' preprocessing
outside groot's own training pipeline: its ``_prepare_video`` lacks the
LIBERO horizontal view concat (it falls into a generic 2×2 grid), and its
batch collate hard-indexes an embodiment-tag mapping with tags a
single-embodiment mapping does not carry.

Everything here imports ``groot`` (installed separately with ``--no-deps``;
see the ``dreamzero-local`` extra); the pure-numpy embodiment facts live in
:mod:`~isaac_rl_posttraining.policies.dreamzero.layouts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.policies.dreamzero.layouts import (
    EmbodimentSpec,
    get_embodiment_spec,
    normalize_instruction_text,
)

__all__ = [
    "InferenceDreamTransform",
    "build_composed_transform",
    "env_action_dim_from_metadata",
    "load_metadata",
]

#: Negative prompt baked into the checkpoints' classifier-free guidance.
_TEXT_NEGATIVE = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
    "style, artwork, painting, image, still, grayscale, dull, worst quality, "
    "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
    "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
    "image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)

_VIDEO_BACKEND = "torchvision"


def _collate_inference(
    features: list[dict[str, Any]], tokenizer: Any, spec: EmbodimentSpec
) -> dict[str, Any]:
    """Stack per-sample transform outputs and tokenize prompts.

    Port of RLinf's ``DreamZeroCollator.collate_batch``: the raw instruction
    is normalized and wrapped in the embodiment's multi-view prompt template
    here — this is the text the checkpoint was trained on, so skipping the
    template degrades the base policy.
    """
    batch: dict[str, Any] = {}
    for key in features[0]:
        if key == "text":
            texts = [
                spec.format_prompt(normalize_instruction_text(elem[key]))
                for elem in features
            ]
            ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask"] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask_negative"] = mask
        else:
            values = [elem[key] for elem in features]
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except ValueError as e:
                shapes = [np.asarray(v).shape for v in values]
                raise ValueError(
                    f"Shape mismatch in collate for key={key!r}: shapes={shapes}"
                ) from e
    return batch


def _make_inference_dream_transform_class():
    """Build the transform subclass lazily so importing this module without
    ``groot`` (or the ``dreamzero-local`` extra) stays possible until a
    chain is actually built."""
    from einops import rearrange
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
        DreamTransform as DreamTransformBase,
    )

    class InferenceDreamTransform(DreamTransformBase):
        """groot ``DreamTransform`` specialized to checkpoint-faithful inference.

        Overrides (each ported from RLinf's subclass, trimmed to the
        inference path — this repo never SFT-trains DreamZero):

        * ``_prepare_video`` — delegates the multi-view tiling to the
          :class:`EmbodimentSpec` (LIBERO: horizontal exterior|wrist;
          2×2 grid with the wrist spanning the top row).
        * ``apply_single`` — the eval-mode sample transform, without groot's
          training-only branches (actions, LAPA/DREAM co-training targets).
        * ``apply_batch`` — per-sample split + a collate that applies the
          embodiment's prompt template (groot's collate assumes its own
          multi-embodiment tag mapping).
        """

        def _spec(self) -> EmbodimentSpec:
            if self.embodiment_tag is None:
                raise RuntimeError(
                    "Embodiment tag not set; call set_metadata before applying "
                    "the transform."
                )
            return get_embodiment_spec(str(self.embodiment_tag.value))

        def _prepare_video(self, data: dict) -> np.ndarray:
            images = rearrange(data["video"], "t v h w c -> v t c h w")
            if images.shape[0] > 1:
                return self._spec().concat_views(images)
            return images

        def apply_single(self, data: dict) -> dict:
            transformed_data: dict[str, Any] = {}

            images = self._prepare_video(data)
            images = images.astype(np.uint8)
            language, is_lapa, is_dream, is_cotrain = self._prepare_language(data)
            if is_lapa or is_dream or is_cotrain:
                raise ValueError(
                    "LAPA/DREAM/COTRAIN-tagged instructions are training-dataset "
                    f"artifacts and unsupported at inference (got {language!r})."
                )
            vlm_outputs = self._apply_vlm_processing(
                {"images": images, "language": language}
            )

            state, state_mask, _ = self._prepare_state(data)
            transformed_data["state"] = state
            transformed_data["state_mask"] = state_mask
            transformed_data["text_negative"] = _TEXT_NEGATIVE

            for key, value in vlm_outputs.items():
                if key in transformed_data:
                    raise KeyError(f"Key {key} already exists in transformed_data.")
                transformed_data[key] = value

            transformed_data["embodiment_id"] = self.get_embodiment_tag()
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
            transformed_data["is_cotrain_instance"] = np.zeros((), dtype=bool)
            return transformed_data

        def apply_batch(self, data: dict, batch_size: int) -> dict:
            import tree

            data_split = [
                tree.map_structure(lambda x, i=i: x[i], data) for i in range(batch_size)
            ]
            data_split_processed = [self.apply_single(elem) for elem in data_split]
            return _collate_inference(data_split_processed, self.tokenizer, self._spec())

    return InferenceDreamTransform


#: Populated on first use; the class object is created once per process.
_INFERENCE_DREAM_TRANSFORM_CLASS = None


def InferenceDreamTransform(**kwargs: Any):  # NOSONAR - public factory with class semantics
    """Instantiate the inference ``DreamTransform`` (lazy groot import)."""
    global _INFERENCE_DREAM_TRANSFORM_CLASS
    if _INFERENCE_DREAM_TRANSFORM_CLASS is None:
        _INFERENCE_DREAM_TRANSFORM_CLASS = _make_inference_dream_transform_class()
    return _INFERENCE_DREAM_TRANSFORM_CLASS(**kwargs)


def build_composed_transform(
    spec: EmbodimentSpec,
    tokenizer_path: str,
    *,
    action_horizon: int | None = None,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    max_seq_len: int = 512,
    default_instruction: str = "Perform the default behavior.",
):
    """Build the checkpoint-faithful ``ComposedModalityTransform`` chain.

    The stage order, augment parameters, per-view resize, and q99
    normalization are pinned to what the released checkpoints saw at SFT
    time (RLinf's per-embodiment transform builders); the composed
    transform is returned in train mode — callers switch it with
    ``.eval()`` after ``set_metadata``.
    """
    from groot.vla.data.transform.base import ComposedModalityTransform
    from groot.vla.data.transform.concat import ConcatTransform
    from groot.vla.data.transform.state_action import (
        StateActionToTensor,
        StateActionTransform,
    )
    from groot.vla.data.transform.video import (
        VideoColorJitter,
        VideoCrop,
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )

    video_keys = list(spec.video_keys)
    state_keys = list(spec.state_keys)
    action_keys = list(spec.action_keys)

    transforms: list[Any] = [
        VideoToTensor(apply_to=video_keys, backend=_VIDEO_BACKEND),
        VideoCrop(apply_to=video_keys, backend=_VIDEO_BACKEND, scale=0.95),
        VideoResize(
            apply_to=video_keys,
            backend=_VIDEO_BACKEND,
            height=spec.video_height,
            width=spec.video_width,
            interpolation="linear",
        ),
        VideoColorJitter(
            apply_to=video_keys,
            backend=_VIDEO_BACKEND,
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08,
        ),
        VideoToNumpy(apply_to=video_keys, backend=_VIDEO_BACKEND),
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes=dict.fromkeys(state_keys, "q99"),
        ),
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes=dict.fromkeys(action_keys, "q99"),
        ),
        ConcatTransform(
            apply_to=[],
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        InferenceDreamTransform(
            default_instruction=default_instruction,
            language_dropout_prob=0.0,
            always_use_default_instruction=False,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            max_length=max_seq_len,
            state_horizon=1,
            action_horizon=action_horizon if action_horizon is not None else spec.action_horizon,
            tokenizer_path=tokenizer_path,
            embodiment_tag_mapping=spec.tag_mapping,
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)


def load_metadata(metadata_json_path: str | Path, spec: EmbodimentSpec):
    """Load the checkpoint's ``DatasetMetadata`` for one embodiment.

    ``metadata.json`` (generated at SFT time, usually under the checkpoint's
    ``experiment_cfg/``) carries the q99 normalization statistics and the
    per-key action shapes the transform chain decodes with.
    """
    from groot.vla.data.schema import DatasetMetadata

    path = Path(metadata_json_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"DreamZero metadata.json not found at {path}. Set "
            "base_model.metadata_json_path, or keep the checkpoint's "
            "experiment_cfg/metadata.json next to its weights."
        )
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if spec.tag not in blob:
        raise KeyError(
            f"embodiment_tag {spec.tag!r} not found in {path} "
            f"(keys: {list(blob)})."
        )
    raw = blob[spec.tag]
    return DatasetMetadata.model_validate(
        {
            "statistics": raw["statistics"],
            "modalities": raw["modalities"],
            "embodiment_tag": spec.tag,
        }
    )


def env_action_dim_from_metadata(metadata: Any, spec: EmbodimentSpec) -> int:
    """Total decoded env action width: the sum of the per-key widths.

    ``metadata.modalities.action`` is keyed by the bare action names
    (``joint_position``), while the transform chain uses prefixed modality
    keys (``action.joint_position``).
    """
    widths = []
    for key in spec.action_keys:
        bare = key.split(".", 1)[1]
        if bare not in metadata.modalities.action:
            raise KeyError(
                f"Action key {bare!r} missing from metadata.modalities.action "
                f"(available: {sorted(metadata.modalities.action)})."
            )
        widths.append(int(metadata.modalities.action[bare].shape[0]))
    return sum(widths)
