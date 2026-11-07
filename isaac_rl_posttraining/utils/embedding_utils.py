# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-encoder embedding helpers for the image and language observation paths.

These back the DINO-embedding and language-instruction wrappers in
:mod:`isaac_rl_posttraining.envs`; only ``torch``/``transformers``-style objects are
required, and the models themselves are passed in already loaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["compute_video_embeddings", "compute_text_embeddings"]


@torch.no_grad()
def compute_video_embeddings(
    frames_array: np.ndarray,
    dinov2_model: Any,
    dinov2_processor: Any,
    batch_size: int = 64,
    use_autocast: bool = True,
    use_tqdm: bool = False,
) -> torch.Tensor:
    """Embed a stack of RGB frames with a DINOv2 model.

    Args:
        frames_array: Frames of shape ``[N, H, W, C]`` (a single ``[H, W, C]``
            frame is also accepted). ``uint8`` is preferred; float inputs in
            ``[0, 1]`` or ``[0, 255]`` are coerced.
        dinov2_model: A loaded HF vision model (e.g. ``facebook/dinov2-base``).
        dinov2_processor: The matching image processor.
        batch_size: Forward-pass micro-batch size.
        use_autocast: Enable autocast when the model lives on CUDA.
        use_tqdm: Show a progress bar over micro-batches.

    Returns:
        CPU float32 tensor of shape ``[N, D]`` — the model's ``pooler_output``,
        falling back to the CLS token of ``last_hidden_state``.
    """
    frames = np.asarray(frames_array)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.dtype != np.uint8:
        peak = float(frames.max()) if frames.size else 0.0
        scaled = frames if peak > 1.0 else frames * 255.0
        frames = np.clip(scaled, 0, 255).astype(np.uint8)

    device = next(dinov2_model.parameters()).device
    n = int(frames.shape[0])
    indices: Any = range(0, n, batch_size)
    if use_tqdm:
        try:
            from tqdm import tqdm

            indices = tqdm(indices, desc="DINO embed")
        except ImportError:
            pass

    autocast_enabled = bool(use_autocast) and device.type == "cuda"
    outs = []
    for i in indices:
        batch = [frames[j] for j in range(i, min(i + batch_size, n))]
        inputs = dinov2_processor(images=batch, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            out = dinov2_model(**inputs)
        emb = getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state[:, 0]
        outs.append(emb.float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def compute_text_embeddings(
    text: str | list[str],
    sentence_model: Any,
    use_autocast: bool = True,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    """Embed text with a SentenceTransformer-style model.

    Args:
        text: A single string or a list of strings.
        sentence_model: A model exposing ``encode(...)``.
        use_autocast: Unused; kept for call-site compatibility.
        show_progress_bar: Forwarded to ``encode``.

    Returns:
        CPU float32 tensor of shape ``[D]`` for a single string or
        ``[len(text), D]`` for a list.
    """
    emb = sentence_model.encode(
        text,
        convert_to_tensor=True,
        show_progress_bar=show_progress_bar,
    )
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(np.asarray(emb))
    return emb.float().cpu()
