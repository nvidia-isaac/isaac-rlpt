# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory management for large DreamZero bundles on VRAM-constrained GPUs.

DreamZero checkpoints bundle the DiT, text encoder, VAE and image encoder in
one module tree. For 14B-scale bundles (e.g. DreamZero-DROID: 22.9B params,
42.7 GiB bf16) the text encoder (umt5-xxl, ~10.6 GiB) is the only component
that tolerates CPU residency: task instructions are constant during rollout,
so its output can be computed once per unique prompt and cached.
"""

from __future__ import annotations

import torch
from loguru import logger

__all__ = ["CpuCachedPromptEncoder", "park_text_encoder"]


class CpuCachedPromptEncoder:
    """Callable replacing ``WANPolicyHead.encode_prompt`` for a CPU-parked
    text encoder.

    Runs the original encode on CPU inputs and caches the embedding per
    unique ``(input_ids, attention_mask)`` pair — one encoder forward per
    unique prompt instead of one per action chunk. The cached embedding is
    copied to the caller's device on each call.
    """

    def __init__(self, action_head: torch.nn.Module) -> None:
        # Bound method captured before the instance attribute shadows it.
        self._encode = action_head.encode_prompt
        self._cache: dict[tuple[bytes, bytes], torch.Tensor] = {}

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        key = (
            input_ids.detach().cpu().numpy().tobytes(),
            attention_mask.detach().cpu().numpy().tobytes(),
        )
        embedding = self._cache.get(key)
        if embedding is None:
            embedding = self._encode(input_ids.cpu(), attention_mask.cpu())
            self._cache[key] = embedding
        return embedding.to(input_ids.device)


def park_text_encoder(model: torch.nn.Module, device: torch.device) -> None:
    """Place the model on ``device`` with the text encoder parked on CPU (fp32).

    Owns the whole device placement: the encoder is pinned to CPU FIRST and
    every other module is moved selectively, so the ~10.6 GiB tower never
    transits GPU memory — a whole-model ``.to(device)`` followed by a
    park-back otherwise peaks at the full bundle size (42.7 GiB for the 14B)
    and OOMs a card shared with the simulator.
    """
    action_head = model.action_head

    # Re-register the encoder LAST among the head's children: the stock
    # first-parameter ``.device``/``.dtype`` properties (HF
    # ``get_parameter_device`` and the head's own property) then resolve
    # from GPU-resident modules instead of the CPU-parked encoder, with no
    # property overrides needed. state_dict keys are name-based and
    # unaffected by registration order.
    modules = action_head._modules
    modules["text_encoder"] = modules.pop("text_encoder")

    # fp32 for the (rare) CPU forward: CPU bf16 matmuls are slow and less
    # accurate, and encode_prompt casts its output to bf16 itself.
    action_head.text_encoder.cpu().float()

    for name, child in model.named_children():
        if name != "action_head":
            child.to(device)
    for name, child in action_head.named_children():
        if name != "text_encoder":
            child.to(device)
    # Parameters/buffers held directly on the module objects (outside any
    # child) are missed by the per-child moves.
    for holder in (model, action_head):
        for parameter in holder.parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        for buffer_name, buffer in holder.named_buffers(recurse=False):
            setattr(holder, buffer_name, buffer.to(device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parked_gib = (
        sum(p.numel() for p in action_head.text_encoder.parameters()) * 2 / 2**30
    )
    action_head.encode_prompt = CpuCachedPromptEncoder(action_head)
    logger.info(
        f"DreamZero text encoder parked on CPU (fp32) with prompt caching; "
        f"GPU residency reduced by ~{parked_gib:.1f} GiB."
    )
