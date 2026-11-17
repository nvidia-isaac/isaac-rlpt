# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached features: activations of frozen sub-modules, computed once at decision time and reused for training.

A policy that runs a frozen encoder (a vision-language backbone, a pretrained
image encoder) produces exactly the same activations at decision time and again
inside the learner. Caching ships those activations with the transition instead
of the raw inputs they were computed from, so the learner skips the frozen
forward pass. The math is unchanged; only recomputation (and the transport of
pixels) is avoided.

Contract:

* A policy that supports caching implements :class:`FeatureCaching`: it names
  the cached tensors, returns them from its decision call, tells the collector
  which inputs the learner still needs, and accepts the tensors in its training
  forward under ``cached_features=``.
* Collectors pack the tensors with :func:`pack_features` (one NumPy record per
  sample; bfloat16 kept as raw bits) and store them in
  ``Transition.cached_features`` and ``ActionChunk.cached_features``; replay
  buffers keep and export them like any other transition field.
* Learners receive them stacked under :data:`CACHED_FEATURES_KEY` in their
  batches and restore tensors with :func:`unpack_features`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

__all__ = [
    "BF16_SUFFIX",
    "CACHED_FEATURES_KEY",
    "FeatureCaching",
    "pack_features",
    "stack_features",
    "unpack_features",
]

CACHED_FEATURES_KEY = "cached_features"
BF16_SUFFIX = "@bf16"  # a bfloat16 tensor stored as the int16 view of its bits


@runtime_checkable
class FeatureCaching(Protocol):
    """A policy whose frozen sub-modules' activations can be computed once and reused.

    ``cached_feature_names`` lists the tensors the policy caches. Its decision
    call returns them (batch-leading) alongside the action; ``learner_inputs``
    reduces processed inputs to what the training forward still needs once the
    cache is present; the training forward accepts ``cached_features=``.
    """

    cached_feature_names: tuple[str, ...]

    def learner_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]: ...


def pack_features(features: Mapping[str, torch.Tensor], batch_size: int) -> list[dict[str, np.ndarray]]:
    """Split batch-leading feature tensors into one NumPy record per sample.

    bfloat16 tensors become ``int16`` arrays under ``name + BF16_SUFFIX`` so the
    exact bits travel through NumPy-only transports; :func:`unpack_features`
    restores them.
    """
    records: list[dict[str, np.ndarray]] = [{} for _ in range(batch_size)]
    for name, tensor in features.items():
        value = tensor.detach()
        if value.ndim < 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"cached feature {name!r} with shape {tuple(value.shape)} cannot be split into {batch_size} samples"
            )
        key = name
        if value.dtype == torch.bfloat16:
            key = name + BF16_SUFFIX
            value = value.view(torch.int16)
        array = value.cpu().numpy()
        for index in range(batch_size):
            records[index][key] = np.array(array[index], copy=True)
    return records


def stack_features(records: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample records (as produced by :func:`pack_features`) into ``[B, ...]`` arrays."""
    if not records:
        raise ValueError("cannot stack an empty sequence of cached-feature records")
    keys = tuple(records[0])
    for record in records[1:]:
        if tuple(record) != keys:
            raise ValueError(f"cached-feature records disagree on keys: {keys} vs {tuple(record)}")
    return {key: np.stack([np.asarray(record[key]) for record in records]) for key in keys}


def unpack_features(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Invert :func:`pack_features` for a stacked ``[B, ...]`` batch (raw bits become bfloat16 again)."""
    features: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.as_tensor(value).to(device)
        if key.endswith(BF16_SUFFIX):
            key = key[: -len(BF16_SUFFIX)]
            tensor = tensor.contiguous().view(torch.bfloat16)
        features[key] = tensor
    return features
