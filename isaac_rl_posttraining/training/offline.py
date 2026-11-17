# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline demonstration sources for IQL and SFT.

A trainable base model must turn a dataset directory into batches its
own ``actor_bc_loss`` (and, for IQL, ``feature_tokens``) read. The builders
live next to the policy stacks and are registered here by base model name,
matching co-training demonstrations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch

__all__ = ["SOURCE_BUILDERS", "OfflineSource", "WorkerSampler", "open_offline_source"]


class OfflineSource(Protocol):
    """Draws offline transitions in one base model's input format."""

    def sample(self, batch_size: int) -> Mapping[str, Any]:
        """Return ``inputs_cur``, ``inputs_next``, ``action``, ``reward``, ``done``."""


class _SampleStream(torch.utils.data.IterableDataset):
    def __init__(self, sample: Callable[[int], Mapping[str, Any]], batch_size: int) -> None:
        self.sample = sample
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.sample(self.batch_size)


def _seed_worker(worker_id: int) -> None:
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class WorkerSampler:
    """Draw ``sample(batch_size)`` batches in DataLoader worker processes, prefetched.

    Each worker holds its own copy of the sampler's owner (for example a buffer
    with per-process video readers), so construct this before sampling in the
    main process. ``num_workers=0`` samples inline.
    """

    def __init__(self, sample: Callable[[int], Mapping[str, Any]], num_workers: int) -> None:
        self._sample = sample
        self._num_workers = int(num_workers)
        self._batch_size: int | None = None
        self._batches: Any = None

    def sample(self, batch_size: int) -> Mapping[str, Any]:
        if self._num_workers <= 0:
            return self._sample(batch_size)
        if batch_size != self._batch_size:
            self._batch_size = batch_size
            loader = torch.utils.data.DataLoader(
                _SampleStream(self._sample, batch_size),
                batch_size=None,
                num_workers=self._num_workers,
                worker_init_fn=_seed_worker,
                prefetch_factor=2,
                persistent_workers=True,
            )
            self._batches = iter(loader)
        return next(self._batches)


#: Trainable base model -> ``"module:function"`` that opens its :class:`OfflineSource`.
SOURCE_BUILDERS: dict[str, str] = {
    "gr00t_n17": "isaac_rl_posttraining.policies.gr00t.offline:open_gr00t_offline_source",
    "pi0": "isaac_rl_posttraining.policies.pi0.offline:open_pi0_offline_source",
}


def open_offline_source(
    base_model: str,
    dataset_dir: str,
    *,
    policy: Any,
    dataset_adapter: Any = None,
) -> OfflineSource:
    """Open the offline source registered for ``base_model``.

    Raises:
        ValueError: If the base model has no builder or ``dataset_dir`` is empty.
    """
    if not dataset_dir:
        raise ValueError(
            "offline training needs base_model.dataset_dir (a LeRobot dataset of demonstrations); "
            f"base_model={base_model!r} has none"
        )
    target = SOURCE_BUILDERS.get(base_model)
    if target is None:
        raise ValueError(
            f"base model {base_model!r} has no offline demonstration source; registered: "
            f"{sorted(SOURCE_BUILDERS)}"
        )
    import importlib

    module_name, function_name = target.split(":", 1)
    builder = getattr(importlib.import_module(module_name), function_name)
    return builder(dataset_dir, policy=policy, dataset_adapter=dataset_adapter)
