# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline batches drawn in DataLoader workers."""

from __future__ import annotations

import numpy as np
import torch

from isaac_rl_posttraining.training.offline import WorkerSampler


class _Buffer:
    def sample(self, batch_size: int) -> dict:
        return {"x": torch.from_numpy(np.random.rand(batch_size))}


def test_inline_when_no_workers():
    sampler = WorkerSampler(_Buffer().sample, num_workers=0)
    assert sampler.sample(3)["x"].shape == (3,)


def test_workers_serve_distinct_batches_of_the_requested_size():
    sampler = WorkerSampler(_Buffer().sample, num_workers=2)
    batches = [sampler.sample(4)["x"] for _ in range(4)]
    assert all(batch.shape == (4,) for batch in batches)
    assert len({tuple(batch.tolist()) for batch in batches}) == 4
