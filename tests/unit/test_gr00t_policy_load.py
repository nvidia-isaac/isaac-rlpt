# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gr00tPolicy.load must merge every safetensors shard.

Regression: sharded checkpoints (model-0000N-of-0000M.safetensors) were
loaded via ``glob(...)[0]`` — one shard of M — leaving the rest of the
model at its pre-load values with only a strict=False "missing" count as
evidence.
"""

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from torch import nn

from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy


class _TwoTensorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 2, bias=False)


def test_load_merges_all_safetensors_shards(tmp_path):
    target = _TwoTensorModel()
    with torch.no_grad():
        target.a.weight.zero_()
        target.b.weight.zero_()

    safetensors_torch.save_file(
        {"a.weight": torch.full((2, 2), 1.5)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    safetensors_torch.save_file(
        {"b.weight": torch.full((2, 2), -2.5)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )

    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.model = type("M", (), {})()
    policy.model.base_model = target

    Gr00tPolicy.load(policy, str(tmp_path))

    assert torch.allclose(target.a.weight, torch.full((2, 2), 1.5)), (
        "shard 1 not applied"
    )
    assert torch.allclose(target.b.weight, torch.full((2, 2), -2.5)), (
        "shard 2 not applied — only the first shard was loaded"
    )
