# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T fine-tuning encoding: per-sample steps, processor in training mode, window from the config."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.policies.gr00t.demonstrations import encode_training_steps


@pytest.fixture
def fake_gr00t(monkeypatch):
    class EmbodimentTag(str, Enum):
        NEW_EMBODIMENT = "new_embodiment"

    class MessageType(Enum):
        EPISODE_STEP = "episode_step"

    @dataclass
    class VLAStepData:
        images: dict
        states: dict
        actions: dict
        text: str = ""
        embodiment: object = None
        metadata: dict = field(default_factory=dict)

    modules = {
        "gr00t": types.ModuleType("gr00t"),
        "gr00t.data": types.ModuleType("gr00t.data"),
        "gr00t.data.embodiment_tags": types.ModuleType("gr00t.data.embodiment_tags"),
        "gr00t.data.types": types.ModuleType("gr00t.data.types"),
    }
    modules["gr00t.data.embodiment_tags"].EmbodimentTag = EmbodimentTag
    modules["gr00t.data.types"].MessageType = MessageType
    modules["gr00t.data.types"].VLAStepData = VLAStepData
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class _Keys:
    def __init__(self, keys, horizon=1):
        self.modality_keys = keys
        self.delta_indices = list(range(horizon))


class _Processor:
    def __init__(self):
        self.training = False
        self.modes_seen = []
        self.modality_configs = {
            "new_embodiment": {
                "video": _Keys(["ego_view"]),
                "state": _Keys(["arm"]),
                "action": _Keys(["arm", "hand"], horizon=3),
            }
        }

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, messages):
        step = messages[0]["content"]
        self.modes_seen.append(self.training)
        return {
            "state": step.states["arm"],
            "action": np.concatenate(
                [step.actions["arm"], step.actions["hand"]], axis=-1
            ),
        }


def _collate(features):
    return {
        "inputs": {
            key: torch.from_numpy(np.stack([f[key] for f in features]))
            for key in features[0]
        }
    }


LAYOUT = {
    "state": {"leg": {"start": 0, "end": 2}, "arm": {"start": 2, "end": 4}},
    "action": {"arm": {"start": 0, "end": 2}, "hand": {"start": 2, "end": 3}},
}


def test_encodes_each_sample_in_training_mode_and_restores_eval(fake_gr00t):
    processor = _Processor()
    states = [np.arange(4, dtype=np.float32) + 10 * i for i in range(2)]
    chunks = [np.arange(15, dtype=np.float32).reshape(5, 3) + 100 * i for i in range(2)]
    frames = [{"ego_view": np.zeros((4, 6, 3), np.uint8)}] * 2

    out = encode_training_steps(
        processor,
        _collate,
        frames,
        states,
        chunks,
        ["lift", "lift"],
        "new_embodiment",
        LAYOUT,
    )

    assert processor.modes_seen == [True, True]
    assert processor.training is False
    np.testing.assert_array_equal(out["state"][1].numpy(), [[12.0, 13.0]])
    assert out["action"].shape == (2, 3, 3)
    np.testing.assert_array_equal(out["action"][0].numpy(), chunks[0][:3])


def test_short_chunk_is_rejected(fake_gr00t):
    processor = _Processor()
    with pytest.raises(ValueError, match="needs 3"):
        encode_training_steps(
            processor,
            _collate,
            [{"ego_view": np.zeros((4, 6, 3), np.uint8)}],
            [np.zeros(4, np.float32)],
            [np.zeros((2, 3), np.float32)],
            ["lift"],
            "new_embodiment",
            LAYOUT,
        )
    assert processor.training is False
