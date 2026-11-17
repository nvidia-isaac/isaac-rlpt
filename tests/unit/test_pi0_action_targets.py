# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0's flow target is the action half of the training pipeline, not the raw column."""

from __future__ import annotations

import numpy as np
import torch

from isaac_rl_posttraining.policies.pi0.flow_sde import (
    action_input_steps,
    encode_action_chunk,
)


class DeltaActions:
    """Subtract state dim 0 from action dim 0, as openpi's ``DeltaActions`` does."""

    def __call__(self, data):
        actions = data["actions"]
        actions[:, 0] -= data["state"][0]
        data["actions"] = actions
        return data


class _LiberoInputs:
    def __call__(self, data):
        raise AssertionError("image parsers must not run on the action pipeline")


class _Normalize:
    def __call__(self, data):
        data["actions"] = (data["actions"] - 3.0) / 2.0
        return data


class _Pad:
    def __call__(self, data):
        data["actions"] = np.pad(data["actions"], ((0, 0), (0, 1)))
        return data


def test_action_steps_keep_delta_normalize_and_pad_and_skip_image_parsers():
    delta, images, normalize, pad = (
        DeltaActions(),
        _LiberoInputs(),
        _Normalize(),
        _Pad(),
    )
    steps = action_input_steps([images, delta], normalize, pad)
    assert steps == [delta, normalize, pad]


def test_encode_action_chunk_applies_delta_norm_and_pad_without_mutating_the_source():
    states = np.array([[10.0, 1.0]], np.float32)
    actions = np.array([[[10.0, 7.0], [12.0, 9.0]]], np.float32)
    original = actions.copy()
    steps = action_input_steps([_LiberoInputs(), DeltaActions()], _Normalize(), _Pad())

    def transform(data):
        for step in steps:
            data = step(data)
        return data

    encoded = encode_action_chunk(transform, states, actions)
    # dim 0 is delta against state 10, then (x - 3) / 2, then a zero pad.
    expected = np.array([[[-1.5, 2.0, 0.0], [-0.5, 3.0, 0.0]]], np.float32)
    np.testing.assert_allclose(encoded, expected)
    np.testing.assert_array_equal(actions, original)


def test_offline_sample_uses_the_encoded_chunk_as_the_flow_target(
    tmp_path, monkeypatch
):
    import pandas as pd

    from isaac_rl_posttraining.policies.pi0.offline import open_pi0_offline_source

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "lift"}\n')
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    state = [np.array([10.0, 1.0], np.float32), np.array([10.0, 1.0], np.float32)]
    action = [np.array([10.0, 7.0], np.float32), np.array([12.0, 9.0], np.float32)]
    pd.DataFrame({"observation.state": state, "action": action}).to_parquet(
        data / "episode_000000.parquet"
    )

    seen = {}

    class _Policy:
        video_keys = ("video.image",)
        state_keys = ("state.q",)
        action_window = 2
        instruction = ""

        def build_inputs(self, frames, states, instruction):
            return {"prompt": instruction}

        def encode_actions(self, states, actions):
            seen["states"] = np.asarray(states).copy()
            seen["actions"] = np.asarray(actions).copy()
            return np.full((actions.shape[0], 2, 3), 4.0, np.float32)

    source = open_pi0_offline_source(str(tmp_path), policy=_Policy())
    monkeypatch.setattr(source, "_frame", lambda *_args: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(np.random, "randint", lambda *_a, **_k: np.array([0]))
    batch = source.sample(1)
    np.testing.assert_array_equal(seen["states"], [[10.0, 1.0]])
    np.testing.assert_array_equal(seen["actions"][0, 0], [10.0, 7.0])
    assert torch.equal(batch["action"], torch.full((1, 2, 3), 4.0))
