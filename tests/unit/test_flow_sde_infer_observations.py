# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-SDE infer() accepts G1 harness observations, not only video.* keys."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_rl_posttraining.policies.flow_sde import (
    FlowSDEActorCritic,
    split_infer_observations,
)


def test_split_infer_observations_maps_ego_view_image_onto_video_keys():
    observations = {
        "ego_view_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "state": np.arange(5, dtype=np.float32),
        "instruction": "Walk to the table and pick up the power drill",
    }
    frames, states = split_infer_observations(
        observations,
        video_keys=("video.ego_view",),
        state_keys=("state.legs", "state.torso"),
        state_slices=(("state.legs", 0, 2), ("state.torso", 2, 5)),
    )
    assert frames["video.ego_view"].shape == (1, 480, 640, 3)
    np.testing.assert_array_equal(states["state.legs"], [[0, 1]])
    np.testing.assert_array_equal(states["state.torso"], [[2, 3, 4]])


def test_split_infer_observations_keeps_model_keys_when_present():
    frame = np.ones((2, 8, 8, 3), dtype=np.uint8)
    legs = np.zeros((2, 2), dtype=np.float32)
    frames, states = split_infer_observations(
        {
            "video.ego_view": frame,
            "state.legs": legs,
            "state.torso": np.ones((2, 3), np.float32),
        },
        video_keys=("video.ego_view",),
        state_keys=("state.legs", "state.torso"),
        state_slices=(("state.legs", 0, 2), ("state.torso", 2, 5)),
    )
    np.testing.assert_array_equal(frames["video.ego_view"], frame)
    np.testing.assert_array_equal(states["state.legs"], legs)


def test_split_infer_observations_returns_none_for_processed_model_inputs():
    assert (
        split_infer_observations(
            {"input_ids": np.zeros((1, 4)), "state": np.zeros((1, 3), np.float32)},
            video_keys=("video.ego_view",),
            state_keys=("state.legs",),
        )
        is None
    )


def test_infer_maps_libero_keys_through_the_observation_key_mapping():
    seen = {}

    class _Policy:
        video_keys = ("video.image", "video.wrist_image")
        state_keys = ("state.q",)

        def __init__(self):
            self.observation_keys = {
                "video.image": "observation/image",
                "video.wrist_image": "observation/wrist_image",
                "state.q": "observation/state",
            }

        def build_inputs(self, frames, states, instruction):
            seen["frames"] = frames
            seen["states"] = states
            seen["instruction"] = instruction
            return {}

    image = np.zeros((8, 8, 3), np.uint8)
    wrist = np.ones((8, 8, 3), np.uint8)
    state = np.arange(8, dtype=np.float32)
    FlowSDEActorCritic._infer_model_inputs(
        _Policy(),
        {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": "pick up the bowl",
        },
    )
    np.testing.assert_array_equal(seen["frames"]["video.image"][0], image)
    np.testing.assert_array_equal(seen["frames"]["video.wrist_image"][0], wrist)
    np.testing.assert_array_equal(seen["states"]["state.q"][0], state)
    assert seen["instruction"] == "pick up the bowl"


def test_infer_still_maps_ego_view_without_an_observation_key_mapping():
    seen = {}

    class _Policy:
        video_keys = ("video.ego_view",)
        state_keys = ("state.q",)

        def build_inputs(self, frames, states, instruction):
            seen["frames"] = frames
            seen["instruction"] = instruction
            return {}

    FlowSDEActorCritic._infer_model_inputs(
        _Policy(),
        {
            "ego_view_image": np.zeros((4, 4, 3), np.uint8),
            "state": np.arange(3, dtype=np.float32),
            "instruction": "walk",
        },
    )
    assert seen["frames"]["video.ego_view"].shape == (1, 4, 4, 3)
    assert seen["instruction"] == "walk"


def test_split_infer_observations_rejects_a_flat_state_without_slices():
    with pytest.raises(KeyError, match="state_slices"):
        split_infer_observations(
            {
                "ego_view_image": np.zeros((4, 4, 3), dtype=np.uint8),
                "state": np.zeros(4, dtype=np.float32),
            },
            video_keys=("video.ego_view",),
            state_keys=("state.a", "state.b"),
        )
