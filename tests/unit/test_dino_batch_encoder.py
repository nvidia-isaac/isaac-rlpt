# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DINO encoding inside runtime-owned policy servers."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_rl_posttraining.envs import dino_wrapper


def test_batch_encoder_preserves_sample_and_key_order(monkeypatch) -> None:
    def fake_embed(images, model, processor):
        del model, processor
        return images[:, 0, 0, :1].astype(np.float32)

    monkeypatch.setattr(dino_wrapper, "_embed_image_batch", fake_embed)
    encoder = dino_wrapper.DinoBatchObservationEncoder(
        object(), object(), image_keys=["front", "wrist"]
    )
    observation = {
        "front": np.stack(
            [np.full((2, 2, 3), 1, np.uint8), np.full((2, 2, 3), 2, np.uint8)]
        ),
        "wrist": np.stack(
            [np.full((2, 2, 3), 3, np.uint8), np.full((2, 2, 3), 4, np.uint8)]
        ),
        "state": np.zeros((2, 4), dtype=np.float32),
    }

    encoded = encoder.observation(observation)

    np.testing.assert_array_equal(
        encoded["dino_embedding"], np.array([[1, 3], [2, 4]], dtype=np.float32)
    )
    assert encoded["state"] is observation["state"]


def test_batch_encoder_rejects_mismatched_image_batches(monkeypatch) -> None:
    monkeypatch.setattr(
        dino_wrapper,
        "_embed_image_batch",
        lambda images, model, processor: np.zeros((len(images), 1), np.float32),
    )
    encoder = dino_wrapper.DinoBatchObservationEncoder(
        object(), object(), image_keys=["front", "wrist"]
    )

    with pytest.raises(ValueError, match="same batch size"):
        encoder.observation(
            {
                "front": np.zeros((2, 2, 2, 3), np.uint8),
                "wrist": np.zeros((1, 2, 2, 3), np.uint8),
            }
        )
