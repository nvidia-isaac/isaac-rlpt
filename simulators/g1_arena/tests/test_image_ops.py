# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side frame shrink: same size rule and same pixels as the policy's own first transform stage."""

from __future__ import annotations

import numpy as np
import pytest

from simulators.g1_arena.env import NATIVE_IMAGE_HW, spec_image_hw
from simulators.g1_arena.image_ops import shrink_shortest_edge, shrunk_hw

cv2 = pytest.importorskip("cv2")


@pytest.mark.parametrize(
    "native, edge, expected",
    [
        (
            (480, 640),
            256,
            (256, 341),
        ),  # the G1 head camera: 640 * 256/480 = 341.33 -> 341
        ((720, 1280), 256, (256, 455)),  # 1280 * 256/720 = 455.1 -> 455
        ((480, 640), 480, (480, 640)),  # already at the edge: unchanged
        ((640, 480), 256, (341, 256)),  # portrait
    ],
)
def test_shrunk_hw_matches_albumentations_rounding(native, edge, expected):
    assert shrunk_hw(native, edge) == expected


def test_shrunk_hw_never_upscales():
    with pytest.raises(ValueError, match="never upscaled"):
        shrunk_hw((200, 300), 256)
    with pytest.raises(ValueError, match="positive"):
        shrunk_hw((480, 640), 0)


def test_shrink_shortest_edge_shapes_and_noop():
    rng = np.random.default_rng(0)
    single = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    batched = rng.integers(0, 255, size=(4, 480, 640, 3), dtype=np.uint8)
    assert shrink_shortest_edge(single, 256, cv2).shape == (256, 341, 3)
    assert shrink_shortest_edge(batched, 256, cv2).shape == (4, 256, 341, 3)
    assert shrink_shortest_edge(single, 480, cv2) is single, (
        "no resize when already at the edge"
    )
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        shrink_shortest_edge(np.zeros((480, 640), np.uint8), 256, cv2)


def test_shrink_is_bit_identical_to_gr00t_first_transform_stage():
    """GR00T-N1.7's eval transform starts with SmallestMaxSize(256, INTER_AREA); ours must equal it."""
    A = pytest.importorskip("albumentations")
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    reference = A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA, p=1.0)(
        image=frame
    )["image"]
    ours = shrink_shortest_edge(frame, 256, cv2)
    assert ours.shape == reference.shape
    assert np.array_equal(ours, reference)
    # applying the transform stage again to our output is the identity, so the
    # rest of the pipeline sees the same pixels whether or not the server shrank
    again = A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA, p=1.0)(
        image=ours
    )["image"]
    assert np.array_equal(again, ours)


def test_spec_image_hw_reads_the_advertised_edge():
    ego_native = {
        "spaces": {"observation": {"ego_view": {"dtype": "uint8", "hw": None}}}
    }
    ego_none = {
        "spaces": {
            "observation": {
                "ego_view": {"dtype": "uint8", "hw": None, "shortest_edge": None}
            }
        }
    }
    ego_256 = {
        "spaces": {
            "observation": {
                "ego_view": {"dtype": "uint8", "hw": None, "shortest_edge": 256}
            }
        }
    }
    assert (
        spec_image_hw(ego_native, NATIVE_IMAGE_HW) == NATIVE_IMAGE_HW
    )  # older server: no key
    assert spec_image_hw(ego_none, NATIVE_IMAGE_HW) == NATIVE_IMAGE_HW
    assert spec_image_hw(ego_256, NATIVE_IMAGE_HW) == (256, 341)


def test_letterbox_square_pads_then_resizes_single_and_batched_frames():
    from simulators.g1_arena.image_ops import letterbox_square

    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :, 0] = 255  # a red frame: the letterbox bands stay black
    out = letterbox_square(frame, 224, cv2)
    assert out.shape == (224, 224, 3)
    assert out[0, 112].tolist() == [0, 0, 0] and out[112, 112].tolist() == [255, 0, 0]
    batched = letterbox_square(np.stack([frame, frame]), 224, cv2)
    assert batched.shape == (2, 224, 224, 3)
    square = np.zeros((224, 224, 3), np.uint8)
    assert letterbox_square(square, 224, cv2) is square
    with pytest.raises(ValueError):
        letterbox_square(frame, 0, cv2)
