#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify zero-latent parity, nonzero steering, and clipping on a GR00T server."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np


def _request(socket: Any, serializer: Any, observation: dict[str, Any], options: dict[str, Any]):
    payload = {
        "endpoint": "get_action",
        "data": {"observation": observation, "options": options},
    }
    socket.send(serializer.to_bytes(payload))
    response = serializer.from_bytes(socket.recv())
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(response["error"])
    return response


def _chunk(response: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    """Flatten the DROID server action groups in the same order as RoboLab."""
    action, _ = response
    joint = np.asarray(action["action.joint_position"], dtype=np.float32)[0]
    gripper = np.asarray(action["action.gripper_position"], dtype=np.float32)[0]
    return np.concatenate([joint, gripper], axis=-1)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    """Run a reproducible live server probe using the public flat observation format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--nonzero-latent", type=float, default=0.25)
    args = parser.parse_args()

    import zmq
    from gr00t.policy.server_client import MsgSerializer

    observation = {
        "video.exterior_image_1_left": np.zeros((1, 1, 180, 320, 3), dtype=np.uint8),
        "video.wrist_image_left": np.zeros((1, 1, 180, 320, 3), dtype=np.uint8),
        # Zero translation plus the first two rows of an identity rotation.
        # An all-zero 6D rotation is not a valid pose and makes GR00T's
        # relative-action decoder fail while orthogonalizing its reference.
        "state.eef_9d": np.array(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=np.float32
        ),
        "state.joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        "state.gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
        "annotation.language.language_instruction": ["pick up the banana"],
    }
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.host}:{args.port}")
    try:
        native = _request(socket, MsgSerializer, observation, {"harness_seed": args.seed})
        native_chunk = _chunk(native)
        latent_width = native_chunk.shape[-1]
        zero = _request(
            socket,
            MsgSerializer,
            observation,
            {"harness_seed": args.seed, "harness_latent": np.zeros(latent_width, np.float32)},
        )
        zero_chunk = _chunk(zero)
        nonzero = _request(
            socket,
            MsgSerializer,
            observation,
            {
                "harness_seed": args.seed,
                "harness_latent": np.full(latent_width, args.nonzero_latent, np.float32),
            },
        )
        nonzero_chunk = _chunk(nonzero)
        clipped = _request(
            socket,
            MsgSerializer,
            observation,
            {"harness_seed": args.seed, "harness_latent": np.full(latent_width, 100.0, np.float32)},
        )
        if not np.array_equal(native_chunk, zero_chunk):
            raise AssertionError("zero additive latent did not reproduce seeded native action chunk")
        if np.array_equal(native_chunk, nonzero_chunk):
            raise AssertionError("bounded nonzero latent did not change the seeded action chunk")
        metadata = clipped[1]["harness_latent"]
        if metadata["clipped_entries"] != latent_width:
            raise AssertionError(f"expected every probe latent entry to clip; got {metadata}")
        print(
            json.dumps(
                {
                    "zero_latent_parity": True,
                    "nonzero_action_changed": True,
                    "latent_width": latent_width,
                    "native_action_sha256": _digest(native_chunk),
                    "nonzero_action_sha256": _digest(nonzero_chunk),
                    "clipping": metadata,
                },
                sort_keys=True,
            )
        )
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
