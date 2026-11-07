# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero base model and wire-protocol tests.

Everything here runs against the in-process ``FakeDreamZeroServer``
(conftest), which implements the exact reference server semantics: a
msgpack-numpy metadata handshake on connect, endpoint-dispatched
request/response frames, binary action chunks, and text frames for reset
acknowledgements and server errors. No network sockets, no DreamZero model
stack — the ``websockets`` package is deliberately not required, and the
whole module skips cleanly when ``msgpack`` (the wire codec) is absent.

Three layers:

1. codec pins — the msgpack-numpy schema must match the reference codec
   byte-for-byte conventions (map keys, dtype strings, rejected dtypes);
2. client protocol — handshake ordering, endpoint framing, error text
   frames, reset acks, reconnect-and-retry;
3. base model contract behavior — batch-1 native inference, the
   ``NotImplementedError`` steering refusal, shape validation, session
   lifecycle, and registry/import-guard wiring.
"""

import numpy as np
import pytest
from conftest import FakeDreamZeroServer

pytest.importorskip("msgpack", reason="msgpack needed for the DreamZero wire codec")

from isaac_rl_posttraining.policies.dreamzero import is_available
from isaac_rl_posttraining.policies.dreamzero.client import (
    DreamZeroClient,
    DreamZeroServerError,
    pack_payload,
    unpack_payload,
)
from isaac_rl_posttraining.policies.dreamzero.server_base_model import (
    DreamZeroServerBaseModelPolicy,
)


def _make_client(server, **kwargs):
    """A client dialing the scripted in-process server."""
    return DreamZeroClient(
        "ws://fake-dreamzero:8000", open_connection=server.connect, **kwargs
    )


def _droid_obs(batched=True):
    """A DreamZero-DROID observation dict (batch-1 axes when ``batched``)."""
    rng = np.random.default_rng(0)

    def batch(arr):
        return arr[None] if batched else arr

    return {
        "observation/exterior_image_0_left": batch(
            rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
        ),
        "observation/exterior_image_1_left": batch(
            rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
        ),
        "observation/wrist_image_left": batch(
            rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
        ),
        "observation/joint_position": batch(np.zeros(7, dtype=np.float32)),
        "observation/cartesian_position": batch(np.zeros(6, dtype=np.float32)),
        "observation/gripper_position": batch(np.zeros(1, dtype=np.float32)),
        "prompt": "pick up the object",
    }


# ---------------------------------------------------------------------------
# Wire codec (msgpack-numpy, pinned to the reference scheme)
# ---------------------------------------------------------------------------


def test_codec_roundtrip_preserves_arrays_and_scalars():
    payload = {
        "image": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        "state": np.linspace(-1.0, 1.0, 7).astype(np.float32),
        "prompt": "brush the pan",
        "count": 3,
        "flag": True,
    }
    out = unpack_payload(pack_payload(payload))
    assert set(out) == set(payload)
    np.testing.assert_array_equal(out["image"], payload["image"])
    assert out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["state"], payload["state"])
    assert out["state"].dtype == np.float32
    assert (
        out["prompt"] == "brush the pan" and out["count"] == 3 and out["flag"] is True
    )


def test_codec_matches_reference_schema():
    """Both directions pin the reference msgpack-numpy map layout.

    Decoding a hand-built buffer with the reference keys must revive the
    array, and our own encoding, unpacked WITHOUT the numpy hook, must show
    exactly those keys — this is what makes the client interoperable with
    the real server, which uses the reference codec.
    """
    import msgpack

    reference = msgpack.packb(
        {
            b"__ndarray__": True,
            b"data": np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
            b"dtype": "<f4",
            b"shape": (3,),
        }
    )
    revived = unpack_payload(reference)
    np.testing.assert_array_equal(revived, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert revived.dtype == np.float32

    raw = msgpack.unpackb(pack_payload(np.array([4, 5], dtype=np.int64)))
    assert raw == {
        b"__ndarray__": True,
        b"data": np.array([4, 5], dtype=np.int64).tobytes(),
        b"dtype": "<i8",
        b"shape": [2],
    }


def test_codec_rejects_unsupported_dtypes():
    """Object and complex arrays must be refused, as the reference codec does."""
    with pytest.raises(ValueError):
        pack_payload({"bad": np.array([{"a": 1}], dtype=object)})
    with pytest.raises(ValueError):
        pack_payload({"bad": np.array([1 + 2j])})


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


def test_handshake_metadata_received_first():
    server = FakeDreamZeroServer()
    client = _make_client(server)
    metadata = client.metadata
    assert metadata["action_space"] == "joint_position"
    assert metadata["needs_session_id"] is True
    assert tuple(metadata["image_resolution"]) == (180, 320)
    assert server.connections == 1
    assert server.infer_requests == []  # nothing sent before the handshake


def test_infer_frames_carry_endpoint_and_roundtrip_obs():
    server = FakeDreamZeroServer()
    client = _make_client(server)
    obs = _droid_obs(batched=False)
    actions = client.infer(obs)
    assert actions.shape == (24, 8) and actions.dtype == np.float32

    # The recorded request has the endpoint popped (reference server behavior)
    # and the observation arrays intact.
    request = server.infer_requests[0]
    assert "endpoint" not in request
    np.testing.assert_array_equal(
        request["observation/exterior_image_0_left"],
        obs["observation/exterior_image_0_left"],
    )
    assert request["prompt"] == "pick up the object"

    # The raw frame on the wire says endpoint=infer, and the caller's dict
    # was not mutated by the client.
    assert "endpoint" not in obs
    # (one connection, one frame so far)
    assert server.connections == 1


def test_reset_endpoint_and_text_ack():
    server = FakeDreamZeroServer()
    client = _make_client(server)
    client.reset({"session_id": "episode-1"})
    assert server.reset_requests == [{"session_id": "episode-1"}]
    # The text acknowledgement is consumed without error, and the connection
    # keeps serving inference afterwards.
    assert client.infer(_droid_obs(batched=False)).shape == (24, 8)


def test_reset_error_text_frame_raises_and_drops_connection():
    """A failed reset must not silently leak recurrent policy state."""
    server = FakeDreamZeroServer()
    client = _make_client(server)
    server.fail_next_reset = (
        "Traceback (most recent call last):\n  ValueError: scripted"
    )
    with pytest.raises(DreamZeroServerError, match=r"(?s)reset error.*scripted"):
        client.reset({"session_id": "episode-1"})

    # The reference server closes after an error response; client state must
    # be dropped so the next request starts with a fresh handshake.
    assert client.infer(_droid_obs(batched=False)).shape == (24, 8)
    assert server.connections == 2


def test_server_error_text_frame_raises_and_drops_connection():
    server = FakeDreamZeroServer()
    client = _make_client(server)
    server.fail_next_infer = (
        "Traceback (most recent call last):\n  ValueError: scripted"
    )
    with pytest.raises(DreamZeroServerError, match="scripted"):
        client.infer(_droid_obs(batched=False))
    # The real server closes after sending the error; the client must re-dial.
    assert client.infer(_droid_obs(batched=False)).shape == (24, 8)
    assert server.connections == 2


def test_reconnect_on_dropped_connection():
    server = FakeDreamZeroServer()
    client = _make_client(server)
    assert client.metadata  # dial once
    server.drop_next_sends = 1
    actions = client.infer(_droid_obs(batched=False))
    assert actions.shape == (24, 8)
    assert server.connections == 2  # re-dialed with a fresh handshake


def test_reconnect_exhaustion_raises_connection_error():
    server = FakeDreamZeroServer()
    client = _make_client(server, max_reconnects=1)
    server.drop_next_sends = 10
    with pytest.raises(ConnectionError, match="unreachable"):
        client.infer(_droid_obs(batched=False))


def test_url_normalization():
    server = FakeDreamZeroServer()
    assert _make_client(server).server_url == "ws://fake-dreamzero:8000"
    bare = DreamZeroClient("host.example:9000", open_connection=server.connect)
    assert bare.server_url == "ws://host.example:9000"
    secure = DreamZeroClient("wss://host.example:9000", open_connection=server.connect)
    assert secure.server_url == "wss://host.example:9000"
    with pytest.raises(ValueError):
        DreamZeroClient("   ")


# ---------------------------------------------------------------------------
# Base model behavior
# ---------------------------------------------------------------------------


def _make_base_model(server=None, **base_model_kwargs):
    server = server or FakeDreamZeroServer()
    return server, DreamZeroServerBaseModelPolicy(
        _make_client(server), **base_model_kwargs
    )


def test_base_model_native_infer_shape_and_unbatching():
    server, base_model = _make_base_model()
    out = base_model.infer(_droid_obs(), None)
    assert set(out) == {"actions"}
    assert out["actions"].shape == (1, 24, 8)
    assert out["actions"].dtype == np.float32

    request = server.infer_requests[0]
    # The contract's batch-1 axis is stripped on the wire...
    assert request["observation/exterior_image_0_left"].shape == (180, 320, 3)
    assert request["observation/joint_position"].shape == (7,)
    # ... and the base model injects the episode's session id.
    assert request["session_id"] == base_model.session_id


def test_base_model_properties_and_capability():
    _, base_model = _make_base_model()
    assert base_model.action_horizon == 24
    assert base_model.action_dim == 8
    assert base_model.latent_dim == 8
    assert base_model.latent_horizon == 24
    assert base_model.supports_steering is False


def test_base_model_noise_raises_not_implemented():
    _, base_model = _make_base_model()
    for noise in (
        np.zeros((1, 8), dtype=np.float32),
        np.zeros((1, 4, 8), dtype=np.float32),
        np.zeros((1, 24, 8), dtype=np.float32),
    ):
        with pytest.raises(NotImplementedError, match="native-only"):
            base_model.infer(_droid_obs(), noise)
    # The refusal happens before any server traffic.
    server, base_model = _make_base_model()
    with pytest.raises(NotImplementedError):
        base_model.infer(_droid_obs(), np.zeros((1, 8), dtype=np.float32))
    assert server.infer_requests == []


def test_base_model_rejects_batched_observations():
    _, base_model = _make_base_model()
    obs = _droid_obs()
    obs["observation/joint_position"] = np.zeros((2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size 2"):
        base_model.infer(obs, None)


def test_base_model_prompt_list_is_unbatched():
    server, base_model = _make_base_model()
    obs = _droid_obs()
    obs["prompt"] = ["place the drill on the table"]
    base_model.infer(obs, None)
    assert server.infer_requests[0]["prompt"] == "place the drill on the table"


def test_base_model_validates_server_chunk_shape():
    server = FakeDreamZeroServer(action_horizon=10, action_dim=8)
    base_model = DreamZeroServerBaseModelPolicy(_make_client(server))  # expects 24 x 8
    with pytest.raises(ValueError, match=r"\(10, 8\).*\(24, 8\)"):
        base_model.infer(_droid_obs(), None)


def test_base_model_shape_configuration_follows_server():
    server = FakeDreamZeroServer(action_horizon=16, action_dim=35)
    base_model = DreamZeroServerBaseModelPolicy(
        _make_client(server), action_horizon=16, action_dim=35
    )
    out = base_model.infer(_droid_obs(), None)
    assert out["actions"].shape == (1, 16, 35)
    assert base_model.latent_dim == 35


def test_start_episode_resets_server_and_rotates_session():
    server, base_model = _make_base_model()
    first_session = base_model.session_id
    base_model.infer(_droid_obs(), None)
    base_model.start_episode()
    assert server.reset_requests == [{"session_id": first_session}]
    assert base_model.session_id != first_session
    base_model.infer(_droid_obs(), None)
    assert server.infer_requests[1]["session_id"] == base_model.session_id


def test_from_server_constructor():
    server = FakeDreamZeroServer()
    base_model = DreamZeroServerBaseModelPolicy.from_server(
        "ws://fake-dreamzero:8000",
        open_connection=server.connect,
    )
    assert base_model.client.server_url == "ws://fake-dreamzero:8000"
    assert base_model.infer(_droid_obs(), None)["actions"].shape == (1, 24, 8)


def test_invalid_action_geometry_rejected():
    server = FakeDreamZeroServer()
    with pytest.raises(ValueError):
        DreamZeroServerBaseModelPolicy(_make_client(server), action_horizon=0)
    with pytest.raises(ValueError):
        DreamZeroServerBaseModelPolicy(_make_client(server), action_dim=-1)


# ---------------------------------------------------------------------------
# Availability probe and registry wiring
# ---------------------------------------------------------------------------


def test_registry_gates_on_optional_deps():
    from isaac_rl_posttraining.policies import available_base_models, get_base_model

    assert ("dreamzero_server" in available_base_models()) == is_available()
    if is_available():
        assert get_base_model("dreamzero_server") is DreamZeroServerBaseModelPolicy
    else:
        with pytest.raises(ImportError, match="dreamzero_server"):
            get_base_model("dreamzero_server")
