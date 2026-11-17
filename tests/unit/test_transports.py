# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-transport tests: native TCP and ZeroMQ behind one contract.

Pins the :class:`~isaac_rl_posttraining.envs.transport.EnvTransport` seam on
in-process loopback servers, parameterized over every implemented transport
(``native`` TCP, ``zmq`` over TCP, ``zmq`` over ``ipc://``):

* request/response round-trips of the documented schema (RESET / STEP with
  single actions and action chunks / SPEC / CLOSE),
* the versioned SPEC handshake (protocol number, space metadata),
* large-frame integrity -- a native-resolution 480x640x3 uint8 ego frame and
  a [40, 35] float32 action chunk round-trip exactly,
* disconnect handling (a severed session raises ``TransportError`` and drops
  ``connected``) and reconnect via a fresh ``connect()``,
* bounded ``connect()`` retry against a dead endpoint,
* :func:`transport_from_uri` scheme dispatch, including the ImportError
  guidance when the ``[zmq]`` extra is missing.

ZeroMQ cases skip cleanly when ``pyzmq``/``msgpack`` are not installed.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
import time

import numpy as np
import pytest

from isaac_rl_posttraining.data import (
    ActionSpec,
    ArraySpec,
    ObservationSpec,
    SpecMismatch,
)
from isaac_rl_posttraining.envs.transport import (
    MSG_CLOSE,
    MSG_RESET,
    MSG_SPEC,
    MSG_STEP,
    PROTOCOL_VERSION,
    NativeTcpServer,
    NativeTcpTransport,
    TransportError,
    check_spec_handshake,
    handshake,
    spec_payload,
    transport_from_uri,
)

ZMQ_AVAILABLE = all(
    importlib.util.find_spec(mod) is not None
    for mod in ("zmq", "msgpack", "msgpack_numpy")
)

TRANSPORT_KINDS = ("native", "zmq-tcp", "zmq-ipc")

EGO_HW = (480, 640)
STATE_DIM = 43
CHUNK, ACTION_DIM = 40, 35


def _free_port() -> int:
    """A port that was free a moment ago (bound and released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_ego_frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, size=(*EGO_HW, 3), dtype=np.uint8
    )


def make_scripted_handler(record: list, ego_frame: np.ndarray, die_flag: dict):
    """A server-side handler implementing the documented schema.

    RESET/STEP return contract-shaped observation payloads; STEP echoes the
    received action back (``action_echo``) so tests can pin wire fidelity;
    SPEC returns the versioned handshake. Setting ``die_flag["die"] = True``
    makes the next request raise, severing the native per-client session
    (the transport must surface this as ``TransportError``).
    """

    def _obs_payload() -> dict:
        return {
            "ego_view": ego_frame,
            "state": np.arange(STATE_DIM, dtype=np.float32),
            "prompt": "pick up the drill",
        }

    def handler(message: dict) -> dict | None:
        if die_flag.get("die"):
            raise TransportError("scripted server failure")
        record.append(message)
        msg_type = message.get("type")
        if msg_type == MSG_RESET:
            payload = _obs_payload()
            payload["info"] = {
                "supports_action_chunking": True,
                "seed": message.get("seed"),
            }
            return payload
        if msg_type == MSG_STEP:
            action = np.asarray(message["action"], dtype=np.float32)
            payload = _obs_payload()
            payload.update(
                reward=float(action.sum()),
                done=False,
                truncated=False,
                success=False,
                num_steps=int(action.shape[0]) if action.ndim == 2 else 1,
                action_echo=action,
                info={},
            )
            return payload
        if msg_type == MSG_SPEC:
            return {
                "protocol": PROTOCOL_VERSION,
                "device": "cpu",
                "task": "ScriptedTask",
                "specs": spec_payload(
                    ObservationSpec(
                        {
                            "ego_view": ArraySpec((None, None, 3), "uint8"),
                            "state": ArraySpec((STATE_DIM,), "float32"),
                        },
                        frozenset({"prompt"}),
                    ),
                    ActionSpec(width=ACTION_DIM, dtype="float32"),
                ),
                "spaces": {
                    "observation": {
                        "ego_view": {"dtype": "uint8", "hw": list(EGO_HW)},
                        "state": {"dtype": "float32", "dim": STATE_DIM},
                    },
                    "action": {"dtype": "float32", "dim": ACTION_DIM},
                },
            }
        if msg_type == MSG_CLOSE:
            return None
        return {"error": f"unknown message type {msg_type}"}

    return handler


class TransportHarness:
    """One loopback server (+ its client URL) for a given transport kind."""

    def __init__(self, kind: str, tmp_path, ego_frame: np.ndarray):
        self.kind = kind
        self.tmp_path = tmp_path
        self.ego_frame = ego_frame
        self.record: list = []
        self.die_flag: dict = {}
        self._server = None
        self._thread = None
        self.url: str | None = None
        self.start()

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        handler = make_scripted_handler(self.record, self.ego_frame, self.die_flag)
        if self.kind == "native":
            port = None
            if self.url is not None:  # restart on the same endpoint
                port = int(self.url.rsplit(":", 1)[1])
            self._server = NativeTcpServer(host="127.0.0.1", port=port or 0)
            self.url = f"tcp://127.0.0.1:{self._server.address[1]}"
        else:
            from isaac_rl_posttraining.envs.zmq_transport import ZmqServer

            if self.kind == "zmq-tcp":
                bind = (
                    self.url.replace("zmq+tcp://", "tcp://")
                    if self.url
                    else "tcp://127.0.0.1:*"
                )
                self._server = ZmqServer(bind)
                host, port = self._server.address
                self.url = f"zmq+tcp://{host}:{port}"
            else:  # zmq-ipc
                ipc_path = self.tmp_path / "env_server.sock"
                self.url = f"ipc://{ipc_path}"
                self._server = ZmqServer(self.url)
        self._thread = threading.Thread(
            target=self._server.serve_forever, args=(handler,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        if self._thread is not None:
            self._thread.join(timeout=10)
            assert not self._thread.is_alive(), (
                f"{self.kind} server thread failed to stop"
            )
        self._server = None
        self._thread = None

    # -- failure injection ----------------------------------------------
    def interrupt(self) -> None:
        """Sever the transport's session (native) or kill the server (zmq)."""
        if self.kind == "native":
            self.die_flag["die"] = True
        else:
            self.stop()

    def resume(self) -> None:
        """Undo :meth:`interrupt` so a fresh ``connect()`` can succeed."""
        if self.kind == "native":
            self.die_flag["die"] = False
        else:
            self.start()


@pytest.fixture(params=TRANSPORT_KINDS)
def harness(request, tmp_path):
    if request.param != "native" and not ZMQ_AVAILABLE:
        pytest.skip("pyzmq/msgpack not installed ([zmq] extra)")
    h = TransportHarness(request.param, tmp_path, _make_ego_frame())
    yield h
    h.stop()


def make_transport(url: str, **kwargs):
    """Client transport for ``url`` with fast test-friendly timeouts."""
    kwargs.setdefault("request_timeout", 2.0)
    kwargs.setdefault("connect_timeout", 10.0)
    kwargs.setdefault("retry_interval", 0.2)
    return transport_from_uri(url, **kwargs)


@pytest.fixture
def transport(harness):
    t = make_transport(harness.url)
    yield t
    t.close()


# ---------------------------------------------------------------------------
# Round-trips and the schema
# ---------------------------------------------------------------------------


def test_reset_step_round_trip(harness, transport):
    transport.connect()
    assert transport.connected

    payload = transport.reset(seed=7)
    assert payload["info"]["seed"] == 7
    assert payload["info"]["supports_action_chunking"] is True
    np.testing.assert_array_equal(
        np.asarray(payload["state"], dtype=np.float32),
        np.arange(STATE_DIM, dtype=np.float32),
    )

    action = [0.5] * ACTION_DIM
    step = transport.step(action)
    assert step["reward"] == pytest.approx(0.5 * ACTION_DIM)
    assert step["num_steps"] == 1
    assert step["done"] is False and step["truncated"] is False

    reset_msg = next(m for m in harness.record if m["type"] == MSG_RESET)
    assert reset_msg["v"] == PROTOCOL_VERSION and reset_msg["seed"] == 7
    step_msg = next(m for m in harness.record if m["type"] == MSG_STEP)
    assert step_msg["v"] == PROTOCOL_VERSION and step_msg["need_obs"] is True


def test_action_chunk_step(harness, transport):
    transport.connect()
    chunk = np.full((CHUNK, ACTION_DIM), 0.25, dtype=np.float32)
    step = transport.step(chunk.tolist(), need_obs=False)
    assert step["num_steps"] == CHUNK
    assert step["reward"] == pytest.approx(float(chunk.sum()))
    step_msg = next(m for m in harness.record if m["type"] == MSG_STEP)
    assert step_msg["need_obs"] is False


def test_spec_handshake(harness, transport):
    transport.connect()
    spec = transport.spec()
    assert spec["protocol"] == PROTOCOL_VERSION
    assert spec["device"] == "cpu"
    assert spec["spaces"]["observation"]["state"]["dim"] == STATE_DIM
    assert spec["spaces"]["action"]["dim"] == ACTION_DIM
    if harness.kind != "native":
        # The zmq connect() handshake caches the server's SPEC payload.
        assert transport.server_spec.get("protocol") == PROTOCOL_VERSION


def test_spec_handshake_checks_what_the_client_reads_and_emits(harness, transport):
    transport.connect()
    reads = ObservationSpec(
        {
            "ego_view": ArraySpec((*EGO_HW, 3), "uint8"),
            "state": ArraySpec((STATE_DIM,), "float32"),
        },
        frozenset({"prompt"}),
    )

    comparison = handshake(
        transport, reads=reads, emits=ActionSpec(width=ACTION_DIM), context="scripted"
    )

    assert comparison.ok and transport.connected
    assert comparison.checks == (
        "observation 'ego_view'",
        "observation 'state'",
        "text 'prompt'",
        "action width",
    )
    # A client reading a wider state than the server emits is refused, and the connection is closed.
    narrow = ObservationSpec({"state": ArraySpec((STATE_DIM + 12,), "float32")})
    with pytest.raises(SpecMismatch, match="scripted: observation 'state': shape"):
        handshake(transport, reads=narrow, context="scripted")
    assert not transport.connected
    # So is a policy emitting a wider action than the server accepts.
    transport.connect()
    with pytest.raises(SpecMismatch, match="action: emitted float32\\[43\\]"):
        handshake(transport, emits=ActionSpec(width=43), context="scripted")
    assert not transport.connected


def test_spec_handshake_against_a_server_without_specs_is_unverified():
    comparison = check_spec_handshake(
        {"protocol": PROTOCOL_VERSION},
        reads=ObservationSpec({"state": ArraySpec((STATE_DIM,), "float32")}),
        emits=ActionSpec(width=ACTION_DIM),
    )
    assert comparison.ok and comparison.unverified == ("server declares no specs",)


# ---------------------------------------------------------------------------
# Large-frame integrity
# ---------------------------------------------------------------------------


def test_large_frame_integrity(harness, transport):
    """A native-resolution ego frame and a [40, 35] chunk round-trip exactly."""
    transport.connect()

    payload = transport.reset()
    ego = np.asarray(payload["ego_view"])
    assert ego.shape == (*EGO_HW, 3) and ego.dtype == np.uint8
    np.testing.assert_array_equal(ego, harness.ego_frame)

    rng = np.random.default_rng(1)
    chunk = rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32)
    step = transport.step(chunk.tolist())
    echo = np.asarray(step["action_echo"], dtype=np.float32)
    assert echo.shape == (CHUNK, ACTION_DIM)
    np.testing.assert_array_equal(echo, chunk)  # exact, not approximate


# ---------------------------------------------------------------------------
# Disconnects, reconnects, timeouts
# ---------------------------------------------------------------------------


def test_disconnect_then_reconnect(harness):
    t = make_transport(harness.url, request_timeout=1.0)
    try:
        t.connect()
        assert t.step([0.0] * ACTION_DIM)["num_steps"] == 1

        harness.interrupt()
        with pytest.raises(TransportError):
            t.step([0.0] * ACTION_DIM)
        assert not t.connected, "a failed request must drop the connection"

        harness.resume()
        t.connect()
        assert t.connected
        assert t.step([0.0] * ACTION_DIM)["num_steps"] == 1
    finally:
        t.close()


def test_request_without_connect_raises(harness):
    t = make_transport(harness.url)
    with pytest.raises(TransportError, match="connect"):
        t.step([0.0] * ACTION_DIM)
    t.close()


def test_close_is_idempotent(harness, transport):
    transport.connect()
    transport.close()
    assert not transport.connected
    transport.close()  # second close: no error


@pytest.mark.parametrize("scheme", ["tcp", "zmq+tcp"])
def test_connect_timeout_against_dead_endpoint(scheme):
    if scheme == "zmq+tcp" and not ZMQ_AVAILABLE:
        pytest.skip("pyzmq/msgpack not installed ([zmq] extra)")
    url = f"{scheme}://127.0.0.1:{_free_port()}"
    t = transport_from_uri(url, connect_timeout=0.5, retry_interval=0.1)
    start = time.monotonic()
    with pytest.raises(TransportError, match="could not connect"):
        t.connect()
    assert time.monotonic() - start < 10.0
    t.close()


# ---------------------------------------------------------------------------
# URI dispatch
# ---------------------------------------------------------------------------


def test_transport_from_uri_dispatch(tmp_path):
    t = transport_from_uri("tcp://127.0.0.1:6000")
    assert isinstance(t, NativeTcpTransport)
    assert (t.host, t.port) == ("127.0.0.1", 6000)

    with pytest.raises(ValueError, match="no transport registered"):
        transport_from_uri("carrier-pigeon://loft:1")

    for uri in (f"ipc://{tmp_path}/x.sock", "zmq+tcp://127.0.0.1:6000"):
        if ZMQ_AVAILABLE:
            from isaac_rl_posttraining.envs.zmq_transport import ZmqTransport

            zmq_t = transport_from_uri(uri)
            assert isinstance(zmq_t, ZmqTransport)
            zmq_t.close()
        else:
            with pytest.raises(ImportError, match=r"\[zmq\]"):
                transport_from_uri(uri)


def test_zmq_uri_normalization():
    if not ZMQ_AVAILABLE:
        pytest.skip("pyzmq/msgpack not installed ([zmq] extra)")
    from isaac_rl_posttraining.envs.zmq_transport import ZmqTransport, parse_zmq_uri

    assert parse_zmq_uri("zmq+tcp://sim-host:6000") == "tcp://sim-host:6000"
    assert parse_zmq_uri("ipc:///tmp/a.sock") == "ipc:///tmp/a.sock"
    with pytest.raises(ValueError):
        parse_zmq_uri("zmq+tcp://nohost")
    t = ZmqTransport("zmq+tcp://127.0.0.1:6000")
    assert t.endpoint == "tcp://127.0.0.1:6000"
    t.close()


def test_zmq_frames_refuse_pickled_object_arrays():
    if not ZMQ_AVAILABLE:
        pytest.skip("pyzmq/msgpack not installed ([zmq] extra)")
    from isaac_rl_posttraining.envs.zmq_transport import pack_frame, unpack_frame

    state = np.arange(3, dtype=np.float32)
    np.testing.assert_array_equal(
        unpack_frame(pack_frame({"state": state}))["state"], state
    )
    with pytest.raises(TransportError, match="object arrays"):
        unpack_frame(pack_frame({"task": np.array(["lift"], dtype=object)}))


def test_arena_env_over_every_transport(harness):
    """The Gymnasium client works unchanged over each transport (the seam's
    whole point): reset/step observation formatting, success propagation,
    and the client-side episode cap."""
    from simulators.g1_arena.env import ArenaEnv

    env = ArenaEnv(
        server_url=harness.url,
        image_resolution=0,
        image_hw=EGO_HW,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_episode_steps=2,
        socket_timeout=5.0,
        connect_timeout=10.0,
        retry_interval=0.2,
    )
    try:
        obs, info = env.reset(seed=3)
        assert obs["ego_view_image"].shape == (*EGO_HW, 3)
        assert obs["ego_view_image"].dtype == np.uint8
        np.testing.assert_array_equal(obs["ego_view_image"], harness.ego_frame)
        assert obs["state"].dtype == np.float32
        assert info["supports_action_chunking"] is True

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, _reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert info["is_success"] is False

        # Second step hits the client-side episode cap (max_episode_steps=2).
        obs, _reward, terminated, truncated, info = env.step(action)
        assert truncated and not terminated
    finally:
        env.close()


def test_arena_env_disconnect_is_not_a_task_truncation(harness):
    """A server failure must escape instead of becoming a scored episode."""
    from simulators.g1_arena.env import ArenaEnv

    env = ArenaEnv(
        server_url=harness.url,
        image_resolution=0,
        image_hw=EGO_HW,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        socket_timeout=2.0,
        connect_timeout=3.0,
        retry_interval=0.1,
    )
    try:
        env.reset(seed=3)
        harness.interrupt()
        with pytest.raises(RuntimeError, match="transition was not recorded"):
            env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    finally:
        env.close()


def test_env_server_script_transport_flag():
    """The Arena server script defaults to the native transport and builds
    the right server binding per flag (the cluster campaign depends on the
    stock default)."""
    import importlib.util as _ilu
    import os
    import sys

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "simulators",
        "g1_arena",
        "env_server.py",
    )
    spec = _ilu.spec_from_file_location("arena_env_server_under_test", script)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = sys.argv
    try:
        sys.argv = ["arena_env_server.py"]
        assert module.parse_args().transport == "native"
        sys.argv = ["arena_env_server.py", "--transport", "zmq"]
        assert module.parse_args().transport == "zmq"
    finally:
        sys.argv = argv

    server = module.make_server("native", "127.0.0.1", 0)
    try:
        assert isinstance(server, NativeTcpServer)
    finally:
        server.close()

    if ZMQ_AVAILABLE:
        from isaac_rl_posttraining.envs.zmq_transport import ZmqServer

        server = module.make_server("zmq", "127.0.0.1", 0)
        try:
            assert isinstance(server, ZmqServer)
        finally:
            server.close()

    with pytest.raises(SystemExit):
        module.make_server("smoke-signals", "127.0.0.1", 0)


def test_native_transport_reports_where_the_round_trip_went(harness, transport):
    """The native client splits each exchange into encode/send and recv/decode; ZeroMQ reports nothing."""
    from isaac_rl_posttraining.envs.transport import NativeTcpTransport

    transport.connect()
    transport.reset()
    if not isinstance(transport, NativeTcpTransport):
        assert transport.last_timing == {}
        return
    assert set(transport.last_timing) == {"encode_s", "send_s", "recv_s", "decode_s"}
    assert all(v >= 0.0 for v in transport.last_timing.values())
    # a fresh dict per request: the previous exchange's numbers do not linger
    first = transport.last_timing
    transport.step([[0.0] * 35])
    assert transport.last_timing is not first
