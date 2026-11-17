# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZeroMQ implementation of the environment transport seam.

This module provides transport #2 behind the
:class:`~isaac_rl_posttraining.envs.transport.EnvTransport` interface:
request/response over a ZeroMQ REQ/REP socket pair, carrying
msgpack(+numpy) frames instead of the native transport's pickle frames.
The *message schema* is byte-for-byte the same set of dictionaries
documented in :mod:`isaac_rl_posttraining.envs.transport` (``RESET`` /
``STEP`` / ``SPEC`` / ``SUCCESS_CHECK`` / ``CLOSE``, all versioned by
``PROTOCOL_VERSION``); only the frame encoding and the socket machinery
differ.

Why a second transport exists:

* ``ipc://`` endpoints -- a learner and simulator sharing a node can skip
  the TCP stack entirely (this is the ``ipc://`` scheme that
  :func:`~isaac_rl_posttraining.envs.transport.transport_from_uri`
  previously reserved).
* msgpack-numpy frames -- a self-describing, cross-language encoding
  (identical scheme to the DreamZero policy protocol and
  ``openpi_client``), safer than pickle across trust boundaries.
* connection supervision -- ZeroMQ reconnects under the hood, so the
  client-side liveness probe below is what defines "connected".

Dependencies (``pyzmq``, ``msgpack``, ``msgpack-numpy``) are optional and
imported lazily: install the ``[zmq]`` extra
(``pip install "isaac-rl-posttraining[zmq]"``). Importing this module
without them succeeds; constructing a transport or server raises
``ImportError`` with that hint.

Wire format (ZeroMQ transport)
------------------------------

Every message is one ZeroMQ frame holding one msgpack document. Numpy
arrays and scalars are encoded with the msgpack-numpy scheme, so
observations round-trip with exact dtype and shape (no pickle involved).
Payloads are the schema dictionaries of
:mod:`~isaac_rl_posttraining.envs.transport`, unchanged.

Endpoint URIs (see :func:`transport_from_uri`):

* ``ipc://<path>`` -- same-node UNIX domain socket (ZeroMQ transport).
* ``zmq+tcp://host:port`` -- cross-node TCP carried by ZeroMQ.
* ``tcp://host:port`` -- remains the *native* pickle transport; the
  default cluster behavior is unchanged.

Handshake, liveness, and reconnect semantics
--------------------------------------------

ZeroMQ connects are asynchronous (``connect()`` on a dead endpoint does
not fail), so :meth:`ZmqTransport.connect` defines connectivity as *the
server answered a versioned* ``SPEC`` *handshake*: it sends ``SPEC`` and
waits up to ``retry_interval`` for the reply, retrying until
``connect_timeout`` -- mirroring the native transport's dial-retry loop.
The answered handshake is cached on the instance as ``server_spec``.

A REQ socket is strictly lockstep (send then receive); a request that
times out or errors leaves it unusable, so the transport drops the socket
(``connected`` becomes ``False``) and raises
:class:`~isaac_rl_posttraining.envs.transport.TransportError` -- exactly
the failure surface :class:`~simulators.g1_arena.env.ArenaEnv`
already handles by ending the episode and re-``connect()``-ing (with a
fresh handshake) on the next call. ``reset`` waits without a timeout, as
in the native transport, because scene (re)builds can take arbitrarily
long.

One asymmetry to know about: REP sockets have no per-connection state, so
:class:`ZmqServer` serves any number of sequential clients on one socket
and cannot observe connects/disconnects (the ``on_connect`` /
``on_disconnect`` callbacks of the native server are accepted for
signature parity but never fire). ``CLOSE`` is acknowledged with a
``CLOSE``-typed frame to keep the REQ/REP state machine valid.
"""

from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import urlparse

# No loguru here: this module is imported by the sim-side env-server, which runs
# under the minimal isaac-sim python (no loguru); its few diagnostics use print.

from isaac_rl_posttraining.envs.transport import (
    MSG_CLOSE,
    MSG_RESET,
    MSG_SPEC,
    MSG_STEP,
    PROTOCOL_VERSION,
    EnvTransport,
    TransportError,
)

__all__ = ["ZmqServer", "ZmqTransport", "pack_frame", "unpack_frame"]

_INSTALL_HINT = "install the [zmq] extra: pip install \"isaac-rl-posttraining[zmq]\""


def _require_zmq() -> Any:
    """Import ``zmq`` lazily so this module imports without the extra.

    Raises:
        ImportError: If ``pyzmq`` is not installed.
    """
    try:
        import zmq
    except ImportError as exc:
        raise ImportError(f"The ZeroMQ transport needs 'pyzmq'; {_INSTALL_HINT}") from exc
    return zmq


def _require_msgpack() -> tuple[Any, Any]:
    """Import ``msgpack`` and ``msgpack_numpy`` lazily.

    Raises:
        ImportError: If either package is not installed.
    """
    try:
        import msgpack
        import msgpack_numpy
    except ImportError as exc:
        raise ImportError(
            f"The ZeroMQ transport needs 'msgpack' and 'msgpack-numpy'; {_INSTALL_HINT}"
        ) from exc
    return msgpack, msgpack_numpy


def pack_frame(payload: dict[str, Any]) -> bytes:
    """Encode one schema message to a msgpack(+numpy) frame."""
    msgpack, msgpack_numpy = _require_msgpack()
    return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)


def unpack_frame(data: bytes) -> dict[str, Any]:
    """Decode one msgpack(+numpy) frame back to a schema message.

    Raises:
        TransportError: If the frame is not a well-formed msgpack document.
    """
    msgpack, msgpack_numpy = _require_msgpack()

    def decode(obj: dict[Any, Any]) -> Any:
        if obj.get(b"kind") == b"O":  # msgpack_numpy unpickles object arrays
            raise TransportError("object arrays are not accepted in a frame")
        return msgpack_numpy.decode(obj)

    try:
        return msgpack.unpackb(data, object_hook=decode, raw=False)
    except Exception as exc:  # malformed frame: unpack errors vary by payload
        raise TransportError(f"failed to decode frame: {exc}") from exc


def parse_zmq_uri(uri: str) -> str:
    """Normalize a ZeroMQ endpoint URI to the form ``zmq_socket.connect`` takes.

    Args:
        uri: ``ipc://<path>`` or ``zmq+tcp://host:port`` (a plain
            ``tcp://host:port`` is also accepted when constructing
            :class:`ZmqTransport` directly).

    Returns:
        The ZeroMQ endpoint string (``ipc://...`` or ``tcp://...``).

    Raises:
        ValueError: If the URI is not a recognized ZeroMQ endpoint.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "ipc":
        # urlparse splits ipc://tmp/x into netloc="tmp"; keep the raw path.
        path = uri[len("ipc://"):]
        if not path:
            raise ValueError(f"expected ipc://<path>, got {uri!r}")
        return f"ipc://{path}"
    if parsed.scheme == "zmq+tcp":
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"expected zmq+tcp://host:port, got {uri!r}")
        return f"tcp://{parsed.hostname}:{parsed.port}"
    if parsed.scheme == "tcp":
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"expected tcp://host:port, got {uri!r}")
        return uri
    raise ValueError(f"no ZeroMQ endpoint form for scheme {parsed.scheme!r} (from {uri!r})")


class ZmqTransport(EnvTransport):
    """``EnvTransport`` over a ZeroMQ REQ socket with msgpack(+numpy) frames.

    Args:
        uri: Endpoint URI: ``ipc://<path>``, ``zmq+tcp://host:port``, or a
            plain ZeroMQ ``tcp://host:port``.
        request_timeout: Seconds to wait for a ``step``/``spec`` response
            before declaring the connection dead. ``reset`` is exempt (it
            waits without a timeout, matching the native transport).
        connect_timeout: Total budget in seconds for :meth:`connect` to
            keep re-sending the ``SPEC`` handshake before raising.
            Defaults to unbounded so a learner can start before its
            simulator.
        retry_interval: Seconds each handshake attempt waits for a reply
            (and therefore the spacing between attempts).
    """

    def __init__(
        self,
        uri: str,
        request_timeout: float = 120.0,
        connect_timeout: float = float("inf"),
        retry_interval: float = 5.0,
    ) -> None:
        self._zmq = _require_zmq()
        _require_msgpack()  # fail at construction, not first use
        self.endpoint = parse_zmq_uri(uri)
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.retry_interval = float(retry_interval)
        self.last_timing = {}  # the ZeroMQ client does not split its round trip
        self.server_spec: dict[str, Any] = {}
        self._ctx: Any = None
        self._sock: Any = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        """Handshake with the server (``SPEC``), retrying until the budget ends.

        Raises:
            TransportError: If no handshake reply arrives within
                ``connect_timeout``.
        """
        if self._sock is not None:
            return
        if self._ctx is None:
            self._ctx = self._zmq.Context()
        start = time.monotonic()
        attempt_ms = max(int(self.retry_interval * 1000), 1)
        while True:
            sock = self._ctx.socket(self._zmq.REQ)
            sock.setsockopt(self._zmq.LINGER, 0)
            try:
                sock.connect(self.endpoint)
                sock.send(pack_frame({"v": PROTOCOL_VERSION, "type": MSG_SPEC}))
                if sock.poll(attempt_ms, self._zmq.POLLIN):
                    payload = unpack_frame(sock.recv())
                    self._sock = sock
                    self.server_spec = {} if "error" in payload else payload
                    print(f"[ZmqTransport] connected to {self.endpoint}", flush=True)
                    return
                failure = "handshake timed out"
            except self._zmq.ZMQError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            sock.close(0)
            elapsed = time.monotonic() - start
            if elapsed > self.connect_timeout:
                raise TransportError(
                    f"could not connect to {self.endpoint} after {elapsed:.0f}s: {failure}"
                )
            print(
                f"[ZmqTransport] connect to {self.endpoint} failed ({failure}); retrying",
                flush=True,
            )

    def spec(self) -> dict[str, Any]:
        response = self._request({"v": PROTOCOL_VERSION, "type": MSG_SPEC})
        if "error" in response:
            # Servers predating SPEC answer with an error payload; treat as "no spec".
            return {}
        return response

    def reset(self, seed: int | None = None, env_ids: list[int] | None = None) -> dict[str, Any]:
        message: dict[str, Any] = {"v": PROTOCOL_VERSION, "type": MSG_RESET, "seed": seed}
        if env_ids is not None:
            # Batched servers reset only these envs (rolling collection);
            # single-env servers never receive the field.
            message["env_ids"] = [int(i) for i in env_ids]
        return self._request(message, unbounded_wait=True)

    def step(self, action: Any, need_obs: bool = True) -> dict[str, Any]:
        return self._request(
            {"v": PROTOCOL_VERSION, "type": MSG_STEP, "action": action, "need_obs": bool(need_obs)}
        )

    def close(self) -> None:
        """Send ``CLOSE`` (briefly waiting for its ack) and release the socket."""
        if self._sock is not None:
            try:
                self._sock.send(pack_frame({"v": PROTOCOL_VERSION, "type": MSG_CLOSE}))
                # Consume the ack so the server's REP state machine stays clean;
                # a dead server just means the poll expires.
                if self._sock.poll(1000, self._zmq.POLLIN):
                    self._sock.recv()
            except self._zmq.ZMQError:
                pass  # closing a dead connection is fine
        self._drop()
        if self._ctx is not None:
            try:
                self._ctx.term()
            except self._zmq.ZMQError:
                pass
            self._ctx = None

    def _drop(self) -> None:
        """Close the socket without notifying the server (connection is dead)."""
        if self._sock is not None:
            try:
                self._sock.close(0)
            except self._zmq.ZMQError:
                pass
            self._sock = None

    def _request(self, message: dict[str, Any], unbounded_wait: bool = False) -> dict[str, Any]:
        """One request/response round-trip; drops the connection on failure."""
        if self._sock is None:
            raise TransportError("not connected; call connect() first")
        try:
            self._sock.send(pack_frame(message))
            timeout_ms = None if unbounded_wait else max(int(self.request_timeout * 1000), 1)
            if self._sock.poll(timeout_ms, self._zmq.POLLIN) == 0:
                raise TransportError(
                    f"no response from {self.endpoint} within {self.request_timeout:.0f}s"
                )
            return unpack_frame(self._sock.recv())
        except TransportError:
            self._drop()
            raise
        except self._zmq.ZMQError as exc:
            self._drop()
            raise TransportError(str(exc)) from exc


class ZmqServer:
    """Server-side binding of the ZeroMQ transport.

    Owns the REP socket and the request loop; environment semantics live in
    the ``handler`` passed to :meth:`serve_forever` -- the same handler
    contract as :class:`~isaac_rl_posttraining.envs.transport.NativeTcpServer`.
    One request is served at a time (REP is strictly lockstep), which
    matches simulators that cannot be shared; unlike the native server,
    successive clients need no re-accept, they simply start sending.

    ZeroMQ sockets are not thread-safe, so shutdown is flag-based: the
    serve loop polls with a short timeout and re-checks the flag, and the
    socket is only ever touched from the serving thread once
    :meth:`serve_forever` is running. :meth:`close` may therefore be
    called from any thread (e.g. a signal handler).

    Args:
        bind_uri: Endpoint to bind: ``tcp://host:port`` (``port`` may be
            ``*`` for an ephemeral port), ``zmq+tcp://host:port``, or
            ``ipc://<path>``.
    """

    #: Poll granularity of the serve loop; bounds shutdown latency.
    POLL_INTERVAL_MS = 200

    def __init__(self, bind_uri: str) -> None:
        self._zmq = _require_zmq()
        _require_msgpack()
        parsed = urlparse(bind_uri)
        if parsed.scheme == "zmq+tcp":
            bind_uri = f"tcp://{bind_uri[len('zmq+tcp://'):]}"
        self._ctx = self._zmq.Context()
        self._sock = self._ctx.socket(self._zmq.REP)
        self._sock.setsockopt(self._zmq.LINGER, 0)
        self._sock.bind(bind_uri)
        self._closing = False
        self._serving = False

    @property
    def endpoint(self) -> str:
        """The bound endpoint (with the real port when ``*`` was requested)."""
        return self._sock.getsockopt(self._zmq.LAST_ENDPOINT).decode()

    @property
    def address(self) -> tuple[str, int]:
        """``(host, port)`` parity helper for ``tcp`` endpoints.

        Raises:
            ValueError: For non-TCP endpoints (``ipc://`` has no port).
        """
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "tcp" or parsed.port is None:
            raise ValueError(f"no (host, port) address for endpoint {self.endpoint!r}")
        return str(parsed.hostname), int(parsed.port)

    def serve_forever(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any] | None],
        on_connect: Callable[[tuple[Any, ...]], None] | None = None,
        on_disconnect: Callable[[tuple[Any, ...], Exception | None], None] | None = None,
        on_response: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Serve requests until :meth:`close` is called (or the process exits).

        ``on_response`` is accepted for interface parity with the native
        server and never invoked: the ZeroMQ server does not time its replies.

        Args:
            handler: Called with each decoded request; its return value is
                sent back as the response. Returning ``None`` (the ``CLOSE``
                path) sends a ``CLOSE``-typed acknowledgement -- REP must
                answer every request -- and the loop keeps serving so the
                next client can connect.
            on_connect: Accepted for signature parity with the native
                server; never called (REP sees no connections).
            on_disconnect: Accepted for signature parity; never called.

        Exceptions raised by ``handler`` (e.g. ``SystemExit`` from a SIGTERM
        handler) propagate to the caller so the process can shut down
        cleanly. A frame that fails to decode is answered with an
        ``{"error": ...}`` payload instead of poisoning the socket.
        """
        del on_connect, on_disconnect  # no connection notion on a REP socket
        self._serving = True
        try:
            while not self._closing:
                try:
                    if not self._sock.poll(self.POLL_INTERVAL_MS, self._zmq.POLLIN):
                        continue
                    data = self._sock.recv()
                except self._zmq.ZMQError:
                    return  # socket/context closed: shutdown requested
                try:
                    message = unpack_frame(data)
                    response = handler(message)
                except TransportError as exc:
                    response = {"error": str(exc)}
                if response is None:
                    response = {"v": PROTOCOL_VERSION, "type": MSG_CLOSE}
                try:
                    self._sock.send(pack_frame(response))
                except self._zmq.ZMQError:
                    return  # closed while replying
        finally:
            self._serving = False
            self._release()

    def close(self) -> None:
        """Request shutdown (idempotent; callable from any thread).

        When :meth:`serve_forever` is running, this only raises the flag --
        the serving thread notices within :data:`POLL_INTERVAL_MS` and
        releases the socket itself. Otherwise the socket is released here.
        """
        self._closing = True
        if not self._serving:
            self._release()

    def _release(self) -> None:
        """Close the socket and terminate the private context (idempotent)."""
        if self._sock is not None:
            try:
                self._sock.close(0)
            except self._zmq.ZMQError:
                pass
            self._sock = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except self._zmq.ZMQError:
                pass
            self._ctx = None
