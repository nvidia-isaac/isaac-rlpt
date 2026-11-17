# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport layer for driving remote environments: simulators, real robots, data.

The learner process and its environment endpoints usually live in different
Python environments, and often on different nodes: an Isaac Lab Arena
simulator, a real-robot bridge, or a data server may each sit behind a
network boundary. Environments are therefore driven over a small
request/response protocol, and this module defines the transport seam:

* :class:`EnvTransport` -- the client-side interface (``connect`` / ``spec`` /
  ``reset`` / ``step`` / ``close``). Environment classes such as
  :class:`simulators.g1_arena.env.ArenaEnv` are written against
  this interface, so adding a new transport only requires a new
  ``EnvTransport`` subclass and a scheme entry in :func:`transport_from_uri`
  (the ZeroMQ implementation in
  :mod:`~isaac_rl_posttraining.envs.zmq_transport` is the worked example).
* :class:`NativeTcpTransport` -- implementation #1: a plain TCP socket
  carrying length-prefixed pickle frames.
* :class:`NativeTcpServer` -- the matching server-side binding, used by
  ``simulators/g1_arena/env_server.py``. It owns the listening socket and the
  per-client receive/dispatch loop; the environment logic is supplied as a
  handler callable.

Endpoints are addressed by connection URI so callers never hard-code a
transport:

* ``tcp://host:port`` -- cross-node stream socket (implemented here).
* ``ipc://path`` -- same-node UNIX domain socket, served by the ZeroMQ
  transport (:mod:`isaac_rl_posttraining.envs.zmq_transport`; needs the
  optional ``[zmq]`` extra).
* ``zmq+tcp://host:port`` -- cross-node TCP carried by ZeroMQ with
  msgpack(+numpy) frames, for endpoints that should not exchange pickle.

Because remote endpoints cannot be introspected like in-process objects, the
``SPEC`` handshake must be self-describing: servers report their protocol
version, compute device, and observation/action space metadata so a client on
another node can size its models without importing the server's stack.

Wire format (native TCP transport)
----------------------------------

Every message is one frame::

    +----------------------+----------------------------+
    | 4 bytes, big endian  | N bytes                    |
    | unsigned length N    | pickle payload (protocol 4)|
    +----------------------+----------------------------+

Pickle protocol 4 keeps frames readable across the CPython versions used by
the learner and the simulator (3.10 through 3.12). Payloads are dictionaries.

Requests (client to server) carry a schema version ``"v"`` and a ``"type"``:

======================  =====================================================
``type``                remaining fields
======================  =====================================================
``"RESET"``             ``"seed"``: ``int | None``
``"STEP"``              ``"action"``: nested ``list`` of floats (a single
                        action ``[action_dim]`` or a chunk
                        ``[chunk, action_dim]``); ``"need_obs"``: ``bool``
``"SPEC"``              --
``"SUCCESS_CHECK"``     --
``"CLOSE"``             --
======================  =====================================================

Responses (server to client):

======================  =====================================================
request                 response payload
======================  =====================================================
``"RESET"``             observation keys (server specific, e.g. ``"ego_view"``,
                        ``"state"``, ``"prompt"``) plus ``"info"``: ``dict``
``"STEP"``              observation keys plus ``"reward"``: ``float``,
                        ``"done"``: ``bool``, ``"truncated"``: ``bool``,
                        ``"success"``: ``bool``, optionally ``"num_steps"``:
                        ``int`` (simulator steps consumed by a chunked
                        action), and ``"info"``: ``dict``
``"SPEC"``              ``"protocol"``: ``int``, ``"device"``: ``str``,
                        ``"spaces"``: ``dict`` describing the observation and
                        action spaces (shapes/dtypes), ``"specs"``: the
                        typed declaration of what the server emits and
                        accepts (:func:`spec_payload`), plus static endpoint
                        metadata (task name, embodiment, image size, ...)
``"SUCCESS_CHECK"``     ``"done"``: ``bool``
unknown ``type``        ``"error"``: ``str``
======================  =====================================================

Compatibility rules: servers ignore request fields they do not understand,
clients tolerate missing optional response fields, and any breaking change to
the schema bumps :data:`PROTOCOL_VERSION`.

The spec handshake
------------------

A server that declares ``"specs"`` in its ``SPEC`` response lets a client check,
before the first episode, that the observation it will read and the action it
will send match what the server emits and accepts. :func:`handshake` runs the
same comparison the launch plan runs and closes the connection on a mismatch;
against a server that predates the declaration the result says ``unverified``
rather than passing silently.
"""

from __future__ import annotations

import abc
import pickle
import socket
import struct
import time
from collections.abc import Mapping
from typing import Any, Callable
from urllib.parse import urlparse

from isaac_rl_posttraining.data.spec import (
    ActionSpec,
    ObservationSpec,
    SpecComparison,
    compare_action_specs,
    compare_observation_specs,
)

# No loguru or torch here: this module is imported by the sim-side env-server,
# which runs under the minimal isaac-sim python; its few diagnostics use print,
# and the spec records it exchanges depend on NumPy only.

PROTOCOL_VERSION = 1
"""Version of the request/response schema documented above."""

PICKLE_PROTOCOL = 4
"""Highest pickle protocol shared by all supported interpreter versions."""

MSG_RESET = "RESET"
MSG_STEP = "STEP"
MSG_SPEC = "SPEC"
MSG_SUCCESS_CHECK = "SUCCESS_CHECK"
MSG_CLOSE = "CLOSE"

SPECS_KEY = "specs"
"""The ``SPEC`` response entry carrying the server's typed observation and action specs."""

_LENGTH_STRUCT = struct.Struct(">I")


class TransportError(ConnectionError):
    """Raised when a transport cannot reach, or loses, its peer."""


def spec_payload(observation: ObservationSpec, action: ActionSpec) -> dict[str, Any]:
    """The ``specs`` entry of a ``SPEC`` response: what the server emits and what it accepts per step."""
    return {"observation": observation.to_json(), "action": action.to_json()}


def check_spec_handshake(
    spec: Mapping[str, Any],
    *,
    reads: ObservationSpec | None = None,
    emits: ActionSpec | None = None,
) -> SpecComparison:
    """Compare a server's ``SPEC`` response with the observation this client reads and the action it emits.

    A server that declares no ``specs`` yields an ``unverified`` comparison
    rather than a passing one, so the report always says what it could not check.
    """
    declared = spec.get(SPECS_KEY)
    if declared is None:
        return SpecComparison(unverified=("server declares no specs",))
    comparison = SpecComparison()
    if reads is not None:
        comparison = comparison.merged(
            compare_observation_specs(ObservationSpec.from_json(declared["observation"]), reads)
        )
    if emits is not None:
        comparison = comparison.merged(compare_action_specs(emits, ActionSpec.from_json(declared["action"])))
    return comparison


def handshake(
    transport: EnvTransport,
    *,
    reads: ObservationSpec | None = None,
    emits: ActionSpec | None = None,
    context: str,
) -> SpecComparison:
    """Run the spec handshake on a connected transport; a mismatch closes it and raises :class:`SpecMismatch`."""
    comparison = check_spec_handshake(transport.spec(), reads=reads, emits=emits)
    if not comparison.ok:
        transport.close()
        comparison.raise_on_mismatch(context)
    return comparison


def parse_tcp_uri(uri: str) -> tuple[str, int]:
    """Split a ``tcp://host:port`` URI into its host and port.

    Args:
        uri: The endpoint URI.

    Returns:
        The ``(host, port)`` pair.

    Raises:
        ValueError: If the URI is not a well-formed ``tcp://host:port``.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
        raise ValueError(f"expected tcp://host:port, got {uri!r}")
    return parsed.hostname, parsed.port


def transport_from_uri(uri: str, **kwargs: Any) -> "EnvTransport":
    """Build the :class:`EnvTransport` implementation for an endpoint URI.

    Args:
        uri: Connection URI. ``tcp://host:port`` maps to
            :class:`NativeTcpTransport`; ``ipc://path`` and
            ``zmq+tcp://host:port`` map to
            :class:`~isaac_rl_posttraining.envs.zmq_transport.ZmqTransport`.
        **kwargs: Forwarded to the transport constructor (timeouts etc.).

    Returns:
        A transport instance (not yet connected).

    Raises:
        ValueError: If the URI scheme has no registered transport.
        ImportError: For ZeroMQ schemes when the ``[zmq]`` extra is not
            installed.
    """
    scheme = urlparse(uri).scheme
    if scheme == "tcp":
        host, port = parse_tcp_uri(uri)
        return NativeTcpTransport(host=host, port=port, **kwargs)
    if scheme in ("ipc", "zmq+tcp"):
        # Function-level import: zmq_transport subclasses EnvTransport, so a
        # module-level import here would be circular.
        from isaac_rl_posttraining.envs.zmq_transport import ZmqTransport

        return ZmqTransport(uri, **kwargs)
    raise ValueError(
        f"no transport registered for scheme {scheme!r} (from {uri!r}); "
        "implemented transports: tcp:// (native), ipc:// and zmq+tcp:// (ZeroMQ)"
    )


def send_frame(sock: socket.socket, payload: dict[str, Any], timing: dict[str, float] | None = None) -> None:
    """Serialize ``payload`` and write it to ``sock`` as one length-prefixed frame.

    Args:
        sock: A connected stream socket.
        payload: The message dictionary to send.
        timing: If given, receives ``encode_s`` (serialization) and ``send_s``
            (the blocking write) for this frame.

    Raises:
        TransportError: If the socket write fails.
    """
    t_start = time.perf_counter()
    data = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
    t_encoded = time.perf_counter()
    try:
        sock.sendall(_LENGTH_STRUCT.pack(len(data)) + data)
    except OSError as exc:
        raise TransportError(f"failed to send frame: {exc}") from exc
    if timing is not None:
        timing["encode_s"] = t_encoded - t_start
        timing["send_s"] = time.perf_counter() - t_encoded


def recv_frame(sock: socket.socket, timing: dict[str, float] | None = None) -> dict[str, Any] | None:
    """Read one length-prefixed frame from ``sock`` and deserialize it.

    Args:
        sock: A connected stream socket.
        timing: If given, receives ``recv_s`` (waiting for and reading the
            bytes) and ``decode_s`` (deserialization) for this frame.

    Returns:
        The decoded message dictionary, or ``None`` if the peer closed the
        connection cleanly before a new frame started.

    Raises:
        TransportError: If the socket read fails or the frame is truncated.
    """
    t_start = time.perf_counter()
    header = _recv_exact(sock, _LENGTH_STRUCT.size)
    if header is None:
        return None
    (length,) = _LENGTH_STRUCT.unpack(header)
    body = _recv_exact(sock, length)
    if body is None:
        raise TransportError("connection closed mid-frame")
    t_received = time.perf_counter()
    try:
        message = pickle.loads(body)
    except Exception as exc:  # corrupt frame: unpickling errors vary by payload
        raise TransportError(f"failed to decode frame: {exc}") from exc
    if timing is not None:
        timing["recv_s"] = t_received - t_start
        timing["decode_s"] = time.perf_counter() - t_received
    return message


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes | None:
    """Receive exactly ``num_bytes`` from ``sock``, or ``None`` on clean EOF."""
    buf = bytearray()
    while len(buf) < num_bytes:
        try:
            chunk = sock.recv(num_bytes - len(buf))
        except OSError as exc:
            raise TransportError(f"failed to receive frame: {exc}") from exc
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class EnvTransport(abc.ABC):
    """Client-side interface for driving a remote environment endpoint.

    Implementations move messages; they attach no environment semantics beyond
    the schema documented in the module docstring. Environment classes own
    observation formatting, episode bookkeeping, and reconnection policy.

    The same interface is meant to front every remote endpoint the learner
    talks to -- simulators, real-robot bridges, and data servers -- whether
    they share the node (``ipc://``) or not (``tcp://``); construct instances
    from a URI via :func:`transport_from_uri`.

    Attributes:
        last_timing: Where the last request's wall time went, as reported by
            the implementation -- ``recv_s`` / ``decode_s`` for the response
            and ``encode_s`` / ``send_s`` for the request. Empty for
            implementations that do not measure it.
    """

    last_timing: dict[str, float]

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the connection to the environment server.

        Raises:
            TransportError: If the server cannot be reached within the
                implementation's connection budget.
        """

    @abc.abstractmethod
    def spec(self) -> dict[str, Any]:
        """Return the server-reported endpoint metadata (the handshake).

        Returns:
            The ``SPEC`` response payload -- protocol version, device, and
            observation/action space metadata -- or an empty dict if the
            server does not implement ``SPEC``.
        """

    @abc.abstractmethod
    def reset(self, seed: int | None = None, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Request an episode reset and return the raw response payload.

        Args:
            seed: Optional episode seed, forwarded to the server.
            env_ids: For natively-batched servers, reset only these env
                indices (rolling collection); ``None`` resets everything.
        """

    @abc.abstractmethod
    def step(self, action: Any, need_obs: bool = True) -> dict[str, Any]:
        """Send one action (or action chunk) and return the raw response payload.

        Args:
            action: Nested list of floats, ``[action_dim]`` or
                ``[chunk, action_dim]``.
            need_obs: Whether the server should include a full observation in
                the response (skipping renders can save time on real robots).
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Tell the server the session is over and release the connection."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """Whether the transport currently holds a live connection."""


class NativeTcpTransport(EnvTransport):
    """``EnvTransport`` over a plain TCP socket with pickle frames.

    Args:
        host: Environment server hostname or address.
        port: Environment server port.
        request_timeout: Socket timeout in seconds for ``step``/``spec``
            round-trips. ``reset`` is exempt: a reset may block on an
            arbitrarily long scene (re)build, so it waits without a timeout.
        connect_timeout: Total budget in seconds for :meth:`connect` to keep
            retrying before raising. Defaults to unbounded so a learner can
            start before its simulator.
        retry_interval: Seconds between connection attempts.
    """

    def __init__(
        self,
        host: str,
        port: int,
        request_timeout: float = 120.0,
        connect_timeout: float = float("inf"),
        retry_interval: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.retry_interval = float(retry_interval)
        self.last_timing = {}
        self._sock: socket.socket | None = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        if self._sock is not None:
            return
        start = time.monotonic()
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.request_timeout)
            try:
                sock.connect((self.host, self.port))
                self._sock = sock
                print(f"[NativeTcpTransport] connected to {self.host}:{self.port}", flush=True)
                return
            except OSError as exc:
                sock.close()
                elapsed = time.monotonic() - start
                if elapsed > self.connect_timeout:
                    raise TransportError(
                        f"could not connect to {self.host}:{self.port} "
                        f"after {elapsed:.0f}s: {exc}"
                    ) from exc
                print(
                    f"[NativeTcpTransport] connect to {self.host}:{self.port} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {self.retry_interval:.0f}s",
                    flush=True,
                )
                time.sleep(self.retry_interval)

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
        if self._sock is None:
            return
        try:
            send_frame(self._sock, {"v": PROTOCOL_VERSION, "type": MSG_CLOSE})
        except TransportError:
            pass  # closing a dead connection is fine
        self._drop()

    def _drop(self) -> None:
        """Close the socket without notifying the server (connection is dead)."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _request(self, message: dict[str, Any], unbounded_wait: bool = False) -> dict[str, Any]:
        """One request/response round-trip; drops the connection on failure."""
        if self._sock is None:
            raise TransportError("not connected; call connect() first")
        try:
            if unbounded_wait:
                self._sock.settimeout(None)
            try:
                self.last_timing = {}
                send_frame(self._sock, message, self.last_timing)
                response = recv_frame(self._sock, self.last_timing)
            finally:
                if unbounded_wait and self._sock is not None:
                    self._sock.settimeout(self.request_timeout)
            if response is None:
                raise TransportError("server closed the connection")
            return response
        except (OSError, EOFError) as exc:
            self._drop()
            if isinstance(exc, TransportError):
                raise
            raise TransportError(str(exc)) from exc


class NativeTcpServer:
    """Server-side binding of the native TCP transport.

    Owns the listening socket and the per-client frame loop; environment
    semantics live in the ``handler`` passed to :meth:`serve_forever`. Serves
    one client at a time, which matches simulators that cannot be shared.

    Args:
        host: Interface to bind; ``"0.0.0.0"`` accepts other hosts, whose frames are unpickled.
        port: Port to bind.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6000) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.host = host
        self.port = int(port)

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)`` pair."""
        addr = self._srv.getsockname()
        return str(addr[0]), int(addr[1])

    def serve_forever(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any] | None],
        on_connect: Callable[[tuple[Any, ...]], None] | None = None,
        on_disconnect: Callable[[tuple[Any, ...], Exception | None], None] | None = None,
        on_response: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Accept clients and dispatch their requests until interrupted.

        Args:
            handler: Called with each decoded request; its return value is
                sent back as the response. Returning ``None`` ends the client
                session (the ``CLOSE`` path) without a response.
            on_connect: Optional callback invoked with the client address when
                a client connects.
            on_disconnect: Optional callback invoked with the client address
                and the error (``None`` for a clean disconnect) when a client
                session ends.
            on_response: Optional callback invoked after each response is
                written, with its ``encode_s`` / ``send_s`` timing.

        Exceptions such as ``SystemExit`` (a SIGTERM handler) and
        ``KeyboardInterrupt`` propagate to the caller so the process can shut
        down cleanly.
        """
        try:
            while True:
                try:
                    conn, addr = self._srv.accept()
                except OSError:
                    return  # listening socket closed: shutdown requested
                if on_connect is not None:
                    on_connect(addr)
                error: Exception | None = None
                try:
                    self._serve_client(conn, handler, on_response)
                except OSError as exc:
                    error = exc
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    if on_disconnect is not None:
                        on_disconnect(addr, error)
        finally:
            self.close()

    def _serve_client(
        self,
        conn: socket.socket,
        handler: Callable[[dict[str, Any]], dict[str, Any] | None],
        on_response: Callable[[dict[str, float]], None] | None,
    ) -> None:
        """Run the request/response loop for one connected client."""
        while True:
            message = recv_frame(conn)
            if message is None:
                return  # client went away
            response = handler(message)
            if response is None:
                return  # handler ended the session (CLOSE)
            timing: dict[str, float] = {}
            send_frame(conn, response, timing)
            if on_response is not None:
                on_response(timing)

    def close(self) -> None:
        """Close the listening socket (callable from any thread).

        The socket is shut down before closing: on Linux a plain ``close``
        does not wake a thread blocked in ``accept``, while ``shutdown``
        makes it return immediately, so a thread-hosted server stops
        promptly.
        """
        try:
            self._srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed / never listening
        try:
            self._srv.close()
        except OSError:
            pass
