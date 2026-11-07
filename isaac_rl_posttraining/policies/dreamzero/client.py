# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket client for a DreamZero policy server.

Experimental — native inference via the DreamZero server; not yet validated
in this repo.

DreamZero (Apache-2.0) serves its world-action model behind a WebSocket policy server
(``eval_utils/policy_server.py`` in that repository). This module implements
the exact wire protocol of that server so the
:class:`~isaac_rl_posttraining.policies.dreamzero.local_base_model.DreamZeroBaseModelPolicy`
can drive a served model without importing any of the DreamZero model stack:

1. **Framing.** Every message is one WebSocket binary frame containing a
   msgpack document, with numpy arrays encoded by the msgpack-numpy scheme
   below. Compression is disabled and frames are unbounded (``max_size``
   unlimited), matching both ends of the reference implementation.
2. **Handshake.** Immediately after the connection opens, the server sends
   one metadata frame: the msgpack-encoded server configuration
   (``image_resolution``, ``needs_wrist_camera``, ``n_external_cameras``,
   ``needs_stereo_camera``, ``needs_session_id``, ``action_space``). The
   client must receive it before sending anything.
3. **Requests.** The client sends a msgpack-encoded observation dict with an
   added ``"endpoint"`` key: ``"infer"`` for an action request or ``"reset"``
   for an episode-boundary reset. The server pops the key and dispatches.
4. **Responses.** For ``"infer"`` the server replies with one binary frame:
   the msgpack-encoded action chunk (a float array of shape
   ``[horizon, action_dim]``). For ``"reset"`` it replies with a plain text
   frame acknowledging the reset. A **text** frame in response to ``"infer"``
   carries a server-side traceback: the server sends it and then closes the
   connection with an internal-error code.

The msgpack-numpy encoding matches ``openpi_client.msgpack_numpy`` (the codec
pinned by the DreamZero server and client): an ``np.ndarray`` is packed as a
map ``{b"__ndarray__": True, b"data": <raw bytes>, b"dtype": <dtype.str>,
b"shape": <shape>}``, a numpy scalar as ``{b"__npgeneric__": True, b"data":
<item>, b"dtype": <dtype.str>}``, and void/object/complex dtypes are
rejected. Only ``msgpack`` is required to speak the codec; the ``websockets``
package is imported lazily and only when a real network connection is opened,
so tests can drive the protocol through an injected in-process connection.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Protocol

import numpy as np

__all__ = [
    "Connection",
    "DreamZeroClient",
    "DreamZeroServerError",
    "pack_payload",
    "unpack_payload",
]

#: Dtype kinds the wire codec rejects, exactly as the reference codec does
#: (void, object, and complex arrays have no portable byte representation).
_UNSUPPORTED_DTYPE_KINDS = ("V", "O", "c")


def _msgpack():
    """Import ``msgpack`` lazily so this module imports without the extra.

    Raises:
        ImportError: If ``msgpack`` is not installed (install the
            ``dreamzero`` extra: ``pip install isaac-rl-posttraining[dreamzero]``).
    """
    try:
        import msgpack
    except ImportError as exc:  # pragma: no cover - exercised via is_available()
        raise ImportError(
            "The DreamZero wire codec needs the 'msgpack' package; install the "
            "'dreamzero' extra (pip install isaac-rl-posttraining[dreamzero])."
        ) from exc
    return msgpack


def _encode_numpy(obj: Any) -> Any:
    """msgpack ``default`` hook: encode numpy arrays and scalars.

    Args:
        obj: Object msgpack cannot encode natively.

    Returns:
        A msgpack-encodable substitute following the msgpack-numpy scheme.

    Raises:
        ValueError: If ``obj`` is a numpy array/scalar of an unsupported
            dtype kind (void, object, complex).
    """
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_DTYPE_KINDS:
        raise ValueError(f"Unsupported dtype for the DreamZero wire codec: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _decode_numpy(obj: dict) -> Any:
    """msgpack ``object_hook``: decode msgpack-numpy maps back to numpy."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def pack_payload(payload: Any) -> bytes:
    """Encode one wire message (dict / array / scalar tree) to msgpack bytes."""
    return _msgpack().packb(payload, default=_encode_numpy)


def unpack_payload(data: bytes) -> Any:
    """Decode one msgpack wire message, reviving numpy arrays and scalars."""
    return _msgpack().unpackb(data, object_hook=_decode_numpy)


class DreamZeroServerError(RuntimeError):
    """The DreamZero server reported an error for an inference request.

    Raised when the server answers an ``"infer"`` request with a text frame
    (a formatted server-side traceback) instead of a binary action frame; the
    server closes the connection immediately after sending it.
    """


class Connection(Protocol):
    """The connection surface :class:`DreamZeroClient` drives.

    A real connection is a synchronous ``websockets`` client connection;
    tests inject an in-process fake with the same three methods.
    """

    def send(self, data: bytes) -> None:
        """Send one binary frame."""

    def recv(self) -> bytes | str:
        """Receive one frame: ``bytes`` for binary, ``str`` for text."""

    def close(self) -> None:
        """Close the connection."""


def _default_open_connection(server_url: str, connect_timeout: float) -> Connection:
    """Open a real WebSocket connection (requires the ``websockets`` package).

    Args:
        server_url: ``ws://`` or ``wss://`` server URL.
        connect_timeout: Seconds to wait for the TCP/WebSocket handshake.

    Returns:
        An open synchronous WebSocket connection with protocol-matching
        settings (no compression, unbounded frames, tolerant keepalive —
        a request can legitimately take minutes while the model denoises).

    Raises:
        ImportError: If ``websockets`` is not installed (install the
            ``dreamzero`` extra: ``pip install isaac-rl-posttraining[dreamzero]``).
    """
    try:
        import websockets.sync.client
    except ImportError as exc:  # pragma: no cover - exercised via is_available()
        raise ImportError(
            "The DreamZero base model needs the 'websockets' package to open a real "
            "server connection; install the 'dreamzero' extra "
            "(pip install isaac-rl-posttraining[dreamzero])."
        ) from exc

    return websockets.sync.client.connect(
        server_url,
        open_timeout=connect_timeout,
        compression=None,
        max_size=None,
        ping_interval=60,
        ping_timeout=600,
    )


def _connection_closed_exceptions() -> tuple[type[BaseException], ...]:
    """Exception types that mean "the connection dropped; reconnect and retry"."""
    closed: tuple[type[BaseException], ...] = (ConnectionError, BrokenPipeError, EOFError, OSError)
    try:
        from websockets.exceptions import WebSocketException
    except ImportError:
        return closed
    return closed + (WebSocketException,)


class DreamZeroClient:
    """Typed client for the DreamZero WebSocket policy protocol.

    Connection management is lazy and self-healing: the first request dials
    the server and consumes the metadata handshake; a dropped connection is
    re-dialed (with a fresh handshake) and the request retried, up to
    ``max_reconnects`` times per call. The client is thread-safe by
    serializing requests with an internal lock — the protocol itself is
    strictly one-request-one-response.

    Args:
        server_url: WebSocket URL of the server. A bare ``host:port`` is
            normalized to ``ws://host:port``.
        connect_timeout: Seconds to wait when opening a connection.
        max_reconnects: How many times one request may re-dial a dropped
            connection before giving up.
        open_connection: Override for opening connections; called as
            ``factory(server_url, connect_timeout)``. Tests inject an
            in-process fake here; the default opens a real WebSocket and
            requires the optional ``websockets`` package.
    """

    def __init__(
        self,
        server_url: str,
        connect_timeout: float = 30.0,
        max_reconnects: int = 1,
        open_connection: Callable[[str, float], Connection] | None = None,
    ) -> None:
        self._server_url = self._normalize_url(server_url)
        self._connect_timeout = float(connect_timeout)
        self._max_reconnects = int(max_reconnects)
        self._open_connection = open_connection or _default_open_connection
        self._connection: Connection | None = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_url(server_url: str) -> str:
        """Prefix bare ``host:port`` addresses with the ``ws://`` scheme."""
        url = server_url.strip()
        if not url:
            raise ValueError("server_url must be a non-empty ws:// or wss:// address")
        if not url.startswith(("ws://", "wss://")):
            url = f"ws://{url}"
        return url

    @property
    def server_url(self) -> str:
        """The normalized server URL this client dials."""
        return self._server_url

    @property
    def metadata(self) -> dict[str, Any]:
        """The server's handshake metadata (connecting first if needed).

        Returns:
            The server-configuration dict sent by the server when the
            connection opened (protocol step 2).
        """
        with self._lock:
            self._ensure_connected()
            assert self._metadata is not None  # set by _ensure_connected
            return dict(self._metadata)

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Request one action chunk for an observation.

        Args:
            observation: Unbatched observation dict in the server's expected
                format (see the base model for the DreamZero-DROID key layout).
                The ``"endpoint"`` key is added by the client; the caller's
                dict is not mutated.

        Returns:
            The decoded action chunk as sent by the server, cast to a float32
            array (shape ``[horizon, action_dim]`` for DreamZero servers).

        Raises:
            DreamZeroServerError: If the server answers with an error text
                frame (it closes the connection after sending it).
            ConnectionError: If the connection keeps dropping after
                ``max_reconnects`` re-dials.
        """
        response = self._request({**observation, "endpoint": "infer"})
        if isinstance(response, str):
            # Protocol step 4: a text frame in response to "infer" is a
            # server-side traceback, and the server is about to close.
            self.close()
            raise DreamZeroServerError(f"DreamZero server error:\n{response}")
        return np.asarray(unpack_payload(response), dtype=np.float32)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        """Signal an episode boundary to the server.

        The server clears its per-session state (frame buffers, KV caches,
        autoregressive video context) and acknowledges with a text frame.

        Args:
            reset_info: Optional extra fields for the reset request; the
                ``"endpoint"`` key is added by the client.

        Raises:
            DreamZeroServerError: If the server does not acknowledge the
                reset. The reference server reports request failures as text
                tracebacks and closes the connection, just as it does for
                inference failures.
        """
        response = self._request({**(reset_info or {}), "endpoint": "reset"})
        if response == "reset successful":
            return

        # The released server's only successful reset response is the exact
        # acknowledgement above. Any other response is a text traceback (or
        # a protocol mismatch) and must not be treated as a successful
        # episode boundary: doing so would leak autoregressive state across
        # episodes.
        self.close()
        if isinstance(response, str):
            raise DreamZeroServerError(f"DreamZero server reset error:\n{response}")
        raise DreamZeroServerError(
            "DreamZero server returned a binary response to reset; expected "
            "the text acknowledgement 'reset successful'."
        )

    def close(self) -> None:
        """Close the connection; the next request re-dials transparently."""
        with self._lock:
            self._drop_connection()

    def __enter__(self) -> "DreamZeroClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def new_session_id() -> str:
        """A fresh session identifier for the server's per-episode state."""
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Dial the server and consume the metadata handshake if needed."""
        if self._connection is not None:
            return
        connection = self._open_connection(self._server_url, self._connect_timeout)
        handshake = connection.recv()
        if isinstance(handshake, str):
            handshake = handshake.encode("utf-8")
        self._metadata = unpack_payload(handshake)
        self._connection = connection

    def _drop_connection(self) -> None:
        """Close and forget the current connection, ignoring close errors."""
        if self._connection is None:
            return
        try:
            self._connection.close()
        # Already tearing the connection down.
        except Exception:  # noqa: BLE001
            pass
        self._connection = None
        self._metadata = None

    def _request(self, payload: dict[str, Any]) -> bytes | str:
        """Send one request frame and return the raw response frame.

        Re-dials and retries on a dropped connection, up to
        ``max_reconnects`` times.
        """
        data = pack_payload(payload)
        closed_excs = _connection_closed_exceptions()
        with self._lock:
            last_error: BaseException | None = None
            for _ in range(self._max_reconnects + 1):
                try:
                    self._ensure_connected()
                    assert self._connection is not None
                    self._connection.send(data)
                    return self._connection.recv()
                except closed_excs as exc:
                    last_error = exc
                    self._drop_connection()
            raise ConnectionError(
                f"DreamZero server at {self._server_url} unreachable after "
                f"{self._max_reconnects + 1} attempt(s): {last_error}"
            ) from last_error
