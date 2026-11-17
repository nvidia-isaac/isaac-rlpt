# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseModelPolicy adapter for a served DreamZero world-action model.

Experimental — native inference via the DreamZero server; not yet validated
in this repo.

:class:`DreamZeroServerBaseModelPolicy` conforms a remote DreamZero policy server
(driven through :class:`~.client.DreamZeroClient`) to the
:class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` contract. For
DSRL steering, use the in-process ``dreamzero`` base model instead
(:mod:`~isaac_rl_posttraining.policies.dreamzero.local_base_model`); this one drives
a stock server whose protocol has no noise-injection channel.

Contract notes, as implemented by this base model:

* **Native-only** (``supports_steering`` is ``False``, contract note 5). The
  DreamZero serving protocol carries observations one way and action chunks
  the other; the initial action-noise tensor that seeds the model's
  flow-matching denoise lives server-side and is not exposed on the wire.
  ``infer(observations, noise=None)`` is exactly the served base policy;
  any other ``noise`` raises ``NotImplementedError``.
* :attr:`latent_dim` mirrors the geometry of the server-side initial action
  noise (one ``[action_horizon, action_dim]`` Gaussian tensor per call), so
  it equals :attr:`action_dim`. It sizes nothing today — the flag above says
  steering is unavailable — but keeps the property contract
  (``0 < latent_dim <= action_dim``) meaningful if a steerable serving
  protocol lands later.
* The wire protocol is unbatched, so observations must have batch size 1;
  the leading axis is stripped for the request and restored on the response
  (``{"actions": [1, H, A]}``).
* The server does not advertise its action horizon or action width in the
  connection handshake (only the action-space family, e.g.
  ``"joint_position"``), so both come from configuration —
  ``action_horizon=24`` and ``action_dim=8`` (7 joints + 1 gripper) for the
  released DreamZero-DROID checkpoints — and every response is validated
  against them so a mis-matched server fails loudly, not silently.

Episode handling: DreamZero servers are stateful within an episode (they
accumulate camera frames and autoregressive video context, keyed by the
``session_id`` observation field). :meth:`start_episode` rotates the session
id and sends the protocol's reset message; call it at every environment
reset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_rl_posttraining.policies.base import BaseModelPolicy
from isaac_rl_posttraining.policies.dreamzero.client import DreamZeroClient

__all__ = ["DreamZeroServerBaseModelPolicy"]

#: Action-chunk length of the released DreamZero-DROID checkpoints.
DEFAULT_ACTION_HORIZON = 24

#: Action width of the released DreamZero-DROID checkpoints (7 joints + gripper).
DEFAULT_ACTION_DIM = 8


class DreamZeroServerBaseModelPolicy(BaseModelPolicy):
    """A served DreamZero policy behind the ``BaseModelPolicy`` contract.

    Args:
        client: Connected (or lazily connecting) protocol client.
        action_horizon: Environment steps per inference call. Must match the
            served checkpoint; the handshake does not advertise it.
        action_dim: Raw action width per step. Must match the served
            checkpoint (8 for DreamZero-DROID: 7 joints + 1 gripper).

    Raises:
        ValueError: If ``action_horizon`` or ``action_dim`` is not positive.
    """

    #: The served model keeps one episode's state for this client (``start_episode()`` takes no session).
    servable = False
    #: The serving protocol exposes no latent channel (a class attribute, so the dry run can read it).
    supports_steering = False

    def __init__(
        self,
        client: DreamZeroClient,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        action_dim: int = DEFAULT_ACTION_DIM,
    ) -> None:
        self._client = client
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        if self._action_horizon <= 0 or self._action_dim <= 0:
            raise ValueError(
                f"action_horizon and action_dim must be positive; got "
                f"({self._action_horizon}, {self._action_dim})"
            )
        self._session_id = DreamZeroClient.new_session_id()

    @classmethod
    def from_server(
        cls,
        server_url: str,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        action_dim: int = DEFAULT_ACTION_DIM,
        **client_kwargs: Any,
    ) -> "DreamZeroServerBaseModelPolicy":
        """Construct a client for a server URL, then adapt it.

        Args:
            server_url: WebSocket URL of the DreamZero policy server
                (``ws://host:port``; a bare ``host:port`` is normalized).
            action_horizon: Environment steps per inference call.
            action_dim: Raw action width per step.
            **client_kwargs: Forwarded to :class:`~.client.DreamZeroClient`
                (``connect_timeout``, ``max_reconnects``, ...).

        Returns:
            The adapted base model.
        """
        client = DreamZeroClient(server_url, **client_kwargs)
        return cls(client, action_horizon=action_horizon, action_dim=action_dim)

    @classmethod
    def from_dsrl_config(cls, cfg: Any, device: Any, dataset_adapter: Any = None) -> "DreamZeroServerBaseModelPolicy":
        """Build the served DreamZero base model from the composed config.

        The server advertises neither its action horizon nor its action width,
        so both come from the ``base_model`` group (defaulting to the released
        DreamZero-DROID geometry). ``device`` is unused: inference is remote.
        """
        from omegaconf import OmegaConf

        server_url = OmegaConf.select(cfg, "base_model.server_url")
        if not server_url:
            raise ValueError("base_model=dreamzero_server requires base_model.server_url (ws://host:port)")
        return cls.from_server(
            server_url=str(server_url),
            action_horizon=int(
                OmegaConf.select(cfg, "base_model.action_horizon", default=DEFAULT_ACTION_HORIZON)
                or DEFAULT_ACTION_HORIZON
            ),
            action_dim=int(
                OmegaConf.select(cfg, "base_model.action_dim", default=DEFAULT_ACTION_DIM)
                or DEFAULT_ACTION_DIM
            ),
        )

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        """Width of the (server-side, non-injectable) per-step action noise."""
        return self._action_dim

    @property
    def client(self) -> DreamZeroClient:
        """The underlying protocol client (for connection management)."""
        return self._client

    @property
    def session_id(self) -> str:
        """The session id attached to every request of the current episode."""
        return self._session_id

    def start_episode(self) -> None:
        """Reset the server's per-episode state and rotate the session id.

        Sends the protocol reset message (the server clears frame buffers and
        autoregressive context) and generates a fresh ``session_id`` so the
        server treats subsequent requests as a new episode even if the reset
        frame was lost to a reconnect.
        """
        self._client.reset({"session_id": self._session_id})
        self._session_id = DreamZeroClient.new_session_id()

    def infer(self, observations: dict, noise: np.ndarray | None = None) -> dict:
        """Run one native inference call against the server.

        Args:
            observations: Batch-1 observation dict in the served model's
                format. For DreamZero-DROID:
                ``observation/exterior_image_{0,1}_left`` and
                ``observation/wrist_image_left`` (``[1, H, W, 3]`` uint8),
                ``observation/joint_position`` (``[1, 7]``),
                ``observation/gripper_position`` (``[1, 1]``), and
                ``prompt`` (a string, or a length-1 list of strings). Array
                entries carry a leading batch axis of 1 (stripped on the
                wire); strings and scalars pass through. The base model adds
                the ``session_id``.
            noise: Must be ``None``. The base model is native-only
                (``supports_steering`` is ``False``).

        Returns:
            ``{"actions": np.ndarray [1, action_horizon, action_dim]}``
            (float32), exactly as decoded by the server.

        Raises:
            NotImplementedError: If ``noise`` is not ``None``.
            ValueError: If an array entry has a batch size other than 1, or
                the server's action chunk does not match
                ``[action_horizon, action_dim]``.
            DreamZeroServerError: If the server reports an inference error.
        """
        if noise is not None:
            raise NotImplementedError(
                "DreamZeroServerBaseModelPolicy is native-only (supports_steering is False): "
                "the DreamZero serving protocol exposes no latent-injection channel, "
                "so infer() only accepts noise=None. For DSRL steering, load the "
                "model in-process with base_model=dreamzero_local, which injects the initial "
                "action noise of the flow-matching denoise directly."
            )
        request = self._unbatch_observations(observations)
        request["session_id"] = self._session_id
        actions = self._client.infer(request)
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self._action_horizon, self._action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"DreamZero server returned an action chunk of shape {actions.shape}; "
                f"expected {expected}. Set the base model's action_horizon/action_dim "
                "to match the served checkpoint."
            )
        return {"actions": actions[None]}

    def _unbatch_observations(self, observations: dict) -> dict[str, Any]:
        """Strip the leading batch-1 axis from every array entry.

        Args:
            observations: The contract-level observation dict.

        Returns:
            The wire-level (unbatched) observation dict; the input is not
            mutated.

        Raises:
            ValueError: If an array (or list) entry is batched with size != 1.
        """
        request: dict[str, Any] = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                if value.shape[0] != 1:
                    raise ValueError(
                        f"DreamZero serving is unbatched: observation {key!r} has "
                        f"batch size {value.shape[0]}, expected 1"
                    )
                request[key] = value[0]
            elif isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
                request[key] = value[0]  # batch-1 prompt list -> plain string
            else:
                request[key] = value
        return request
