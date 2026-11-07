# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded GR00T latent decoding through RoboLab's native client hooks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from simulators.robolab.runtime.trace import TraceRecorder

__all__ = ["Gr00tRoboLabLatentPolicy"]


@dataclass(frozen=True)
class _PendingDecision:
    """Server-authoritative data retained until the collector reports its outcome."""

    context: dict[str, Any]
    requested_latent: np.ndarray
    clipping: dict[str, Any]


class Gr00tRoboLabLatentPolicy:
    """Decode B=1 DSRL latents through the opt-in GR00T smoke endpoint.

    The base model delegates the DROID observation extraction, request packing,
    response unpacking, and action postprocessing to RoboLab's stock GR00T
    client.  Only the transport call differs: it forwards the server's
    documented ``options.harness_latent`` field to the dedicated smoke server.
    """

    supports_steering = True

    def __init__(
        self,
        client: Any,
        *,
        trace_recorder: TraceRecorder,
        request_with_options: Callable[[Any, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.client = client
        self.trace_recorder = trace_recorder
        self._request_with_options = request_with_options or self._send_with_options
        self._pending: _PendingDecision | None = None
        self._action_horizon: int | None = None
        self._action_dim: int | None = None

    @property
    def action_horizon(self) -> int:
        """Decoded action horizon, available after the first server response."""
        if self._action_horizon is None:
            raise RuntimeError("GR00T action horizon is not known before the first decode")
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        """Decoded action width, available after the first server response."""
        if self._action_dim is None:
            raise RuntimeError("GR00T action dimension is not known before the first decode")
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        """The smoke actor steers the eight decoded DROID action dimensions."""
        return 8

    def infer(self, observations: dict[str, Any], noise: np.ndarray | None) -> dict[str, np.ndarray]:
        """Decode one B=1 DSRL latent, or sample the native policy when ``noise`` is absent."""
        extracted = self.client._extract_observation(observations["robolab_raw"], env_id=0)
        request = self.client._pack_request(extracted, str(observations["prompt"]))
        started = perf_counter()
        if noise is None:
            response = self._request_with_options(request, {})
        else:
            latent = _single_env_latent(noise, latent_dim=self.latent_dim)
            response = self._request_with_options(request, {"harness_latent": latent})
        inference_ms = (perf_counter() - started) * 1000.0
        actions = np.asarray(
            self.client._postprocess_chunk(self.client._unpack_response(response)), dtype=np.float32
        )
        if actions.ndim != 2 or actions.shape[1] != self.latent_dim:
            raise ValueError(f"expected GR00T decoded actions [H, {self.latent_dim}], got {actions.shape}")

        if noise is None:
            # Native evaluation is part of the same harness path, but it has
            # no steering decision to join with a learner outcome.
            self._pending = None
            self._action_horizon, self._action_dim = map(int, actions.shape)
            return {"actions": actions[None, ...]}

        metadata = response[1].get("harness_latent")
        if not isinstance(metadata, dict):
            raise RuntimeError("latent server response omitted harness_latent clipping metadata")
        context = dict(observations["robolab_context"])
        self._action_horizon, self._action_dim = map(int, actions.shape)
        self._pending = _PendingDecision(
            context={**context, "inference_ms": inference_ms},
            requested_latent=latent.copy(),
            clipping=metadata,
        )
        return {"actions": actions[None, ...]}

    def record_collector_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        executed_action_count: int,
        update_count: int,
    ) -> None:
        """Join the pending decoded latent to the collector's real outcome."""
        if self._pending is None:
            raise RuntimeError("collector reported an outcome before any latent decode")
        context = self._pending.context
        self.trace_recorder.record_steering_outcome(
            episode_id=str(context["episode_id"]),
            env_id=int(context["env_id"]),
            step_idx=int(context["step_idx"]),
            task=str(context["task"]),
            requested_latent=self._pending.requested_latent,
            clipping=self._pending.clipping,
            executed_action_count=executed_action_count,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            update_count=update_count,
        )
        self._pending = None

    def discard_pending_decision(self) -> None:
        """Drop an unscored decision produced by evaluation rather than collection."""
        self._pending = None

    def _send_with_options(self, request: Any, options: dict[str, Any]) -> Any:
        """Use the stock GR00T ZMQ transport with its existing serializer."""
        from policies.gr00t.client import _MsgSerializer

        policy_client = self.client.client
        payload: dict[str, Any] = {
            "endpoint": "get_action",
            "data": {"observation": request, "options": options},
        }
        if getattr(policy_client, "api_token", None):
            payload["api_token"] = policy_client.api_token
        policy_client.socket.send(_MsgSerializer.to_bytes(payload))
        response = _MsgSerializer.from_bytes(policy_client.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return tuple(response)


def _single_env_latent(noise: np.ndarray, *, latent_dim: int) -> np.ndarray:
    """Validate the deliberate B=1, one-latent-per-decision smoke contract."""
    latent = np.asarray(noise, dtype=np.float32)
    if latent.shape != (1, latent_dim):
        raise ValueError(
            f"RoboLab DSRL smoke expects noise shape (1, {latent_dim}), got {latent.shape}"
        )
    if not np.isfinite(latent).all():
        raise ValueError("steering latent must be finite")
    return latent
