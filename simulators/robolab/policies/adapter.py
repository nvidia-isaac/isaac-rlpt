# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt RoboLab hook-based clients to the harness proposal contract."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from simulators.robolab.runtime.capabilities import PolicyCapabilities
from simulators.robolab.runtime.types import PolicyProposal


class RoboLabHookClient(Protocol):
    """The stable hook surface documented by ``robolab.eval.InferenceClient``."""

    open_loop_horizon: int

    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict[str, Any]:
        """Convert a native RoboLab observation to policy-specific arrays."""

    def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> Any:
        """Build the policy server request without changing preprocessing."""

    def _query_server(self, request: Any) -> Any:
        """Send one request through the original policy transport."""

    def _unpack_response(self, response: Any) -> np.ndarray:
        """Convert the original policy response to a raw action chunk."""

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Apply the policy client's original action normalization."""

    def _build_visualization(self, extracted_obs: dict[str, Any]) -> np.ndarray | None:
        """Build the policy client's original visualization output."""

    def reset(self, *, env_id: int | None = None) -> None:
        """Reset policy-client episode state."""

    def close(self) -> None:
        """Release policy-client resources."""


class RoboLabPolicyAdapter:
    """Reuse a RoboLab client's exact four hooks behind one proposal API.

    The adapter never normalizes images or actions itself.  It intentionally
    delegates observation extraction, request packing, transport, response
    unpacking, and postprocessing to the upstream client so no-op parity can
    be meaningful.
    """

    def __init__(self, client: RoboLabHookClient, capabilities: PolicyCapabilities) -> None:
        self.client = client
        self._capabilities = capabilities

    @property
    def capabilities(self) -> PolicyCapabilities:
        """Return the explicit handshake, updated from validated chunks."""
        return self._capabilities

    @property
    def open_loop_horizon(self) -> int:
        """Preserve the exact policy-client refresh cadence."""
        return int(self.client.open_loop_horizon)

    def extract_observation(self, raw_obs: Any, *, env_id: int) -> dict[str, Any]:
        """Delegate native observation extraction without copying or remapping."""
        return self.client._extract_observation(raw_obs, env_id=env_id)

    def propose(self, extracted_obs: dict[str, Any], instruction: str) -> PolicyProposal:
        """Query once through the original client hooks and normalize its chunk."""
        request = self.client._pack_request(extracted_obs, instruction)
        started = perf_counter()
        response = self.client._query_server(request)
        inference_ms = (perf_counter() - started) * 1000.0
        actions = self.client._postprocess_chunk(self.client._unpack_response(response))
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError(
                "RoboLab policy client must return an action chunk shaped "
                f"[horizon, action_dim]; got {actions.shape}"
            )
        if (
            self._capabilities.action_horizon != int(actions.shape[0])
            or self._capabilities.action_dim != int(actions.shape[1])
        ):
            self._capabilities = replace(
                self._capabilities,
                action_horizon=int(actions.shape[0]),
                action_dim=int(actions.shape[1]),
            )
        return PolicyProposal(
            actions=actions,
            latent=None,
            predicted_future=None,
            predicted_value=None,
            base_model_name=self._capabilities.name,
            inference_ms=inference_ms,
        )

    def visualization(self, extracted_obs: dict[str, Any]) -> np.ndarray | None:
        """Delegate the original visualization behavior."""
        return self.client._build_visualization(extracted_obs)

    def reset(self, *, env_id: int | None = None) -> None:
        """Forward reset so server-specific episode state is preserved."""
        self.client.reset(env_id=env_id)

    def close(self) -> None:
        """Forward close to the original transport owner."""
        self.client.close()
