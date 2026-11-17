# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data exchanged between policy adapters and the runtime executive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class RuntimeObservation:
    """One policy observation with enough identity for a replay timeline."""

    obs: dict[str, Any]
    timestamp: float
    episode_id: str
    step_idx: int
    simulator_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolicyProposal:
    """A normalized policy action-chunk proposal.

    ``actions`` is always ``[horizon, action_dim]`` for the single environment
    represented by one policy inference call.
    """

    actions: np.ndarray
    latent: np.ndarray | None
    predicted_future: dict[str, Any] | None
    predicted_value: float | None
    base_model_name: str
    inference_ms: float
    control_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError(
                "PolicyProposal.actions must have shape [horizon, action_dim] "
                f"with positive dimensions; got {actions.shape}"
            )


DecisionKind = Literal["execute", "steer", "recover", "reset", "hold", "abort"]


@dataclass(frozen=True)
class RuntimeDecision:
    """A bounded, explicit decision about a policy proposal."""

    kind: DecisionKind
    execute_prefix_len: int
    reason: str
    selected_base_model: str
    steering_payload: dict[str, Any] | None = None
    recovery_request: dict[str, Any] | None = None

    def validate_for(self, proposal: PolicyProposal) -> None:
        """Reject decisions that would read outside the proposed action chunk."""
        if self.kind not in {"execute", "steer"}:
            raise ValueError(
                "The runtime currently only executes policy proposals; "
                f"got unsupported decision kind {self.kind!r}"
            )
        horizon = int(np.asarray(proposal.actions).shape[0])
        if not 1 <= self.execute_prefix_len <= horizon:
            raise ValueError(
                "execute_prefix_len must be in [1, proposal horizon]; "
                f"got {self.execute_prefix_len} for horizon {horizon}"
            )
