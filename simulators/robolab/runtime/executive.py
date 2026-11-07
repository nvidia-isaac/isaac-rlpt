# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal simulator-neutral runtime executive for policy proposals."""

from __future__ import annotations

from typing import Protocol

from simulators.robolab.runtime.types import (
    PolicyProposal,
    RuntimeDecision,
    RuntimeObservation,
)


class Arbiter(Protocol):
    """The narrow interface required to approve a policy proposal."""

    def decide(
        self, observation: RuntimeObservation, proposal: PolicyProposal
    ) -> RuntimeDecision:
        """Return a bounded decision for ``proposal``."""


class RuntimeExecutive:
    """Validate the arbiter decision before the client caches any action."""

    def __init__(self, arbiter: Arbiter) -> None:
        self._arbiter = arbiter

    def decide(
        self, observation: RuntimeObservation, proposal: PolicyProposal
    ) -> RuntimeDecision:
        """Ask the arbiter and enforce the action-chunk boundary."""
        decision = self._arbiter.decide(observation, proposal)
        decision.validate_for(proposal)
        return decision
