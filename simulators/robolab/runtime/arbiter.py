# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The first runtime arbiter: deliberately behaviorally invisible."""

from __future__ import annotations

from simulators.robolab.runtime.types import (
    PolicyProposal,
    RuntimeDecision,
    RuntimeObservation,
)


class NoOpArbiter:
    """Approve an entire proposal without changing policy behavior.

    Keeping the no-op policy as a real arbiter makes the behavior-preserving
    path testable and gives future monitor/steering arbiters one narrow
    replacement point.
    """

    def decide(
        self, observation: RuntimeObservation, proposal: PolicyProposal
    ) -> RuntimeDecision:
        del observation
        return RuntimeDecision(
            kind="execute",
            execute_prefix_len=int(proposal.actions.shape[0]),
            reason="no_op",
            selected_base_model=proposal.base_model_name,
        )
