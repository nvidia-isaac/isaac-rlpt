# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free runtime contracts for the policy harness.

The runtime package intentionally does not import a simulator or a policy
implementation.  Policy-specific adapters live next to their integrations and
translate external clients into these contracts.
"""

from simulators.robolab.runtime.arbiter import NoOpArbiter
from simulators.robolab.runtime.capabilities import PolicyCapabilities
from simulators.robolab.runtime.executive import RuntimeExecutive
from simulators.robolab.runtime.steering import BoundedLatent, bound_latent
from simulators.robolab.runtime.trace import TraceRecorder
from simulators.robolab.runtime.types import (
    PolicyProposal,
    RuntimeDecision,
    RuntimeObservation,
)

__all__ = [
    "NoOpArbiter",
    "BoundedLatent",
    "PolicyCapabilities",
    "PolicyProposal",
    "RuntimeDecision",
    "RuntimeExecutive",
    "RuntimeObservation",
    "TraceRecorder",
    "bound_latent",
]
