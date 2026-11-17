# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual Flow Steering over a frozen generative policy.

RFS jointly controls the initial flow noise and a bounded additive correction
to the decoded action.
The optimizer remains the generic REDQ-style SAC core; this package owns the
RFS-specific action geometry and modulation semantics.
"""

from isaac_rl_posttraining.algorithms.rfs.config import RFSConfig, RFSModulationConfig
from isaac_rl_posttraining.algorithms.rfs.distribution import (
    LatentStochasticResidualDeterministicDistribution,
    configure_rfs_actor_distribution,
    enable_proposal_conditioned_residual,
)
from isaac_rl_posttraining.algorithms.rfs.modulation import (
    RFSActionModulator,
    initialize_rfs_residual_head,
)
from isaac_rl_posttraining.algorithms.rfs.rfs import RFS


__all__ = [
    "RFS",
    "RFSActionModulator",
    "RFSConfig",
    "RFSModulationConfig",
    "LatentStochasticResidualDeterministicDistribution",
    "configure_rfs_actor_distribution",
    "enable_proposal_conditioned_residual",
    "initialize_rfs_residual_head",
]
