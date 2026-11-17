# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RFS learner identity over the shared SAC optimizer."""

from __future__ import annotations

from isaac_rl_posttraining.algorithms.rfs.config import RFSConfig
from isaac_rl_posttraining.algorithms.sac import SAC


__all__ = ["RFS"]


class RFS(SAC):
    """SAC optimization of a joint flow-noise and residual-action policy.

    The rollout path presents ``[flow_noise, residual_action]`` as the SAC
    action.
    The generic optimizer therefore remains unchanged while RFS checkpoints
    and configuration retain a distinct algorithm identity.
    """

    def __init__(self, config: RFSConfig) -> None:
        super().__init__(config)
