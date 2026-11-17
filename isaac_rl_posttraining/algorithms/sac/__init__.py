# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic REDQ-style SAC core (algorithm-side of the steering split).

This package is base-model- and task-agnostic: it optimizes a stochastic actor
against a randomized Q-ensemble over whatever action space it is given.
Latent-steering layers (e.g. :mod:`isaac_rl_posttraining.algorithms.dsrl`)
reuse it unchanged by presenting the steering latent as the action.
"""

from isaac_rl_posttraining.algorithms.sac.config import SACConfig
from isaac_rl_posttraining.algorithms.sac.sac import SAC

__all__ = ["SAC", "SACConfig"]
