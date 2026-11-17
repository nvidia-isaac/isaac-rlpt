# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implicit Q-Learning: token critics and a model-agnostic learner.

The math lives in :mod:`isaac_rl_posttraining.algorithms.iql.critics`.
:class:`IQL` trains any :class:`~isaac_rl_posttraining.policies.trainable.TrainablePolicy`
that implements ``feature_tokens`` and ``actor_bc_loss``.
"""

from isaac_rl_posttraining.algorithms.iql.config import IQLConfig
from isaac_rl_posttraining.algorithms.iql.critics import (
    TokenCriticEnsemble,
    TokenValueHead,
    discounted_reward_sum,
    expectile_loss,
    iql_losses,
    reduce_q,
)
from isaac_rl_posttraining.algorithms.iql.iql import IQL

__all__ = [
    "IQL",
    "IQLConfig",
    "TokenCriticEnsemble",
    "TokenValueHead",
    "discounted_reward_sum",
    "expectile_loss",
    "iql_losses",
    "reduce_q",
]
