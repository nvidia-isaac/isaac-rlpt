# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discount helpers for algorithms that execute action chunks."""

from __future__ import annotations

__all__ = ["chunk_discounted_gamma"]


def chunk_discounted_gamma(gamma: float, action_exec_len: int) -> float:
    """Fold receding-horizon execution into the discount factor.

    An algorithm that stores one transition per *executed chunk*
    (``action_exec_len`` environment steps, with the chunk's terminal reward)
    must discount each transition by ``gamma ** action_exec_len`` rather than
    ``gamma``. DSRL passes the result as the SAC ``gamma`` and disables
    ``compute_chunked_gamma`` (the reward is already per chunk). PPO discounts
    at the decision level instead: its semi-Markov GAE raises ``gamma`` to each
    transition's own executed length (``algorithms/ppo/advantages.py``), which
    also covers chunks of varying length. For a fixed ``action_exec_len`` the
    two agree; neither is expressed through the other.

    Args:
        gamma: Per-environment-step discount factor.
        action_exec_len: Environment steps executed per decoded chunk.

    Returns:
        The effective per-transition discount ``gamma ** action_exec_len``.
    """
    return float(gamma) ** int(action_exec_len)
