# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generalized advantage estimation for variable-duration action chunks."""

from __future__ import annotations

import numpy as np

__all__ = ["generalized_advantage_estimate"]


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    bootstrap_steps: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute semi-Markov GAE over arrays shaped ``[time, environments]``.

    True termination suppresses value bootstrapping. Time-limit truncation
    bootstraps from ``next_values`` but stops the eligibility trace so the next
    reset episode cannot leak into the current trajectory. A chunk that
    executes ``k`` simulator steps uses ``gamma**k`` and ``lambda**k``.
    """

    arrays = tuple(
        np.asarray(value)
        for value in (
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            bootstrap_steps,
        )
    )
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays[1:]):
        raise ValueError("GAE inputs must share shape [time, environments]")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if not np.all(np.isfinite(arrays[0:3])):
        raise ValueError("rewards and values must be finite")
    if not np.issubdtype(arrays[5].dtype, np.integer):
        raise TypeError("bootstrap_steps must use an integer dtype")
    if np.any(arrays[5] < 1):
        raise ValueError("bootstrap_steps must be positive")

    reward, value, successor = (
        item.astype(np.float64, copy=False) for item in arrays[:3]
    )
    terminal = arrays[3].astype(bool, copy=False)
    timeout = arrays[4].astype(bool, copy=False)
    duration = arrays[5].astype(np.float64, copy=False)
    discount = np.power(float(gamma), duration)
    trace_discount = discount * np.power(float(gae_lambda), duration)
    delta = reward + discount * (~terminal) * successor - value
    episode_continues = ~(terminal | timeout)

    advantages = np.empty(shape, dtype=np.float64)
    running = np.zeros(shape[1], dtype=np.float64)
    for index in range(shape[0] - 1, -1, -1):
        running = (
            delta[index] + trace_discount[index] * episode_continues[index] * running
        )
        advantages[index] = running
    returns = advantages + value
    return advantages.astype(np.float32), returns.astype(np.float32)
