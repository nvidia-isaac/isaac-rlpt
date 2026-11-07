# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived reset seeds stay 32-bit: a 500-iteration run used to die at iteration 430."""

from __future__ import annotations

from isaac_rl_posttraining.runtimes.base import SEED_MODULUS
from isaac_rl_posttraining.training.on_policy import training_reset_seed


def test_training_reset_seed_is_unchanged_below_the_limit() -> None:
    assert training_reset_seed(0, 1) == 0
    assert training_reset_seed(0, 429) == 428 * 10_000_019
    assert training_reset_seed(7, 2) == 7 + 10_000_019


def test_training_reset_seed_wraps_instead_of_overflowing() -> None:
    for iteration in (430, 431, 500, 5000):
        seed = training_reset_seed(0, iteration)
        assert 0 <= seed < SEED_MODULUS
    assert training_reset_seed(0, 430) == (429 * 10_000_019) % SEED_MODULUS
