# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-retention selection: the best-evaluating weights must survive.

Regression for a real loss: a keep-last+final pruning policy deleted the
15.6k-step peak checkpoint of a completed DSRL run. The selection rule is a
pure function so the guarantee is testable without a Trainer.
"""

from __future__ import annotations

import pytest

from isaac_rl_posttraining.utils.checkpoint_retention import (
    select_protected_checkpoints,
    validate_checkpoint_retention,
)


def test_protects_best_and_last() -> None:
    tags = [5000, 10000, 15000, 20000]
    history = {5000: 0.4, 10000: 0.7, 15000: 0.6, 20000: 0.5}
    assert select_protected_checkpoints(tags, history) == {10000, 20000}


def test_best_eval_between_saves_maps_to_nearest_checkpoint() -> None:
    tags = [10000, 20000]
    history = {5200: 0.5, 15600: 0.76, 20800: 0.7}
    # Peak eval at 15.6k: nearest saved weights are 20000 (4.4k vs 5.6k away).
    assert select_protected_checkpoints(tags, history) == {20000}


def test_no_eval_history_protects_everything() -> None:
    tags = [2000, 4000]
    assert select_protected_checkpoints(tags, {}) == {2000, 4000}
    assert select_protected_checkpoints([], {0: 0.5}) == set()


def test_tie_prefers_later_step() -> None:
    tags = [1000, 3000]
    history = {1000: 0.5, 3000: 0.5}
    # Equal SR: the later eval wins; the later checkpoint is protected (and is
    # also the last, so the set collapses to one tag).
    assert select_protected_checkpoints(tags, history) == {3000}


def test_unknown_retention_mode_is_rejected() -> None:
    assert validate_checkpoint_retention("best_last_final") == "best_last_final"
    with pytest.raises(ValueError, match="checkpoint_retention"):
        validate_checkpoint_retention("newest")
