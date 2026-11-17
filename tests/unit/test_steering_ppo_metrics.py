# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO entries log their namespaced metrics without a launcher prefix."""

from __future__ import annotations

from typing import cast

import pytest

from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.runners.ppo_launch import log_namespaced_metrics


class _Logger:
    def __init__(self) -> None:
        self.calls = []

    def log(self, metrics, step=None, prefix=None):
        self.calls.append((prefix, dict(metrics), step))


def test_metrics_are_logged_under_their_own_namespaces() -> None:
    recorder = _Logger()
    logger = cast(Logger, recorder)
    log_namespaced_metrics(
        logger,
        {
            "train/success_rate": 1.0,
            "train/episodes": 2.0,
            "learner/policy_loss": 0.1,
            "system/env_steps": 8.0,
        },
        3,
    )

    assert recorder.calls == [
        ("train", {"success_rate": 1.0, "episodes": 2.0}, 3),
        ("learner", {"policy_loss": 0.1}, 3),
        ("system", {"env_steps": 8.0}, 3),
    ]
    with pytest.raises(ValueError, match="namespace"):
        log_namespaced_metrics(logger, {"loss": 1.0}, 1)
