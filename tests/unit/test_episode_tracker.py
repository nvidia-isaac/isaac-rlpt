# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode metrics keep a bounded window and running totals, however long the run."""

import pytest

from isaac_rl_posttraining.rollouts import EpisodeTracker


def test_metrics_are_running_totals_over_a_bounded_window() -> None:
    tracker = EpisodeTracker(num_envs=2)
    for episode in range(1000):
        info = {
            "env_reward": float(episode),
            "relabeled_reward": None,
            "success_prob": 0.5,
        }
        tracker.add_step(
            episode % 2, float(episode), info, True, False, success=episode % 4 == 0
        )
        tracker.end_episode(episode % 2)

    metrics = tracker.get_metrics()
    assert len(tracker.recent_rewards) == 2
    assert (
        metrics["ep_avg_reward"],
        metrics["ep_min_reward"],
        metrics["ep_max_reward"],
    ) == (998.5, 998, 999)
    assert metrics["success_rate"] == 0.0 and metrics["ep_overall_success_rate"] == 0.25
    assert (
        metrics["ep_overall_reward"] == pytest.approx(499.5)
        and metrics["num_episodes"] == 1000
    )
    assert (metrics["ep_min_env_reward"], metrics["ep_max_env_reward"]) == (0.0, 999.0)
    assert (
        metrics["ep_avg_env_reward"] == pytest.approx(499.5)
        and metrics["ep_avg_success_prob"] == 0.5
    )
    assert "ep_avg_progress_reward" not in metrics
