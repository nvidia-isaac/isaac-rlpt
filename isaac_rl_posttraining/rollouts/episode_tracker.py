# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-environment episode accounting for collected rollouts."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["EpisodeTracker"]

#: Reward-model relabeling diagnostics: (info key, metric name).
_RELABEL_KEYS = (("env_reward", "env_reward"), ("relabeled_reward", "progress_reward"), ("success_prob", "success_prob"))


class EpisodeTracker:
    """Per-environment episode accounting and rollout metrics."""

    def __init__(self, num_envs: int) -> None:
        """Args:
        num_envs: Number of parallel environments tracked.
        """
        self.num_envs = num_envs
        self.reset()
        self.total_steps = 0
        self.episodes_collected = [0] * self.num_envs

    def reset(self) -> None:
        """Reset all running episode state and metric accumulators."""
        self.current_rewards = [0.0] * self.num_envs
        self.current_successes = [False] * self.num_envs
        self.current_steps = [0] * self.num_envs  # env steps in each env's running episode
        # The recent window (the last num_envs episodes) and running totals; nothing grows with the run.
        self.recent_rewards: deque[float] = deque(maxlen=self.num_envs)
        self.recent_successes: deque[bool] = deque(maxlen=self.num_envs)
        self.num_episodes = 0
        self.reward_sum = 0.0
        self.success_count = 0
        #: Per relabeling metric: running ``[count, sum, min, max]``.
        self.relabel_stats: dict[str, list[float]] = {}

    def add_step(
        self,
        env_idx: int,
        reward: float,
        info: Mapping[str, Any],
        done: bool,
        truncated: bool,
        actual_env_steps: int | None = None,
        *,
        success: bool,
    ) -> None:
        """Record one collected step.

        Args:
            env_idx: Environment the step came from.
            reward: Step reward.
            info: Step info dict (relabeling metadata).
            done: Terminal flag.
            truncated: Truncation flag.
            actual_env_steps: Environment steps this logical step covered
                (chunked execution advances several sim steps at once).
            success: Whether this step ended the episode successfully.
        """
        env_steps = actual_env_steps if actual_env_steps is not None else 1
        self.total_steps += env_steps
        self.current_steps[env_idx] += env_steps
        self.current_rewards[env_idx] += reward

        if (done or truncated) and success:
            self.current_successes[env_idx] = True

        # Reward-model relabeling diagnostics, when the env reports them.
        if info is not None and "env_reward" in info:
            for key, name in _RELABEL_KEYS:
                value = info[key]
                if isinstance(info["env_reward"], (list, np.ndarray)):
                    value = value[env_idx]
                if value is not None:
                    value = float(value)
                    stats = self.relabel_stats.setdefault(name, [0, 0.0, value, value])
                    stats[0] += 1
                    stats[1] += value
                    stats[2] = min(stats[2], value)
                    stats[3] = max(stats[3], value)

    def end_episode(self, env_idx: int) -> None:
        """Close out the running episode for one environment."""
        self.recent_rewards.append(self.current_rewards[env_idx])
        self.recent_successes.append(self.current_successes[env_idx])
        self.num_episodes += 1
        self.reward_sum += self.current_rewards[env_idx]
        self.success_count += self.current_successes[env_idx]
        self.current_rewards[env_idx] = 0.0
        self.current_successes[env_idx] = False
        self.current_steps[env_idx] = 0
        self.episodes_collected[env_idx] += 1

    def min_episodes_collected(self) -> int:
        """Fewest episodes collected by any environment."""
        return min(self.episodes_collected) if self.episodes_collected else 0

    def get_metrics(self) -> dict[str, float]:
        """Aggregate rollout metrics (recent-window plus overall)."""
        if not self.num_episodes:
            return {
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "success_rate": 0.0,
                "num_episodes": 0,
                "total_steps": self.total_steps,
            }

        metrics: dict[str, float] = {
            "ep_avg_reward": np.mean(self.recent_rewards),
            "ep_min_reward": np.min(self.recent_rewards),
            "ep_max_reward": np.max(self.recent_rewards),
            "success_rate": np.mean(self.recent_successes),
            "ep_overall_success_rate": self.success_count / self.num_episodes,
            "ep_overall_reward": self.reward_sum / self.num_episodes,
            "num_episodes": self.num_episodes,
            "total_steps": self.total_steps,
        }
        for _, name in _RELABEL_KEYS:
            if name in self.relabel_stats:
                count, total, low, high = self.relabel_stats[name]
                metrics[f"ep_avg_{name}"] = total / count
                metrics[f"ep_min_{name}"] = low
                metrics[f"ep_max_{name}"] = high
        return metrics
