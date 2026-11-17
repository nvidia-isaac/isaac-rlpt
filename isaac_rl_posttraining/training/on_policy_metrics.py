# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics the on-policy loop logs: episodes, learner ranks and the policy service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from isaac_rl_posttraining.training.trajectories import ChunkTrajectory

__all__ = [
    "drain_policy_service_metrics",
    "learner_metrics",
    "policy_service_metrics",
    "rank_mean_metrics",
    "trajectory_metrics",
]


def policy_service_metrics(rollout_group: Any) -> dict[str, float]:
    """Policy-server compute, batch sizes and latencies, when the rollout group serves its policy."""
    read_stats = getattr(rollout_group, "policy_server_stats", None)
    if not callable(read_stats):
        return {}
    server_stats = tuple(read_stats())
    operation_seconds: dict[str, float] = {}
    operation_batch_sizes: dict[str, list[int]] = {}
    latencies: list[float] = []
    for server in server_stats:
        for name, seconds in server.get("operation_seconds", {}).items():
            operation_seconds[name] = operation_seconds.get(name, 0.0) + float(seconds)
        for name, sizes in server.get("operation_batch_sizes", {}).items():
            operation_batch_sizes.setdefault(name, []).extend(
                int(size) for size in sizes
            )
        latencies.extend(
            float(value) for value in server.get("request_latencies_seconds", ())
        )
    metrics = {
        f"policy_service/{name}_compute_seconds": seconds
        for name, seconds in operation_seconds.items()
    }
    if operation_seconds:
        metrics["policy_service/total_compute_seconds"] = sum(
            operation_seconds.values()
        )
    for name, sizes in operation_batch_sizes.items():
        if sizes:
            metrics[f"policy_service/{name}_calls"] = float(len(sizes))
            metrics[f"policy_service/{name}_mean_batch_size"] = float(np.mean(sizes))
            metrics[f"policy_service/{name}_max_batch_size"] = float(max(sizes))
    if latencies:
        metrics["policy_service/action_mean_latency_seconds"] = float(
            np.mean(latencies)
        )
        metrics["policy_service/action_p95_latency_seconds"] = float(
            np.percentile(latencies, 95)
        )
    return metrics


def drain_policy_service_metrics(rollout_group: Any) -> dict[str, float]:
    """The policy-server metrics since the last call, then reset the servers' counters."""
    if rollout_group is None:
        return {}
    metrics = policy_service_metrics(rollout_group)
    reset_stats = getattr(rollout_group, "reset_policy_server_stats", None)
    if callable(reset_stats):
        reset_stats()
    return metrics


def trajectory_metrics(
    trajectories: Sequence[ChunkTrajectory], prefix: str
) -> dict[str, float]:
    """Success rate, episode count, mean length and mean return of a set of episodes."""
    return {
        f"{prefix}/success_rate": float(
            np.mean([trajectory.success for trajectory in trajectories])
        ),
        f"{prefix}/episodes": float(len(trajectories)),
        f"{prefix}/mean_env_steps": float(
            np.mean([trajectory.env_steps for trajectory in trajectories])
        ),
        f"{prefix}/mean_return": float(
            np.mean([trajectory.rewards.sum() for trajectory in trajectories])
        ),
    }


def learner_metrics(results: Sequence[Any]) -> dict[str, float]:
    """The metrics every rank reports, averaged over ranks, under ``learner/``."""
    learner_names = set.intersection(*(set(result.metrics) for result in results))
    return {
        f"learner/{name}": float(np.mean([result.metrics[name] for result in results]))
        for name in sorted(learner_names)
    }


def rank_mean_metrics(results: Sequence[Any]) -> Mapping[str, float]:
    """One update's metrics: rank 0's on one rank, else the mean over ranks of the metrics every rank reports."""
    if len(results) == 1:
        return results[0].metrics
    names = set.intersection(*(set(result.metrics) for result in results))
    return {name: float(np.mean([result.metrics[name] for result in results])) for name in sorted(names)}
