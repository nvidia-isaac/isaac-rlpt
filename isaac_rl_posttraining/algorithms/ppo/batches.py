# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semi-Markov GAE over action-chunk trajectories, sharded into PPO learner batches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from isaac_rl_posttraining.algorithms.ppo.advantages import (
    generalized_advantage_estimate,
)
from isaac_rl_posttraining.policies.feature_cache import (
    CACHED_FEATURES_KEY,
    stack_features,
)
from isaac_rl_posttraining.runtimes.learner import LearnerBatch
from isaac_rl_posttraining.training.trajectories import ChunkTrajectory, DiscountPer

__all__ = ["build_rank_batches"]


def _stack_mapping_observations(
    observations: Sequence[Any],
) -> dict[str, np.ndarray]:
    if not all(isinstance(observation, Mapping) for observation in observations):
        raise TypeError("PPO observations cannot mix mappings and arrays")
    keys = tuple(observations[0])
    if any(tuple(observation) != keys for observation in observations[1:]):
        raise ValueError("PPO observations must carry identical ordered keys")
    return {
        key: np.stack([np.asarray(observation[key]) for observation in observations])
        for key in keys
    }


def _stack_observations(
    observations: Sequence[Any],
) -> dict[str, np.ndarray] | np.ndarray:
    if not observations:
        raise ValueError("cannot stack an empty observation sequence")
    if isinstance(observations[0], Mapping):
        return _stack_mapping_observations(observations)
    if any(isinstance(observation, Mapping) for observation in observations[1:]):
        raise TypeError("PPO observations cannot mix arrays and mappings")
    return np.stack([np.asarray(observation) for observation in observations])


def _trajectory_samples(
    trajectory: ChunkTrajectory,
    *,
    gamma: float,
    gae_lambda: float,
    discount_per: DiscountPer,
    bootstrap_at_time_limit: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """GAE for one trajectory, returned as per-chunk samples plus advantages and returns."""
    if discount_per == "decision":
        bootstrap_steps = np.ones_like(trajectory.bootstrap_steps)
    else:
        bootstrap_steps = trajectory.bootstrap_steps
    next_values = trajectory.next_values
    if not bootstrap_at_time_limit:
        next_values = np.where(trajectory.truncated, 0.0, next_values)
    advantages, returns = generalized_advantage_estimate(
        trajectory.rewards[:, None],
        trajectory.values[:, None],
        next_values[:, None],
        trajectory.terminated[:, None],
        trajectory.truncated[:, None],
        bootstrap_steps[:, None],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = advantages[:, 0]
    returns = returns[:, 0]
    samples = [
        {
            "observation": trajectory.observations[index],
            "action": trajectory.actions[index],
            "old_log_prob": trajectory.log_probs[index],
            "advantage": advantages[index],
            "return": returns[index],
            "old_value": trajectory.values[index],
            "cached_features": (
                None
                if trajectory.cached_features is None
                else trajectory.cached_features[index]
            ),
        }
        for index in range(len(trajectory.rewards))
    ]
    return samples, advantages, returns


def _rank_batch(
    chosen: Sequence[dict[str, Any]],
    *,
    with_cached_features: bool,
    samples_per_rank: int,
) -> LearnerBatch:
    tensors: dict[str, Any] = {
        "observations": _stack_observations(
            [sample["observation"] for sample in chosen]
        ),
        "actions": np.stack([sample["action"] for sample in chosen]).astype(np.float32),
        "old_log_probs": np.asarray(
            [sample["old_log_prob"] for sample in chosen],
            dtype=np.float32,
        ),
        "advantages": np.asarray(
            [sample["advantage"] for sample in chosen],
            dtype=np.float32,
        ),
        "returns": np.asarray(
            [sample["return"] for sample in chosen],
            dtype=np.float32,
        ),
        "old_values": np.asarray(
            [sample["old_value"] for sample in chosen],
            dtype=np.float32,
        ),
    }
    if with_cached_features:
        tensors[CACHED_FEATURES_KEY] = stack_features(
            [sample["cached_features"] for sample in chosen]
        )
    return LearnerBatch(tensors=tensors, samples=samples_per_rank)


def build_rank_batches(
    trajectories: Sequence[ChunkTrajectory],
    *,
    num_learner_ranks: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    rng: np.random.Generator,
    discount_per: DiscountPer = "simulator_step",
    bootstrap_at_time_limit: bool = True,
) -> tuple[tuple[LearnerBatch, ...], dict[str, float]]:
    """Compute semi-Markov GAE, pad, shuffle, and shard a global batch.

    ``discount_per="decision"`` treats every chunk as one time step regardless
    of how many simulator steps it executed. With ``bootstrap_at_time_limit``
    off, a truncated episode ends with a zero successor value instead of
    bootstrapping from the last observation's value.
    """
    samples: list[dict[str, Any]] = []
    all_advantages: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    for trajectory in trajectories:
        trajectory_samples, advantages, returns = _trajectory_samples(
            trajectory,
            gamma=gamma,
            gae_lambda=gae_lambda,
            discount_per=discount_per,
            bootstrap_at_time_limit=bootstrap_at_time_limit,
        )
        all_advantages.append(advantages)
        all_returns.append(returns)
        samples.extend(trajectory_samples)
    if not samples:
        raise ValueError("no PPO samples were collected")
    cached_flags = {sample["cached_features"] is not None for sample in samples}
    if len(cached_flags) != 1:
        raise ValueError("PPO samples must all carry cached features or none of them")
    with_cached_features = cached_flags.pop()
    if num_learner_ranks < 1 or minibatch_size < 1:
        raise ValueError("learner ranks and minibatch size must be positive")
    granule = num_learner_ranks * minibatch_size
    padded_size = int(math.ceil(len(samples) / granule) * granule)
    duplicated = padded_size - len(samples)
    order = np.concatenate(
        (
            np.arange(len(samples), dtype=np.int64),
            rng.integers(0, len(samples), size=duplicated, dtype=np.int64),
        )
    )
    rng.shuffle(order)
    samples_per_rank = padded_size // num_learner_ranks
    batches: list[LearnerBatch] = []
    for learner_rank in range(num_learner_ranks):
        rank_order = order[
            learner_rank * samples_per_rank : (learner_rank + 1) * samples_per_rank
        ]
        chosen = [samples[int(index)] for index in rank_order]
        batches.append(
            _rank_batch(
                chosen,
                with_cached_features=with_cached_features,
                samples_per_rank=samples_per_rank,
            )
        )
    advantages = np.concatenate(all_advantages)
    returns = np.concatenate(all_returns)
    return tuple(batches), {
        "samples": float(len(samples)),
        "duplicated_samples": float(duplicated),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "return_mean": float(returns.mean()),
    }
