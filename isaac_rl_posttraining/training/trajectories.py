# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-chunk trajectories: the on-policy loop's view of the episodes it collected."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from isaac_rl_posttraining.runtimes.base import CollectedDecision, RolloutResult
from isaac_rl_posttraining.utils.hashing import digest_value

__all__ = ["ChunkTrajectory", "DiscountPer", "trajectories_from_results"]

#: What one discount factor spans: every simulator step a chunk executes, or the
#: whole chunk decision (RLinf's chunk-level convention for flow-matching heads).
DiscountPer = Literal["simulator_step", "decision"]


@dataclass(frozen=True)
class ChunkTrajectory:
    """One ordered simulator episode represented by action-chunk decisions."""

    environment_index: int
    episode_id: Any
    observations: tuple[Any, ...]
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_steps: np.ndarray
    successes: np.ndarray
    initial_observation_sha256: str
    first_execution_action_sha256: str
    execution_action_sha256: str
    reset_seed: int | None = None
    reset_seed_scope: Literal["environment", "worker_batch"] | None = None
    initial_observation_sha256_by_key: tuple[tuple[str, str], ...] = ()
    observation_sha256_by_decision: tuple[str, ...] = ()
    execution_action_sha256_by_decision: tuple[str, ...] = ()
    initial_raw_observation_sha256: str = ""
    initial_raw_observation_sha256_by_key: tuple[tuple[str, str], ...] = ()
    #: Per-decision activations recorded by the rollout side (see
    #: ``policies/feature_cache.py``), or ``None`` when none were recorded.
    cached_features: tuple[Mapping[str, np.ndarray], ...] | None = None

    @property
    def success(self) -> bool:
        return bool(self.successes.any())

    @property
    def env_steps(self) -> int:
        return int(self.bootstrap_steps.sum())


def _policy_output(decision: CollectedDecision, name: str) -> float:
    value = getattr(decision.transition.policy_outputs, name)
    if value is None:
        raise ValueError(f"PPO transition is missing policy output {name!r}")
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"PPO policy output {name!r} must contain one scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"PPO policy output {name!r} must be finite")
    return result


def _chunk_reward(
    decision: CollectedDecision, gamma: float, discount_per: DiscountPer, reward_channel: str
) -> float:
    """Collapse a chunk's per-step rewards into the one reward of its decision.

    Per simulator step, later steps inside the chunk are discounted; per
    decision, the chunk is one time step and its rewards simply add up.
    """
    rewards = np.asarray(decision.transition.reward(reward_channel), dtype=np.float64)
    if discount_per == "decision":
        return float(rewards.sum())
    discounts = np.power(float(gamma), np.arange(len(rewards), dtype=np.float64))
    return float(np.dot(discounts, rewards))


def _episode_cached_features(
    decisions: Sequence[CollectedDecision],
) -> tuple[Mapping[str, np.ndarray], ...] | None:
    cached = [decision.transition.policy_outputs.features for decision in decisions]
    if all(item is None for item in cached):
        return None
    if any(item is None for item in cached):
        raise ValueError(
            "one PPO trajectory mixes decisions with and without cached features"
        )
    return tuple(cached)


def _learner_action(decision: CollectedDecision) -> np.ndarray:
    learner_action = decision.transition.learner_action
    if learner_action is None:
        raise ValueError("PPO trajectories need the learner action of every decision")
    return np.asarray(learner_action.value)


def _behavior_outputs(
    decisions: Sequence[CollectedDecision], require_behavior_metadata: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-decision ``(values, next_values, log_probs)``; zeros when the behaviour policy recorded none."""
    if require_behavior_metadata:
        values = np.asarray(
            [_policy_output(item, "value") for item in decisions], dtype=np.float32
        )
        next_values = np.asarray(
            [_policy_output(item, "next_value") for item in decisions], dtype=np.float32
        )
        log_probs = np.asarray(
            [_policy_output(item, "log_prob") for item in decisions], dtype=np.float32
        )
    else:
        values = np.zeros(len(decisions), dtype=np.float32)
        next_values = np.zeros(len(decisions), dtype=np.float32)
        log_probs = np.zeros(len(decisions), dtype=np.float32)
    return values, next_values, log_probs


def _chunk_trajectory(
    environment_index: int,
    episode_id: Any,
    decisions: list[CollectedDecision],
    *,
    gamma: float,
    require_behavior_metadata: bool,
    discount_per: DiscountPer,
    reward_channel: str,
) -> ChunkTrajectory:
    """One complete episode's decisions, ordered and checked, as a trajectory."""
    decisions.sort(key=lambda item: item.transition.metadata.step_in_episode)
    steps = [item.transition.metadata.step_in_episode for item in decisions]
    if len(steps) != len(set(steps)):
        raise ValueError(
            f"environment {environment_index} episode {episode_id!r} has "
            "duplicate step indices"
        )
    if not decisions[-1].transition.ended:
        raise ValueError(
            "episode-bounded PPO collection returned an incomplete trajectory"
        )
    observations: list[dict[str, np.ndarray]] = [
        {
            name: np.asarray(value)
            for name, value in decision.transition.observation.to_mapping().items()
        }
        for decision in decisions
    ]
    values, next_values, log_probs = _behavior_outputs(
        decisions, require_behavior_metadata
    )
    terminated = np.asarray(
        [item.transition.terminated[-1] for item in decisions], dtype=bool
    )
    next_values[terminated] = 0.0
    metadata = [item.transition.metadata for item in decisions]
    initial_raw_observation_hash = metadata[0].extra.get("raw_observation_sha256")
    raw_observation_hashes_by_key = metadata[0].extra.get(
        "raw_observation_sha256_by_key", {}
    )
    if not isinstance(raw_observation_hashes_by_key, Mapping):
        raise TypeError("raw_observation_sha256_by_key must be a mapping")
    reset_seeds = {entry.reset_seed for entry in metadata}
    reset_seed_scopes = {entry.reset_seed_scope for entry in metadata}
    if len(reset_seeds) != 1 or len(reset_seed_scopes) != 1:
        raise ValueError("one PPO trajectory must use one reset seed and scope")
    executed_actions = [
        np.asarray(item.action_chunk.execution_actions)[0, : item.transition.steps]
        for item in decisions
    ]
    return ChunkTrajectory(
        environment_index=environment_index,
        episode_id=episode_id,
        observations=tuple(observations),
        actions=np.stack([_learner_action(item) for item in decisions]).astype(
            np.float32
        ),
        log_probs=log_probs,
        values=values,
        next_values=next_values,
        rewards=np.asarray(
            [
                _chunk_reward(item, gamma, discount_per, reward_channel)
                for item in decisions
            ],
            dtype=np.float32,
        ),
        terminated=terminated,
        truncated=np.asarray(
            [item.transition.truncated[-1] for item in decisions], dtype=bool
        ),
        bootstrap_steps=np.asarray(
            [item.transition.steps for item in decisions], dtype=np.int64
        ),
        successes=np.asarray(
            [item.transition.succeeded for item in decisions], dtype=bool
        ),
        initial_observation_sha256=digest_value(observations[0]),
        first_execution_action_sha256=digest_value(executed_actions[0]),
        execution_action_sha256=digest_value(executed_actions),
        reset_seed=reset_seeds.pop(),
        reset_seed_scope=reset_seed_scopes.pop(),
        initial_observation_sha256_by_key=tuple(
            (name, digest_value(value))
            for name, value in sorted(observations[0].items())
        ),
        observation_sha256_by_decision=tuple(
            digest_value(observation) for observation in observations
        ),
        execution_action_sha256_by_decision=tuple(
            digest_value(action) for action in executed_actions
        ),
        initial_raw_observation_sha256=(
            str(initial_raw_observation_hash)
            if initial_raw_observation_hash is not None
            else ""
        ),
        initial_raw_observation_sha256_by_key=tuple(
            (str(name), str(value))
            for name, value in sorted(raw_observation_hashes_by_key.items())
        ),
        cached_features=_episode_cached_features(decisions),
    )


def trajectories_from_results(
    results: Sequence[RolloutResult],
    *,
    gamma: float,
    require_behavior_metadata: bool = True,
    discount_per: DiscountPer = "simulator_step",
    reward_channel: str = "env",
) -> tuple[ChunkTrajectory, ...]:
    """Group runtime results into complete, ordered PPO trajectories.

    ``reward_channel`` names the reward channel the trajectories learn from;
    the workers record the simulator's reward as ``env`` and a shaped reward
    under its transform's channel.
    """
    grouped: dict[tuple[int, Any], list[CollectedDecision]] = {}
    for result in sorted(results, key=lambda item: item.worker_index):
        for decision in result.decisions:
            key = (decision.environment_index, decision.transition.episode_id)
            grouped.setdefault(key, []).append(decision)

    return tuple(
        _chunk_trajectory(
            environment_index,
            episode_id,
            decisions,
            gamma=gamma,
            require_behavior_metadata=require_behavior_metadata,
            discount_per=discount_per,
            reward_channel=reward_channel,
        )
        for (environment_index, episode_id), decisions in grouped.items()
    )
