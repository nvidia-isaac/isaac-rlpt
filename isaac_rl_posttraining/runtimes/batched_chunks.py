# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode and per-chunk bookkeeping of the batched rollout worker."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from isaac_rl_posttraining.runtimes.base import CollectedDecision

__all__ = ["ChunkOutcome", "EpisodeStart", "chain_next_values", "unbatch_observation"]


def unbatch_observation(
    observation: Mapping[str, Any], index: int, batch_size: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in observation.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            result[key] = np.array(value[index], copy=True)
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = copy.deepcopy(value[index])
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class EpisodeStart:
    """How one batched episode began, by environment index; its first decisions record it."""

    reset_seed: int | None
    episode_ids: tuple[int, ...]
    policy_seeds: tuple[int | None, ...]
    reset_info: tuple[dict[str, Any], ...]
    raw_observation_sha256: tuple[str, ...]
    raw_observation_sha256_by_key: tuple[dict[str, str], ...]

    def decision_extra(
        self,
        environment_index: int,
        info: Mapping[str, Any],
        *,
        first_decision: bool,
        decision_cap_reached: bool,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if first_decision:
            extra.update(self.reset_info[environment_index])
        extra.update(info)
        if decision_cap_reached:
            extra["decision_cap_reached"] = True
            extra["terminal_reason"] = "decision_cap"
        if first_decision:
            extra["policy_seed"] = self.policy_seeds[environment_index]
            extra["raw_observation_sha256"] = self.raw_observation_sha256[
                environment_index
            ]
            extra["raw_observation_sha256_by_key"] = self.raw_observation_sha256_by_key[
                environment_index
            ]
        return extra


@dataclass
class ChunkOutcome:
    """What one lockstep action chunk produced for the active environments, by environment index."""

    active_indices: tuple[int, ...]
    batch_size: int
    #: The latest batched observation.
    observations: Mapping[str, np.ndarray]
    still_executing: set[int]
    rewards: dict[int, list[float]]
    terminal_info: dict[int, Mapping[str, Any]]
    terminated: np.ndarray
    truncated: np.ndarray
    successors: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    replay_successors: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    next_values: dict[int, float] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        active_indices: Sequence[int],
        observations: Mapping[str, np.ndarray],
        batch_size: int,
    ) -> ChunkOutcome:
        return cls(
            active_indices=tuple(active_indices),
            batch_size=batch_size,
            observations=observations,
            still_executing=set(active_indices),
            rewards={index: [] for index in active_indices},
            terminal_info={index: {} for index in active_indices},
            terminated=np.zeros(batch_size, dtype=bool),
            truncated=np.zeros(batch_size, dtype=bool),
        )

    def record_step(
        self,
        next_observations: Mapping[str, np.ndarray],
        step_rewards: np.ndarray,
        step_terminated: np.ndarray,
        step_truncated: np.ndarray,
        infos: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record one batched step for the environments still executing their chunk."""
        for environment_index in tuple(self.still_executing):
            self.rewards[environment_index].append(
                float(step_rewards[environment_index])
            )
            self.terminal_info[environment_index] = dict(infos[environment_index])
            if step_terminated[environment_index] or step_truncated[environment_index]:
                self.terminated[environment_index] = bool(
                    step_terminated[environment_index]
                )
                self.truncated[environment_index] = bool(
                    step_truncated[environment_index]
                )
                self.successors[environment_index] = unbatch_observation(
                    next_observations, environment_index, self.batch_size
                )
                self.still_executing.remove(environment_index)
        self.observations = next_observations

    def finish(self, *, final_chunk: bool) -> None:
        """Environments that ran the whole chunk continue from the latest observation."""
        for environment_index in tuple(self.still_executing):
            self.successors[environment_index] = unbatch_observation(
                self.observations, environment_index, self.batch_size
            )
            if final_chunk:
                self.truncated[environment_index] = True


def chain_next_values(
    decisions: Sequence[CollectedDecision],
) -> list[CollectedDecision]:
    """Give each valued decision the next decision's value in its episode; 0 at the episode's end."""
    following: dict[int, CollectedDecision] = {}
    chained: list[CollectedDecision] = []
    for decision in reversed(decisions):
        outputs = decision.transition.policy_outputs
        if outputs.value is None:
            chained.append(decision)
            continue
        successor = following.get(decision.environment_index)
        if successor is None:
            next_value = np.asarray(0.0, dtype=np.float32)
        else:
            next_value = np.asarray(
                successor.transition.policy_outputs.value, dtype=np.float32
            ).reshape(())
        transition = replace(
            decision.transition,
            policy_outputs=replace(outputs, next_value=next_value),
        )
        decision = replace(decision, transition=transition)
        chained.append(decision)
        following[decision.environment_index] = decision
    chained.reverse()
    return chained
