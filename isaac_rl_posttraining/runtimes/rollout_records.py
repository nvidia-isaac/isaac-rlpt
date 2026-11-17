# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every rollout worker records: shaped rewards, camera frames, timings, and the typed transition of one decision."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np

from isaac_rl_posttraining.data import (
    DynamicObservation,
    EnvAction,
    LearnerAction,
    PolicyOutputs,
    Transition,
    TransitionMetadata,
    episode_success,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.policy import (
    InfoValue,
    Observation,
    ObservationValue,
)

__all__ = [
    "NEXT_OBSERVATION_TIMING",
    "RewardTransform",
    "SparseOutcomeReward",
    "TransformedReward",
    "decision_transition",
    "image_frames",
    "learning_reward_channel",
    "mirror_renamed_timings",
    "timed",
]

#: Seconds spent encoding (and, for valued decisions, valuing) next observations.
NEXT_OBSERVATION_TIMING = "next_observation_inference_seconds"
#: The timing names before the rename, reported alongside for one release.
_RENAMED_TIMINGS = {NEXT_OBSERVATION_TIMING: "successor_inference_seconds"}


def mirror_renamed_timings(timings: dict[str, float]) -> dict[str, float]:
    """Also report each renamed timing under its old name."""
    for name, old_name in _RENAMED_TIMINGS.items():
        if name in timings:
            timings[old_name] = timings[name]
    return timings


def image_frames(observation: Observation) -> dict[str, np.ndarray]:
    """The camera images of one observation: its ``uint8 [H, W, 3]`` entries."""
    if not isinstance(observation, Mapping):
        return {}
    frames: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.uint8
            and value.ndim == 3
            and value.shape[-1] == 3
        ):
            frames[key] = np.array(value, copy=True)
    return frames


@dataclass(frozen=True)
class TransformedReward:
    """A shaped reward channel: its per-simulator-step values and the decision's own scalar."""

    value: float
    sequence: tuple[float, ...]


class RewardTransform(Protocol):
    """Shape one executed action chunk's rewards into a named reward channel.

    ``channel``, a class constant, names the channel the worker records the
    shaped per-step values under; the simulator's own rewards stay in the
    ``env`` channel.
    """

    channel: ClassVar[str]

    def __call__(
        self,
        rewards: Sequence[float],
        *,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, InfoValue],
    ) -> TransformedReward: ...


@dataclass(frozen=True)
class SparseOutcomeReward:
    """The terminal-outcome reward: one value on the step that ends the episode, another elsewhere."""

    channel: ClassVar[str] = "sparse_outcome"

    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_reward: float = 0.0

    def __call__(
        self,
        rewards: Sequence[float],
        *,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, InfoValue],
    ) -> TransformedReward:
        if not rewards:
            raise ValueError("rewards must not be empty")
        succeeded = bool(info.get("is_success", info.get("success", False)))
        if succeeded:
            final_reward = float(self.success_reward)
        elif terminated or truncated:
            final_reward = float(self.failure_reward)
        else:
            final_reward = float(self.step_reward)
        sequence = (float(self.step_reward),) * (len(rewards) - 1) + (final_reward,)
        return TransformedReward(value=final_reward, sequence=sequence)


def learning_reward_channel(reward_transform: RewardTransform | None) -> str:
    """The reward channel a worker's decisions are learned and scored from: the transform's, else ``env``."""
    return "env" if reward_transform is None else reward_transform.channel


@contextmanager
def timed(timings: dict[str, float], name: str) -> Iterator[None]:
    """Add the block's wall-clock seconds to ``timings[name]`` when it completes."""
    started = time.perf_counter()
    yield
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started)


def _named_observation(
    name: str, observation: Observation
) -> Mapping[str, ObservationValue]:
    """Replay records name their entries; a bare array observation cannot be recorded."""
    if isinstance(observation, Mapping):
        return observation
    raise TypeError(
        f"environments emit name-keyed observation mappings; {name} is a {type(observation).__name__}"
    )


def decision_transition(
    chunk: ActionChunk,
    *,
    episode_id: int | str,
    observation: Observation,
    next_observation: Observation,
    rewards: Sequence[float],
    terminated: bool,
    truncated: bool,
    info: Mapping[str, InfoValue],
    decision_cap_reached: bool,
    reward_transform: RewardTransform | None,
    metadata: TransitionMetadata,
    next_value: float | None = None,
) -> Transition:
    """File one executed decision as a typed transition.

    ``rewards`` are the simulator's rewards of the executed steps; a
    ``reward_transform`` adds its shaped channel next to them. Success is
    read only from the step that ended the episode, and never when the
    worker's own decision cap ended it: a capped episode is a failure
    whatever flags the simulator still reports.
    """
    steps = len(rewards)
    if steps < 1:
        raise RuntimeError("an ActionChunk executed zero simulator steps")
    ended = terminated or truncated
    channels: dict[str, np.ndarray] = {"env": np.asarray(rewards, dtype=np.float32)}
    if reward_transform is not None:
        shaped = reward_transform(
            rewards, terminated=terminated, truncated=truncated, info=info
        )
        if len(shaped.sequence) != steps:
            raise ValueError(
                "reward transform must return one sequence value per executed simulator step"
            )
        channels[reward_transform.channel] = np.asarray(
            shaped.sequence, dtype=np.float32
        )
    terminated_flags: np.ndarray = np.zeros(steps, dtype=bool)
    truncated_flags: np.ndarray = np.zeros(steps, dtype=bool)
    success_flags: np.ndarray = np.zeros(steps, dtype=bool)
    terminated_flags[-1] = terminated
    truncated_flags[-1] = truncated
    success_flags[-1] = ended and not decision_cap_reached and episode_success(info)

    learner_action = None
    if chunk.learner_action is not None:
        learner_action = LearnerAction(
            chunk.learner_action_kind, chunk.learner_action[0].copy()
        )
    outputs: dict[str, Any] = {}
    if chunk.behavior_log_prob is not None:
        outputs["log_prob"] = np.asarray(chunk.behavior_log_prob[0], dtype=np.float32)
    if chunk.behavior_value is not None:
        outputs["value"] = np.asarray(chunk.behavior_value[0], dtype=np.float32)
    if next_value is not None:
        outputs["next_value"] = np.asarray(next_value, dtype=np.float32)
    if chunk.cached_features is not None:
        outputs["features"] = {
            key: np.array(value[0], copy=True)
            for key, value in chunk.cached_features.items()
        }
    return Transition(
        episode_id=episode_id,
        observation=DynamicObservation.from_mapping(
            _named_observation("observation", observation)
        ),
        next_observation=DynamicObservation.from_mapping(
            _named_observation("next_observation", next_observation)
        ),
        learner_action=learner_action,
        env_action=EnvAction(
            commands=chunk.execution_actions[0, :steps].copy(),
            execution_length=chunk.execution_length,
        ),
        rewards=channels,
        terminated=terminated_flags,
        truncated=truncated_flags,
        success=success_flags,
        behavior_source=chunk.source,
        metadata=metadata,
        policy_outputs=PolicyOutputs(**outputs),
    )
