# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstrations held in a replay buffer, on the controller.

A demonstration is a replay of a successful episode, so it is stored the way
every other trajectory is, and it lives where every other buffer in this repo
lives -- with the controller, which draws each rank's rows and ships them
inside that rank's ``LearnerBatch``. What is stored is env space; the rank
that scores a batch encodes it, because only the policy family knows its own
processor and uint8 frames are far smaller on the wire than encoded ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from isaac_rl_posttraining.buffers.replay import ReplayBuffer
from isaac_rl_posttraining.buffers.samplers import Sampler
from isaac_rl_posttraining.data.transitions import (
    LearnerAction,
    Observation,
    Transition,
    TransitionMetadata,
)

__all__ = [
    "DEMONSTRATIONS_KEY",
    "demonstration_slice",
    "draw_demonstration_tensors",
    "build_demonstration_buffer",
    "demonstration_transition",
    "load_demonstration_buffer",
]

DEMONSTRATION_SOURCE = "expert"

#: Where a rank's demonstrations travel inside its ``LearnerBatch``.
DEMONSTRATIONS_KEY = "demonstrations"


def demonstration_transition(
    episode_id: int | str,
    step: int,
    window: Mapping[str, np.ndarray] | Observation,
    action: np.ndarray,
    *,
    last: bool,
    next_window: Observation | None = None,
    reward: float = 0.0,
    terminal: bool | None = None,
    task: str | None = None,
) -> Transition:
    """One window as a transition: env-space ``window``, the expert's ``action``.

    A BC target needs no successor, so ``next_window`` defaults to an alias of
    ``window`` rather than a second copy of its frames. ``terminal`` (default
    ``last``) says whether the episode's end is a true terminal; when it is
    not, the last window is marked truncated, so a TD target bootstraps there.
    """
    observation = window if isinstance(window, Observation) else Observation.from_mapping(dict(window))
    terminated = bool(last) if terminal is None else bool(last and terminal)
    return Transition(
        episode_id=episode_id,
        observation=observation,
        next_observation=observation if next_window is None else next_window,
        learner_action=LearnerAction(kind="chunk", value=np.asarray(action)),
        env_action=None,
        rewards={"env": np.array([reward], dtype=np.float32)},
        terminated=np.array([terminated]),
        truncated=np.array([bool(last) and not terminated]),
        success=np.array([True]),
        behavior_source=DEMONSTRATION_SOURCE,
        metadata=TransitionMetadata(
            step_in_episode=int(step), extra={} if not task else {"task": task}
        ),
    )


def build_demonstration_buffer(
    windows: Iterable[tuple[int | str, int, Mapping[str, np.ndarray], np.ndarray, bool]],
) -> ReplayBuffer:
    """Fill a buffer from ``(episode, step, observation, action, last)`` windows.

    Sized to hold every window: a demonstration set is fixed, and evicting part
    of it would narrow the anchor without saying so.
    """
    materialized = list(windows)
    if not materialized:
        raise ValueError("a demonstration buffer needs at least one window")
    buffer = ReplayBuffer(capacity=len(materialized))
    for episode_id, step, window, action, last in materialized:
        buffer.add(demonstration_transition(episode_id, step, window, action, last=last))
    return buffer


def load_demonstration_buffer(
    root: str | Path,
    *,
    action_horizon: int,
    cameras: Sequence[str],
    episodes: Sequence[int] | None = None,
    pad_episode_end: bool = False,
    assemble: Callable[[Any], tuple[np.ndarray, np.ndarray]] | None = None,
) -> ReplayBuffer:
    """Every action-chunk window of every episode, in one buffer.

    Each episode is decoded once, whole: building a decoder and seeking costs
    far more than reading from one, ~17 frames/s against ~270.

    By default only windows with a full chunk are kept, as upstream GR00T
    fine-tuning does. ``pad_episode_end`` keeps every step and repeats the last
    action past the end (upstream's ``allow_padding``), so the terminal steps an
    offline critic learns its reward from stay in. Each window's successor is
    the next step's observation, shared rather than copied, and its reward and
    done are the dataset's.
    """
    from isaac_rl_posttraining.training.demonstrations import LeRobotDemonstrationReader

    reader = LeRobotDemonstrationReader(
        root,
        action_horizon=action_horizon,
        cameras=tuple(cameras),
        episodes=episodes,
        assemble=assemble,
    )
    transitions: list[Transition] = []
    for episode in reader.episodes:
        state, action, task = reader.episode_arrays(episode)
        reward, done = reader.episode_outcomes(episode)
        length = len(state)
        count = length if pad_episode_end else length - action_horizon
        if count <= 0:
            continue  # too short to hold one chunk
        frames = {camera: reader.decode_episode(episode, camera) for camera in cameras}
        observations = [
            Observation.from_mapping(
                {"state": state[step]}
                | {camera: decoded[min(step, len(decoded) - 1)] for camera, decoded in frames.items()}
            )
            for step in range(min(count + 1, length))
        ]
        for step in range(count):
            chunk = np.minimum(np.arange(step, step + action_horizon), length - 1)
            transitions.append(
                demonstration_transition(
                    episode,
                    step,
                    observations[step],
                    action[chunk],
                    last=step == count - 1,
                    next_window=observations[min(step + 1, len(observations) - 1)],
                    reward=float(reward[step]),
                    terminal=bool(done[step]),
                    task=task or None,
                )
            )
    if not transitions:
        raise ValueError("a demonstration buffer needs at least one window")
    buffer = ReplayBuffer(capacity=len(transitions))
    for transition in transitions:
        buffer.add(transition)
    return buffer


def draw_demonstration_tensors(sampler: Sampler, count: int) -> dict[str, np.ndarray]:
    """``count`` demonstrations as one flat mapping, the shape the transport takes."""
    batch = sampler.sample(count)
    drawn: dict[str, np.ndarray] = {
        name: np.asarray(value) for name, value in batch["obs"].items()
    }
    drawn["action"] = np.asarray(batch["action"])
    return drawn


def demonstration_slice(
    tensors: Mapping[str, np.ndarray], start: int, size: int
) -> dict[str, object]:
    """One co-training batch's rows, back in the drawn-batch shape.

    Taken from what the controller shipped, so a rank never touches the dataset.
    """
    stop = start + size
    observation = {
        name: value[start:stop] for name, value in tensors.items() if name != "action"
    }
    return {"obs": observation, "action": tensors["action"][start:stop]}
