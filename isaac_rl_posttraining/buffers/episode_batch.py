# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete-episode batching for recurrent off-policy learning."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.buffers.replay import ReplayBuffer
from isaac_rl_posttraining.data import Episode

__all__ = ["sample_episode_batch"]


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), device=device)


def _allocate_padded(example: torch.Tensor, batch_size: int, time_size: int) -> torch.Tensor:
    return torch.zeros((batch_size, time_size, *example.shape), dtype=example.dtype, device=example.device)


def _learner_action(episode: Episode, step: int) -> np.ndarray:
    transition = episode.transitions[step]
    if transition.learner_action is None:
        raise ValueError(f"episode {episode.id!r} has a decision without a learner action")
    return transition.learner_action.value


def sample_episode_batch(
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Sample complete episodes uniformly and pad them for masked full-episode BPTT.

    Rewards are the buffer's configured channel aggregated per decision.
    ``states`` has one more timestep than every transition tensor:
    ``[obs_0, next_obs_0, ..., next_obs_T]``.
    Consequently, the exact previous action for each state is
    ``[zero, action_0, ..., action_T]``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    episodes = buffer.episodes(complete=True)
    if not episodes:
        return {}
    generator = rng if rng is not None else np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))  # NOSONAR - draws from the globally seeded NumPy stream
    chosen = generator.choice(len(episodes), size=batch_size, replace=len(episodes) < batch_size)
    selected = [episodes[int(index)] for index in chosen]
    max_length = max(len(episode.transitions) for episode in selected)

    first = selected[0].transitions[0]
    states: dict[str, torch.Tensor] = {}
    for key, value in first.observation.arrays().items():
        states[key] = _allocate_padded(_to_tensor(value, device), batch_size, max_length + 1)
    actions = _allocate_padded(_to_tensor(_learner_action(selected[0], 0), device).float(), batch_size, max_length)
    rewards = torch.zeros(batch_size, max_length, 1, device=device)
    dones = torch.zeros(batch_size, max_length, 1, dtype=torch.bool, device=device)
    truncateds = torch.zeros(batch_size, max_length, 1, dtype=torch.bool, device=device)
    mask = torch.zeros(batch_size, max_length, 1, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for batch_index, episode in enumerate(selected):
        length = len(episode.transitions)
        lengths[batch_index] = length
        mask[batch_index, :length] = 1.0
        for key, state in states.items():
            state[batch_index, 0] = _to_tensor(episode.transitions[0].observation.arrays()[key], device)
        for step, transition in enumerate(episode.transitions):
            actions[batch_index, step] = _to_tensor(_learner_action(episode, step), device).float()
            rewards[batch_index, step, 0] = transition.decision_reward(
                buffer.reward_channel, aggregation=buffer.reward_aggregation
            )
            dones[batch_index, step, 0] = bool(transition.terminated[-1])
            truncateds[batch_index, step, 0] = bool(transition.truncated[-1])
            for key, state in states.items():
                state[batch_index, step + 1] = _to_tensor(transition.next_observation.arrays()[key], device)

    previous_actions = torch.zeros(batch_size, max_length + 1, *actions.shape[2:], dtype=actions.dtype, device=device)
    previous_actions[:, 1:] = actions
    return {
        "states": states,
        "previous_actions": previous_actions,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "truncateds": truncateds,
        "mask": mask,
        "lengths": lengths,
        "num_complete_episodes": len(episodes),
    }
