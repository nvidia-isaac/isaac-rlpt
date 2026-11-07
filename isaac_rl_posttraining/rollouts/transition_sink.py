# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write destination for collected transitions.

The data source writes each decision through ``sink.push(transition)``;
``InProcessBufferSink`` forwards to a replay buffer's ``add``, and
``discard_open`` drops the stored head of an episode cut short.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from isaac_rl_posttraining.buffers.replay import ReplayBuffer
from isaac_rl_posttraining.data import Transition


@runtime_checkable
class TransitionSink(Protocol):
    """A destination for collected transitions. ``push`` mirrors ``ReplayBuffer.add``."""

    def push(self, transition: Transition) -> None: ...

    def discard_open(self, episode_id: int | str) -> None: ...


class InProcessBufferSink:
    """Forward ``push(transition)`` to an in-process replay buffer's ``add``."""

    def __init__(self, buffer: ReplayBuffer):
        self._buffer = buffer

    def push(self, transition: Transition) -> None:
        self._buffer.add(transition)

    def discard_open(self, episode_id: int | str) -> None:
        self._buffer.discard_open(episode_id)

    def __len__(self) -> int:
        return len(self._buffer)
