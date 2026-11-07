# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in robot: seven joints that move toward their targets, a camera that tracks them.

Replace the dynamics with your simulator or robot bridge; keep the contract
(``docs/pages/extending/add_a_simulator.md``, "Environment contract").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np

from isaac_rl_posttraining.data import Array, Observation, dim

NUM_JOINTS = 7
IMAGE_SIZE = 16


@dataclass(frozen=True)
class MyRobotObservation(Observation):
    """What the environment emits per step; the launch plan checks the policy against it."""

    state: Array[np.float32, dim.state_dim]
    image: Array[np.uint8, dim.H, dim.W, 3]


class MyRobotEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self, task: str, max_episode_steps: int, seed: int | None = None
    ) -> None:
        super().__init__()
        self.task = task
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (NUM_JOINTS,), np.float32),
                "image": gym.spaces.Box(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (NUM_JOINTS,), np.float32)
        self._joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._goal = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._steps = 0
        if seed is not None:
            self.reset(seed=seed)

    def _observation(self) -> dict[str, np.ndarray]:
        frame = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 96, dtype=np.uint8)
        row = int((np.clip(self._joints[0], -1.0, 1.0) + 1.0) * 0.5 * (IMAGE_SIZE - 4))
        frame[row : row + 4, :4] = 230
        return {"state": self._joints.copy(), "image": frame}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._goal = self.np_random.uniform(-0.6, 0.6, size=NUM_JOINTS).astype(
            np.float32
        )
        self._joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._steps = 0
        return self._observation(), {"success": False, "is_success": False}

    def step(self, action):
        targets = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._joints = self._joints + 0.1 * (targets - self._joints)
        self._steps += 1
        success = bool(np.linalg.norm(self._joints - self._goal) < 0.15)
        truncated = (not success) and self._steps >= self.max_episode_steps
        return (
            self._observation(),
            0.0,
            success,
            truncated,
            {"success": success, "is_success": success},
        )

    def close(self) -> None:
        pass
