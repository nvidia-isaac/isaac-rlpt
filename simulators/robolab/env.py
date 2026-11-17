# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deliberately narrow DROID RoboLab adapter for DSRL smoke runs.

The actor sees only an explicit proprioceptive state and a fixed single-task
language vector.  The untouched RoboLab observation is carried privately for
the frozen GR00T client, which remains the sole owner of camera and DROID
preprocessing.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaac_rl_posttraining.envs.steering_wrappers import LANGUAGE_EMBEDDING_DIM

__all__ = ["ROBOLAB_DROID_DSRL_KEY_MAPPING", "RoboLabDroidEnv"]


ROBOLAB_DROID_DSRL_KEY_MAPPING = {
    "state": "state",
    "language": "language",
    "image": [],
}
"""Actor-facing keys for the intentionally state-only smoke adapter."""


class RoboLabDroidEnv(gym.Env):
    """Expose one RoboLab DROID environment to the DSRL stack.

    This is intentionally not a generic RoboLab environment abstraction.
    It supports exactly one DROID joint-position environment because the first
    smoke must preserve GR00T N1.7's native preprocessing and action contract.
    Returned rewards and termination flags keep a leading one-element axis,
    the batch-one vector-environment interface the DSRL stack was built on.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: Any,
        *,
        instruction: str,
        task: str,
        max_episode_steps: int | None = None,
    ) -> None:
        self.env = env
        self.instruction = str(instruction)
        self.task = str(task)
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when provided")
        self.max_episode_steps = max_episode_steps
        self.steering_key_mapping = dict(ROBOLAB_DROID_DSRL_KEY_MAPPING)
        self._episode_index = -1
        self._step_index = 0

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
                ),
                "language": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(LANGUAGE_EMBEDDING_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._action_dim(),),
            dtype=np.float32,
        )

    @property
    def episode_id(self) -> str:
        """Stable identifier included in each harness outcome trace."""
        return f"robolab-{self.task}-episode-{self._episode_index}"

    @property
    def step_index(self) -> int:
        """Current physical RoboLab action index within this episode."""
        return self._step_index

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh RoboLab episode and retain its native observation."""
        del kwargs
        if hasattr(self.env, "reset_eval_state"):
            self.env.reset_eval_state()
        raw_obs, _ = self.env.reset()
        self._episode_index += 1
        self._step_index = 0
        return self._format_obs(raw_obs), {"supports_action_chunking": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Execute exactly one DROID joint-position action through RoboLab."""
        action_arr = np.asarray(action, dtype=np.float32)
        expected = (1, self._action_dim())
        if action_arr.shape != expected:
            raise ValueError(f"expected one RoboLab DROID action {expected}, got {action_arr.shape}")
        device = getattr(self.env, "device", None)
        action_tensor = torch.as_tensor(action_arr, device=device)
        raw_obs, reward, terminated, truncated, _ = self.env.step(action_tensor)
        terminated_value = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
        raw_truncated_value = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
        raw_terminal = terminated_value or raw_truncated_value
        reward_value = float(torch.as_tensor(reward).reshape(-1)[0].item())
        self._step_index += 1
        capped = (
            not raw_terminal
            and self.max_episode_steps is not None
            and self._step_index >= self.max_episode_steps
        )
        truncated_value = raw_truncated_value or capped
        info = {
            "step_in_episode": self._step_index - 1,
            "num_steps": 1,
            # A client-side time limit is a failure, not a RoboLab success.
            "is_success": self._success(raw_terminal),
        }
        return (
            self._format_obs(raw_obs),
            np.asarray([reward_value], dtype=np.float32),
            np.asarray([terminated_value]),
            np.asarray([truncated_value]),
            [info],
        )

    def close(self) -> None:
        """Close the native simulator when it exposes a close method."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _format_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        robot = raw_obs["proprio_obs"]
        state = torch.cat(
            [
                robot["arm_joint_pos"],
                robot["gripper_pos"],
                robot["ee_pos"],
                robot["ee_quat"],
            ],
            dim=-1,
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        if state.shape != (1, 15):
            raise ValueError(f"expected one DROID state shaped (1, 15), got {state.shape}")
        return {
            "state": state,
            "language": np.zeros((1, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
            "prompt": self.instruction,
            "robolab_raw": raw_obs,
            "robolab_context": {
                "episode_id": self.episode_id,
                "env_id": 0,
                "step_idx": self.step_index,
                "task": self.task,
            },
        }

    def _action_dim(self) -> int:
        action_manager = getattr(self.env, "action_manager", None)
        dim = getattr(action_manager, "total_action_dim", None)
        if dim is None:
            dim = self.env.action_space.shape[-1]
        return int(dim)

    def _success(self, terminal: bool) -> bool:
        """Read RoboLab's success result only once the single env is terminal."""
        if not terminal or not hasattr(self.env, "get_env_results"):
            return False
        results = self.env.get_env_results()
        return bool(results and results[0].get("success", False))
