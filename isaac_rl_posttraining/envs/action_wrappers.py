# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-chunking wrappers for open-loop execution of policy action chunks.

Chunked policies emit ``[chunk_size, action_dim]`` blocks of future actions.
These wrappers buffer a chunk and execute its first ``n_action_steps`` actions
one environment step at a time, which is the contract the rollout workers rely
on: one decoded chunk == exactly ``n_action_steps`` environment steps (unless
the episode ends early).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np


def _validate_chunking(chunk_size: int, n_action_steps: int) -> None:
    """Check chunking parameters, raising ``ValueError`` on misuse."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if n_action_steps <= 0:
        raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
    if n_action_steps > chunk_size:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) cannot be greater than "
            f"chunk_size ({chunk_size})"
        )


class ActionChunkingWrapper(gym.ActionWrapper):
    """Execute action chunks open-loop on a single environment.

    Each call to :meth:`action` pops one action from the internal buffer; a
    new chunk (``[chunk_size, action_dim]``, optionally with a leading
    singleton batch dimension) refills the buffer with its first
    ``n_action_steps`` rows.

    Args:
        env: The environment to wrap.
        chunk_size: Number of actions per policy chunk.
        n_action_steps: How many actions of each chunk are executed before a
            new chunk is required.
    """

    def __init__(self, env: gym.Env, chunk_size: int, n_action_steps: int) -> None:
        super().__init__(env)
        _validate_chunking(chunk_size, n_action_steps)
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self.action_buffer: np.ndarray | None = None
        self.last_action: np.ndarray | None = None

    def action(self, action: np.ndarray | None) -> np.ndarray:
        """Return the next single action to execute.

        Args:
            action: A new ``[chunk_size, action_dim]`` chunk (a leading
                singleton batch dimension is tolerated), consumed only when
                the buffer is empty.

        Returns:
            One ``[action_dim]`` action.

        Raises:
            ValueError: If a new chunk is needed but ``action`` is not 2-D.
        """
        if action is not None:
            if isinstance(action, np.ndarray) and action.ndim == 3 and action.shape[0] == 1:
                action = action[0]
            if len(action.shape) != 2:
                raise ValueError(
                    "Action must be chunked into a sequence of shape "
                    f"(chunk_size, action_dim), got shape {action.shape}"
                )

        if self.action_buffer is None or self._remaining() == 0:
            # Refill: only the first n_action_steps of the chunk are executed.
            self.action_buffer = action[: self.n_action_steps, :]

        next_action = self.action_buffer[0, :]
        self.action_buffer = self.action_buffer[1:, :]
        self.last_action = next_action
        return next_action

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and discard any buffered actions."""
        self.action_buffer = None
        return super().reset(**kwargs)

    @property
    def is_chunk_empty(self) -> bool:
        """``True`` when a new chunk is required before the next step."""
        return self._remaining() == 0

    def _remaining(self) -> int:
        """Number of buffered actions not yet executed."""
        return 0 if self.action_buffer is None else int(self.action_buffer.shape[0])

    def _get_last_action(self) -> np.ndarray | None:
        """The most recently executed single action."""
        return self.last_action


class VectorActionChunkingWrapper(gym_vector.VectorWrapper):
    """Execute action chunks open-loop on a vectorized environment.

    :meth:`step` accepts three action forms:

    * ``[num_envs, chunk_size, action_dim]`` -- start new per-env buffers and
      execute each env's first action;
    * ``None`` -- continue executing from the existing buffers;
    * ``[num_envs, action_dim]`` -- bypass chunking and forward as-is
      (invalidates the buffers so the two modes never mix).

    Args:
        env: The vectorized environment to wrap.
        chunk_size: Number of actions per policy chunk.
        n_action_steps: How many actions of each chunk are executed before a
            new chunk is required.
    """

    def __init__(
        self, env: gym_vector.VectorEnv, chunk_size: int, n_action_steps: int
    ) -> None:
        super().__init__(env)
        _validate_chunking(int(chunk_size), int(n_action_steps))
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self.action_buffers: list[np.ndarray | None] = [None] * self.env.num_envs
        self.last_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset all environments and discard the per-env buffers.

        Calls ``self.env.reset()`` directly rather than the vector base
        class's reset: the base class dispatches through reset_async/
        reset_wait straight to the innermost env, which would skip the
        ``reset`` overrides of inner wrappers (for example the embedding
        wrapper that attaches ``dino_embedding``). :meth:`step` forwards the
        same way, so both paths see every wrapper.
        """
        self.action_buffers = [None] * self.env.num_envs
        self.last_action = None
        return self.env.reset(**kwargs)

    def step(
        self, actions: np.ndarray | None
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every environment with the next buffered (or given) action."""
        if actions is None:
            next_actions = self._pop_next_actions()
        elif isinstance(actions, np.ndarray) and actions.ndim == 3:
            self._initialize_buffers(actions)
            next_actions = self._pop_next_actions()
        elif isinstance(actions, np.ndarray) and actions.ndim == 2:
            next_actions = actions
            self.action_buffers = [None] * self.env.num_envs
        else:
            raise ValueError(
                "Unsupported actions format. Expected None, "
                "(num_envs, chunk_size, action_dim) or (num_envs, action_dim); "
                f"got {type(actions)} with shape {getattr(actions, 'shape', None)}"
            )

        self.last_action = next_actions.copy()
        obs, reward, terminated, truncated, info = self.env.step(next_actions)
        self._clear_finished(terminated, truncated)
        return obs, reward, terminated, truncated, info

    @property
    def is_chunk_empty(self) -> bool:
        """``True`` when every per-env buffer is exhausted."""
        return all(buf is None or buf.shape[0] == 0 for buf in self.action_buffers)

    def _initialize_buffers(self, chunk_actions: np.ndarray) -> None:
        """Fill the per-env buffers from a ``[num_envs, chunk, dim]`` block."""
        if not isinstance(chunk_actions, np.ndarray) or chunk_actions.ndim != 3:
            raise ValueError(
                "Chunk actions must be ndarray with shape "
                "(num_envs, chunk_size, action_dim); got "
                f"{type(chunk_actions)} with shape {getattr(chunk_actions, 'shape', None)}"
            )
        num_envs = self.env.num_envs
        if chunk_actions.shape[0] != num_envs:
            raise ValueError(
                f"chunk_actions first dim must equal num_envs={num_envs}, "
                f"got {chunk_actions.shape[0]}"
            )
        if chunk_actions.shape[1] < self.n_action_steps:
            raise ValueError(
                f"chunk_actions second dim (chunk_size={chunk_actions.shape[1]}) "
                f"must be >= n_action_steps={self.n_action_steps}"
            )
        self.action_buffers = [
            chunk_actions[i, : self.n_action_steps, :].copy() for i in range(num_envs)
        ]

    def _pop_next_actions(self) -> np.ndarray:
        """Pop one action per environment; ``[num_envs, action_dim]``."""
        num_envs = self.env.num_envs
        action_dim = int(np.prod(self.single_action_space.shape))
        next_actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        for i in range(num_envs):
            buf = self.action_buffers[i]
            if buf is None or buf.shape[0] == 0:
                raise RuntimeError(
                    "Action buffer empty for at least one env; a new chunk must "
                    "be provided before stepping."
                )
            next_actions[i] = buf[0]
            remaining = buf[1:]
            self.action_buffers[i] = remaining if remaining.shape[0] > 0 else None
        return next_actions

    def _clear_finished(self, terminated: Any, truncated: Any) -> None:
        """Drop buffered actions for environments that just finished."""
        try:
            done = np.asarray(terminated).astype(bool) | np.asarray(truncated).astype(bool)
            for i in range(self.env.num_envs):
                if i < done.shape[0] and done[i]:
                    self.action_buffers[i] = None
        except Exception:
            pass  # tolerate unexpected terminated/truncated shapes

    def _get_last_action(self) -> np.ndarray | None:
        """The most recently executed per-env action batch."""
        return self.last_action

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)
