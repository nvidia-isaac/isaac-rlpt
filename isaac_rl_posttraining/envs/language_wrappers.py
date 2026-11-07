# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrappers that attach a fixed language-instruction embedding to observations.

For fixed-instruction tasks the instruction can be embedded once at
construction time; these wrappers add that embedding under the ``"language"``
observation key and mirror the raw instruction into ``info``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np
from loguru import logger

LANGUAGE_EMBEDDING_DIM = 384
"""Dimension of the sentence-embedding observation entry."""


def _embed(instruction: str, sentence_model: Any) -> np.ndarray:
    """Embed ``instruction`` with ``sentence_model``; ``float32 [dim]``."""
    # Local import: the embedding stack (torch models) is only needed when a
    # language wrapper is actually constructed.
    from isaac_rl_posttraining.utils.embedding_utils import compute_text_embeddings

    encoding = compute_text_embeddings(instruction, sentence_model)
    return encoding.cpu().numpy().astype(np.float32)


class LanguageInstructionWrapper(gym.ObservationWrapper):
    """Attach a fixed instruction embedding to a single (non-vector) env.

    Args:
        env: The environment to wrap.
        instruction: Instruction text, embedded once at construction.
        sentence_model: ``sentence_transformers`` model used for embedding.
    """

    def __init__(self, env: gym.Env, instruction: str, sentence_model: Any) -> None:
        super().__init__(env)
        self.language_instruction = instruction
        self.language_encoding = _embed(instruction, sentence_model)

        orig_space = self.env.observation_space
        spaces_dict = (
            dict(orig_space.spaces)
            if isinstance(orig_space, gym.spaces.Dict)
            else {"obs": orig_space}
        )
        spaces_dict["language"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.language_encoding.shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces_dict)

    def observation(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            out = dict(obs)
            out["language"] = self.language_encoding
            return out
        return obs

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        return obs, self._with_instruction(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, self._with_instruction(info)

    def _with_instruction(self, info: Any) -> dict[str, Any]:
        """Return ``info`` with the raw instruction mirrored in."""
        out = dict(info) if isinstance(info, dict) else {}
        out["language_instruction"] = self.language_instruction
        return out

    def get_language_instruction(self) -> str:
        """The raw instruction text this wrapper embeds."""
        return self.language_instruction

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)


class LanguageInstructionVectorWrapper(gym_vector.VectorWrapper):
    """Attach a fixed instruction embedding across a vectorized env.

    Adds ``language`` (``[num_envs, 384]``) to the batched observation dict,
    updates both the batched and single observation spaces, and mirrors the
    per-env instruction list into ``info``.

    Args:
        env: The vectorized environment to wrap.
        instruction: Instruction text, embedded once at construction.
        sentence_model: ``sentence_transformers`` model used for embedding.
    """

    def __init__(
        self,
        env: gym_vector.VectorEnv,
        instruction: str | None = None,
        sentence_model: Any = None,
    ) -> None:
        super().__init__(env)
        logger.info(f"Using language instruction: {instruction}")
        self.language_instruction = instruction
        self.language_encoding = _embed(instruction, sentence_model)

        self.observation_space = self._with_language_space(self.env.observation_space)
        single = getattr(self.env, "single_observation_space", None)
        self.single_observation_space = self._with_language_space(
            single if single is not None else self.env.observation_space
        )

    @staticmethod
    def _with_language_space(space: gym.spaces.Space) -> gym.spaces.Dict:
        """Return ``space`` as a Dict space with a ``language`` entry added."""
        spaces_dict = (
            dict(space.spaces) if isinstance(space, gym.spaces.Dict) else {"obs": space}
        )
        spaces_dict["language"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(LANGUAGE_EMBEDDING_DIM,), dtype=np.float32
        )
        return gym.spaces.Dict(spaces_dict)

    def reset(self, **kwargs: Any) -> tuple[Any, Any]:
        obs, info = self.env.reset(**kwargs)
        n = self._num_envs_for(obs)
        return self._add_language_to_obs(obs, n), self._with_instructions(info, n)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        n = self._num_envs_for(obs)
        return (
            self._add_language_to_obs(obs, n),
            reward,
            terminated,
            truncated,
            self._with_instructions(info, n),
        )

    def _num_envs_for(self, obs: Any) -> int:
        """Best-effort environment count (falls back to the obs length)."""
        try:
            return self.env.num_envs
        except Exception:
            return len(obs) if hasattr(obs, "__len__") else 1

    def _add_language_to_obs(self, obs: Any, n: int) -> Any:
        """Attach the embedding to a batched dict obs (or per-entry dicts)."""
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["language"] = np.tile(self.language_encoding, (n, 1))  # [n, 384]
            return obs
        if isinstance(obs, (list, tuple)):
            for i in range(n):
                if isinstance(obs[i], dict):
                    obs[i] = dict(obs[i])
                    obs[i]["language"] = self.language_encoding  # [384]
            return obs
        return obs  # unknown observation format: pass through

    def _with_instructions(self, info: Any, n: int) -> Any:
        """Return ``info`` with a per-env instruction list mirrored in."""
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = [self.language_instruction] * n
            return info
        return [{"language_instruction": self.language_instruction} for _ in range(n)]

    def get_language_instruction(self) -> str | None:
        """The raw instruction text this wrapper embeds."""
        return self.language_instruction

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)
