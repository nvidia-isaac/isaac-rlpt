# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment glue for latent-steering algorithms (DSRL and relatives).

Latent-steering learners act in a base policy's noise space rather than the
environment's action space, and their actor/critic consume a specific
observation layout. Two pieces bridge that gap:

* :class:`SteeringObsWrapper` formats a remote env's observations for the
  steering learner: the state vector and camera image pass through untouched,
  a ``language`` embedding is attached (384-d zeros when no sentence model is
  configured -- the single-task case), the raw ``prompt`` string is
  preserved, and the wrapped env's ``steering_key_mapping`` is re-exposed so the
  rollout workers can locate each modality.
* :class:`LatentSpaceEnv` is a space-only ``gym.Env`` that reports the
  observation/action spaces the steering actor/critic are sized from: the env
  observation (plus optional base-policy features) and a latent action box of
  ``noise_dim`` per step, or ``[chunk_size, noise_dim]`` for chunked actors.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np

LANGUAGE_EMBEDDING_DIM = 384
"""Dimension of the sentence-embedding observation entry."""


def _embed_instruction(instruction: str, sentence_model: Any) -> np.ndarray:
    """Embed an instruction string, or return zeros without a sentence model.

    Args:
        instruction: The language instruction to embed.
        sentence_model: A ``sentence_transformers`` model, or ``None`` for the
            single-task case where no language conditioning is used.

    Returns:
        ``float32 [LANGUAGE_EMBEDDING_DIM]`` embedding.
    """
    if sentence_model is None:
        return np.zeros((LANGUAGE_EMBEDDING_DIM,), dtype=np.float32)
    # Local import: the embedding stack (torch models) is only needed when a
    # sentence model is actually configured.
    from isaac_rl_posttraining.utils.embedding_utils import compute_text_embeddings

    encoding = compute_text_embeddings(instruction, sentence_model)
    return encoding.cpu().numpy().astype(np.float32)


class SteeringObsWrapper(gym.ObservationWrapper):
    """Format remote-env observations for a latent-steering learner.

    The wrapped environment must expose ``steering_key_mapping`` (see
    :data:`simulators.g1_arena.env.ARENA_KEY_MAPPING`); the
    mapping is re-exposed on the wrapper for the rollout workers.

    Args:
        env: Environment whose observations include a ``"prompt"`` string.
        sentence_model: Optional ``sentence_transformers`` model; when absent
            the ``language`` entry is a zero vector.
    """

    def __init__(self, env: gym.Env, sentence_model: Any = None) -> None:
        super().__init__(env)
        assert hasattr(env, "steering_key_mapping"), (
            f"env {env.__class__.__name__} must expose a steering_key_mapping attribute"
        )
        self.steering_key_mapping: dict[str, Any] = env.steering_key_mapping
        self.sentence_model = sentence_model
        self.language_instruction: str | None = None
        self.language_encoding: np.ndarray | None = None
        self.observation_space = env.observation_space

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        assert obs.get("prompt") is not None, "observation is missing its prompt"
        # Drop the cached embedding so the new episode's prompt is re-embedded.
        self.language_instruction = None
        return self._format_obs(obs), info

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._format_obs(obs)

    def _format_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Attach the language embedding, re-embedding if the prompt changed."""
        prompt = obs["prompt"]
        if prompt != self.language_instruction or self.language_encoding is None:
            self.language_instruction = prompt
            self.language_encoding = _embed_instruction(prompt, self.sentence_model)
        obs["language"] = self.language_encoding
        obs["prompt"] = self.language_instruction
        return obs


class LatentSpaceEnv(gym.Env):
    """Space-only environment used to size the steering actor/critic.

    Nothing is simulated here: the class exists so network constructors that
    are written against ``env.observation_space`` / ``env.action_space`` can
    be sized for latent-noise steering without a live environment.

    Args:
        observation_space: The (dict) observation space the learner sees.
        action_space: The wrapped environment's raw action space, kept for
            reference alongside the latent action box.
        base_model: Base-policy wrapper; when ``use_vlm_features`` is set its
            ``get_features`` output determines the ``vlm_features`` entry size.
        noise_dim: Steerable latent dimensions per action step.
        chunk_size: When given, also expose ``chunk_action_space`` of shape
            ``[chunk_size, noise_dim]`` for chunked actors.
        action_bound: Symmetric bound of the latent action box.
        warmup_exploration: Distribution :meth:`sample_action` draws from:
            ``"uniform"`` over the bounded box, or ``"normal"`` -- unbounded
            N(0, 1), the frozen flow's native noise, so warmup rollouts
            behave near the base policy (injection-time clipping still
            applies base-model-side).
        use_vlm_features: Whether to add a ``vlm_features`` observation entry
            sized from the base policy.

    Raises:
        ValueError: For an unknown ``warmup_exploration`` name.
    """

    #: Plain latent steering never attaches the frozen policy's decoded chunk
    #: to the learner observation; the residual-flow subclass may.
    include_reference_action: bool = False
    reference_action_key: str = "reference_action"

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        base_model: Any,
        noise_dim: int,
        chunk_size: int | None = None,
        action_bound: float = 1.0,
        warmup_exploration: str = "uniform",
        use_vlm_features: bool = True,
    ) -> None:
        super().__init__()
        self.env_action_space = action_space
        self.noise_dim = int(noise_dim)
        if warmup_exploration not in ("uniform", "normal"):
            raise ValueError(
                f"warmup_exploration must be 'uniform' or 'normal', got {warmup_exploration!r}"
            )
        self.warmup_exploration = warmup_exploration
        self.use_vlm_features = bool(use_vlm_features)

        obs_space_dict = copy.deepcopy(observation_space.spaces)
        if self.use_vlm_features:
            obs_space_dict["vlm_features"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._probe_feature_dim(observation_space, base_model),),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(obs_space_dict)

        self.action_space = gym.spaces.Box(
            low=-action_bound, high=action_bound, shape=(self.noise_dim,), dtype=float
        )
        self.chunk_action_space: gym.spaces.Box | None = None
        if chunk_size is not None:
            self.chunk_action_space = gym.spaces.Box(
                low=-action_bound,
                high=action_bound,
                shape=(int(chunk_size), self.noise_dim),
                dtype=float,
            )

    @staticmethod
    def _probe_feature_dim(observation_space: gym.spaces.Dict, base_model: Any) -> int:
        """Measure the base policy's feature width on a sampled observation."""
        import torch  # local: only the feature probe needs torch

        example_obs = observation_space.sample()
        example_obs.pop("dino_embedding", None)  # features come from the policy, not DINO
        example_obs["prompt"] = "test"
        with torch.inference_mode():
            features = base_model.get_features(example_obs).detach().cpu().numpy()
        return int(features.shape[-1]) if features.ndim > 1 else int(features.shape[0])

    def sample_action(self) -> np.ndarray:
        """Sample a warmup latent action with a leading batch dimension."""
        space = self.chunk_action_space if self.chunk_action_space is not None else self.action_space
        if self.warmup_exploration == "normal":
            return np.random.standard_normal(space.shape).astype(space.dtype)[None, ...]  # NOSONAR - draws from the globally seeded NumPy stream
        # The globally seeded stream (not the space's own unseeded generator), so runs reproduce.
        return np.random.uniform(space.low, space.high).astype(space.dtype)[None, ...]  # NOSONAR - draws from the globally seeded NumPy stream

    # Space-only environment.
    def reset(self, **kwargs: Any):  # noqa: D102
        return None, None

    # Space-only environment.
    def step(self, action: Any):  # noqa: D102
        return None, 0.0, True, {}


class ResidualFlowSpaceEnv(LatentSpaceEnv):
    """Space-only environment for joint flow-noise and residual actions.

    Residual Flow Steering acts in ``[latent, normalized residual]`` per step:
    the latent steers the frozen sampler and the residual is a bounded
    correction added to the decoded action. Both branches use their normalized
    actor ranges here; the action modulator applies the smaller physical
    residual bound after the frozen-policy decode. Warmup exploration samples
    only the latent and keeps the residual exactly zero, so RFS starts from the
    same behavior distribution as DSRL.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        base_model: Any,
        noise_dim: int,
        residual_action_dim: int,
        chunk_size: int | None,
        noise_action_bound: float,
        residual_action_bound: float,
        warmup_exploration: str,
        use_vlm_features: bool,
        include_reference_action: bool = False,
        reference_action_key: str = "reference_action",
    ) -> None:
        if residual_action_dim <= 0:
            raise ValueError(f"residual_action_dim must be positive, got {residual_action_dim}")
        if noise_action_bound <= 0:
            raise ValueError(f"noise_action_bound must be positive, got {noise_action_bound}")
        if residual_action_bound <= 0:
            raise ValueError(f"residual_action_bound must be positive, got {residual_action_bound}")

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            base_model=base_model,
            noise_dim=noise_dim,
            chunk_size=chunk_size,
            action_bound=noise_action_bound,
            warmup_exploration=warmup_exploration,
            use_vlm_features=use_vlm_features,
        )
        self.residual_action_dim = int(residual_action_dim)
        self.noise_action_bound = float(noise_action_bound)
        self.residual_action_bound = float(residual_action_bound)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.include_reference_action = bool(include_reference_action)
        self.reference_action_key = str(reference_action_key)
        if self.include_reference_action:
            if not self.reference_action_key:
                raise ValueError("reference_action_key must not be empty")
            if not callable(getattr(base_model, "infer_reference", None)):
                raise TypeError(
                    "reference-conditioned RFS requires a base model with infer_reference()"
                )
            # GR00T's decoded action chunk for the current image is deployable
            # policy output, not simulator state, so it may condition the learner.
            self.observation_space.spaces[self.reference_action_key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(base_model.action_horizon), int(base_model.action_dim)),
                dtype=np.float32,
            )

        low = np.concatenate(
            [
                np.full(self.noise_dim, -self.noise_action_bound, dtype=np.float32),
                np.full(self.residual_action_dim, -1.0, dtype=np.float32),
            ]
        )
        high = -low
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.chunk_action_space = None
        if self.chunk_size is not None:
            self.chunk_action_space = gym.spaces.Box(
                low=np.broadcast_to(low, (self.chunk_size, low.size)).copy(),
                high=np.broadcast_to(high, (self.chunk_size, high.size)).copy(),
                dtype=np.float32,
            )

    def sample_action(self) -> np.ndarray:
        """Sample warmup flow noise and concatenate a zero residual."""
        leading_shape = () if self.chunk_size is None else (self.chunk_size,)
        latent_shape = (*leading_shape, self.noise_dim)
        if self.warmup_exploration == "uniform":
            latent = np.random.uniform(  # NOSONAR - draws from the globally seeded NumPy stream
                -self.noise_action_bound,
                self.noise_action_bound,
                size=latent_shape,
            ).astype(np.float32)
        elif self.warmup_exploration == "normal":
            latent = np.random.standard_normal(latent_shape).astype(np.float32)  # NOSONAR - draws from the globally seeded NumPy stream
        else:
            raise ValueError(
                f"warmup_exploration must be 'uniform' or 'normal', got {self.warmup_exploration!r}"
            )
        residual = np.zeros((*leading_shape, self.residual_action_dim), dtype=np.float32)
        return np.concatenate([latent, residual], axis=-1)[None, ...]


# Names used by earlier revisions of this package.
SimpleDSRLWrapper = SteeringObsWrapper
DummyDSRLEnv = LatentSpaceEnv
