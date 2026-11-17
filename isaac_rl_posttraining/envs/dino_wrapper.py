# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrappers that add DINOv2 image embeddings to environment observations.

The steering critic consumes compact visual features rather than raw pixels.
These wrappers compute a DINOv2 embedding for each configured image key and
attach the concatenation under ``"dino_embedding"``, while keeping the
original image entries in the observation (the base policy still needs them).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np
import torch


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Normalize one image to ``uint8 [H, W, 3]``.

    Accepts ``[C, H, W]`` (transposed), grayscale ``[H, W]`` (replicated to
    three channels), and float images (rescaled from ``[0, 1]`` when needed).
    """
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 2:
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    return image


def _embed_image_batch(
    images: np.ndarray, dinov2_model: Any, dinov2_processor: Any
) -> np.ndarray:
    """Embed a batch of images with DINOv2.

    Args:
        images: ``[N, H, W, C]`` batch (a single ``[H, W, C]`` image gains a
            batch dimension).
        dinov2_model: The DINOv2 model (already on its device).
        dinov2_processor: The matching image processor.

    Returns:
        ``float32 [N, embedding_dim]`` pooled embeddings.
    """
    # Local import: the embedding helpers pull in the full model stack.
    from isaac_rl_posttraining.utils.embedding_utils import compute_video_embeddings

    if images.ndim == 3:
        images = images[None, ...]
    frames = np.stack([_to_uint8_hwc(img) for img in images], axis=0)
    embeddings = compute_video_embeddings(
        frames_array=frames,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        batch_size=frames.shape[0],
        use_autocast=True,
        use_tqdm=False,
    )
    return embeddings.cpu().numpy().astype(np.float32)


def _probe_embedding_dim(dinov2_model: Any, dinov2_processor: Any, device: str) -> int:
    """Run one dummy image through the model to measure its pooled width.

    Also warms the processor with an out-of-range float image so its
    "images already scaled" warning fires here once instead of during
    rollouts.
    """
    dummy = torch.ones((1, 3, 84, 84)).to(device).float() * 2
    inputs = dinov2_processor(images=dummy, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    outputs = dinov2_model(**inputs)
    return int(outputs.pooler_output.shape[1])


def _spaces_with_embedding(
    space: gym.spaces.Space, embedding_dim: int, critic_image_size: int | None = None
) -> gym.spaces.Dict:
    """Return ``space`` as a Dict space with a ``dino_embedding`` entry added.

    With ``critic_image_size`` set, a ``critic_image`` entry (small uint8
    frame for pixel-input critics) is added as well.
    """
    spaces_dict = (
        dict(space.spaces) if isinstance(space, gym.spaces.Dict) else {"obs": space}
    )
    if embedding_dim:
        spaces_dict["dino_embedding"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32
        )
    if critic_image_size:
        spaces_dict["critic_image"] = gym.spaces.Box(
            low=0, high=255, shape=(critic_image_size, critic_image_size, 3), dtype=np.uint8
        )
    return gym.spaces.Dict(spaces_dict)


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Downsize one ``[H, W, 3]`` frame to ``uint8 [size, size, 3]``."""
    import cv2

    return cv2.resize(_to_uint8_hwc(image), (size, size), interpolation=cv2.INTER_AREA)


class DinoEmbeddingWrapper(gym.ObservationWrapper):
    """Attach DINOv2 embeddings to a single (non-vector) environment.

    Args:
        env: The environment to wrap.
        dinov2_model: DINOv2 model used for the embeddings.
        dinov2_processor: The matching image processor.
        device: Device the model lives on; defaults to CUDA when available.
        image_keys: Observation keys holding images; their embeddings are
            concatenated in key order. Defaults to ``["image"]``.
    """

    def __init__(
        self,
        env: gym.Env,
        dinov2_model: Any,
        dinov2_processor: Any,
        device: str | None = None,
        image_keys: list[str] | None = None,
    ) -> None:
        super().__init__(env)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        self.image_keys = image_keys if image_keys is not None else ["image"]

        per_image_dim = _probe_embedding_dim(dinov2_model, dinov2_processor, device)
        self.output_dim = per_image_dim * len(self.image_keys)
        self.observation_space = _spaces_with_embedding(env.observation_space, self.output_dim)

    def observation(self, obs: Any) -> dict[str, Any]:
        """Attach the concatenated per-key embeddings under ``dino_embedding``."""
        if not isinstance(obs, dict):
            embeddings = _embed_image_batch(obs, self.dinov2_model, self.dinov2_processor)
            return {"dino_embedding": embeddings[0]}

        new_obs = dict(obs)
        images = []
        for key in self.image_keys:
            if key not in obs:
                raise ValueError(f"Image key {key} not found in observation")
            image = obs[key]
            images.append(image[None, ...] if image.ndim == 3 else image)
        embeddings = _embed_image_batch(
            np.concatenate(images, axis=0), self.dinov2_model, self.dinov2_processor
        )
        # One embedding per image key; concatenate them for the single env.
        new_obs["dino_embedding"] = (
            np.concatenate(list(embeddings), axis=0) if len(images) > 1 else embeddings[0]
        )
        return new_obs


class DinoBatchObservationEncoder:
    """Attach DINO embeddings to an already-batched observation mapping.

    Runtime policy servers do not own a Gym environment, so they cannot use
    :class:`VectorDinoEmbeddingWrapper`.  This encoder exposes the same
    ``observation`` operation without requiring an environment instance.
    """

    def __init__(
        self,
        dinov2_model: Any,
        dinov2_processor: Any,
        image_keys: list[str] | None = None,
    ) -> None:
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        self.image_keys = image_keys if image_keys is not None else ["image"]

    def observation(self, obs: Any) -> dict[str, Any]:
        """Return the observation with one concatenated embedding per sample."""
        if not isinstance(obs, dict):
            return {
                "dino_embedding": _embed_image_batch(
                    obs, self.dinov2_model, self.dinov2_processor
                )
            }

        new_obs = dict(obs)
        num_envs: int | None = None
        images: list[np.ndarray] = []
        for key in self.image_keys:
            if key not in obs:
                raise ValueError(f"Image key {key} not found in observation")
            batch = obs[key]
            if not isinstance(batch, np.ndarray) or batch.ndim != 4:
                raise ValueError(f"Image key {key} must be a 4D numpy array")
            if num_envs is None:
                num_envs = batch.shape[0]
            elif batch.shape[0] != num_envs:
                raise ValueError("Image keys must have the same batch size")
            images.extend(batch[index] for index in range(batch.shape[0]))

        assert num_envs is not None
        embeddings = _embed_image_batch(
            np.stack(images, axis=0), self.dinov2_model, self.dinov2_processor
        )
        new_obs["dino_embedding"] = np.stack(
            [
                np.concatenate(embeddings[index::num_envs], axis=0)
                for index in range(num_envs)
            ],
            axis=0,
        )
        return new_obs


class VectorDinoEmbeddingWrapper(gym_vector.VectorWrapper):
    """Attach DINOv2 embeddings across a vectorized environment.

    The batched observation gains ``dino_embedding`` of shape
    ``[num_envs, len(image_keys) * embedding_dim]``.

    Args:
        env: The vectorized environment to wrap.
        dinov2_model: DINOv2 model used for the embeddings.
        dinov2_processor: The matching image processor.
        device: Device the model lives on; defaults to CUDA when available.
        image_keys: Observation keys holding images; their embeddings are
            concatenated in key order. Defaults to ``["image"]``.
        critic_image_size: When set, also attach ``critic_image`` -- the
            first image key downsized to this square size (uint8) -- for
            pixel-input critics trained alongside the frozen embeddings.

    Pixel-only mode: with ``dinov2_model=None`` (and ``critic_image_size``
    set) no embedding is computed or attached -- the wrapper only serves the
    downsized ``critic_image``, for agents that train their own encoders.
    """

    def __init__(
        self,
        env: gym_vector.VectorEnv,
        dinov2_model: Any,
        dinov2_processor: Any,
        device: str | None = None,
        image_keys: list[str] | None = None,
        critic_image_size: int | None = None,
    ) -> None:
        super().__init__(env)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        self.image_keys = image_keys if image_keys is not None else ["image"]
        self.critic_image_size = critic_image_size

        if dinov2_model is None:
            if not critic_image_size:
                raise ValueError(
                    "VectorDinoEmbeddingWrapper without a DINO model (pixel-only "
                    "mode) needs critic_image_size; otherwise it would attach "
                    "nothing."
                )
            self.output_dim = 0
        else:
            self.output_dim = _probe_embedding_dim(dinov2_model, dinov2_processor, device)
        total_dim = len(self.image_keys) * self.output_dim
        self.single_observation_space = _spaces_with_embedding(
            env.single_observation_space, total_dim, critic_image_size
        )
        if hasattr(env, "observation_space"):
            self.observation_space = _spaces_with_embedding(
                env.observation_space, total_dim, critic_image_size
            )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        obs, reward, terminated, truncated, info = super().step(actions)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs: Any) -> dict[str, Any]:
        """Attach per-env concatenated embeddings under ``dino_embedding``."""
        if not isinstance(obs, dict):
            embeddings = _embed_image_batch(obs, self.dinov2_model, self.dinov2_processor)
            return {"dino_embedding": embeddings}

        new_obs = dict(obs)
        if self.dinov2_model is None:
            first = obs[self.image_keys[0]]
            new_obs["critic_image"] = np.stack(
                [_resize_uint8(first[i], self.critic_image_size) for i in range(first.shape[0])],
                axis=0,
            )
            return new_obs
        num_envs = None
        images: list[np.ndarray] = []
        # Key-major layout: all envs for key 0, then all envs for key 1, ...
        for key in self.image_keys:
            if key not in obs:
                raise ValueError(f"Image key {key} not found in observation")
            batch = obs[key]
            assert batch.ndim == 4, (
                "Observation must be a 4D array as it should be wrapped around "
                "a vectorized environment"
            )
            num_envs = batch.shape[0]
            images.extend(batch[i] for i in range(num_envs))

        embeddings = _embed_image_batch(
            np.stack(images, axis=0), self.dinov2_model, self.dinov2_processor
        )
        # embeddings[i :: num_envs] gathers every key's embedding for env i;
        # concatenating them yields [num_keys * embedding_dim] per env.
        new_obs["dino_embedding"] = np.stack(
            [np.concatenate(embeddings[i::num_envs], axis=0) for i in range(num_envs)],
            axis=0,
        )
        if self.critic_image_size:
            first = obs[self.image_keys[0]]
            new_obs["critic_image"] = np.stack(
                [_resize_uint8(first[i], self.critic_image_size) for i in range(first.shape[0])],
                axis=0,
            )
        return new_obs

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)
