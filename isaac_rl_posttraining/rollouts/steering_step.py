# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared steering decision step.

The steering rollout policy
(:class:`~isaac_rl_posttraining.runtimes.steering_policy.SteeringActionChunkPolicy`)
runs the same decision step for collection and evaluation — project the raw
observation onto the actor's input dict, select a steering latent, decode it
through the frozen base model — and differs only in the decision mode. This
module holds that step once, so training and evaluation semantics cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from isaac_rl_posttraining.utils.image_utils import resize_images

__all__ = ["SteeringStep"]


class SteeringStep:
    """Observation projection, latent selection, and frozen decode.

    Args:
        base_model: Frozen base model exposing ``infer(observations, noise)``
            (and optionally ``get_features``).
        device: Device for actor inputs.
        steering_key_mapping: Which observation entries are the state, the
            language embedding (``None`` for single-task envs), and the
            image list.
        dino_wrapper: Optional DINO-embedding wrapper handle, used to
            re-embed observations that arrive without ``dino_embedding``.
    """

    def __init__(
        self,
        base_model: Any,
        device: torch.device,
        steering_key_mapping: dict[str, Any],
        dino_wrapper: Any = None,
        action_modulator: Any = None,
        reference_action_key: str | None = None,
        reference_image_transform_mode: str = "eval",
    ) -> None:
        self.base_model = base_model
        self.device = device
        self.steering_key_mapping = steering_key_mapping
        self.dino_wrapper = dino_wrapper
        # RFS splits the joint actor action into flow noise and a residual, then
        # adds the bounded residual back after the frozen decode.
        self.action_modulator = action_modulator
        # When set (RFS reference-action conditioning), the frozen policy's own
        # decoded chunk is attached to the learner observation under this key.
        self.reference_action_key = reference_action_key
        self.reference_image_transform_mode = reference_image_transform_mode
        if self.reference_image_transform_mode not in ("train", "eval"):
            raise ValueError("reference_image_transform_mode must be train or eval")
        self._noise_log_count = 0

    # ------------------------------------------------------------------
    # Environment introspection (shared helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def find_dino_wrapper(env: Any) -> Any:
        """Locate the DINO-embedding wrapper in the env wrapper chain."""
        current = env
        for _ in range(8):
            if hasattr(current, "dinov2_model") and callable(getattr(current, "observation", None)):
                return current
            current = getattr(current, "env", None)
            if current is None:
                break
        return None

    @staticmethod
    def base_env(env: Any, env_idx: int) -> Any:
        """The unwrapped per-environment handle, for chunked stepping."""
        if hasattr(env, "envs"):
            return env.envs[env_idx]
        return env

    @staticmethod
    def vectorize_obs(obs: Any) -> Any:
        """Re-add the batch dimension after a direct base-env step."""
        if isinstance(obs, dict):
            vectorized_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    vectorized_obs[key] = value[None, ...]
                else:
                    vectorized_obs[key] = value
            return vectorized_obs
        return np.array([obs])

    # ------------------------------------------------------------------
    # The decision step
    # ------------------------------------------------------------------

    def _mapped_entry(self, obs: dict[str, Any], role: str, key: str) -> Any:
        """``obs[key]``, naming the simulator entry's key mapping when the observation lacks it."""
        try:
            return obs[key]
        except KeyError:
            raise KeyError(
                f"the environment observation has no {role} entry {key!r} (observation keys: "
                f"{sorted(obs)}); the simulator entry's steering_key_mapping() names the entries "
                "the steering policies read"
            ) from None

    def process_obs(
        self,
        obs: dict[str, Any],
        *,
        squeeze: bool = False,
        vlm_features: str = "off",
    ) -> dict[str, torch.Tensor]:
        """Project a raw env observation onto the actor/critic input dict.

        Keeps ``{state, dino_embedding[, language, vlm_features]}`` per the
        env's key mapping; the raw image stays only in the env observation
        (the base model's ``infer`` consumes it directly).

        Args:
            obs: Raw (batched) observation dict.
            squeeze: Drop the leading batch dimension (single-env actors).
            vlm_features: ``"off"`` — never attach base model VLM features;
                ``"enabled"`` — extract them (the training-time contract:
                the config asked for them, so failures raise);
                ``"probe"`` — attach them only if the base model exposes
                ``get_features`` (evaluation of arbitrary base models).
        """
        if "prompt" in obs and isinstance(obs["prompt"], (list, tuple)):
            obs["prompt"] = obs["prompt"][0]

        if "dino_embedding" not in obs and self.dino_wrapper is not None:
            try:
                obs = self.dino_wrapper.observation(obs)
            except Exception as exc:
                logger.warning(f"[SteeringStep] DINO re-embed failed: {exc}")

        def _maybe_squeeze(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.squeeze(0) if squeeze else tensor

        state_key = self.steering_key_mapping["state"]
        proprio = _maybe_squeeze(
            torch.from_numpy(self._mapped_entry(obs, "state", state_key)).to(self.device).float()
        )
        dsrl_obs: dict[str, torch.Tensor] = {state_key: proprio}

        self._attach_base_model_features(obs, dsrl_obs, vlm_features, _maybe_squeeze)

        lang_key = self.steering_key_mapping.get("language")
        if lang_key is not None and lang_key in obs:
            language = torch.from_numpy(obs[lang_key]).to(self.device).float()
            dsrl_obs["language"] = _maybe_squeeze(language)

        if "dino_embedding" in obs:
            dino = torch.from_numpy(obs["dino_embedding"]).to(self.device).float()
            dsrl_obs["dino_embedding"] = _maybe_squeeze(dino)
        else:
            for img_key in self.steering_key_mapping["image"]:
                img = resize_images(self._mapped_entry(obs, "image", img_key), 128)
                dsrl_obs[img_key] = torch.from_numpy(img).to(self.device)

        # Pixel-critic input (env.critic_image_size): the small frame rides
        # alongside the embedding for trainable image encoders.
        if "critic_image" in obs:
            pixels = torch.from_numpy(np.ascontiguousarray(obs["critic_image"])).to(self.device)
            dsrl_obs["critic_image"] = _maybe_squeeze(pixels)

        return dsrl_obs

    def _attach_base_model_features(
        self,
        obs: dict[str, Any],
        dsrl_obs: dict[str, torch.Tensor],
        vlm_features: str,
        maybe_squeeze: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Add the frozen policy's reference action and/or VLM features to ``dsrl_obs``."""
        if self.reference_action_key is not None:
            # The frozen policy's decoded action chunk is deployable output, not
            # simulator state, so it may condition the learner directly.
            infer_reference = getattr(self.base_model, "infer_reference", None)
            if not callable(infer_reference):
                raise TypeError(
                    "reference-conditioned steering requires a base model with infer_reference()"
                )
            with torch.inference_mode():
                features, reference_action = infer_reference(
                    obs,
                    image_transform_mode=self.reference_image_transform_mode,
                )
            if vlm_features != "off":
                dsrl_obs["vlm_features"] = maybe_squeeze(features.to(self.device)).clone()
            reference_tensor = torch.as_tensor(
                np.asarray(reference_action),
                device=self.device,
                dtype=torch.float32,
            )
            dsrl_obs[self.reference_action_key] = maybe_squeeze(reference_tensor).clone()
        elif vlm_features == "enabled":
            with torch.inference_mode():
                # Extract before any resizing so the base model sees full-res.
                features = self.base_model.get_features(obs).to(self.device)
            dsrl_obs["vlm_features"] = maybe_squeeze(features).clone()
        elif vlm_features == "probe":
            try:
                features = self.base_model.get_features(obs).to(self.device)
                dsrl_obs["vlm_features"] = maybe_squeeze(features)
            except (NotImplementedError, AttributeError):
                pass

    def select_latent(
        self,
        actor: Any,
        dsrl_obs: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        actor_state: Any = None,
    ) -> tuple[np.ndarray, Any]:
        """Sample (or take the mean of) the actor's steering latent.

        Returns:
            Tuple ``(latents, new_actor_state)``; the state passes through
            for recurrent actors and is ``None``-safe for feed-forward ones.
        """
        noise_a, new_state = actor.act(
            dict(dsrl_obs), actor_state=actor_state, deterministic=deterministic
        )
        return noise_a.detach().cpu().numpy(), new_state

    def decode(self, obs: dict[str, Any], policy_actions: Optional[np.ndarray]) -> np.ndarray:
        """Decode a steering action and apply optional output modulation.

        ``policy_actions=None`` is native mode — the base model samples its own
        noise and no residual is applied. Standard DSRL passes the policy
        action straight through as the flow latent. RFS supplies an action
        modulator that splits the joint policy action into flow noise and a
        residual, then adds the residual after decode. Returns decoded (or
        output-modulated) actions with shape ``[B, H, A]``.
        """
        latents = policy_actions
        residuals = None
        if policy_actions is not None and self.action_modulator is not None:
            latents, residuals = self.action_modulator.split(policy_actions)

        if self._noise_log_count < 3:
            self._noise_log_count += 1
            if latents is None:
                logger.info("[noise-evidence] base model noise=None (native/base path)")
            else:
                arr = np.asarray(latents)
                logger.info(
                    f"[noise-evidence] base model noise shape={arr.shape} norm={float(np.linalg.norm(arr)):.3f} "
                    f"std={float(arr.std()):.3f} max|.|={float(np.abs(arr).max()):.3f}"
                )
        with torch.inference_mode():
            result = self.base_model.infer(observations=obs, noise=latents)
        decoded_actions = result["actions"]
        if residuals is None:
            return decoded_actions
        return self.action_modulator.apply(decoded_actions, residuals)

    def decode_batch(self, obs: dict[str, Any], latents: np.ndarray | None) -> np.ndarray:
        """Decode all ``B`` envs in one base model forward (``[B, H, A]``).

        The batched counterpart of :meth:`decode` for base models advertising
        ``supports_batched_infer``: ``obs`` is the batched observation dict and
        ``latents`` is ``[B, ...]`` (or ``None`` native). Steers every env with
        a single call instead of one decode per env.
        """
        if self.action_modulator is not None:
            raise NotImplementedError("batched decode does not apply residual flow steering; use num_envs=1")
        with torch.inference_mode():
            result = self.base_model.infer_batch(observations=obs, noise=latents)
        return result["actions"]

    def decode_prepared_batch(
        self, obs: dict[str, Any], initial_noise: np.ndarray
    ) -> np.ndarray:
        """Decode pre-sampled per-session noise in one base model forward."""
        with torch.inference_mode():
            result = self.base_model.infer_prepared_batch(
                observations=obs,
                initial_noise=initial_noise,
            )
        return result["actions"]
