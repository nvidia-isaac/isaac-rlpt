# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseModelPolicy adapter for an in-process DreamZero world-action model.

:class:`DreamZeroBaseModelPolicy` conforms
:class:`~.policy.DreamZeroLocalPolicy` to the
:class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` contract. The
frozen model loads in this process (like the gr00t and pi0 base models) and
DSRL steering injects the initial action-latent noise of the WAN
flow-matching denoise via a verified sampler patch — the mechanism RLinf
validated (LIBERO-Spatial 81.25% SFT → 90.6% DSRL with the 5B model).

Documented deviations from the contract:

* when per-step latents cover only ``K < H`` steps, the remaining steps are
  filled by tiling the ``K``-step sequence (the pi0 deviation; the validated
  recipe uses ``K = 1``, one latent tiled across the whole chunk), not by
  fresh N(0, 1) noise;
* ``latent_dim`` (32, the WAN head's padded action-latent width — the
  flow's noise space) exceeds the decoded ``action_dim`` (the checkpoint's
  env action width): steering happens in the padded space, decoding
  slices it;
* native mode (``noise=None``) is deterministic per observation: the model
  draws its action latent with a fixed seed (1140) — the base policy's own
  evaluation protocol, kept as-is.

For driving a stock DreamZero *server* (native evaluation only), use
``base_model=dreamzero_server`` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP, BaseModelPolicy
from isaac_rl_posttraining.policies.dreamzero.policy import DreamZeroLocalPolicy

__all__ = ["DreamZeroBaseModelPolicy"]


class DreamZeroBaseModelPolicy(BaseModelPolicy):
    """An in-process DreamZero policy behind the ``BaseModelPolicy`` contract."""

    def __init__(
        self,
        policy: DreamZeroLocalPolicy,
        latent_dim: int | None = None,
        latent_clip: float = DEFAULT_LATENT_CLIP,
    ) -> None:
        """Adapt an already-constructed DreamZero policy.

        Args:
            policy: The loaded :class:`~.policy.DreamZeroLocalPolicy`.
            latent_dim: Steerable latent width per step; must equal the WAN
                head's action-latent width (the policy's ``noise_dim``, 32
                for the released checkpoints). Defaults to it.
            latent_clip: Symmetric bound applied to injected latents at
                injection time; ``<= 0`` disables clipping.

        Raises:
            ValueError: If ``latent_dim`` disagrees with the checkpoint.
        """
        self._policy = policy
        head_width = int(policy.noise_dim)
        if latent_dim is not None and int(latent_dim) != head_width:
            raise ValueError(
                f"latent_dim ({latent_dim}) must equal the WAN action head's "
                f"latent width ({head_width}): DSRL steers the head's padded "
                "action-latent space."
            )
        self._latent_dim = head_width
        self._latent_clip = float(latent_clip)

    @classmethod
    def from_dsrl_config(
        cls, cfg: Any, device: Any, dataset_adapter: Any = None
    ) -> "DreamZeroBaseModelPolicy":
        """Build the in-process DreamZero base model from the composed config.

        Reads the checkpoint/runtime knobs from the ``base_model`` group and the
        steering geometry from ``dsrl``/``training``, then fail-fast validates
        the geometry against the loaded checkpoint before any env traffic.
        """
        from omegaconf import OmegaConf

        from isaac_rl_posttraining.policies.dreamzero.model import load_dreamzero_model
        from isaac_rl_posttraining.policies.dreamzero.validation import (
            validate_dreamzero_dsrl_geometry,
        )

        base_ckpt = OmegaConf.select(cfg, "base_model.ckpt")
        if not base_ckpt:
            raise ValueError(
                "base_model=dreamzero_local requires base_model.ckpt (a bundled SFT "
                "checkpoint directory with config.json + model.safetensors)"
            )
        embodiment_tag = OmegaConf.select(cfg, "base_model.embodiment_tag")
        if not embodiment_tag:
            raise ValueError(
                "base_model=dreamzero_local requires base_model.embodiment_tag "
                "(a groot tag with a registered EmbodimentSpec, e.g. libero_sim)"
            )
        tokenizer_path = OmegaConf.select(cfg, "base_model.tokenizer_path")
        if not tokenizer_path:
            raise ValueError(
                "base_model=dreamzero_local requires base_model.tokenizer_path "
                "(e.g. google/umt5-xxl)"
            )

        target_height = OmegaConf.select(cfg, "base_model.target_video_height")
        target_width = OmegaConf.select(cfg, "base_model.target_video_width")
        if (target_height is None) != (target_width is None):
            raise ValueError(
                "base_model.target_video_height and base_model.target_video_width "
                "must be set together (or both left null)."
            )
        target_video = (
            None
            if target_height is None
            else (int(target_height), int(target_width))
        )

        num_dit_steps = OmegaConf.select(cfg, "base_model.num_dit_steps")
        loaded = load_dreamzero_model(
            str(base_ckpt),
            embodiment_tag=str(embodiment_tag),
            metadata_json_path=OmegaConf.select(cfg, "base_model.metadata_json_path"),
            tokenizer_path=str(tokenizer_path),
            device=device,
            text_encoder_cpu_offload=bool(
                OmegaConf.select(cfg, "base_model.text_encoder_cpu_offload")
            ),
            torch_compile=bool(OmegaConf.select(cfg, "base_model.torch_compile")),
            num_dit_steps=None if num_dit_steps is None else int(num_dit_steps),
            target_video=target_video,
        )

        policy = DreamZeroLocalPolicy(
            loaded,
            predict_micro_batch_size=int(
                OmegaConf.select(cfg, "base_model.predict_micro_batch_size", default=1)
                or 1
            ),
        )
        task_instruction = OmegaConf.select(cfg, "base_model.task_instruction")
        if task_instruction:
            policy.task = str(task_instruction)

        base_model_action_horizon = OmegaConf.select(cfg, "base_model.action_horizon")
        base_model_action_dim = OmegaConf.select(cfg, "base_model.action_dim")
        chunk_size = OmegaConf.select(cfg, "training.chunk_size")
        validate_dreamzero_dsrl_geometry(
            head_action_dim=loaded.head_action_dim,
            head_action_horizon=loaded.head_action_horizon,
            env_action_dim=policy.env_action_dim,
            noise_dim=int(OmegaConf.select(cfg, "dsrl.noise_dim")),
            chunk_size=None if chunk_size is None else int(chunk_size),
            action_exec_len=int(OmegaConf.select(cfg, "dsrl.action_exec_len")),
            base_model_action_horizon=(
                None if base_model_action_horizon is None else int(base_model_action_horizon)
            ),
            base_model_action_dim=(
                None if base_model_action_dim is None else int(base_model_action_dim)
            ),
        )
        return cls(policy, latent_dim=int(OmegaConf.select(cfg, "dsrl.noise_dim")))

    @property
    def policy(self) -> DreamZeroLocalPolicy:
        """The wrapped policy (the trainer syncs ``task`` onto it)."""
        return self._policy

    @property
    def action_horizon(self) -> int:
        return self._policy.action_horizon

    @property
    def action_dim(self) -> int:
        return self._policy.env_action_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def infer(self, observations: dict, noise: np.ndarray | None = None) -> dict:
        """Run one inference call, optionally steering the initial latent.

        Args:
            observations: Batched steering-env observation dict
                (``main_image``, ``wrist_image``, ``state``, ``prompt``;
                extra keys are ignored).
            noise: ``None`` (native mode — the model's own seeded draw),
                ``[B, latent_dim]`` (broadcast across the horizon; the
                validated recipe), or ``[B, K, latent_dim]`` with ``K <= H``
                (per-step; see the module docstring for the tiling
                deviation).

        Returns:
            ``{"actions": np.ndarray [B, H, A_raw]}`` (float32).

        Raises:
            ValueError: If ``noise`` has an unsupported shape.
            RuntimeError: If an injected latent was never consumed by the
                sampler (the patch's consumed-flag guard).
        """
        chunk = None
        if noise is not None:
            steer = np.asarray(noise, dtype=np.float32)
            valid = (
                steer.ndim in (2, 3)
                and steer.shape[-1] == self._latent_dim
                and (steer.ndim == 2 or steer.shape[1] <= self.action_horizon)
            )
            if not valid:
                raise ValueError(
                    f"noise must have shape [B, {self._latent_dim}] or "
                    f"[B, K, {self._latent_dim}] with K <= {self.action_horizon}; "
                    f"got {np.shape(noise)}"
                )
            if self._latent_clip > 0:
                steer = np.clip(steer, -self._latent_clip, self._latent_clip)
            from isaac_rl_posttraining.policies.pi0.policy import build_noise_chunk

            chunk = build_noise_chunk(steer, self.action_horizon)
        actions = self._policy.predict_chunk(observations, chunk)
        return {"actions": np.asarray(actions, dtype=np.float32)}
