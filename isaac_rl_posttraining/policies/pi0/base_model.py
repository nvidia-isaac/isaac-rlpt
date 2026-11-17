# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseModelPolicy adapter for pi0.

Experimental -- ported, not yet validated in this repo.

:class:`Pi0BaseModelPolicy` conforms :class:`~.policy.Pi0Policy` to the
:class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` contract.

Documented deviations from the contract:

* when per-step latents cover only ``K < H`` steps, the remaining steps
  are filled by tiling the ``K``-step sequence (the openpi steering
  convention this port was validated with), not by fresh N(0, 1) noise;
* ``latent_dim`` (32, pi0's padded internal action width — the flow's
  noise space) exceeds the decoded ``action_dim`` (7 env dims for
  LIBERO): steering happens in the padded space, decoding slices it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP, BaseModelPolicy
from isaac_rl_posttraining.policies.pi0.policy import Pi0Policy

__all__ = ["Pi0BaseModelPolicy"]


class Pi0BaseModelPolicy(BaseModelPolicy):
    """A pi0 policy behind the ``BaseModelPolicy`` contract (experimental)."""

    def __init__(
        self,
        policy: Pi0Policy,
        action_dim: int | None = None,
        latent_dim: int | None = None,
        latent_clip: float = DEFAULT_LATENT_CLIP,
    ) -> None:
        """Adapt an already-constructed pi0 policy.

        Args:
            policy: The loaded :class:`~.policy.Pi0Policy`.
            action_dim: Width of one decoded action step; read from the
                policy when it exposes ``action_dim``, otherwise required.
            latent_dim: Steerable latent width per step; defaults to
                ``action_dim``. For LIBERO checkpoints this is 32 -- the
                padded width the flow denoises -- while decoded actions
                are 7-dim.
            latent_clip: Symmetric bound applied to injected latents at
                injection time; ``<= 0`` disables clipping.

        Raises:
            ValueError: If ``action_dim`` is neither given nor exposed by
                the policy.
        """
        self._policy = policy
        inferred = action_dim if action_dim is not None else getattr(
            policy.policy, "action_dim", None
        )
        if inferred is None:
            raise ValueError(
                "action_dim is required: the pi0 policy does not expose one."
            )
        self._action_dim = int(inferred)
        self._latent_dim = int(latent_dim) if latent_dim is not None else self._action_dim
        self._latent_clip = float(latent_clip)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, device: str = "cuda", **base_model_kwargs: Any
    ) -> "Pi0BaseModelPolicy":
        """Load the policy from a checkpoint, then adapt it.

        Args:
            checkpoint_dir: Path to the pi0 checkpoint directory.
            device: Torch device for converted feature tensors.
            **base_model_kwargs: Forwarded to the constructor.

        Returns:
            The adapted base model.
        """
        return cls(Pi0Policy(checkpoint_dir, device=device), **base_model_kwargs)

    @classmethod
    def from_dsrl_config(cls, cfg: Any, device: Any, dataset_adapter: Any = None) -> "Pi0BaseModelPolicy":
        """Build the pi0 base model from the composed config.

        Steering happens in pi0's padded internal action space, so
        ``latent_dim`` comes from ``dsrl.noise_dim``; ``action_dim`` is read
        from the checkpoint when ``base_model.action_dim`` is unset.
        """
        from omegaconf import OmegaConf

        base_ckpt = OmegaConf.select(cfg, "base_model.ckpt")
        if not base_ckpt:
            raise ValueError("base_model=pi0 requires base_model.ckpt (checkpoint directory)")
        action_dim = OmegaConf.select(cfg, "base_model.action_dim")
        return cls.from_checkpoint(
            str(base_ckpt),
            device=str(device),
            action_dim=None if action_dim is None else int(action_dim),
            latent_dim=int(OmegaConf.select(cfg, "dsrl.noise_dim")),
        )

    @property
    def policy(self) -> Pi0Policy:
        """The wrapped policy (the trainer syncs ``task`` onto it)."""
        return self._policy

    @property
    def action_horizon(self) -> int:
        horizon = self._policy.action_horizon
        if horizon is None:
            raise ValueError("pi0 policy does not expose an action horizon")
        return int(horizon)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def infer(self, observations: dict, noise: np.ndarray | None = None) -> dict:
        """Run one inference call, optionally steering the initial latent.

        Args:
            observations: pi0-format observation dict.
            noise: ``None`` (native mode), ``[B, latent_dim]`` (broadcast
                across the horizon), or ``[B, K, latent_dim]`` with
                ``K <= H`` (per-step; see the module docstring for the
                tiling deviation).

        Returns:
            ``{"actions": np.ndarray [B, H, A_raw]}`` (float32).

        Raises:
            ValueError: If ``noise`` has an unsupported shape.
        """
        steer = None
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
        result = self._policy.infer(observations, steer)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 2:  # unbatched [H, A] -> [1, H, A]
            actions = actions[None]
        return {"actions": actions}
