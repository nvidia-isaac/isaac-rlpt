# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseModelPolicy adapter for the GR00T steering wrapper.

:class:`Gr00tBaseModelPolicy` conforms :class:`~.wrapper.SteeredGr00tWrapper`
to the :class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` contract:
batched ``infer(observations, noise)`` with native mode on ``noise=None``,
broadcast and per-step latent shapes, and the
``action_horizon`` / ``action_dim`` / ``latent_dim`` properties.

Contract notes, as implemented by this base model:

* Native mode (``noise=None``) runs the wrapper's un-steered path -- the
  model samples its own Gaussian flow noise, identical to native GR00T
  evaluation.
* Injected latents are clipped at injection time (default plus/minus
  :data:`~isaac_rl_posttraining.policies.base.DEFAULT_LATENT_CLIP`) and
  un-steered noise entries are drawn fresh from N(0, 1) per call -- both
  governed by the wrapper's
  :class:`~.model.LatentInjectionConfig` defaults.
* Decoding is per-dimension: the head output is sliced into action groups
  and un-normalized (plus relative-to-absolute conversion) element-wise; no
  cross-dimension mixing happens after the head.
* The wrapped policy decodes one sample at a time, so batches are processed
  sample-by-sample. Observation entries with a leading batch axis are
  sliced per sample; entries without one are shared across the batch.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Optional

import numpy as np
import torch

from isaac_rl_posttraining.policies.base import BaseModelPolicy
from isaac_rl_posttraining.policies.gr00t.wrapper import SteeredGr00tWrapper

__all__ = ["Gr00tBaseModelPolicy"]


def _expand_multitask_dirs(spec: Any) -> Optional[list[str]]:
    """Expand a ``base_model.multitask_dirs`` config value into dataset roots.

    Accepts a list/tuple (returned as-is), a brace/space/comma-separated
    string of paths, or a single parent directory whose immediate
    subdirectories are the task datasets.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return [str(p) for p in spec]

    text = str(spec).strip()
    # Shell-style brace lists: "/root/{taskA,taskB}" -> two paths.
    if "{" in text and "}" in text:
        prefix, rest = text.split("{", 1)
        inner, suffix = rest.split("}", 1)
        candidates = [f"{prefix}{part}{suffix}" for part in inner.split(",")]
    else:
        candidates = text.replace(",", " ").split()

    expanded: list[str] = []
    for token in candidates:
        matches = sorted(glob.glob(token))
        expanded.extend(matches or [token])

    if len(expanded) == 1 and os.path.isdir(expanded[0]):
        children = sorted(p for p in glob.glob(os.path.join(expanded[0], "*")) if os.path.isdir(p))
        if children:
            return children
    return expanded


class Gr00tBaseModelPolicy(BaseModelPolicy):
    """The GR00T steering wrapper behind the ``BaseModelPolicy`` contract."""

    def __init__(self, wrapper: SteeredGr00tWrapper) -> None:
        """Adapt an already-constructed wrapper.

        Args:
            wrapper: A (typically frozen) steering wrapper exposing
                ``infer`` and the shape properties.
        """
        self._wrapper = wrapper

    @classmethod
    def from_checkpoint(cls, base_ckpt: str, **wrapper_kwargs: Any) -> "Gr00tBaseModelPolicy":
        """Construct the wrapper from checkpoints, then adapt it.

        Args:
            base_ckpt: GR00T base checkpoint directory (or model id).
            **wrapper_kwargs: Forwarded to
                :class:`~.wrapper.SteeredGr00tWrapper`.

        Returns:
            The adapted base model.
        """
        return cls(SteeredGr00tWrapper(base_ckpt=base_ckpt, **wrapper_kwargs))

    @classmethod
    def from_dsrl_config(
        cls, cfg: Any, device: Any, dataset_adapter: Any = None
    ) -> SteeredGr00tWrapper:
        """Build the frozen GR00T steering wrapper from the composed config.

        Returns the :class:`~.wrapper.SteeredGr00tWrapper` directly (not a
        ``Gr00tBaseModelPolicy``): DSRL steers the wrapper, and the training
        script reads ``wrapper.policy`` to sync the task instruction. The
        embodiment adapter is resolved by the launcher (which may touch the env
        layer) and passed in, so this base model stays embodiment-agnostic.
        """
        from omegaconf import OmegaConf

        def field(name: str, default: Any = None) -> Any:
            return OmegaConf.select(cfg, f"base_model.{name}", default=default)

        embodiment_tag = field("embodiment_tag")
        denoising_steps = field("denoising_steps")
        return SteeredGr00tWrapper(
            base_ckpt=field("ckpt"),
            backbone_model_name_or_path=field("backbone_model_name_or_path"),
            bc_ckpt=field("bc_ckpt"),
            dataset_dir=field("dataset_dir"),
            embodiment_tag=(
                "new_embodiment" if embodiment_tag is None else str(embodiment_tag)
            ),
            device=str(device),
            noise_dim=OmegaConf.select(cfg, "dsrl.noise_dim"),
            action_bound=float(OmegaConf.select(cfg, "dsrl.noise_action_bound")),
            multitask_dirs=_expand_multitask_dirs(field("multitask_dirs")),
            task_dir=field("task_dir"),
            task_instruction=field("task_instruction"),
            dataset_adapter=dataset_adapter,
            data_config=field("data_config"),
            denoising_steps=None if denoising_steps is None else int(denoising_steps),
        )

    @property
    def action_horizon(self) -> int:
        return int(self._wrapper.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self._wrapper.action_dim)

    @property
    def latent_dim(self) -> int:
        return int(self._wrapper.latent_dim)

    @property
    def latent_horizon(self) -> int:
        return int(getattr(self._wrapper, "latent_horizon", self._wrapper.action_horizon))

    def infer(self, observations: dict, noise: np.ndarray | None = None) -> dict:
        """Run one (batched) inference call; see the class docstring.

        Args:
            observations: Observation dict; array entries may carry a
                leading batch axis.
            noise: ``None`` (native mode), ``[B, latent_dim]`` (broadcast
                across the horizon), or ``[B, K, latent_dim]`` with
                ``K <= H`` (per-step latents).

        Returns:
            ``{"actions": np.ndarray [B, H, A_raw]}`` (float32).

        Raises:
            ValueError: If ``noise`` has an unsupported shape.
        """
        batch_size = self._batch_size(observations, noise)
        per_sample_noise = self._split_noise(noise, batch_size)
        actions = [
            self._infer_single(self._slice_observations(observations, b, batch_size), noise_b)
            for b, noise_b in enumerate(per_sample_noise)
        ]
        return {"actions": np.stack(actions).astype(np.float32)}

    def _infer_single(self, observations: dict, noise: np.ndarray | None) -> np.ndarray:
        """Decode one sample; returns ``[H, A_raw]``."""
        if noise is None:
            return np.asarray(self._wrapper.infer(observations, None)["actions"][0])
        # [latent_dim] -> [1, latent_dim] (broadcast) or
        # [K, latent_dim] -> [1, K, latent_dim] (per-step).
        return np.asarray(self._wrapper.infer(observations, noise[None])["actions"][0])

    def _split_noise(self, noise: np.ndarray | None, batch_size: int) -> list[np.ndarray | None]:
        """Validate the contract shapes and split the latent per sample."""
        if noise is None:
            return [None] * batch_size
        latent = np.asarray(noise, dtype=np.float32)
        latent_dim, window = self.latent_dim, self.latent_horizon
        valid = (
            latent.ndim in (2, 3)
            and latent.shape[0] == batch_size
            and latent.shape[-1] == latent_dim
            and (latent.ndim == 2 or latent.shape[1] <= window)
        )
        if not valid:
            raise ValueError(
                f"noise must have shape [B, {latent_dim}] or [B, K, {latent_dim}] with "
                f"B == {batch_size} and K <= {window} (the steering window); "
                f"got {np.shape(noise)}"
            )
        return [latent[b] for b in range(batch_size)]

    @staticmethod
    def _batch_size(observations: dict, noise: np.ndarray | None) -> int:
        """Infer the batch size from the observations, then the noise."""
        state = observations.get("state")
        if state is not None:
            state = np.asarray(state.detach().cpu() if torch.is_tensor(state) else state)
            if state.ndim >= 2:
                return int(state.shape[0])
        images = [observations.get("ego_view_image", observations.get("ego_view"))]
        images.extend(v for k, v in observations.items() if k.startswith("video."))
        for image in images:
            if image is not None and hasattr(image, "ndim") and image.ndim == 4:
                return int(image.shape[0])
        if noise is not None:
            return int(np.shape(noise)[0])
        return 1

    @staticmethod
    def _slice_observations(observations: dict, index: int, batch_size: int) -> dict:
        """Select one sample from batched observation entries.

        Entries whose leading axis equals the batch size are indexed;
        everything else (shared metadata, unbatched arrays) passes through.
        """
        if batch_size == 1:
            return observations
        sliced: dict[str, Any] = {}
        for key, value in observations.items():
            if (
                isinstance(value, np.ndarray) or torch.is_tensor(value)
            ) and value.ndim > 0 and value.shape[0] == batch_size:
                sliced[key] = value[index]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                sliced[key] = value[index]
            else:
                sliced[key] = value
        return sliced
