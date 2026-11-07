# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process DreamZero inference with DSRL noise injection.

:class:`DreamZeroLocalPolicy` owns one predict step: env observations →
checkpoint-faithful transform chain → the WAN joint video/action denoise
(with the SAC latent injected as the action's initial noise) → decoded
environment actions. The mechanism is RLinf's validated one
(``dreamzero_policy.predict_action_batch``): the latent rides a side
channel on the action head, the patched ``generate_noise`` consumes it, and
a consumed-flag check makes "the latent never reached the sampler" a hard
error instead of a silent no-op.

Inference is **stateless per chunk**: observations enter as single frames
(``T=1``), which resets the head's autoregressive video context on every
call. Accumulating that context across calls measurably hurt DSRL training
in RLinf's campaigns, so no session mode is offered.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.policies.dreamzero.layouts import (
    convert_env_obs,
    ensure_batched_env_obs,
)
from isaac_rl_posttraining.policies.dreamzero.model import LoadedDreamZero

__all__ = ["DreamZeroLocalPolicy"]


class DreamZeroLocalPolicy:
    """Frozen DreamZero policy with steered, batched chunk prediction.

    Attributes:
        task: Default language instruction, injected as ``"prompt"`` when an
            observation carries none (vectorized envs drop the
            per-observation prompt string).
    """

    def __init__(
        self,
        loaded: LoadedDreamZero,
        *,
        predict_micro_batch_size: int = 1,
    ) -> None:
        """Wrap a loaded DreamZero bundle.

        Args:
            loaded: The model bundle from
                :func:`~isaac_rl_posttraining.policies.dreamzero.model.
                load_dreamzero_model`.
            predict_micro_batch_size: Rows per model call. The default 1 is
                deliberate: it sidesteps a groot batched-transform hazard
                (a non-ndarray language batch gets split into *characters*)
                and larger checkpoints' batch>1 OOM on 48 GiB GPUs; raise it
                only on large-memory GPUs after verifying prompts survive
                intact.
        """
        self._loaded = loaded
        self._predict_micro_batch_size = int(predict_micro_batch_size)
        if self._predict_micro_batch_size < 1:
            raise ValueError(
                f"predict_micro_batch_size must be >= 1, got {predict_micro_batch_size}"
            )
        self.task: str | None = None
        from isaac_rl_posttraining.policies.dreamzero.transforms import (
            env_action_dim_from_metadata,
        )

        self._env_action_dim = env_action_dim_from_metadata(
            loaded.metadata, loaded.spec
        )

    @property
    def action_horizon(self) -> int:
        """Env steps decoded per predict (the checkpoint's chunk length)."""
        return self._loaded.head_action_horizon

    @property
    def noise_dim(self) -> int:
        """The WAN head's latent action width — the DSRL steering width."""
        return self._loaded.head_action_dim

    @property
    def env_action_dim(self) -> int:
        """Decoded environment action width (from the checkpoint metadata)."""
        return self._env_action_dim

    def predict_chunk(
        self, env_obs: dict[str, Any], noise: np.ndarray | None
    ) -> np.ndarray:
        """Decode one action chunk per environment, optionally steered.

        Args:
            env_obs: Batched steering-env observation dict (``main_image``,
                ``wrist_image``, ``state``, ``prompt``; extra keys are
                ignored). When ``prompt`` is absent, :attr:`task` fills in.
            noise: ``None`` for the native (seeded) draw, or
                ``[B, action_horizon, noise_dim]`` — the full action-latent
                tensor the sampler would otherwise draw.

        Returns:
            Decoded absolute env actions, float32
            ``[B, action_horizon, env_action_dim]``.
        """
        if "main_image" not in env_obs:
            raise KeyError(
                f"env_obs must contain 'main_image' (got keys {sorted(env_obs)})."
            )
        env_obs = ensure_batched_env_obs(env_obs)
        if "prompt" not in env_obs or env_obs["prompt"] is None:
            if self.task is None:
                raise KeyError(
                    "Observation carries no 'prompt' and no default task "
                    "instruction is set; set base_model.task_instruction or expose "
                    "the instruction on the environment."
                )
            env_obs = {**env_obs, "prompt": self.task}

        converted = convert_env_obs(env_obs, self._loaded.spec)
        batch_size = int(np.asarray(env_obs["main_image"]).shape[0])
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            expected = (batch_size, self.action_horizon, self.noise_dim)
            if noise.shape != expected:
                raise ValueError(
                    f"predict_chunk noise must be {expected}, got {noise.shape}."
                )

        chunks: list[np.ndarray] = []
        step = self._predict_micro_batch_size
        for start in range(0, batch_size, step):
            stop = min(start + step, batch_size)
            rows = {key: value[start:stop] for key, value in converted.items()}
            row_noise = None if noise is None else noise[start:stop]
            chunks.append(self._predict_rows(rows, row_noise))
        return np.concatenate(chunks, axis=0)

    def _predict_rows(
        self, converted: dict[str, Any], noise: np.ndarray | None
    ) -> np.ndarray:
        """One model call on already-converted modality rows."""
        loaded = self._loaded
        normalized_input = loaded.transforms(dict(converted))

        action_head = loaded.model.action_head
        injected = None
        if noise is not None:
            injected = torch.as_tensor(noise, dtype=torch.float32)
        try:
            if injected is not None:
                # Side channel consumed by the patched generate_noise (see
                # policies/dreamzero/patch.py).
                action_head._dsrl_external_action_noise = injected
                action_head._dsrl_noise_consumed = False
            with torch.no_grad():
                model_pred = loaded.model.lazy_joint_video_action_causal(
                    normalized_input
                )
            if injected is not None and not action_head._dsrl_noise_consumed:
                raise RuntimeError(
                    "DSRL noise was set but never consumed by the sampler — the "
                    "generate_noise patch did not take effect (incompatible groot "
                    "version?). Refusing to continue: SAC would train on latents "
                    "that were never applied."
                )
        finally:
            if injected is not None:
                action_head._dsrl_external_action_noise = None

        normalized_action = model_pred["action_pred"].float()
        unnormalized = loaded.transforms.unapply({"action": normalized_action.cpu()})

        parts: list[np.ndarray] = []
        for key in loaded.spec.action_keys:
            if key not in unnormalized:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; available: "
                    f"{sorted(unnormalized)}."
                )
            value = unnormalized[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32))
        actions = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)

        if loaded.spec.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )
        return actions
