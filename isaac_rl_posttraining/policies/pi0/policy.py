# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0 policy integration over the ``openpi`` runtime.

Experimental -- ported, not yet validated in this repo.

Loads a pretrained pi0 / pi0.5 policy through ``openpi`` and exposes the
pieces latent-steering RL needs: native inference, inference with an
injected flow initial noise, and vision-language feature extraction. The
policy runs in openpi's own (JAX) runtime; this module converts at the
boundary to numpy/torch.

The ``openpi`` and ``openpi_client`` packages are optional: this module
imports without them, and only the functions that exercise the pi0 path
require them.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.utils.torch_utils import require_cuda_kernels

try:  # Optional dependency: needed only when the pi0 path is exercised.
    from openpi_client import image_tools
except ImportError:  # pragma: no cover - exercised only without openpi installed
    image_tools = None

logger = logging.getLogger(__name__)

__all__ = [
    "load_pi0_policy",
    "extract_vlm_features",
    "build_noise_chunk",
    "infer_with_noise",
    "Pi0Policy",
    "keep_jax_on",
]


def _config_name(checkpoint_dir: str) -> str:
    """Name of the openpi training config that produced ``checkpoint_dir``."""
    if "libero" in checkpoint_dir:
        return "pi05_libero" if "pi05" in checkpoint_dir else "pi0_libero"
    if "bridge" in checkpoint_dir:
        if "pi05" in checkpoint_dir:
            raise ValueError("pi05_bridge is not supported")
        return "pi0_lora_bridge_1_cam" if "lora" in checkpoint_dir else "pi0_bridge_1_cam"
    if "droid" in checkpoint_dir:
        if "jointpos" in checkpoint_dir:
            return "pi05_droid_jointpos"
        return "pi05_droid" if "pi05" in checkpoint_dir else "pi0_droid"
    raise ValueError(f"Cannot infer a pi0 config from checkpoint directory: {checkpoint_dir}")


def keep_jax_on(device: torch.device) -> None:
    """Run openpi's JAX on ``device`` alone, allocating as it goes.

    By default JAX reserves 75% of every visible GPU, so a process holding pi0
    would starve the rollout workers, policy servers and learner ranks that
    share those GPUs. ``XLA_PYTHON_CLIENT_PREALLOCATE`` set by the user wins.
    """
    import jax

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device.type == "cuda":
        jax.config.update("jax_cuda_visible_devices", str(device.index or 0))


def load_pi0_policy(checkpoint_dir: str) -> Any:
    """Load a pretrained pi0 policy from a checkpoint directory.

    The openpi training config is inferred from the checkpoint path
    (benchmark family and pi0 vs pi0.5 variant), matching how the
    checkpoints were produced.

    Args:
        checkpoint_dir: Path to the pi0 checkpoint directory.

    Returns:
        The loaded openpi policy object.

    Raises:
        ValueError: If no known config matches the checkpoint path.
    """
    from openpi.policies import policy_config
    from openpi.training import config

    pi0_config = config.get_config(_config_name(checkpoint_dir))

    policy = policy_config.create_trained_policy(pi0_config, checkpoint_dir)
    logger.info("loaded pi0 policy from %s", checkpoint_dir)
    return policy


def extract_vlm_features(pi0_policy: Any, observations: dict[str, Any]) -> torch.Tensor:
    """Extract vision-language hidden states from the pi0 prefix encoder.

    Args:
        pi0_policy: Loaded openpi policy.
        observations: pi0-format observation dict (images, state, prompt).

    Returns:
        Last-token hidden states as a float32 tensor of shape ``[B, D]``.
    """
    with torch.no_grad():
        hidden_states, _ = pi0_policy.get_prefix_rep(observations)
        hidden_states = hidden_states[:, -1, :]  # [B, seq, D] -> [B, D]
        # openpi returns JAX arrays (possibly bfloat16); convert via numpy.
        if hasattr(hidden_states, "__array__"):
            hidden_states = torch.from_numpy(np.array(hidden_states, dtype=np.float32))
        elif not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
        else:
            hidden_states = hidden_states.float()
    return hidden_states


def build_noise_chunk(noise: np.ndarray, horizon: int) -> np.ndarray:
    """Expand a steering latent to the policy's full flow horizon.

    Args:
        noise: ``[B, latent_dim]`` (one latent per sample, repeated over the
            horizon) or ``[B, K, latent_dim]`` with ``K <= horizon`` (the
            ``K``-step sequence is tiled to fill the remaining steps -- the
            openpi steering convention).
        horizon: The policy's action horizon.

    Returns:
        Noise of shape ``[B, horizon, latent_dim]``.

    Raises:
        ValueError: If ``noise`` has an unsupported rank or more than
            ``horizon`` steps.
    """
    noise = np.asarray(noise)
    if noise.ndim == 2:
        return np.repeat(noise[:, None, :], horizon, axis=1)
    if noise.ndim != 3 or noise.shape[1] > horizon:
        raise ValueError(
            f"noise must have shape [B, D] or [B, K, D] with K <= {horizon}; "
            f"got {noise.shape}"
        )
    steps = noise.shape[1]
    missing = horizon - steps
    if missing == 0:
        return noise
    # Tile the whole K-step sequence until the horizon is filled.
    full_repeats, remainder = divmod(missing, steps)
    parts = [noise]
    if full_repeats > 0:
        parts.append(np.tile(noise, (1, full_repeats, 1)))
    if remainder > 0:
        parts.append(noise[:, :remainder, :])
    return np.concatenate(parts, axis=1)


def infer_with_noise(
    pi0_policy: Any, observations: dict[str, Any], noise: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Run pi0 inference, optionally steering the flow initial noise.

    Args:
        pi0_policy: Loaded openpi policy; its ``infer`` accepts a ``noise``
            keyword of shape ``[B, horizon, action_dim]``.
        observations: pi0-format observation dict.
        noise: Steering latent (see :func:`build_noise_chunk`), or ``None``
            for native inference.

    Returns:
        The openpi result dict with ``"actions"`` converted to float32.

    Raises:
        ValueError: If the policy exposes no ``action_horizon`` while noise
            is given, or the noise shape is unsupported.
    """
    if noise is not None:
        horizon = getattr(pi0_policy, "action_horizon", None)
        if horizon is None:
            raise ValueError(
                "pi0 policy is missing the action_horizon attribute; cannot shape noise."
            )
        noise = build_noise_chunk(noise, int(horizon))

    result = pi0_policy.infer(observations, noise=noise)
    if "actions" in result:
        result["actions"] = np.array(result["actions"], dtype=np.float32)
    return result


class Pi0Policy:
    """Frozen pi0 policy with feature extraction and steered inference.

    Experimental -- ported, not yet validated in this repo.

    Attributes:
        policy: The loaded openpi policy object.
        device: Torch device features are returned on (the policy itself
            runs in openpi's runtime).
        task: Default language instruction, injected as ``"prompt"`` when
            an observation carries none (vectorized envs drop the
            per-observation prompt string).
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda") -> None:
        """Load and freeze the policy.

        Args:
            checkpoint_dir: Path to the pi0 checkpoint directory.
            device: Torch device for converted feature tensors.
        """
        self.device = torch.device(device)
        require_cuda_kernels(self.device)
        keep_jax_on(self.device)
        self.policy = load_pi0_policy(checkpoint_dir)
        self.task: str | None = None
        # The policy is steering-only and must never train.
        model = getattr(self.policy, "_model", None)
        if model is not None and hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad = False
        logger.info("pi0 policy wrapper initialized (device: %s)", device)

    @property
    def action_horizon(self) -> int | None:
        """The policy's action horizon, when exposed."""
        return getattr(self.policy, "action_horizon", None)

    def get_features(self, observations: dict[str, Any]) -> torch.Tensor:
        """Extract vision-language features for the learner.

        Args:
            observations: pi0-format observation dict; image entries are
                resized to 224x224 first.

        Returns:
            Feature tensor of shape ``[B, D]`` on :attr:`device`.
        """
        resized = {key: self._resize_if_image(value) for key, value in observations.items()}
        return extract_vlm_features(self.policy, resized).to(self.device)

    @staticmethod
    def _resize_if_image(value: Any) -> Any:
        """Resize 4-D image arrays to 224x224; pass everything else through."""
        if image_tools is None:
            return value
        if isinstance(value, np.ndarray) and value.ndim == 4:
            return image_tools.resize_with_pad(value, 224, 224)
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return image_tools.resize_with_pad(value.cpu().numpy(), 224, 224)
        return value

    def infer(
        self, observations: dict[str, Any], noise: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Run inference, optionally steered by ``noise``.

        Args:
            observations: pi0-format observation dict; image entries are
                resized to 224x224 first. The input dict is not mutated.
                When it carries no ``"prompt"``, :attr:`task` fills in.
            noise: Steering latent (see :func:`build_noise_chunk`) or
                ``None`` for native inference.

        Returns:
            The openpi result dict with float32 ``"actions"``.
        """
        # openpi's tokenizer takes one prompt (it calls ``prompt.item()``); a
        # batched observation carries one identical prompt per env, so use a
        # single entry. An explicit prompt wins, else ``task`` fills in.
        raw = observations.get("prompt")
        if raw is None:
            prompt = self.task
        elif isinstance(raw, (list, tuple, np.ndarray)):
            flat = np.asarray(raw).reshape(-1)
            prompt = flat[0] if flat.size else self.task
        else:
            prompt = raw
        resized = {
            key: self._resize_if_image(value)
            for key, value in observations.items()
            if key != "prompt"
        }
        if prompt is not None:
            resized["prompt"] = prompt
        return infer_with_noise(self.policy, resized, noise)

    def __call__(
        self, observations: dict[str, Any], noise: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Alias for :meth:`infer`."""
        return self.infer(observations, noise)
