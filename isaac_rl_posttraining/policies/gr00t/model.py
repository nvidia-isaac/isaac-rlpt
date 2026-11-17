# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T base-model wrapper and steering-noise construction.

Wraps a loaded GR00T N1.7 model (vision-language backbone plus flow-matching
action head) in the small surface the RL layer builds on:

* :meth:`Gr00tModel.extract_features` -- gradient-carrying feature tokens for
  critic heads.
* :meth:`Gr00tModel.actor_bc_loss` -- the model's own action loss, per sample.
* :meth:`Gr00tModel.get_action` -- native diffusion sampling (the un-steered
  base policy).
* :meth:`Gr00tModel.get_action_with_noise` -- decoding steered by an external
  latent, for steering algorithms such as DSRL that run RL in the flow
  initial-noise space.

:func:`build_base_noise` is the pure function at the heart of steering: it
turns a latent of shape ``[B, latent_dim]`` or ``[B, K, latent_dim]`` into the
``[B, H, D]`` flow initial-noise tensor under an explicit, fully
parameterized :class:`LatentInjectionConfig` -- no hidden state and no
environment variables.

This module never imports the external ``gr00t`` package; it operates on an
already-loaded model, so it is importable and unit-testable without the GR00T
stack installed.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn

from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP

__all__ = ["LatentInjectionConfig", "build_base_noise", "Gr00tModel"]


@dataclass(frozen=True)
class LatentInjectionConfig:
    """How a steering latent becomes the flow-matching initial noise.

    The defaults reproduce the validated steering configuration: clipped,
    unscaled latents that replace the steered block, tiled across the full
    flow horizon, on top of a freshly sampled Gaussian base.

    Attributes:
        latent_clip: Symmetric bound applied to the injected latent at
            injection time; a Gaussian actor's rare tail draws would
            otherwise push the flow decoder off-distribution. Values
            ``<= 0`` disable clipping.
        latent_scale: Multiplier applied after clipping; values below 1
            shrink the steering toward the native noise distribution.
        additive: When ``True`` the latent perturbs the sampled base noise
            instead of replacing the steered block, so a zero latent decodes
            exactly like the native policy.
        full_horizon_tile: When per-step latents cover ``K < H`` steps, tile
            them across all ``H`` flow steps (the whole horizon the decoder
            couples over). ``False`` steers only the first ``K`` steps.
        fresh_base: Draw the un-steered entries (padding dimensions and any
            un-tiled steps) fresh from N(0, 1) on every call, so a standard
            normal latent reproduces the true base-policy distribution.
            ``False`` uses a fixed seed-0 base, which makes decoding
            deterministic but no longer matches the native distribution.
        tile_dims: Repeat the steered block across the full action width
            ``D`` as well, so no entry of the ``[B, H, D]`` initial noise is
            left random. The flow head denoises all ``D`` dimensions
            jointly, so the otherwise-random padding dimensions couple into
            the steered ones.
    """

    latent_clip: float = DEFAULT_LATENT_CLIP
    latent_scale: float = 1.0
    additive: bool = False
    full_horizon_tile: bool = True
    fresh_base: bool = True
    tile_dims: bool = False


def _latent_as_steps(latent: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Reshape a steering latent to ``[B, K, latent_dim]`` and validate it."""
    lat = torch.as_tensor(latent)
    if lat.ndim == 1:
        lat = lat[None]  # [latent_dim] -> [1, latent_dim]
    if lat.ndim == 2:
        lat = lat[:, None, :]  # [B, latent_dim] -> [B, 1, latent_dim]
    if lat.ndim != 3:
        raise ValueError(
            f"latent must have shape [latent_dim], [B, latent_dim], or [B, K, latent_dim]; "
            f"got {tuple(torch.as_tensor(latent).shape)}"
        )
    if lat.shape[-1] > action_dim:
        raise ValueError(
            f"latent width {lat.shape[-1]} exceeds the action width {action_dim}"
        )
    return lat


def build_base_noise(
    latent: torch.Tensor,
    horizon: int,
    action_dim: int,
    config: LatentInjectionConfig | None = None,
) -> torch.Tensor:
    """Build the ``[B, H, D]`` flow initial noise from a steering latent.

    Args:
        latent: Steering latent. Accepted shapes: ``[latent_dim]`` (single
            sample), ``[B, latent_dim]`` (one latent per sample, broadcast
            across the horizon), or ``[B, K, latent_dim]`` (per-step latents
            for the first ``K`` steps, placed according to
            ``config.full_horizon_tile``). ``latent_dim`` may be smaller
            than ``action_dim``; the remaining dimensions are padding.
        horizon: Flow horizon ``H`` of the action head.
        action_dim: Padded action width ``D`` of the action head.
        config: Injection knobs; defaults to :class:`LatentInjectionConfig`.

    Returns:
        Initial-noise tensor of shape ``[B, H, D]`` on ``latent``'s device
        with ``latent``'s dtype.

    Raises:
        ValueError: If ``latent`` has more than three dimensions or is wider
            than ``action_dim``.
    """
    config = config if config is not None else LatentInjectionConfig()

    lat = _latent_as_steps(latent, action_dim)
    if config.latent_clip > 0:
        lat = lat.clamp(-config.latent_clip, config.latent_clip)
    lat = lat * config.latent_scale

    batch, steps, width = lat.shape

    # The base tensor: everything not steered below stays exactly this noise.
    if config.fresh_base:
        base = torch.randn(batch, horizon, action_dim, device=lat.device, dtype=torch.float32)
    else:
        generator = torch.Generator(device=lat.device).manual_seed(0)
        base = torch.randn(
            batch, horizon, action_dim, generator=generator, device=lat.device, dtype=torch.float32
        )
    base = base.to(lat.dtype)

    if steps == 1 or config.full_horizon_tile:
        # Repeat the K-step sequence to cover all H steps (a single latent
        # broadcasts across every flow step).
        repeats = (horizon + steps - 1) // steps
        steered = lat.repeat(1, repeats, 1)[:, :horizon, :]
        block = base[:, :, :width]
        base[:, :, :width] = block + steered if config.additive else steered
    else:
        # Per-step latents on the first K steps only; later steps stay base noise.
        k = min(steps, horizon)
        block = base[:, :k, :width]
        base[:, :k, :width] = block + lat[:, :k, :] if config.additive else lat[:, :k, :]

    if config.tile_dims:
        # Repeat the steered block across the full action width so no
        # padding dimension is left as uncontrolled noise.
        steered_block = base[:, :, :width]
        repeats = (action_dim + width - 1) // width
        base = steered_block.repeat(1, 1, repeats)[:, :, :action_dim].contiguous()

    return base


# torch.randn is process-global; concurrent decodes (in-process base replicas) must not interleave.
_RANDN_LOCK = threading.Lock()


@contextmanager
def _randn_override(base: torch.Tensor, shape: tuple[int, ...]) -> Iterator[None]:
    """Make ``torch.randn`` return ``base`` for calls requesting ``shape``.

    The flow-matching head samples its initial noise internally with
    ``torch.randn``; this shim substitutes the steering noise without
    modifying the frozen model. Calls with any other shape pass through
    unchanged, and the original function is always restored.

    Args:
        base: Tensor to return for matching calls.
        shape: Exact shape that triggers the substitution.
    """
    def _shim(*args: Any, **kwargs: Any) -> torch.Tensor:
        size = kwargs.get("size")
        if size is None and args:
            # torch.randn((B, H, D)) / torch.randn(torch.Size) vs torch.randn(B, H, D)
            packed = len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size))
            size = args[0] if packed else args
        try:
            match = size is not None and tuple(size) == tuple(shape)
        except TypeError:
            match = False
        if match:
            return base.to(
                dtype=kwargs.get("dtype", base.dtype), device=kwargs.get("device", base.device)
            )
        return original_randn(*args, **kwargs)

    with _RANDN_LOCK:
        original_randn = torch.randn
        torch.randn = _shim
        try:
            yield
        finally:
            torch.randn = original_randn


class Gr00tModel(nn.Module):
    """Thin wrapper over a loaded GR00T N1.7 model.

    Exposes feature extraction for critic heads, the model's own action loss
    for behavior-cloning terms, and both native and latent-steered action
    decoding. The wrapped model is expected to provide ``prepare_input``,
    ``backbone``, ``action_head`` (with ``_encode_features`` and ``forward``),
    and ``get_action``.

    Attributes:
        base_model: The wrapped GR00T model.
        latent_injection: Knobs for :meth:`get_action_with_noise`.
        action_horizon: Flow horizon ``H`` read from the model config.
        action_dim: Padded action width ``D`` read from the model config.
        feat_dim: Backbone feature width used to size critic heads.
    """

    def __init__(
        self, base_model: nn.Module, latent_injection: LatentInjectionConfig | None = None
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.latent_injection = (
            latent_injection if latent_injection is not None else LatentInjectionConfig()
        )
        self.action_horizon = int(base_model.config.action_horizon)
        self.action_dim = int(base_model.config.max_action_dim)
        self.feat_dim = int(getattr(base_model.config, "backbone_embedding_dim", 1536))

    def extract_features(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone and state encoder, keeping gradients.

        Uses the action head's feature encoder, which needs no action and
        does not run the diffusion transformer -- cheap enough to evaluate
        for both the current and the next observation.

        Args:
            inputs: Already-processed GR00T model inputs.

        Returns:
            ``(backbone_features, state_features)`` of shapes ``[B, T, D]``
            and ``[B, S, D]``.
        """
        backbone_inputs, action_inputs = self.base_model.prepare_input(inputs)
        backbone_output = self.base_model.backbone(backbone_inputs)
        encoded = self.base_model.action_head._encode_features(backbone_output, action_inputs)
        return encoded["backbone_features"], encoded["state_features"]

    def feature_tokens(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Concatenate backbone and state feature tokens to ``[B, T + S, D]``."""
        backbone_features, state_features = self.extract_features(inputs)
        return torch.cat([backbone_features, state_features], dim=1)

    def actor_bc_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Return the model's own action loss, reduced per sample to ``[B]``.

        The head emits a fixed ``[B, max_horizon, max_action_dim]`` grid and
        reports which part the embodiment occupies in ``action_mask``, so the
        reduction is a masked mean. Averaging the whole grid divides by the
        padding as well: a 28-DoF embodiment whose 16x28 window sits in a
        40x132 grid reads 11.8x low, and any coefficient applied to this loss
        is scaled down by that factor with nothing in the metrics to show it.
        The masked mean is also what the head's own scalar loss reports.

        Sibling reductions over the same padded grid crop instead of masking --
        the flow log-prob in ``isaac_rl_posttraining/policies/flow_sde.py``,
        and RLinf's N1.7 actor, which slices to
        ``[:action_chunk, :valid_action_dim]`` before averaging. Cropping works
        there because those callers own the layout and the padding is a
        trailing block. Here the head hands back its loss already reduced over
        a grid whose valid region only ``action_mask`` describes, so the mask
        is the reliable source rather than a width this method would have to be
        told separately.
        """
        backbone_inputs, action_inputs = self.base_model.prepare_input(inputs)
        backbone_output = self.base_model.backbone(backbone_inputs)
        out = self.base_model.action_head.forward(backbone_output, action_inputs)
        action_loss = out["action_loss"]  # [B, H, A]
        samples = action_loss.shape[0]
        flat_loss = action_loss.reshape(samples, -1)
        flat_mask = out["action_mask"].to(action_loss.dtype).reshape(samples, -1)
        return (flat_loss * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def get_action(self, inputs: dict[str, Any]) -> Any:
        """Sample an action chunk natively (the un-steered base policy)."""
        return self.base_model.get_action(inputs)

    @torch.no_grad()
    def get_action_with_noise(self, inputs: dict[str, Any], latent: torch.Tensor) -> torch.Tensor:
        """Decode an action chunk with an external latent as the flow noise.

        The latent is expanded to the full ``[B, H, D]`` initial noise by
        :func:`build_base_noise` under :attr:`latent_injection`, then
        substituted for the head's internal ``torch.randn`` draw.

        Args:
            inputs: Already-processed GR00T model inputs (batch size ``B``
                must match the latent's).
            latent: Steering latent; see :func:`build_base_noise` for
                accepted shapes.

        Returns:
            Predicted normalized actions of shape ``[B, H, D]``.
        """
        base = self.prepare_initial_noise(latent)
        return self.get_action_with_initial_noise(inputs, base)

    def prepare_initial_noise(self, latent: torch.Tensor) -> torch.Tensor:
        """Expand one or more steering latents using the current RNG stream."""
        param = next(self.base_model.parameters())
        lat = torch.as_tensor(latent, device=param.device, dtype=param.dtype)
        return build_base_noise(
            lat,
            self.action_horizon,
            self.action_dim,
            self.latent_injection,
        )

    def sample_native_initial_noise(self, batch_size: int = 1) -> torch.Tensor:
        """Draw the native flow noise explicitly for reproducible batched decode."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        param = next(self.base_model.parameters())
        return torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=param.device,
            dtype=param.dtype,
        )

    @torch.no_grad()
    def get_action_with_initial_noise(
        self, inputs: dict[str, Any], initial_noise: torch.Tensor
    ) -> torch.Tensor:
        """Decode an already expanded ``[B, H, D]`` initial-noise tensor."""
        param = next(self.base_model.parameters())
        base = torch.as_tensor(
            initial_noise,
            device=param.device,
            dtype=param.dtype,
        )
        expected_tail = (self.action_horizon, self.action_dim)
        if base.ndim != 3 or tuple(base.shape[1:]) != expected_tail:
            raise ValueError(
                "initial_noise must have shape [B, H, D] with "
                f"(H, D)={expected_tail}; got {tuple(base.shape)}"
            )
        with _randn_override(
            base,
            (base.shape[0], self.action_horizon, self.action_dim),
        ):
            out = self.base_model.get_action(inputs)
        return out["action_pred"] if hasattr(out, "__getitem__") else out
