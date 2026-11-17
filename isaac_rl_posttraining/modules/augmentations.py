# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel augmentations for image observations in replay batches."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["PixelAugmentation", "color_jitter_images", "random_shift_images"]


def color_jitter_images(
    images: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.03,
    apply_prob: float = 0.8,
) -> torch.Tensor:
    """Reference-DSRL color jitter for a ``[B, H, W, C]`` batch.

    Per-sample factors, random op order per call, per-sample ``apply_prob``
    gate; operates in the input value range (0..255 or 0..1). Returns
    float32.
    """
    if images.dim() != 4:
        raise ValueError(f"expected [B, H, W, C] images, got shape {tuple(images.shape)}")
    x = images.float()
    scale = 255.0 if x.max() > 1.0 else 1.0
    x = x / scale
    b = x.shape[0]

    def _rand(low: float, high: float) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=x.device).uniform_(low, high)

    def _brightness(img: torch.Tensor) -> torch.Tensor:
        return img * _rand(1 - brightness, 1 + brightness)

    def _contrast(img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        return (img - mean) * _rand(1 - contrast, 1 + contrast) + mean

    def _saturation(img: torch.Tensor) -> torch.Tensor:
        grey = img.mean(dim=-1, keepdim=True)
        return grey + (img - grey) * _rand(1 - saturation, 1 + saturation)

    def _hue(img: torch.Tensor) -> torch.Tensor:
        # Approximate hue rotation via channel rotation in YIQ space.
        theta = torch.empty(b, 1, 1, device=img.device).uniform_(-hue, hue) * 2 * math.pi
        rgb = img.permute(0, 3, 1, 2)
        y = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        i = 0.596 * rgb[:, 0] - 0.274 * rgb[:, 1] - 0.322 * rgb[:, 2]
        q = 0.211 * rgb[:, 0] - 0.523 * rgb[:, 1] + 0.312 * rgb[:, 2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        i2 = i * cos_t - q * sin_t
        q2 = i * sin_t + q * cos_t
        r = y + 0.956 * i2 + 0.621 * q2
        g = y - 0.272 * i2 - 0.647 * q2
        b2 = y - 1.106 * i2 + 1.703 * q2
        return torch.stack([r, g, b2], dim=-1)

    ops = [_brightness, _contrast, _saturation, _hue]
    for idx in torch.randperm(len(ops)).tolist():
        x = ops[idx](x)
    x = x.clamp(0.0, 1.0)
    keep = (torch.rand(b, 1, 1, 1, device=x.device) < apply_prob).float()
    out = keep * x + (1 - keep) * (images.float() / scale)
    return out * scale


def random_shift_images(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random-shift augmentation for a ``[B, H, W, C]`` image batch.

    The DrQ-style regularizer for pixel critics at high update ratios:
    replicate-pad by ``pad`` and resample each image at a per-sample integer
    shift (bilinear grid sample). Returns float32 in the input value range.
    """
    if images.dim() != 4:
        raise ValueError(f"expected [B, H, W, C] images, got shape {tuple(images.shape)}")
    x = images.permute(0, 3, 1, 2).float()
    n, _, h, w = x.shape
    if h != w:
        raise ValueError(f"random shift expects square images, got {h}x{w}")
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift = shift * 2.0 / (h + 2 * pad)
    out = F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)
    return out.permute(0, 2, 3, 1)


class PixelAugmentation:
    """DrQ-style regularization of the image keys of an update batch.

    Every observation key whose name contains ``"image"`` and holds a
    ``[B, H, W, C]`` tensor is random-shifted, then optionally color-jittered,
    with independent draws for the current and next observations. The draw
    order per key is fixed (shift obs, shift next_obs, jitter obs, jitter
    next_obs) so a seeded run reproduces the same augmentations.

    Args:
        color_jitter: Also apply :func:`color_jitter_images` after the shift.
        pad: Replicate padding of :func:`random_shift_images`.
    """

    def __init__(self, *, color_jitter: bool, pad: int = 4) -> None:
        self.color_jitter = color_jitter
        self.pad = pad

    def apply_(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        next_obs: dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        """Augment the image keys of ``obs`` and ``next_obs`` in place."""
        if not isinstance(obs, dict):
            # Image keys only exist in dict observations.
            return
        for key in obs:
            if "image" in key and torch.is_tensor(obs[key]) and obs[key].dim() == 4:
                obs[key] = random_shift_images(obs[key], pad=self.pad)
                next_obs[key] = random_shift_images(next_obs[key], pad=self.pad)
                if self.color_jitter:
                    obs[key] = color_jitter_images(obs[key])
                    next_obs[key] = color_jitter_images(next_obs[key])
