# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running statistics for SACEdit's low-dimensional simulator state."""

from __future__ import annotations

import torch
from torch import nn


class RunningMeanStdNormalizer(nn.Module):
    """Normalize a fixed-width vector using streaming population statistics."""

    def __init__(
        self,
        input_dim: int,
        *,
        epsilon: float = 1.0e-5,
        clip: float = 5.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if clip <= 0.0:
            raise ValueError("clip must be positive")
        self.input_dim = int(input_dim)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.register_buffer(
            "running_mean",
            torch.zeros(self.input_dim, dtype=torch.float64, device=device),
        )
        self.register_buffer(
            "running_var",
            torch.ones(self.input_dim, dtype=torch.float64, device=device),
        )
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=device))

    def _flatten(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        else:
            value = value.flatten(start_dim=1)
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input width {self.input_dim}, got {value.shape[-1]}"
            )
        return value

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = self._flatten(value).detach().to(
            device=self.running_mean.device,
            dtype=torch.float64,
        )
        if value.shape[0] == 0:
            return
        batch_mean = value.mean(dim=0)
        batch_var = (value - batch_mean).square().mean(dim=0)
        batch_count = torch.as_tensor(
            value.shape[0],
            device=value.device,
            dtype=torch.float64,
        )
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / total_count
        current_m2 = self.running_var * self.count
        batch_m2 = batch_var * batch_count
        correction = delta.square() * self.count * batch_count / total_count
        self.running_mean.copy_(new_mean)
        self.running_var.copy_((current_m2 + batch_m2 + correction) / total_count)
        self.count.copy_(total_count)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        flattened = self._flatten(value)
        mean = self.running_mean.to(device=value.device, dtype=value.dtype)
        variance = self.running_var.to(device=value.device, dtype=value.dtype)
        normalized = (flattened - mean) / torch.sqrt(variance + self.epsilon)
        return normalized.clamp(-self.clip, self.clip).reshape(original_shape)
