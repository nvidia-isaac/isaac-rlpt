# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference-DSRL 'small' pixel encoder.

Four 32-channel 3x3 VALID convolutions with strides (2, 1, 1, 1) and ReLU,
orthogonal(sqrt(2)) kernels with zero bias, then flatten ->
Dense(bottleneck) -> LayerNorm -> tanh: a bounded feature box downstream
value heads cannot extrapolate far from. Matches the encoder the official
DSRL pi0 implementation trains its actor and critic on (64x64 inputs).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from isaac_rl_posttraining.modules.cnn.impala import _normalize_input_shape, _to_bchw_normalized

__all__ = ["SmallConvEncoder"]


class SmallConvEncoder(nn.Module):
    """Reference 'small' conv encoder with a bounded bottleneck head.

    Args:
        input_shape: Image shape, ``(C, H, W)`` or ``(H, W, C)``.
        bottleneck_dim: Width of the Dense -> LayerNorm -> tanh head.
    """

    def __init__(self, input_shape: tuple[int, ...], bottleneck_dim: int = 50) -> None:
        super().__init__()
        self.input_shape = _normalize_input_shape(input_shape)
        channels = self.input_shape[0]
        convs = []
        in_ch = channels
        for stride in (2, 1, 1, 1):
            convs.extend([nn.Conv2d(in_ch, 32, kernel_size=3, stride=stride, padding=0), nn.ReLU()])
            in_ch = 32
        self.convs = nn.Sequential(*convs)

        with torch.no_grad():
            flat = self.convs(torch.zeros(1, *self.input_shape)).flatten(1).shape[1]
        self.projection = nn.Linear(flat, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.output_dim = bottleneck_dim
        self.reinitialize_parameters()

    def reinitialize_parameters(self) -> None:
        """(Re-)apply the reference init: orthogonal convs, xavier bottleneck.

        Critic-ensemble construction re-initializes each copied member; this
        hook keeps the encoder's bespoke scheme instead of PyTorch defaults.
        """
        for module in self.convs:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images (any layout/range) to ``[B, bottleneck_dim]`` in [-1, 1]."""
        x = _to_bchw_normalized(x)
        features = self.convs(x).flatten(1)
        return torch.tanh(self.norm(self.projection(features)))
