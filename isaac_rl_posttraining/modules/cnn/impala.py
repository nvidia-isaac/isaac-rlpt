# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IMPALA-style CNN encoders for image observations.

PyTorch adaptation of the IMPALA residual encoder popularized for pixel-based
RL (Espeholt et al., 2018), with global average pooling on the output. Two
variants are provided: the standard three-stack encoder and a shallower
variant for smaller budgets.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ResnetStack", "ImpalaEncoder", "SmallerImpalaEncoder"]


def _normalize_input_shape(input_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Coerce an image shape to channels-first ``(C, H, W)``.

    A trailing dimension of at most 4 is treated as channels-last; 2-D shapes
    are treated as grayscale.
    """
    if len(input_shape) == 3:
        if input_shape[2] <= 4:
            return (input_shape[2], input_shape[0], input_shape[1])
        return (input_shape[0], input_shape[1], input_shape[2])
    if len(input_shape) == 2:
        return (1, input_shape[0], input_shape[1])
    raise ValueError(f"Unsupported input shape: {input_shape}")


def _to_bchw_normalized(x: torch.Tensor) -> torch.Tensor:
    """Batch, channel-order, and value-normalize an image tensor.

    Accepts ``[B, C, H, W]``, ``[B, H, W, C]``, or single unbatched frames;
    values above 1.0 are assumed to be ``[0, 255]`` and rescaled.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.size(-1) <= 4:
        x = x.permute(0, 3, 1, 2)
    if x.max() > 1.0:
        x = x.float() / 255.0
    return x


class ResnetStack(nn.Module):
    """One IMPALA stack: conv, optional max-pool, then residual blocks."""

    def __init__(
        self,
        num_ch: int,
        num_blocks: int,
        in_channels: int,
        use_max_pooling: bool = True,
    ) -> None:
        """Args:
        num_ch: Output channels of the stack.
        num_blocks: Number of two-conv residual blocks.
        in_channels: Input channels.
        use_max_pooling: Downsample 2x after the initial conv.
        """
        super().__init__()
        self.num_ch = num_ch
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling

        self.initial_conv = nn.Conv2d(in_channels, num_ch, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.initial_conv.weight)
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

        self.residual_blocks = nn.ModuleList(self._make_residual_block() for _ in range(num_blocks))

    def _make_residual_block(self) -> nn.Module:
        """Build a ReLU-conv-ReLU-conv residual block with Xavier init."""
        block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.num_ch, self.num_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_ch, self.num_ch, kernel_size=3, stride=1, padding=1),
        )
        for module in block:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stack to a ``[B, C, H, W]`` tensor."""
        out = self.initial_conv(x)
        if self.use_max_pooling:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        for block in self.residual_blocks:
            out = out + block(out)
        return out


class _ImpalaBase(nn.Module):
    """Shared forward path for the IMPALA encoder variants."""

    stack_blocks: nn.ModuleList
    global_pool: nn.AdaptiveAvgPool2d
    projection: Optional[nn.Module]
    output_dim: int

    def _init_pool_and_projection(
        self, output_dim: Optional[int], bounded_head: bool = False
    ) -> None:
        """Probe the feature width and attach the optional projection head.

        ``bounded_head`` appends LayerNorm + tanh after the projection (the
        DrQ/official-DSRL encoder head): features live in ``[-1, 1]``, which
        bounds what a downstream Q head can extrapolate from them.
        """
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            pooled = self.global_pool(self._forward_features(dummy))
            feature_dim = int(pooled.shape[1])

        if output_dim is not None:
            head: nn.Module = nn.Linear(feature_dim, output_dim)
            if bounded_head:
                head = nn.Sequential(head, nn.LayerNorm(output_dim), nn.Tanh())
            self.projection = head
        else:
            if bounded_head:
                raise ValueError("bounded_head requires output_dim (the head wraps the projection)")
            self.projection = None
            output_dim = feature_dim
        self.output_dim = output_dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run all stacks and the final activation on ``[B, C, H, W]`` input."""
        out = x
        for stack in self.stack_blocks:
            out = stack(out)
        return F.relu(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to a ``[B, output_dim]`` feature vector.

        Accepts channels-first or channels-last, batched or single frames,
        and values in ``[0, 1]`` or ``[0, 255]``.
        """
        x = _to_bchw_normalized(x)
        features = self._forward_features(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        if self.projection is not None:
            features = self.projection(features)
        return features


class ImpalaEncoder(_ImpalaBase):
    """Standard IMPALA encoder: three stacks of ``[16, 32, 32] * nn_scale`` channels."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_scale: int = 1,
        num_blocks_per_stack: int = 2,
        output_dim: Optional[int] = None,
        bounded_head: bool = False,
    ) -> None:
        """Args:
        input_shape: Image shape, ``(C, H, W)`` or ``(H, W, C)``.
        nn_scale: Channel-width multiplier.
        num_blocks_per_stack: Residual blocks per stack.
        output_dim: Optional linear projection width; ``None`` keeps the
            pooled channel count.
        """
        super().__init__()
        self.nn_scale = nn_scale
        self.num_blocks_per_stack = num_blocks_per_stack
        self.input_shape = _normalize_input_shape(input_shape)

        stack_sizes = [16, 32, 32]
        in_channels = [self.input_shape[0], stack_sizes[0] * nn_scale, stack_sizes[1] * nn_scale]
        self.stack_blocks = nn.ModuleList(
            ResnetStack(
                num_ch=size * nn_scale,
                num_blocks=num_blocks_per_stack,
                in_channels=in_ch,
                use_max_pooling=True,
            )
            for size, in_ch in zip(stack_sizes, in_channels)
        )
        self._init_pool_and_projection(output_dim, bounded_head=bounded_head)


class SmallerImpalaEncoder(_ImpalaBase):
    """Reduced-depth IMPALA encoder (2/1/1 residual blocks per stack)."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_scale: int = 1,
        output_dim: Optional[int] = None,
        bounded_head: bool = False,
    ) -> None:
        """Args:
        input_shape: Image shape, ``(C, H, W)`` or ``(H, W, C)``.
        nn_scale: Channel-width multiplier.
        output_dim: Optional linear projection width.
        """
        super().__init__()
        self.nn_scale = nn_scale
        self.input_shape = _normalize_input_shape(input_shape)

        stack_sizes = [16, 32, 32]
        num_blocks = [2, 1, 1]
        in_channels = [self.input_shape[0], stack_sizes[0] * nn_scale, stack_sizes[1] * nn_scale]
        self.stack_blocks = nn.ModuleList(
            ResnetStack(
                num_ch=size * nn_scale,
                num_blocks=blocks,
                in_channels=in_ch,
                use_max_pooling=True,
            )
            for size, blocks, in_ch in zip(stack_sizes, num_blocks, in_channels)
        )
        self._init_pool_and_projection(output_dim, bounded_head=bounded_head)
