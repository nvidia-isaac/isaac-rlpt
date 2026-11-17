# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint latent/residual action geometry for Residual Flow Steering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


__all__ = [
    "RFSActionModulator",
    "initialize_rfs_residual_head",
]


@dataclass(frozen=True)
class RFSActionModulator:
    """Split a joint policy action and add its bounded residual after decode."""

    latent_dim: int
    residual_action_dim: int
    residual_action_bound: float

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.residual_action_dim <= 0:
            raise ValueError(
                f"residual_action_dim must be positive, got {self.residual_action_dim}"
            )
        if self.residual_action_bound <= 0.0:
            raise ValueError(
                f"residual_action_bound must be positive, got {self.residual_action_bound}"
            )

    @property
    def policy_action_dim(self) -> int:
        """Width of one joint actor token."""
        return self.latent_dim + self.residual_action_dim

    def split(self, policy_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return flow latents and raw additive residuals."""
        actions = np.asarray(policy_actions)
        if actions.ndim not in (2, 3):
            raise ValueError(
                f"RFS policy actions must be [B, A] or [B, K, A], got shape {actions.shape}"
            )
        if actions.shape[-1] != self.policy_action_dim:
            raise ValueError(
                f"RFS policy action width must be {self.policy_action_dim} "
                f"({self.latent_dim} latent + {self.residual_action_dim} residual), "
                f"got {actions.shape[-1]}"
            )
        return (
            actions[..., : self.latent_dim],
            actions[..., self.latent_dim :],
        )

    def apply(self, decoded_actions: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Add residuals to the corresponding decoded action steps."""
        decoded = np.asarray(decoded_actions)
        residual = np.asarray(residuals)
        squeeze_batch = decoded.ndim == 2
        if squeeze_batch:
            decoded = decoded[None, ...]
        if decoded.ndim != 3:
            raise ValueError(
                f"Decoded actions must be [H, A] or [B, H, A], got shape {decoded.shape}"
            )
        if decoded.shape[-1] != self.residual_action_dim:
            raise ValueError(
                f"Decoded action width must match residual_action_dim="
                f"{self.residual_action_dim}, got {decoded.shape[-1]}"
            )

        if residual.ndim == 2:
            residual = residual[:, None, :]
        if residual.ndim != 3:
            raise ValueError(f"Residuals must be [B, A] or [B, K, A], got shape {residual.shape}")
        if residual.shape[0] != decoded.shape[0]:
            raise ValueError(
                f"Residual batch {residual.shape[0]} does not match decoded "
                f"batch {decoded.shape[0]}"
            )
        if residual.shape[-1] != decoded.shape[-1]:
            raise ValueError(
                f"Residual width {residual.shape[-1]} does not match decoded "
                f"action width {decoded.shape[-1]}"
            )
        if residual.shape[1] == 1:
            aligned_residual = residual
        elif residual.shape[1] >= decoded.shape[1]:
            aligned_residual = residual[:, : decoded.shape[1]]
        else:
            raise ValueError(
                f"Residual horizon {residual.shape[1]} is shorter than decoded "
                f"horizon {decoded.shape[1]}"
            )

        modulated = decoded + aligned_residual * self.residual_action_bound
        if squeeze_batch:
            return modulated[0]
        return modulated


def initialize_rfs_residual_head(
    actor: nn.Module,
    *,
    latent_dim: int,
    initial_residual_log_std: float,
) -> None:
    """Initialize the residual branch at zero without changing latent outputs.

    The residual mean starts exactly at zero and its normalized standard
    deviation starts at ``exp(initial_residual_log_std)``.
    This preserves the DSRL behavior distribution at initialization while
    leaving both residual parameters trainable.
    """
    if hasattr(actor, "mean_processor") and hasattr(actor, "log_std_processor"):
        mean_head = actor.mean_processor[-1]
        log_std_head = actor.log_std_processor[-1]
        chunk_size = 1
    elif hasattr(actor, "mean_layer") and hasattr(actor, "log_std_layer"):
        mean_head = actor.mean_layer
        log_std_head = actor.log_std_layer
        chunk_size = int(getattr(actor.config, "chunk_size", 1))
    else:
        raise TypeError(
            "RFS actor must expose linear mean/log-std output heads, "
            f"got {actor.__class__.__name__}"
        )
    if not isinstance(mean_head, nn.Linear) or not isinstance(log_std_head, nn.Linear):
        raise TypeError("RFS expected linear mean and log-std output heads")
    if mean_head.out_features != log_std_head.out_features:
        raise ValueError(
            f"RFS mean/log-std widths differ: {mean_head.out_features} "
            f"vs {log_std_head.out_features}"
        )
    if mean_head.out_features % chunk_size != 0:
        raise ValueError(
            f"RFS output width {mean_head.out_features} is not divisible by chunk size {chunk_size}"
        )
    action_dim = mean_head.out_features // chunk_size
    if not 0 < latent_dim < action_dim:
        raise ValueError(
            f"RFS latent_dim must be in (0, {action_dim}), got {latent_dim}"
        )

    with torch.no_grad():
        residual_indices = torch.arange(mean_head.out_features, device=mean_head.weight.device)
        residual_mask = (residual_indices % action_dim) >= latent_dim
        mean_head.weight[residual_mask] = 0.0
        mean_head.bias[residual_mask] = 0.0
        log_std_head.weight[residual_mask] = 0.0
        log_std_head.bias[residual_mask] = float(initial_residual_log_std)
