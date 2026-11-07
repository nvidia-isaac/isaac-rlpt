# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive reward scaling (opt-in; off in the shipped SAC-edit recipe)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
from torch import nn


__all__ = ["AdaptiveRewardScaler"]


class AdaptiveRewardScaler(nn.Module):
    """Scale rewards from streaming discounted-return statistics.

    Each episode accumulates its own discounted return, so decisions from
    several environments may arrive interleaved.
    """

    def __init__(
        self,
        *,
        gamma: float,
        support_g_max: float,
        epsilon: float = 1.0e-8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if support_g_max <= 0.0:
            raise ValueError("support_g_max must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.gamma = float(gamma)
        self.support_g_max = float(support_g_max)
        self.epsilon = float(epsilon)
        # Running episodes restart after a resume, so their returns are not checkpointed.
        self._episode_returns: dict[Hashable, torch.Tensor] = {}
        self.register_buffer(
            "return_mean", torch.zeros((), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "return_var", torch.ones((), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "return_count", torch.zeros((), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "return_max", torch.zeros((), dtype=torch.float64, device=device)
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Older checkpoints kept one return shared by all episodes; running episodes restart on resume.
        state_dict.pop(f"{prefix}discounted_return", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        truncateds: Sequence[bool],
        episode_ids: Sequence[Hashable],
    ) -> None:
        """Add one decision per entry, each to the discounted return of its episode."""
        device = self.return_max.device
        rewards = torch.as_tensor(rewards, dtype=torch.float64, device=device).flatten()
        dones = torch.as_tensor(dones, dtype=torch.bool, device=device).flatten()
        truncateds = torch.as_tensor(truncateds, dtype=torch.bool, device=device).flatten()
        if not rewards.shape == dones.shape == truncateds.shape == (len(episode_ids),):
            raise ValueError("reward, done, truncated and episode streams must have equal lengths")
        for reward, done, truncated, episode in zip(
            rewards, dones, truncateds, episode_ids, strict=True
        ):
            discounted = self._episode_returns.pop(
                episode, torch.zeros((), dtype=torch.float64, device=device)
            )
            discounted = discounted.mul_(self.gamma).add_(reward)
            self._update_running_stats(discounted.reshape(1))
            self.return_max.copy_(torch.maximum(self.return_max, discounted))
            if not bool(done or truncated):
                self._episode_returns[episode] = discounted

    def denominator(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        std_denominator = torch.sqrt(self.return_var.clamp_min(0.0)) + self.epsilon
        max_denominator = self.return_max / self.support_g_max
        return torch.maximum(std_denominator, max_denominator).clamp_min(
            self.epsilon
        ).to(device=device, dtype=dtype)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards / self.denominator(device=rewards.device, dtype=rewards.dtype)

    def _update_running_stats(self, returns: torch.Tensor) -> None:
        values = returns.detach().flatten(start_dim=0).to(dtype=torch.float64)
        batch_count = torch.as_tensor(
            values.numel(), dtype=torch.float64, device=values.device
        )
        batch_mean = values.mean()
        batch_var = (values - batch_mean).square().mean()
        if self.return_count.item() == 0:
            self.return_mean.copy_(batch_mean)
            self.return_var.copy_(batch_var)
            self.return_count.copy_(batch_count)
            return
        delta = batch_mean - self.return_mean
        total_count = self.return_count + batch_count
        new_mean = self.return_mean + delta * batch_count / total_count
        current_m2 = self.return_var * self.return_count
        batch_m2 = batch_var * batch_count
        new_m2 = (
            current_m2
            + batch_m2
            + delta.square() * self.return_count * batch_count / total_count
        )
        self.return_mean.copy_(new_mean)
        self.return_var.copy_(new_m2 / total_count)
        self.return_count.copy_(total_count)
