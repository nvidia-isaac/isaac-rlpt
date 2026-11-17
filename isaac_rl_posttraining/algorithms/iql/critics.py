# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-attention critic heads and Implicit Q-Learning losses.

A policy exposes per-step feature tokens (backbone plus state embeddings)
and a per-sample action loss; this module adds the RL half:

* Token-based V and Q heads: a learnable query token attends over the
  feature tokens (plus an action token for Q) and is read out to a scalar,
  instead of mean-pooling the token sequence.
* Implicit Q-Learning losses: expectile V regression, TD Q with a
  discounted-reward target and a V(s') bootstrap, and advantage-weighted
  regression (AWR) weights for the actor.

The functions are model-agnostic: any policy that yields feature tokens and
a per-sample action loss can train with them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TokenValueHead",
    "TokenCriticEnsemble",
    "reduce_q",
    "discounted_reward_sum",
    "expectile_loss",
    "iql_losses",
]


class TokenValueHead(nn.Module):
    """Scalar value head that attends over feature tokens with a query token.

    A learnable query token cross-attends over the feature tokens (plus an
    optional projected action token) and is decoded to a scalar. Passing
    ``action_dim`` makes this a Q head; leaving it ``None`` makes a V head.
    """

    def __init__(
        self,
        feat_dim: int,
        n_heads: int = 8,
        hidden: int = 1024,
        action_dim: int | None = None,
        action_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.is_q = action_dim is not None
        self.action_proj = (
            nn.Linear(action_dim * action_horizon, feat_dim) if self.is_q else None
        )
        self.out = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        feats: torch.Tensor,
        action: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of token sequences.

        Args:
            feats: Feature tokens, shape ``[B, T, D]``.
            action: Action chunk, shape ``[B, H, A]``; required for Q heads.
            key_padding_mask: Optional ``[B, T]`` mask; True marks tokens to ignore.

        Returns:
            Values of shape ``[B, 1]``.
        """
        batch = feats.shape[0]
        context = feats
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=feats.device, dtype=torch.bool)
        if self.is_q:
            assert action is not None, "Q head requires an action"
            action_token = self.action_proj(action.reshape(batch, -1)).unsqueeze(1)  # [B, 1, D]
            context = torch.cat([feats, action_token], dim=1)
            if key_padding_mask is not None:
                # The action token is real. True marks padding.
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(batch, 1)], dim=1
                )
        query = self.token.expand(batch, -1, -1)
        pooled, _ = self.attn(query, context, context, key_padding_mask=key_padding_mask)
        return self.out(pooled.squeeze(1))  # [B, 1]


class TokenCriticEnsemble(nn.Module):
    """Ensemble of token Q heads plus a single token V head."""

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        action_horizon: int,
        num_q: int = 2,
        n_heads: int = 8,
        hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.q_heads = nn.ModuleList(
            [
                TokenValueHead(feat_dim, n_heads, hidden, action_dim, action_horizon)
                for _ in range(num_q)
            ]
        )
        self.v_head = TokenValueHead(feat_dim, n_heads, hidden, action_dim=None)

    def q_values(
        self,
        feats: torch.Tensor,
        action: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return stacked Q estimates of shape ``[num_q, B, 1]``."""
        return torch.stack(
            [head(feats, action, key_padding_mask=key_padding_mask) for head in self.q_heads], dim=0
        )

    def v_value(
        self, feats: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return V estimates of shape ``[B, 1]``."""
        return self.v_head(feats, key_padding_mask=key_padding_mask)


def reduce_q(q: torch.Tensor, mode: str = "min", indices: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce an ensemble of Q values ``[num_q, B, 1]`` to ``[B, 1]``.

    Args:
        q: Stacked Q estimates.
        mode: ``"min"`` (clipped double-Q), ``"mean"``, or ``"max"``.
        indices: Optional subset of ensemble members ``[n_sample]`` to reduce
            over (REDQ-style random 2-of-N). All heads still exist on ``q``.

    Returns:
        The reduced Q values.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    if indices is not None:
        q = q[indices]
    if mode == "min":
        return q.min(dim=0)[0]
    if mode == "mean":
        return q.mean(dim=0)
    if mode == "max":
        return q.max(dim=0)[0]
    raise ValueError(mode)


def discounted_reward_sum(
    reward: torch.Tensor, gamma: float, done: torch.Tensor | None = None
) -> torch.Tensor:
    """Discount per-step chunk rewards into a single return per sample.

    Rewards at and after the first ``done`` inside the chunk are zeroed, so
    nothing is accumulated across an episode boundary.

    Args:
        reward: Per-step rewards over the chunk, shape ``[B, H]``.
        gamma: Discount factor.
        done: Optional per-step done flags, shape ``[B, H]``.

    Returns:
        Discounted sums of shape ``[B, 1]``.
    """
    batch, horizon = reward.shape
    discounts = (gamma ** torch.arange(horizon, device=reward.device)).view(1, horizon)
    if done is not None:
        not_done = (done == 0).float()
        keep = torch.cumprod(
            torch.cat([torch.ones(batch, 1, device=reward.device), not_done[:, :-1]], dim=1),
            dim=1,
        )
        reward = reward * keep
    return (reward * discounts).sum(dim=1, keepdim=True)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric L2 used by IQL's expectile regression.

    Args:
        diff: ``target - value`` residuals.
        expectile: Expectile in ``(0, 1)``; values above 0.5 weight positive
            residuals more.

    Returns:
        The scalar loss.
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_losses(
    critic: TokenCriticEnsemble,
    feats_cur: torch.Tensor,
    feats_next: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.99,
    expectile: float = 0.7,
    advantage_temp: float = 1.0,
    clip_score: float = 20.0,
    q_reduce: str = "min",
    bootstrap_horizon: int = 1,
    critic_target: TokenCriticEnsemble | None = None,
    n_q_to_sample: int | None = None,
    key_padding_mask: torch.Tensor | None = None,
    next_key_padding_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the Implicit Q-Learning losses on one batch.

    Args:
        critic: The trainable critic ensemble.
        feats_cur: Feature tokens for the current observations, ``[B, T, D]``.
        feats_next: Feature tokens for the next observations, ``[B, T, D]``.
        action: Normalized action chunks, ``[B, H, A]``.
        reward: Per-step rewards over the chunk, ``[B, H]``.
        done: Per-step done flags over the chunk, ``[B, H]``.
        gamma: Discount factor.
        expectile: Expectile for the V regression.
        advantage_temp: Temperature multiplying advantages inside the AWR
            exponent.
        clip_score: Upper bound on the AWR weights.
        q_reduce: Ensemble reduction mode for Q targets (see :func:`reduce_q`).
        bootstrap_horizon: Exponent on ``gamma`` for the V(s') bootstrap.
        critic_target: Optional target-network ensemble; falls back to
            ``critic`` when absent.
        n_q_to_sample: If set and smaller than the ensemble, V/AWR targets
            use a random subset of this many heads (all heads still regress).
        key_padding_mask: Optional ``[B, T]`` mask for ``feats_cur`` (True = ignore).
        next_key_padding_mask: Optional ``[B, T]`` mask for ``feats_next``.

    Returns:
        ``{"q_loss", "v_loss", "awr_weights", "advantage"}`` plus detached
        ``q_mean`` / ``v_mean`` / ``adv_mean`` / ``reward_mean`` diagnostics.
        ``awr_weights`` and ``advantage`` have shape ``[B, 1]``.
    """
    target = critic_target if critic_target is not None else critic

    # V regresses toward the (target) Q of the dataset action, expectile-weighted.
    # The TD/AWR target may use a random subset of the ensemble (REDQ-style);
    # every head still trains on the Q loss.
    with torch.no_grad():
        q_sa_all = target.q_values(feats_cur, action, key_padding_mask=key_padding_mask)  # [num_q, B, 1]
        num_q = int(q_sa_all.shape[0])
        q_indices = None
        if n_q_to_sample is not None and int(n_q_to_sample) < num_q:
            q_indices = torch.randperm(num_q, device=q_sa_all.device)[: int(n_q_to_sample)]
        q_sa_target = reduce_q(q_sa_all, q_reduce, indices=q_indices)  # [B, 1]
    v_s = critic.v_value(feats_cur, key_padding_mask=key_padding_mask)  # [B, 1]
    v_loss = expectile_loss(q_sa_target - v_s, expectile)

    # Q regresses toward discounted chunk reward plus the V(s') bootstrap.
    with torch.no_grad():
        v_next = critic.v_value(feats_next, key_padding_mask=next_key_padding_mask)  # [B, 1]
        chunk_return = discounted_reward_sum(reward, gamma, done)  # [B, 1]
        not_done = (done.sum(dim=1, keepdim=True) == 0).float()
        q_target = chunk_return + (gamma**bootstrap_horizon) * not_done * v_next
    q_sa = critic.q_values(feats_cur, action, key_padding_mask=key_padding_mask)  # [num_q, B, 1]
    q_loss = sum(F.mse_loss(q_sa[i], q_target) for i in range(q_sa.shape[0])) / q_sa.shape[0]

    # AWR weights for the actor: exponentiated, clipped advantages.
    with torch.no_grad():
        advantage = q_sa_target - v_s
        awr_weights = torch.clamp(torch.exp(advantage * advantage_temp), 0.0, clip_score)

    return {
        "q_loss": q_loss,
        "v_loss": v_loss,
        "awr_weights": awr_weights,
        "advantage": advantage,
        "q_mean": q_sa.mean().detach(),
        "v_mean": v_s.mean().detach(),
        "adv_mean": advantage.mean().detach(),
        "reward_mean": reward.mean().detach(),
    }
