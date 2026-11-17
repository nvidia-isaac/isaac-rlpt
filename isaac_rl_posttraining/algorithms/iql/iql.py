# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implicit Q-Learning over a trainable base model.

The learner owns token Q/V critics and updates the base model's action head with
advantage-weighted regression. It never imports a model stack: observation
tokens and the per-sample action loss come from the injected base model.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from isaac_rl_posttraining.algorithms.actor_optimizer import ActorOptimizer
from isaac_rl_posttraining.algorithms.base import BaseAlgorithm
from isaac_rl_posttraining.algorithms.iql.config import IQLConfig
from isaac_rl_posttraining.algorithms.iql.critics import TokenCriticEnsemble, iql_losses
from isaac_rl_posttraining.utils.networks import polyak_update

__all__ = ["IQL"]


class IQL(BaseAlgorithm):
    """Offline IQL: expectile V, TD Q, AWR-weighted ``actor_bc_loss``."""

    def __init__(self, config: IQLConfig) -> None:
        super().__init__(config)
        if config.policy is None:
            raise ValueError("IQLConfig.policy must be the trainable base model")
        if config.offline_source is None:
            raise ValueError("IQLConfig.offline_source must provide sample(batch_size)")
        self.policy = config.policy
        self.offline_source = config.offline_source
        actor_params = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not actor_params:
            raise ValueError("IQL needs at least one trainable parameter on the policy")
        self.actor_optimizer = ActorOptimizer(
            actor_params,
            lr=config.actor_optimizer_lr,
            weight_decay=config.actor_optimizer_weight_decay,
            warmup_steps=int(config.actor_lr_warmup_ratio * config.total_updates),
            schedule=config.actor_lr_schedule,
            total_steps=config.total_updates,
            max_grad_norm=config.actor_max_grad_norm,
        )
        self.critic: TokenCriticEnsemble | None = None
        self.critic_target: TokenCriticEnsemble | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None

    def train_step(self, rollout_step: int | None = None) -> dict[str, Any]:
        """Run ``num_updates_per_train_step`` gradient steps on offline batches; the loop logs the metrics."""
        if rollout_step is not None and rollout_step < self.config.learning_starts:
            return {}
        metrics: dict[str, Any] = {}
        for _ in range(self.config.num_updates_per_train_step):
            batch = self.offline_source.sample(self.config.batch_size)
            metrics = self._update(batch)
            self.step_counter += 1
        return metrics

    def _update(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs_cur = batch["inputs_cur"]
        inputs_next = batch["inputs_next"]
        action = batch["action"]
        # Critics sit on frozen observation features. Detach so a bf16 VLM
        # does not store a critic graph through the backbone, and cast to
        # float32 so the token heads match the action tensors.
        with torch.no_grad():
            tokens_cur, mask_cur = self.policy.feature_tokens(inputs_cur)
            tokens_next, mask_next = self.policy.feature_tokens(inputs_next)
        tokens_cur = tokens_cur.float()
        tokens_next = tokens_next.float()
        mask_cur = _as_padding_mask(mask_cur, tokens_cur)
        mask_next = _as_padding_mask(mask_next, tokens_next)
        action = action.to(device=tokens_cur.device, dtype=torch.float32)
        reward = _as_horizon(batch["reward"], action).to(device=tokens_cur.device)
        done = _as_horizon(batch["done"], action).to(device=tokens_cur.device)
        self._ensure_critics(tokens_cur, action)
        losses = iql_losses(
            self.critic,
            tokens_cur,
            tokens_next,
            action,
            reward,
            done,
            gamma=self.config.gamma,
            expectile=self.config.expectile,
            advantage_temp=self.config.advantage_temp,
            clip_score=self.config.clip_score,
            q_reduce=self.config.q_reduce,
            bootstrap_horizon=self.config.bootstrap_horizon,
            critic_target=self.critic_target,
            n_q_to_sample=self.config.n_q_to_sample,
            key_padding_mask=mask_cur,
            next_key_padding_mask=mask_next,
        )
        actor_inputs = dict(inputs_cur)
        actor_inputs["action"] = action
        self.policy.train()
        try:
            bc_loss = self.policy.actor_bc_loss(actor_inputs)
        finally:
            self.policy.eval()
        actor_loss = (losses["awr_weights"].reshape(-1) * bc_loss).mean()

        # A non-finite critic loss would enter the Adam moments and the target for good.
        q_loss, v_loss = losses["q_loss"], losses["v_loss"]
        if bool(torch.isfinite(q_loss).item() and torch.isfinite(v_loss).item()):
            self.critic_optimizer.zero_grad(set_to_none=True)
            (q_loss + v_loss).backward()
            self.critic_optimizer.step()
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), tau=self.config.tau)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        step = self.actor_optimizer.step()
        return {
            "actor_grad_norm": step["grad_norm"],
            "actor_lr": step["lr"],
            "q_loss": float(losses["q_loss"].detach()),
            "v_loss": float(losses["v_loss"].detach()),
            "actor_loss": float(actor_loss.detach()),
            "bc_loss": float(bc_loss.mean().detach()),
            "awr_weight_mean": float(losses["awr_weights"].mean().detach()),
            "q_mean": float(losses["q_mean"]),
            "v_mean": float(losses["v_mean"]),
            "adv_mean": float(losses["adv_mean"]),
        }

    def _ensure_critics(self, tokens: torch.Tensor, action: torch.Tensor) -> None:
        if self.critic is not None:
            return
        if tokens.ndim != 3:
            raise ValueError(f"feature_tokens must return [B, T, D], got {tuple(tokens.shape)}")
        if action.ndim != 3:
            raise ValueError(f"action must have shape [B, H, A], got {tuple(action.shape)}")
        feat_dim = int(tokens.shape[-1])
        n_heads = _compatible_heads(feat_dim, int(self.config.critic_n_heads))
        critic = TokenCriticEnsemble(
            feat_dim,
            int(action.shape[-1]),
            int(action.shape[1]),
            num_q=self.config.num_q,
            n_heads=n_heads,
            hidden=self.config.critic_hidden,
        ).to(device=tokens.device, dtype=torch.float32)
        target = copy.deepcopy(critic)
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        self.critic = critic
        self.critic_target = target
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=self.config.critic_optimizer_lr,
            weight_decay=self.config.critic_optimizer_weight_decay,
        )


def _compatible_heads(feat_dim: int, requested: int) -> int:
    """Largest attention-head count at most ``requested`` that divides ``feat_dim``."""
    if requested < 1:
        raise ValueError("critic_n_heads must be a positive integer")
    for heads in range(min(requested, feat_dim), 0, -1):
        if feat_dim % heads == 0:
            return heads
    return 1


def _as_padding_mask(mask: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor | None:
    """Boolean key-padding mask on the token device, or ``None`` when every token is valid."""
    if mask is None:
        return None
    return mask.to(device=tokens.device, dtype=torch.bool)


def _as_horizon(value: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[B]`` reward/done onto the action chunk, placing the scalar on step 0."""
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.float()
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim != 1:
        raise ValueError(f"reward/done must be [B] or [B, H], got {tuple(tensor.shape)}")
    out = torch.zeros(action.shape[0], action.shape[1], dtype=torch.float32, device=action.device)
    out[:, 0] = tensor.to(device=action.device)
    return out
