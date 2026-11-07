# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-update statistics of one SAC ``train_step`` and the latent metrics derived from them."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["TrainStepStats", "latent_noise_metrics"]


@dataclass
class TrainStepStats:
    """Per-update diagnostics accumulated over the gradient steps of one ``train_step``."""

    # Critic-step values stay on the device (0-d tensors) until the metrics are assembled.
    critic_losses: list[torch.Tensor] = field(default_factory=list)
    q_values: list[torch.Tensor] = field(default_factory=list)
    q_next_values: list[torch.Tensor] = field(default_factory=list)
    reward_values: list[torch.Tensor] = field(default_factory=list)
    ent_coef_losses: list[float] = field(default_factory=list)
    ent_coefs: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    actor_log_pis: list[float] = field(default_factory=list)
    min_qf_pis: list[float] = field(default_factory=list)
    actor_q_loss_active_values: list[float] = field(default_factory=list)
    actor_q_loss_coef_values: list[float] = field(default_factory=list)
    actor_regularization_metrics: dict[str, float] = field(default_factory=dict)
    # KL(actor || frozen snapshot), when anchoring is on
    kl_divs: list[float] = field(default_factory=list)
    # KL(actor || fixed BC-handoff actor)
    behavior_kl_divs: list[float] = field(default_factory=list)
    # KL(actor || N(0,1)) diagnostic
    prior_kls: list[float] = field(default_factory=list)
    prior_diagnostics: dict[str, list[float]] = field(default_factory=dict)
    td_targets: list[torch.Tensor] = field(default_factory=list)
    td_terminals: list[torch.Tensor] = field(default_factory=list)
    target_network_updates: int = 0
    expert_bc_losses: list[float] = field(default_factory=list)
    expert_bc_diagnostics: dict[str, list[float]] = field(default_factory=dict)
    online_success_bc_losses: list[float] = field(default_factory=list)
    online_success_bc_diagnostics: dict[str, list[float]] = field(default_factory=dict)
    progress_rewards: list[float] = field(default_factory=list)
    success_probs: list[float] = field(default_factory=list)
    # Latent-distribution diagnostics: is the actor's output still on the
    # N(0,1) manifold the decoder was trained on, or drifting/saturating?
    w_means: list[float] = field(default_factory=list)
    w_stds: list[float] = field(default_factory=list)
    w_absmax: list[float] = field(default_factory=list)
    w_satur: list[float] = field(default_factory=list)
    mu_absmax: list[float] = field(default_factory=list)
    sig_means: list[float] = field(default_factory=list)
    # diag/* metrics, recorded every config.diagnostics_interval updates
    diagnostics: dict[str, list[float]] = field(default_factory=dict)


def latent_noise_metrics(stats: TrainStepStats) -> dict[str, float]:
    """Statistics of the executed latents (the decoder's input): ~0 / ~1 under the N(0,1) prior."""
    metrics: dict[str, float] = {
        "noise/w_mean": float(np.mean(stats.w_means)),
        "noise/w_std": float(np.mean(stats.w_stds)),
        # Reaches the bound when the actor pins
        "noise/w_absmax": float(np.max(stats.w_absmax)),
        # Large pre-tanh mean => saturation
        "noise/mu_absmax": float(np.max(stats.mu_absmax)),
        # Collapse => deterministic latents
        "noise/sigma_mean": float(np.mean(stats.sig_means)),
    }
    if stats.w_satur:
        metrics["noise/w_frac_saturated"] = float(np.mean(stats.w_satur))
    return metrics
