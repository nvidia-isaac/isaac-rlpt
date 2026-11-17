# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actor-update diagnostics for the SAC core: can the critic move the actor?

Recorded only when ``SACConfig.diagnostics_interval`` is set. Everything here
reads the graph of one actor update without touching it: the gradient of the
actor's Q objective with respect to its sampled action, the actor-parameter
gradient norms of the Q and entropy terms taken separately, and the critic's
value of the actor's action, of the replayed action, of a latent from the
decoder's N(0, 1) prior and of a uniform latent on the same observations,
plus the spread of Q across latents at a fixed observation against its spread
across observations at a fixed latent. All actions are in the actor's
normalized coordinates; Q is the mean over the full critic ensemble.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from isaac_rl_posttraining.algorithms.sac.sac import SAC, _ActorForward
    from isaac_rl_posttraining.algorithms.sac.train_stats import TrainStepStats

__all__ = [
    "diagonal_gaussian_prior_kl",
    "gaussian_prior_diagnostics",
    "record_actor_diagnostics",
    "record_latent_diagnostics",
    "summarize_learning_diagnostics",
    "td_target_diagnostics",
]


def _norm(grads: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def record_actor_diagnostics(
    sac: SAC,
    forward: _ActorForward,
    log_prob: torch.Tensor,
    min_qf_pi: torch.Tensor,
    *,
    ent_coef: torch.Tensor,
    actor_observations: Any,
    replay_actions: torch.Tensor | None,
    stats: TrainStepStats,
) -> None:
    """Append this actor update's diagnostics to ``stats.diagnostics`` (see the module docstring)."""
    record = stats.diagnostics
    params = [p for p in sac.actor.parameters() if p.requires_grad]

    # d(sum_i Q_i)/d a_i: the per-sample gradient of the actor objective with
    # respect to the action, as its per-element RMS.
    (dq_da,) = torch.autograd.grad(
        min_qf_pi.sum(), forward.actions_pi, retain_graph=True, allow_unused=True
    )
    if dq_da is not None:
        record.setdefault("dq_da_rms", []).append(
            float(dq_da.detach().float().pow(2).mean().sqrt().item())
        )
    q_grads = torch.autograd.grad(
        (-min_qf_pi).mean(), params, retain_graph=True, allow_unused=True
    )
    record.setdefault("actor_grad_norm_q", []).append(_norm(q_grads))
    if sac.config.train_actor_with_entropy:
        ent_grads = torch.autograd.grad(
            (ent_coef.float() * log_prob).mean(),
            params,
            retain_graph=True,
            allow_unused=True,
        )
        record.setdefault("actor_grad_norm_entropy", []).append(_norm(ent_grads))

    with torch.no_grad(), sac._autocast():
        sac.critic.eval()
        try:
            actions_pi = forward.actions_pi.detach()

            def q_mean(actions: torch.Tensor) -> torch.Tensor:
                outputs = sac._critic_outputs(sac.critic, actor_observations, actions)
                return sac._critic_values_from_outputs(outputs).float().mean(dim=1)

            q_actor = q_mean(actions_pi)
            record.setdefault("q_actor", []).append(float(q_actor.mean().item()))
            if replay_actions is not None and replay_actions.shape == actions_pi.shape:
                q_replay = q_mean(replay_actions.to(actions_pi.dtype))
                record.setdefault("q_replay", []).append(float(q_replay.mean().item()))
                record.setdefault("q_actor_minus_replay", []).append(
                    float((q_actor - q_replay).mean().item())
                )
            prior = torch.tanh(torch.randn_like(actions_pi))
            record.setdefault("q_prior_latent", []).append(
                float(q_mean(prior).mean().item())
            )
            uniform = torch.rand_like(actions_pi) * 2.0 - 1.0
            record.setdefault("q_uniform_latent", []).append(
                float(q_mean(uniform).mean().item())
            )
            # Spread across latents at one observation versus across observations
            # at one latent: a critic that only tracks the state has the first near zero.
            num_obs = min(8, int(actions_pi.shape[0]))
            num_latents = 16
            obs_subset = sac._slice_batch_data(actor_observations, np.arange(num_obs))
            latent_shape = tuple(actions_pi.shape[1:])
            latents = torch.tanh(
                torch.randn(
                    (num_latents, *latent_shape),
                    device=actions_pi.device,
                    dtype=actions_pi.dtype,
                )
            )

            def q_subset(actions: torch.Tensor) -> torch.Tensor:
                outputs = sac._critic_outputs(sac.critic, obs_subset, actions)
                return sac._critic_values_from_outputs(outputs).float().mean(dim=1)

            table = torch.stack(
                [
                    q_subset(latents[j : j + 1].expand(num_obs, *latent_shape))
                    for j in range(num_latents)
                ],
                dim=1,
            )  # [num_obs, num_latents]
            record.setdefault("q_std_over_latents", []).append(
                float(table.std(dim=1).mean().item())
            )
            record.setdefault("q_std_over_obs", []).append(
                float(table.std(dim=0).mean().item())
            )
        finally:
            sac.critic.train()


def record_latent_diagnostics(sac: "SAC", forward: "_ActorForward", stats: "TrainStepStats") -> None:
    """Track whether the latent the decoder receives stays on its N(0,1) scale.

    A bounded actor samples in normalized ``[-1, 1]`` coordinates; the
    statistics are taken on the executed latent (mapped onto the action
    box), so a wider ``noise_action_bound`` reads as a wider latent and
    saturation at the bound is visible.
    """
    with torch.no_grad():
        w = forward.actions_pi.detach().float()
        if getattr(sac.actor, "normalize_actions", False):
            # Saturation is relative to the normalized bound, regardless of its executed scale.
            stats.w_satur.append((w.abs() > 0.95).float().mean().item())
            w = sac.actor.unnormalize_action(w).float()
        stats.w_means.append(w.mean().item())
        stats.w_stds.append(w.std().item())
        stats.w_absmax.append(w.abs().max().item())
        stats.mu_absmax.append(forward.mean_actions.detach().abs().max().item())
        stats.sig_means.append(torch.exp(forward.log_std.detach()).mean().item())


def diagonal_gaussian_prior_kl(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    prior_std: float,
) -> torch.Tensor:
    """Per-dimension ``KL(N(mean, std) || N(0, prior_std²))``."""
    if not math.isfinite(prior_std) or prior_std <= 0.0:
        raise ValueError(f"prior_std must be finite and positive, got {prior_std}")
    prior_variance = prior_std**2
    return 0.5 * (
        (torch.exp(2.0 * log_std) + mean.pow(2)) / prior_variance
        - 1.0
        + 2.0 * math.log(prior_std)
        - 2.0 * log_std
    )


@torch.no_grad()
def td_target_diagnostics(
    targets: torch.Tensor, terminals: torch.Tensor
) -> dict[str, float]:
    """Summarize actual regression targets, retaining counts for empty subsets."""
    targets = targets.detach().float().reshape(-1)
    terminals = terminals.detach().bool().reshape(-1)
    metrics: dict[str, float] = {}
    for name, values in (
        ("td_target", targets),
        ("td_target_terminal", targets[terminals]),
        ("td_target_nonterminal", targets[~terminals]),
    ):
        metrics[f"{name}_count"] = float(values.numel())
        if values.numel() == 0:
            continue
        metrics[f"{name}_mean"] = values.mean().item()
        metrics[f"{name}_max"] = values.max().item()
        metrics[f"{name}_frac_above_one"] = (values > 1.0).float().mean().item()
    return metrics


def summarize_learning_diagnostics(stats: TrainStepStats) -> dict[str, float]:
    """Aggregate regression targets and prior diagnostics across learner updates."""
    metrics: dict[str, float] = {}
    if stats.td_targets:
        metrics.update(
            td_target_diagnostics(
                torch.cat(stats.td_targets), torch.cat(stats.td_terminals)
            )
        )
    for name, values in stats.prior_diagnostics.items():
        metrics[name] = float(np.mean(values))
    return metrics


@torch.no_grad()
def gaussian_prior_diagnostics(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    prior_std: float,
) -> dict[str, float]:
    """Decompose pre-tanh KL in float32 without changing the actor objective."""
    mean = mean.detach().float()
    log_std = log_std.detach().float()
    mean_component = 0.5 * mean.square() / prior_std**2
    variance_component = 0.5 * (
        torch.exp(2.0 * log_std) / prior_std**2
        - 1.0
        + 2.0 * math.log(prior_std)
        - 2.0 * log_std
    )
    return {
        "prior_kl_mean_component": mean_component.flatten(1).sum(1).mean().item(),
        "prior_kl_variance_component": variance_component.flatten(1)
        .sum(1)
        .mean()
        .item(),
        "prior_kl_per_dim": (mean_component + variance_component).mean().item(),
    }
