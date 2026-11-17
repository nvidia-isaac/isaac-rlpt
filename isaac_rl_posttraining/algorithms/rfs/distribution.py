# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed stochastic-latent and deterministic-residual action distribution."""

from __future__ import annotations

import math

import torch
from torch import nn

from isaac_rl_posttraining.modules.base.distributions import SquashedDiagGaussianDistribution


__all__ = [
    "LatentStochasticResidualDeterministicDistribution",
    "configure_rfs_actor_distribution",
    "enable_proposal_conditioned_residual",
]


class LatentStochasticResidualDeterministicDistribution(SquashedDiagGaussianDistribution):
    """Sample flow latents while taking the residual mean deterministically.

    SAC entropy and temperature tuning see only the latent dimensions.
    Residual actions remain differentiable through ``tanh(mean)`` so the
    critic can train precise corrections without injecting residual noise.
    This distribution returns normalized residuals in ``[-1, 1]``; the
    actor's action-space transform applies the physical residual bound once.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int,
        residual_action_bound: float,
        residual_exploration_std: float = 0.0,
        residual_proposal_conditioner: nn.Module | None = None,
    ) -> None:
        if not 0 < latent_dim < action_dim:
            raise ValueError(f"latent_dim must be in (0, {action_dim}), got {latent_dim}")
        if residual_action_bound <= 0:
            raise ValueError(f"residual_action_bound must be positive, got {residual_action_bound}")
        if not math.isfinite(residual_exploration_std) or residual_exploration_std < 0:
            raise ValueError(
                "residual_exploration_std must be finite and non-negative, "
                f"got {residual_exploration_std}"
            )
        super().__init__(action_dim=action_dim)
        self.latent_dim = int(latent_dim)
        self.residual_action_bound = float(residual_action_bound)
        self.residual_exploration_std = float(residual_exploration_std)
        self.residual_exploration_enabled = True
        self.residual_proposal_conditioner = residual_proposal_conditioner
        self.latent_distribution = SquashedDiagGaussianDistribution(action_dim=self.latent_dim)
        self._last_normalized_residual: torch.Tensor | None = None
        self._last_latent_mean: torch.Tensor | None = None
        self._last_latent_log_std: torch.Tensor | None = None
        self._last_proposal_conditioning: torch.Tensor | None = None

    def set_residual_exploration_enabled(self, enabled: bool) -> None:
        """Enable collection noise or disable it for clean policy evaluation."""
        self.residual_exploration_enabled = bool(enabled)

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> "LatentStochasticResidualDeterministicDistribution":
        """Parameterize the full diagnostic and latent sampling distributions."""
        super().proba_distribution(mean_actions, log_std)
        latent_mean = mean_actions[..., : self.latent_dim]
        latent_log_std = log_std[..., : self.latent_dim]
        self._last_latent_mean = latent_mean
        self._last_latent_log_std = latent_log_std
        self.latent_distribution.proba_distribution(latent_mean, latent_log_std)
        return self

    def _condition_residual(
        self,
        residual_pre_tanh: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_proposal_conditioner is None:
            return residual_pre_tanh
        conditioning = self.residual_proposal_conditioner(latent_actions)
        self._last_proposal_conditioning = conditioning
        return residual_pre_tanh + conditioning

    def residual_from_params(
        self,
        mean_actions: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic residual for specific flow proposals."""
        residual_pre_tanh = self._condition_residual(
            mean_actions[..., self.latent_dim :],
            latent_actions,
        )
        return torch.tanh(residual_pre_tanh)

    def _residual_mode(
        self,
        latent_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distribution is None:
            raise RuntimeError("RFS distribution must be parameterized before sampling")
        residual_gaussian = self._condition_residual(
            self.distribution.mean[..., self.latent_dim :],
            latent_actions,
        )
        normalized_residual = torch.tanh(residual_gaussian)
        self._last_normalized_residual = normalized_residual
        return normalized_residual, residual_gaussian

    def sample(self) -> torch.Tensor:
        """Sample the latent branch and optionally perturb the residual."""
        latent = self.latent_distribution.sample()
        residual_mean = self.distribution.mean[..., self.latent_dim :]
        residual_gaussian = self._condition_residual(residual_mean, latent)
        if self.residual_exploration_enabled and self.residual_exploration_std > 0.0:
            residual_gaussian = residual_gaussian + (
                torch.randn_like(residual_gaussian) * self.residual_exploration_std
            )
        normalized_residual = torch.tanh(residual_gaussian)
        self._last_normalized_residual = normalized_residual
        self.gaussian_actions = torch.cat(
            [
                self.latent_distribution.gaussian_actions,
                residual_gaussian,
            ],
            dim=-1,
        )
        return torch.cat([latent, normalized_residual], dim=-1)

    def mode(self) -> torch.Tensor:
        """Return deterministic latent and residual modes."""
        latent = self.latent_distribution.mode()
        residual, residual_gaussian = self._residual_mode(latent)
        self.gaussian_actions = torch.cat(
            [
                self.latent_distribution.gaussian_actions,
                residual_gaussian,
            ],
            dim=-1,
        )
        return torch.cat([latent, residual], dim=-1)

    def log_prob(
        self,
        actions: torch.Tensor,
        gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the stochastic latent log probability."""
        latent_gaussian = None
        if gaussian_actions is not None:
            latent_gaussian = gaussian_actions[..., : self.latent_dim]
        return self.latent_distribution.log_prob(
            actions[..., : self.latent_dim],
            latent_gaussian,
        )

    def actions_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        return actions, self.log_prob(actions, self.gaussian_actions)

    def diagnostics(self) -> dict[str, float]:
        """Report normalized and raw residual magnitude from the latest action."""
        if self._last_normalized_residual is None:
            return {}
        residual = self._last_normalized_residual.detach().float()
        raw = residual * self.residual_action_bound
        diagnostics = {
            "rfs/residual_exploration_std": self.residual_exploration_std,
            "rfs/residual_exploration_enabled": float(self.residual_exploration_enabled),
            "rfs/residual_normalized_abs_mean": residual.abs().mean().item(),
            "rfs/residual_normalized_abs_max": residual.abs().max().item(),
            "rfs/residual_raw_abs_mean": raw.abs().mean().item(),
            "rfs/residual_raw_rms": raw.square().mean().sqrt().item(),
            "rfs/residual_raw_abs_max": raw.abs().max().item(),
        }
        if self._last_latent_mean is not None and self._last_latent_log_std is not None:
            latent_mean = self._last_latent_mean.detach().float()
            latent_std = self._last_latent_log_std.detach().float().exp()
            diagnostics.update(
                {
                    "rfs/latent_mean_abs_mean": latent_mean.abs().mean().item(),
                    "rfs/latent_mean_rms": latent_mean.square().mean().sqrt().item(),
                    "rfs/latent_mean_abs_max": latent_mean.abs().max().item(),
                    "rfs/latent_std_mean": latent_std.mean().item(),
                }
            )
        if self._last_proposal_conditioning is not None:
            conditioning = self._last_proposal_conditioning.detach().float()
            diagnostics.update(
                {
                    "rfs/proposal_conditioning_abs_mean": conditioning.abs().mean().item(),
                    "rfs/proposal_conditioning_abs_max": conditioning.abs().max().item(),
                }
            )
        return diagnostics


def configure_rfs_actor_distribution(
    actor: torch.nn.Module,
    *,
    latent_dim: int,
    residual_action_bound: float,
    residual_exploration_std: float = 0.0,
    proposal_conditioned_residual: bool = False,
) -> None:
    """Install latent-only entropy and deterministic residual sampling."""
    if hasattr(actor, "action_token_dim"):
        action_dim = int(actor.action_token_dim)
    elif hasattr(actor, "mean_processor"):
        action_dim = int(actor.mean_processor[-1].out_features)
    else:
        raise TypeError(
            "RFS actor must expose action_token_dim or a transformer mean_processor, "
            f"got {actor.__class__.__name__}"
        )
    actor.action_dist = LatentStochasticResidualDeterministicDistribution(
        action_dim=action_dim,
        latent_dim=latent_dim,
        residual_action_bound=residual_action_bound,
        residual_exploration_std=residual_exploration_std,
    )
    if proposal_conditioned_residual:
        enable_proposal_conditioned_residual(actor)


def enable_proposal_conditioned_residual(actor: nn.Module) -> nn.Linear:
    """Attach a zero-initialized flow-proposal conditioner to an RFS actor."""
    action_dist = getattr(actor, "action_dist", None)
    latent_dim = getattr(action_dist, "latent_dim", None)
    if hasattr(actor, "action_token_dim"):
        action_dim = int(actor.action_token_dim)
    elif hasattr(actor, "mean_processor"):
        action_dim = int(actor.mean_processor[-1].out_features)
    else:
        raise TypeError(
            "RFS actor must expose action_token_dim or a transformer mean_processor, "
            f"got {actor.__class__.__name__}"
        )
    if latent_dim is None:
        raise TypeError("Configure the RFS action distribution before enabling conditioning")
    residual_dim = action_dim - int(latent_dim)
    conditioner = getattr(actor, "residual_proposal_conditioner", None)
    if conditioner is None:
        conditioner = nn.Linear(int(latent_dim), residual_dim)
        conditioner = conditioner.to(next(actor.parameters()).device)
        nn.init.zeros_(conditioner.weight)
        nn.init.zeros_(conditioner.bias)
        actor.add_module("residual_proposal_conditioner", conditioner)
    if not isinstance(conditioner, nn.Linear):
        raise TypeError("actor.residual_proposal_conditioner must be an nn.Linear")
    action_dist.residual_proposal_conditioner = conditioner
    return conditioner
