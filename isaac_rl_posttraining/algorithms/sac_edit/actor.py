# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAC-edit direct action-chunk actor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from isaac_rl_posttraining.modules.base.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from isaac_rl_posttraining.modules.mlp.actor import MLPActor, MLPActorConfig


__all__ = [
    "ReferenceEditActor",
    "ReferenceEditActorConfig",
    "ReferenceCenteredSquashedGaussianDistribution",
    "ReferenceEditDistribution",
    "SimNorm",
]


_LOG_2 = math.log(2.0)


def _require_finite(**tensors: torch.Tensor) -> None:
    """Fail at the source when a distribution input is NaN/inf (one device sync)."""
    finite = torch.stack([torch.isfinite(t).all() for t in tensors.values()]).tolist()
    bad = [name for name, ok in zip(tensors, finite) if not ok]
    if bad:
        detail = ", ".join(f"{n}{tuple(tensors[n].shape)} {tensors[n].dtype}" for n in bad)
        raise FloatingPointError(
            f"non-finite actor distribution input: {detail}. Check the reference action "
            "and observation features, and the update precision (algorithm.use_amp)."
        )


def _validated_reference(
    reference_action: torch.Tensor | None,
    mean_actions: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Check the reference action against the parameters and move it onto their device/dtype."""
    if reference_action is None:
        raise ValueError("reference-edit distributions require a reference_action")
    if reference_action.shape != mean_actions.shape:
        raise ValueError(
            "reference action shape must match edit distribution parameters, got "
            f"{tuple(reference_action.shape)} and {tuple(mean_actions.shape)}"
        )
    _require_finite(mean_actions=mean_actions, log_std=log_std, reference_action=reference_action)
    return reference_action.to(
        device=mean_actions.device,
        dtype=mean_actions.dtype,
    )


class ReferenceEditDistribution(SquashedDiagGaussianDistribution):
    """Gaussian over bounded edits around a per-observation reference action.

    The stochastic variable lives in edit space: ``edit_scale * tanh(u)`` for
    ``u ~ Normal(mean, std)``. The final environment command is the reference
    plus that edit, clipped to the actor's normalized ``[-1, 1]`` bounds.

    The density (and therefore the entropy) is taken over the normalized edit
    ``tanh(u)`` in ``[-1, 1]``, the same convention as a squashed Gaussian over
    normalized actions. Measuring it in action units would add the constant
    ``-D * log(edit_scale)`` (about +2200 nats for a 16x35 chunk at 0.02): that
    constant cancels in the policy gradient but makes any finite entropy target
    unreachable, so the auto-tuned temperature grows without bound.
    """

    def __init__(self, action_dim: int, edit_scale: float) -> None:
        super().__init__(action_dim=action_dim)
        if edit_scale <= 0.0:
            raise ValueError("edit_scale must be positive")
        self.edit_scale = float(edit_scale)
        self.reference_action: torch.Tensor | None = None

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> ReferenceEditDistribution:
        self.reference_action = _validated_reference(reference_action, mean_actions, log_std)
        DiagGaussianDistribution.proba_distribution(self, mean_actions, log_std)
        return self

    def _require_reference(self) -> torch.Tensor:
        if self.reference_action is None:
            raise RuntimeError("parameterize the reference-edit distribution before sampling")
        return self.reference_action

    def _final_action(self, gaussian_edit: torch.Tensor) -> torch.Tensor:
        reference = self._require_reference()
        action = reference + self.edit_scale * torch.tanh(gaussian_edit)
        return action.clamp(-1.0, 1.0)

    def sample(self) -> torch.Tensor:
        self.gaussian_actions = self.distribution.rsample()
        return self._final_action(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = self.distribution.mean
        return self._final_action(self.gaussian_actions)

    def log_prob(
        self,
        actions: torch.Tensor,
        gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outside_support: torch.Tensor | None = None
        if gaussian_actions is None:
            reference = self._require_reference()
            raw_normalized_edit = (actions - reference) / self.edit_scale
            outside_support = raw_normalized_edit.abs().ge(1.0).any(dim=-1)
            normalized_edit = raw_normalized_edit.clamp(
                -1.0 + self.epsilon,
                1.0 - self.epsilon,
            )
            gaussian_actions = torch.atanh(normalized_edit)
        log_prob = DiagGaussianDistribution.log_prob(self, gaussian_actions)
        # log(1 - tanh(u)^2), the squash Jacobian; no edit_scale term (see class docstring)
        log_abs_det = 2.0 * (
            _LOG_2 - gaussian_actions - torch.nn.functional.softplus(-2.0 * gaussian_actions)
        )
        log_prob = log_prob - log_abs_det.sum(dim=-1)
        if outside_support is not None:
            log_prob = log_prob.masked_fill(outside_support, -torch.inf)
        return log_prob

    def project_actions_to_support(
        self,
        actions: torch.Tensor,
        *,
        reference_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replay targets into the bounded edit actor's feasible interval."""
        reference = reference_action.to(device=actions.device, dtype=actions.dtype)
        if reference.shape != actions.shape:
            raise ValueError(
                "reference action shape must match replay actions for projection, got "
                f"{tuple(reference.shape)} and {tuple(actions.shape)}"
            )
        lower = torch.clamp(reference - self.edit_scale, min=-1.0)
        upper = torch.clamp(reference + self.edit_scale, max=1.0)
        projected = torch.maximum(torch.minimum(actions, upper), lower)
        clipped = projected.ne(actions)
        return projected, clipped

    def actions_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.proba_distribution(
            mean_actions,
            log_std,
            reference_action=reference_action,
        )
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(
            mean_actions,
            log_std,
            reference_action=reference_action,
        )
        return action, self.log_prob(action, self.gaussian_actions)


class ReferenceCenteredSquashedGaussianDistribution(DiagGaussianDistribution):
    """Full-support squashed Gaussian whose zero edit is the reference action.

    The stochastic variable is an additive edit in pre-tanh coordinates:
    ``action = tanh(atanh(reference) + edit)``. Unlike a clipped additive edit,
    this remains a proper density over the entire open action interval while
    making a zero-initialized actor reproduce the frozen policy exactly.
    """

    def __init__(self, action_dim: int, epsilon: float = 1.0e-6) -> None:
        super().__init__(action_dim=action_dim)
        self.epsilon = float(epsilon)
        self.gaussian_actions: torch.Tensor | None = None
        self.reference_action: torch.Tensor | None = None

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> ReferenceCenteredSquashedGaussianDistribution:
        self.reference_action = _validated_reference(reference_action, mean_actions, log_std)
        return super().proba_distribution(mean_actions, log_std)

    def _reference_logits(self) -> torch.Tensor:
        if self.reference_action is None:
            raise RuntimeError("parameterize the reference-centered distribution before sampling")
        return torch.atanh(self.reference_action.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon))

    def sample(self) -> torch.Tensor:
        self.gaussian_actions = self.distribution.rsample()
        return torch.tanh(self._reference_logits() + self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = self.distribution.mean
        return torch.tanh(self._reference_logits() + self.gaussian_actions)

    def log_prob(
        self,
        actions: torch.Tensor,
        gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gaussian_actions is None:
            squashed_logits = torch.atanh(actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon))
            gaussian_actions = squashed_logits - self._reference_logits()
        else:
            squashed_logits = self._reference_logits() + gaussian_actions
        log_prob = super().log_prob(gaussian_actions)
        log_abs_det = 2.0 * (
            _LOG_2 - squashed_logits - torch.nn.functional.softplus(-2.0 * squashed_logits)
        )
        return log_prob - log_abs_det.sum(dim=-1)

    def actions_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.proba_distribution(
            mean_actions,
            log_std,
            reference_action=reference_action,
        )
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        *,
        reference_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(
            mean_actions,
            log_std,
            reference_action=reference_action,
        )
        return action, self.log_prob(action, self.gaussian_actions)


class SimNorm(nn.Module):
    """Softmax-normalize fixed-width feature groups."""

    def __init__(self, group_dim: int = 64) -> None:
        super().__init__()
        if group_dim <= 0:
            raise ValueError("group_dim must be positive")
        self.group_dim = int(group_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % self.group_dim:
            raise ValueError(
                f"feature width {value.shape[-1]} is not divisible by {self.group_dim}"
            )
        grouped = value.reshape(*value.shape[:-1], -1, self.group_dim)
        return torch.softmax(grouped, dim=-1).reshape_as(value)


@dataclass
class ReferenceEditActorConfig(MLPActorConfig):
    """MLP actor configuration with a SimNorm action head."""

    simnorm_dim: int = 64
    simnorm_groups: int = 8
    reference_action_key: str = "reference_action"
    edit_scale: float | None = None
    reference_centered_full_support: bool = False

    @property
    def actor_class(self) -> type:
        return ReferenceEditActor


class ReferenceEditActor(MLPActor):
    """State-dependent Gaussian actor producing a full action chunk."""

    def __init__(self, config: ReferenceEditActorConfig) -> None:
        super().__init__(config)
        if config.edit_scale is not None and config.edit_scale <= 0.0:
            raise ValueError("edit_scale must be positive when set")
        if config.edit_scale is not None and config.reference_centered_full_support:
            raise ValueError(
                "edit_scale and reference_centered_full_support are mutually exclusive"
            )
        action_width = int(self.mean_layer.out_features)
        hidden_width = int(config.simnorm_dim) * int(config.simnorm_groups)
        trunk_width = int(config.hidden_dims[-1])
        self.mean_layer = nn.Sequential(
            nn.Linear(trunk_width, hidden_width),
            SimNorm(config.simnorm_dim),
            nn.Linear(hidden_width, action_width),
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(trunk_width, hidden_width),
            SimNorm(config.simnorm_dim),
            nn.Linear(hidden_width, action_width),
        )
        nn.init.zeros_(self.mean_layer[-1].weight)
        nn.init.zeros_(self.mean_layer[-1].bias)
        nn.init.zeros_(self.log_std_layer[-1].weight)
        log_std_span = float(config.log_std_max - config.log_std_min)
        if log_std_span <= 0.0:
            raise ValueError("log_std_max must be greater than log_std_min")
        normalized_log_std = (
            2.0 * (float(config.log_std_init) - config.log_std_min) / log_std_span - 1.0
        )
        normalized_log_std = max(-1.0 + 1.0e-6, min(1.0 - 1.0e-6, normalized_log_std))
        self.log_std_layer[-1].bias.data.fill_(math.atanh(normalized_log_std))
        self._action_shape = tuple(config.action_space.shape)
        if config.edit_scale is not None:
            if not config.use_tanh_output:
                raise ValueError("SAC-edit edit actor requires use_tanh_output=true")
            self.action_dist = ReferenceEditDistribution(
                action_dim=action_width,
                edit_scale=float(config.edit_scale),
            )
        elif config.reference_centered_full_support:
            if not config.use_tanh_output:
                raise ValueError("reference-centered SAC-edit requires use_tanh_output=true")
            self.action_dist = ReferenceCenteredSquashedGaussianDistribution(
                action_dim=action_width
            )

    def _hidden(self, obs: Any) -> torch.Tensor:
        """Accept both vector-batched and evaluator-style unbatched dicts."""
        if isinstance(obs, dict):
            observation_space = self.config.observation_space
            obs = dict(obs)
            for key, value in obs.items():
                space = observation_space.spaces.get(key)
                if torch.is_tensor(value) and space is not None:
                    if value.ndim == len(space.shape):
                        obs[key] = value.unsqueeze(0)
        normalizer = getattr(self, "input_normalizer", None)
        normalizer_key = getattr(self, "input_normalizer_key", None)
        if normalizer is not None and isinstance(obs, dict):
            if normalizer_key not in obs:
                raise KeyError(f"SAC-edit observation is missing {normalizer_key!r}")
            obs = dict(obs)
            obs[normalizer_key] = normalizer(obs[normalizer_key])
        return super()._hidden(obs)

    def _act(self, obs: Any, deterministic: bool = False) -> torch.Tensor:
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        hidden = self._hidden(obs)
        mean, log_std, distribution_kwargs = self.get_action_dist_params(
            obs,
            hidden=hidden,
        )
        return self.action_dist.actions_from_params(
            mean,
            log_std,
            deterministic=deterministic,
            **distribution_kwargs,
        )

    def log_prob(self, obs: Any, action: torch.Tensor) -> torch.Tensor:
        """Evaluate an action while retaining edit-distribution conditioning."""
        hidden = self._hidden(obs)
        mean, log_std, distribution_kwargs = self.get_action_dist_params(
            obs,
            hidden=hidden,
        )
        distribution = self.action_dist.proba_distribution(
            mean,
            log_std,
            **distribution_kwargs,
        )
        return distribution.log_prob(action)

    def get_action_dist_params(
        self,
        obs: Any,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if hidden is None:
            hidden = self._hidden(obs)
        batch_size = hidden.shape[0]
        mean = self.mean_layer(hidden).reshape(batch_size, *self._action_shape)
        dist_kwargs: dict[str, Any] = {}
        if self.config.edit_scale is not None or self.config.reference_centered_full_support:
            if not isinstance(obs, dict) or self.config.reference_action_key not in obs:
                raise KeyError(
                    "SAC-edit edit actor requires observation key "
                    f"{self.config.reference_action_key!r}"
                )
            reference = obs[self.config.reference_action_key]
            if reference.ndim == len(self._action_shape):
                reference = reference.unsqueeze(0)
            reference = reference.reshape(batch_size, *self._action_shape).to(
                device=mean.device,
                dtype=mean.dtype,
            )
            dist_kwargs["reference_action"] = self.normalize_action(reference)
        raw_log_std = self.log_std_layer(hidden).reshape(batch_size, *self._action_shape)
        log_std = self.config.log_std_min + 0.5 * (
            self.config.log_std_max - self.config.log_std_min
        ) * (torch.tanh(raw_log_std) + 1.0)
        return mean, log_std, dist_kwargs
