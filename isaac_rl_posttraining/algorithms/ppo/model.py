# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference continuous-action actor-critic used by PPO and its benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

__all__ = ["DSRLPPOActorCritic", "GaussianActorCritic"]


def _sum_non_batch(value: torch.Tensor) -> torch.Tensor:
    """Reduce distribution terms to one scalar per policy decision."""
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value.reshape(value.shape[0], -1).sum(dim=-1)


class DSRLPPOActorCritic(nn.Module):
    """Adapt RLPT's existing steering actor and V-network to PPO.

    The wrapped actor samples in its normalized policy space and separately
    exposes the environment-range action consumed by the frozen decoder.
    PPO therefore recomputes log-probabilities for exactly the action stored
    in a rollout, even when the steering action bounds are not ``[-1, 1]``.
    """

    def __init__(self, actor: nn.Module, value: nn.Module) -> None:
        super().__init__()
        required = ("get_action_dist_params", "action_dist", "unnormalize_action")
        missing = [name for name in required if not hasattr(actor, name)]
        if missing:
            raise TypeError(f"DSRL PPO actor is missing {missing}")
        self.actor = actor
        self.value_model = value

    def _actor_observation(self, observations: Any) -> Any:
        remove = getattr(self.actor, "_remove_obs_keys", None)
        return remove(observations) if callable(remove) else observations

    @staticmethod
    def _value_output(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value.reshape(1)
        if value.ndim == 1:
            return value
        flattened = value.reshape(value.shape[0], -1)
        if flattened.shape[1] != 1:
            raise ValueError(
                "PPO value model must return one scalar per observation, got "
                f"{tuple(value.shape)}"
            )
        return flattened[:, 0]

    def _distribution(self, observations: Any):
        actor_observation = self._actor_observation(observations)
        mean, log_std, kwargs = self.actor.get_action_dist_params(actor_observation)
        if log_std is None:
            distribution = self.actor.action_dist.proba_distribution(mean, **kwargs)
        else:
            distribution = self.actor.action_dist.proba_distribution(
                mean, log_std, **kwargs
            )
        return distribution

    @torch.no_grad()
    def sample_action_chunk(
        self,
        observations: Any,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return decoder action, learner action, log-probability, and value."""
        was_training = self.training
        self.eval()
        distribution = self._distribution(observations)
        learner_action = distribution.get_actions(deterministic=deterministic)
        log_prob = _sum_non_batch(distribution.log_prob(learner_action))
        decoder_action = self.actor.unnormalize_action(learner_action)
        value = self._value_output(self.value_model(observations))
        if was_training:
            self.train()
        return decoder_action, learner_action, log_prob, value

    @torch.no_grad()
    def values(self, observations: Any) -> torch.Tensor:
        """Evaluate successor values without sampling a second action."""
        was_training = self.training
        self.eval()
        result = self._value_output(self.value_model(observations))
        if was_training:
            self.train()
        return result

    def forward(
        self,
        observations: Any,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate normalized rollout actions through the PPO loss path."""
        was_training = self.training
        self.eval()
        try:
            distribution = self._distribution(observations)
            log_prob = _sum_non_batch(distribution.log_prob(actions))
            base_distribution = getattr(distribution, "distribution", None)
            if base_distribution is None:
                entropy = -log_prob
            else:
                entropy = _sum_non_batch(base_distribution.entropy())
            values = self._value_output(self.value_model(observations))
        finally:
            if was_training:
                self.train()
        return log_prob, entropy, values


class GaussianActorCritic(nn.Module):
    """MLP actor-critic with a tanh-squashed diagonal Gaussian policy."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if not hidden_sizes or any(width < 1 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        layers: list[nn.Module] = []
        width = observation_dim
        for next_width in hidden_sizes:
            layers.extend((nn.Linear(width, next_width), nn.Tanh()))
            width = next_width
        self.encoder = nn.Sequential(*layers)
        self.policy_mean = nn.Linear(width, action_dim)
        self.value_head = nn.Linear(width, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.action_dim = action_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_mean.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def _distribution_and_value(
        self, observations: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        # The MLP reads one flat vector: a bare tensor, or the single component of a typed observation.
        if isinstance(observations, Mapping):
            if len(observations) != 1:
                raise ValueError(
                    f"GaussianActorCritic reads one flat observation component, got {sorted(observations)}"
                )
            (observations,) = observations.values()
        features = self.encoder(observations)
        mean = self.policy_mean(features)
        log_std = self.log_std.clamp(-20.0, 2.0).expand_as(mean)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        return distribution, self.value_head(features).squeeze(-1)

    @staticmethod
    def _atanh(actions: torch.Tensor) -> torch.Tensor:
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions)
        return actions, log_prob, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        raw_actions = self._atanh(actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions)
        # Base-Gaussian entropy is stable and exact. The squashed entropy has
        # no closed form, so this is the conventional optimization surrogate.
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, values

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate fixed actions through the FSDP-safe forward path."""
        return self.evaluate_actions(observations, actions)
