# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from Stable-Baselines3 (MIT) and modified; see THIRD_PARTY_NOTICES.md.
#
# SPDX-FileCopyrightText: Copyright (c) 2019 Antonin Raffin
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Action distributions for stochastic policies.

Adapted from Stable-Baselines3 (MIT License, (c) 2019 Antonin Raffin et al.),
trimmed to the distributions used by this package: diagonal Gaussians (plain
and tanh-squashed) for continuous control and a categorical head for discrete
actions. Sampling uses the reparameterization trick so gradients flow to the
distribution parameters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch as th
from torch import nn
from torch.distributions import Categorical, Normal

_LOG_2 = math.log(2.0)

__all__ = [
    "Distribution",
    "DiagGaussianDistribution",
    "SquashedDiagGaussianDistribution",
    "CategoricalDistribution",
    "TanhBijector",
    "sum_independent_dims",
    "kl_divergence",
]


class Distribution(ABC):
    """Abstract base class for policy action distributions."""

    def __init__(self) -> None:
        super().__init__()
        self.distribution: Any = None

    @abstractmethod
    def proba_distribution_net(self, *args: Any, **kwargs: Any) -> Union[nn.Module, tuple[nn.Module, nn.Parameter]]:
        """Create the layers/parameters that produce the distribution parameters."""

    @abstractmethod
    def proba_distribution(self, *args: Any, **kwargs: Any) -> "Distribution":
        """Set the distribution parameters and return ``self``."""

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """Log-likelihood of ``x`` under the current parameters."""

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """Shannon entropy, or ``None`` if no analytical form exists."""

    @abstractmethod
    def sample(self) -> th.Tensor:
        """Draw a reparameterized sample."""

    @abstractmethod
    def mode(self) -> th.Tensor:
        """Most likely action (deterministic output)."""

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """Return the mode when ``deterministic`` else a sample."""
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args: Any, **kwargs: Any) -> th.Tensor:
        """Set parameters, then return actions."""

    @abstractmethod
    def log_prob_from_params(self, *args: Any, **kwargs: Any) -> tuple[th.Tensor, th.Tensor]:
        """Set parameters, then return ``(actions, log_prob)``."""


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """Sum per-dimension log-probs/entropies over the action dimension.

    Args:
        tensor: Shape ``[..., n_actions]`` or ``[n]``.

    Returns:
        The input summed over its last dimension (or fully summed for 1-D
        input), reflecting independence across action dimensions.
    """
    if len(tensor.shape) > 1:
        return tensor.sum(dim=-1)
    return tensor.sum()


class DiagGaussianDistribution(Distribution):
    """Diagonal-covariance Gaussian for continuous actions."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions: Optional[th.Tensor] = None
        self.log_std: Optional[th.Tensor] = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        """Create a mean head and a state-independent ``log_std`` parameter.

        Args:
            latent_dim: Width of the features feeding the mean head.
            log_std_init: Initial value for every ``log_std`` entry.

        Returns:
            Tuple ``(mean_head, log_std_parameter)``.
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """Parameterize the Gaussian; std is floored at ``1e-6`` for stability."""
        action_std = th.ones_like(mean_actions) * log_std.exp()
        action_std = th.clamp(action_std, min=1e-6)
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Log-likelihood summed over action dimensions."""
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """Diagonal Gaussian squashed through ``tanh`` to enforce ``[-1, 1]`` bounds.

    Log-probabilities include the tanh change-of-variables correction from
    the original SAC derivation.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6) -> None:
        """Args:
        action_dim: Action dimensionality.
        epsilon: Numerical floor inside the squash correction's log.
        """
        super().__init__(action_dim)
        self.epsilon = epsilon
        self.gaussian_actions: Optional[th.Tensor] = None

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> "SquashedDiagGaussianDistribution":
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        """Log-likelihood of squashed actions.

        Args:
            actions: Post-tanh actions in ``[-1, 1]``.
            gaussian_actions: Matching pre-tanh values if already known;
                recovered via a clipped ``atanh`` otherwise.
        """
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        # Change-of-variables correction: subtract log|d tanh(u)/du| per
        # dim, computed forward from the pre-tanh value:
        # log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u)). Same value as
        # the log(1 - a^2 + eps) form but exact at saturation, where the
        # eps floor biases entropy by tens of nats and whipsaws the
        # temperature (the reference DSRL stack computes it this way).
        log_prob -= th.sum(
            2.0 * (_LOG_2 - gaussian_actions - th.nn.functional.softplus(-2.0 * gaussian_actions)),
            dim=-1,
        )
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        """No analytical form; estimate as ``-log_prob.mean()`` if needed."""
        return None

    def sample(self) -> th.Tensor:
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class CategoricalDistribution(Distribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the logits head."""
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=-1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class TanhBijector:
    """Bijective ``tanh`` squashing with a numerically safe inverse."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        """Args:
        epsilon: Numerical floor for the log-prob correction.
        """
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        """Apply ``tanh``."""
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """Numerically stable ``atanh`` via ``0.5 * (log1p(x) - log1p(-x))``."""
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """Invert ``tanh`` with inputs clipped away from ±1 to avoid NaNs."""
        eps = th.finfo(y.dtype).eps
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        """Per-dimension change-of-variables term ``log(1 - tanh(x)^2)``."""
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> th.Tensor:
    """Compute ``KL(dist_true || dist_pred)`` for same-type distributions.

    Args:
        dist_true: The ``p`` distribution.
        dist_pred: The ``q`` distribution (must be the same class as ``p``).

    Returns:
        The elementwise KL divergence from PyTorch's implementation.
    """
    assert dist_true.__class__ == dist_pred.__class__, "Input distributions must be the same type"
    return th.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)
