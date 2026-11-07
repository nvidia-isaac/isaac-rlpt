# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan §3 file 2 — action-distribution tests.

Covers the SAC actor's distribution stack in
``isaac_rl_posttraining.modules.base.distributions``: the tanh-squashed
diagonal Gaussian (bounds, reparameterized gradients, log-prob
change-of-variables) and the tanh bijector round trip.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.modules.base.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    TanhBijector,
)


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _squashed(batch: int = 4, dim: int = 3, mean: float = 0.3, log_std: float = -0.5):
    dist = SquashedDiagGaussianDistribution(action_dim=dim)
    mean_t = torch.full((batch, dim), mean)
    log_std_t = torch.full((batch, dim), log_std)
    dist.proba_distribution(mean_t, log_std_t)
    return dist, mean_t, log_std_t


def test_samples_within_bounds():
    dist, _, _ = _squashed(batch=512, dim=5)
    samples = dist.sample()
    assert samples.shape == (512, 5)
    assert torch.all(samples <= 1.0) and torch.all(samples >= -1.0)
    # tanh output at moderate std never pins to the exact bound
    assert float(samples.abs().max()) < 1.0


def test_rsample_gradient_flows_to_mean_and_logstd():
    dist = SquashedDiagGaussianDistribution(action_dim=3)
    mean = torch.zeros(8, 3, requires_grad=True)
    log_std = torch.full((8, 3), -0.5, requires_grad=True)
    dist.proba_distribution(mean, log_std)
    sample = dist.sample()  # reparameterized draw
    sample.sum().backward()
    assert mean.grad is not None and torch.any(mean.grad != 0)
    assert log_std.grad is not None and torch.any(log_std.grad != 0)


def test_log_prob_shape_and_finite():
    dist, mean, log_std = _squashed(batch=6, dim=4)
    actions, log_prob = dist.log_prob_from_params(mean, log_std)
    assert actions.shape == (6, 4)
    assert log_prob.shape == (6,)  # summed over action dims
    assert torch.isfinite(log_prob).all()
    # Chunked parameters keep the chunk axis: [B, K, A] -> [B, K].
    dist3 = SquashedDiagGaussianDistribution(action_dim=4)
    _, log_prob3 = dist3.log_prob_from_params(
        torch.zeros(2, 5, 4), torch.zeros(2, 5, 4)
    )
    assert log_prob3.shape == (2, 5)
    assert torch.isfinite(log_prob3).all()


def test_deterministic_mode_repeatable():
    dist, mean, log_std = _squashed()
    first = dist.actions_from_params(mean, log_std, deterministic=True)
    second = dist.actions_from_params(mean, log_std, deterministic=True)
    assert torch.equal(first, second)
    torch.testing.assert_close(first, torch.tanh(mean))


def test_stochastic_draws_differ():
    dist, mean, log_std = _squashed(batch=16)
    first = dist.actions_from_params(mean, log_std, deterministic=False)
    second = dist.actions_from_params(mean, log_std, deterministic=False)
    assert not torch.allclose(first, second)


def test_tanh_bijector_roundtrip():
    x = torch.linspace(-3.0, 3.0, 41)
    torch.testing.assert_close(
        TanhBijector.inverse(TanhBijector.forward(x)), x, atol=1e-4, rtol=0
    )
    y = torch.linspace(-0.999, 0.999, 41)
    torch.testing.assert_close(
        TanhBijector.forward(TanhBijector.inverse(y)), y, atol=1e-6, rtol=0
    )


def test_log_prob_change_of_variables_vs_numerical():
    """log p(a) = log N(atanh(a); mu, sigma) - log(1 - a^2 + eps), and integrates to 1."""
    mean = torch.tensor([[0.3]])
    log_std = torch.tensor([[-0.2]])
    grid = torch.linspace(-1.0 + 1e-4, 1.0 - 1e-4, 20001).unsqueeze(-1)

    dist = SquashedDiagGaussianDistribution(action_dim=1)
    dist.proba_distribution(mean.expand_as(grid), log_std.expand_as(grid))
    log_prob = dist.log_prob(grid)

    gaussian_actions = TanhBijector.inverse(grid)
    base_log_prob = (
        torch.distributions.Normal(
            mean.expand_as(grid), torch.exp(log_std).expand_as(grid)
        )
        .log_prob(gaussian_actions)
        .sum(-1)
    )
    # Exact change-of-variables term, computed forward from the pre-tanh
    # value: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u)).
    manual = base_log_prob - 2.0 * (
        torch.log(torch.tensor(2.0))
        - gaussian_actions.squeeze(-1)
        - torch.nn.functional.softplus(-2.0 * gaussian_actions.squeeze(-1))
    )
    torch.testing.assert_close(log_prob, manual, atol=1e-5, rtol=0)

    integral = torch.trapz(log_prob.exp(), grid.squeeze(-1))
    assert float(integral) == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Gap cases: entropy API semantics the SAC update relies on.
# ---------------------------------------------------------------------------


def test_squashed_entropy_is_none():
    """No analytic entropy post-squash; SAC must estimate it via -log_prob."""
    dist, _, _ = _squashed()
    assert dist.entropy() is None


def test_diag_gaussian_entropy_finite_and_summed():
    dist = DiagGaussianDistribution(action_dim=3)
    dist.proba_distribution(torch.zeros(4, 3), torch.full((4, 3), -0.5))
    entropy = dist.entropy()
    assert entropy.shape == (4,)
    assert torch.isfinite(entropy).all()
