# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression-target and actor-prior diagnostics must preserve optimization."""

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.algorithms.sac.diagnostics import (
    gaussian_prior_diagnostics,
    td_target_diagnostics,
)
from tests.unit.test_sac_core import _make_sac


def test_td_subsets_report_masked_values_and_empty_counts():
    targets = torch.tensor([[1.0], [0.0], [1.5], [0.5]], requires_grad=True)
    metrics = td_target_diagnostics(targets, torch.tensor([1, 1, 0, 0]))
    assert metrics["td_target_mean"] == pytest.approx(0.75)
    assert metrics["td_target_frac_above_one"] == pytest.approx(0.25)
    assert metrics["td_target_terminal_count"] == 2
    assert metrics["td_target_terminal_mean"] == pytest.approx(0.5)
    assert metrics["td_target_nonterminal_max"] == pytest.approx(1.5)
    assert metrics["td_target_nonterminal_frac_above_one"] == pytest.approx(0.5)
    empty = td_target_diagnostics(targets, torch.zeros(4))
    assert empty["td_target_terminal_count"] == 0
    assert "td_target_terminal_mean" not in empty
    assert targets.grad is None


@pytest.mark.parametrize("prior_std", [1.0, 2.0])
def test_prior_components_match_distribution_kl_without_rng_or_gradients(prior_std):
    mean = torch.tensor([[[0.0, 1.0], [-0.5, 0.3]]], requires_grad=True)
    log_std = torch.tensor([[[-0.4, 0.0], [0.1, -0.2]]], requires_grad=True)
    before_rng = torch.get_rng_state().clone()
    metrics = gaussian_prior_diagnostics(mean, log_std, prior_std)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, log_std.exp()),
        torch.distributions.Normal(
            torch.zeros_like(mean), torch.full_like(mean, prior_std)
        ),
    )
    assert metrics["prior_kl_per_dim"] == pytest.approx(expected.mean().item())
    assert metrics["prior_kl_mean_component"] + metrics[
        "prior_kl_variance_component"
    ] == pytest.approx(expected.flatten(1).sum(1).mean().item())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert mean.grad is None and log_std.grad is None


def test_logged_targets_include_all_critic_updates_and_mask_terminal_q():
    sac = _make_sac(critic_optimizer_lr=0.0, gamma=1.0)
    sac.sampler.done = 1.0
    sac.sampler.reward = 0.0
    metrics = sac.train_step(rollout_step=10000)
    assert metrics["target_q_mean"] == pytest.approx(1.0)
    assert metrics["td_target_mean"] == pytest.approx(0.0)
    assert metrics["td_target_max"] == pytest.approx(0.0)
    assert metrics["td_target_terminal_count"] == 8 * 4
    assert metrics["td_target_nonterminal_count"] == 0
    assert "td_target_nonterminal_mean" not in metrics


def test_saturation_uses_normalized_actor_bounds(monkeypatch):
    sac = _make_sac()
    original = sac.actor.action_dist.log_prob_from_params

    def saturated(mean, log_std, **kwargs):
        actions, log_prob = original(mean, log_std, **kwargs)
        return actions * 0 + 0.98, log_prob

    monkeypatch.setattr(sac.actor.action_dist, "log_prob_from_params", saturated)
    metrics = sac.train_step(rollout_step=10000)
    assert metrics["noise/w_frac_saturated"] == pytest.approx(1.0)


def test_diagnostic_helpers_do_not_change_updates_or_randomness(monkeypatch):
    sac = _make_sac()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    sac.train_step(rollout_step=10000)
    expected_weights = {
        key: value.clone() for key, value in sac.actor.state_dict().items()
    }
    expected_critic = {
        key: value.clone() for key, value in sac.critic.state_dict().items()
    }
    expected_rng = torch.get_rng_state().clone()
    reference = _make_sac()
    monkeypatch.setattr(
        "isaac_rl_posttraining.algorithms.sac.sac.gaussian_prior_diagnostics",
        lambda *args: {},
    )
    monkeypatch.setattr(
        "isaac_rl_posttraining.algorithms.sac.diagnostics.td_target_diagnostics",
        lambda *args: {},
    )
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    reference.train_step(rollout_step=10000)
    for key, value in reference.actor.state_dict().items():
        assert torch.equal(value, expected_weights[key])
    for key, value in reference.critic.state_dict().items():
        assert torch.equal(value, expected_critic[key])
    assert torch.equal(torch.get_rng_state(), expected_rng)
