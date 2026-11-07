# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic IQL components: critic heads, reductions, and losses.

These are backbone-agnostic (any feature-token producer can use them);
the tests pin the math on tiny tensors.
"""

import pytest

torch = pytest.importorskip("torch")

from isaac_rl_posttraining.algorithms.iql import (
    TokenCriticEnsemble,
    TokenValueHead,
    discounted_reward_sum,
    expectile_loss,
    iql_losses,
    reduce_q,
)


def test_reduce_q_modes():
    q = torch.tensor([[[1.0]], [[3.0]]])  # [num_q=2, B=1, 1]
    assert reduce_q(q, "min").item() == 1.0
    assert reduce_q(q, "mean").item() == 2.0
    assert reduce_q(q, "max").item() == 3.0
    with pytest.raises(ValueError):
        reduce_q(q, "median")


def test_expectile_loss_asymmetry():
    positive = expectile_loss(torch.tensor([1.0]), expectile=0.9)
    negative = expectile_loss(torch.tensor([-1.0]), expectile=0.9)
    assert positive.item() == pytest.approx(0.9)
    assert negative.item() == pytest.approx(0.1)


def test_discounted_reward_sum_stops_at_done():
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    done = torch.tensor([[0.0, 1.0, 0.0]])
    # Steps after the first done contribute nothing.
    value = discounted_reward_sum(reward, gamma=0.5, done=done)
    assert value.item() == pytest.approx(1.0 + 0.5 * 1.0)


def test_iql_losses_shapes_and_finiteness():
    torch.manual_seed(0)
    batch, tokens, feat_dim, horizon, action_dim = 3, 4, 16, 2, 5
    critic = TokenCriticEnsemble(
        feat_dim, action_dim, horizon, num_q=2, n_heads=2, hidden=8
    )
    losses = iql_losses(
        critic,
        feats_cur=torch.randn(batch, tokens, feat_dim),
        feats_next=torch.randn(batch, tokens, feat_dim),
        action=torch.randn(batch, horizon, action_dim),
        reward=torch.zeros(batch, horizon),
        done=torch.zeros(batch, horizon),
    )
    assert losses["awr_weights"].shape == (batch, 1)
    for key in ("q_loss", "v_loss"):
        assert torch.isfinite(losses[key])
    assert (losses["awr_weights"] >= 0).all()


class _FixedCritic:
    """Q and V that ignore their inputs, so the TD target is the only unknown."""

    def __init__(self, q: float, v: float) -> None:
        self._q = q
        self._v = v

    def q_values(self, feats, action, key_padding_mask=None):
        del action, key_padding_mask
        return torch.full((2, feats.shape[0], 1), self._q)

    def v_value(self, feats, key_padding_mask=None):
        del key_padding_mask
        return torch.full((feats.shape[0], 1), self._v)


def test_q_target_is_the_reward_at_done_and_bootstraps_otherwise():
    critic = _FixedCritic(q=0.0, v=5.0)
    feats = torch.zeros(1, 1, 1)
    action = torch.zeros(1, 1, 1)
    reward = torch.tensor([[1.0]])
    stopped = iql_losses(
        critic,
        feats,
        feats,
        action,
        reward,
        torch.tensor([[1.0]]),
        gamma=0.5,
        bootstrap_horizon=1,
        critic_target=critic,
    )
    # Heads score 0, so the MSE is the squared target. done drops V(s'): target == 1.
    assert stopped["q_loss"].item() == pytest.approx(1.0)
    alive = iql_losses(
        critic,
        feats,
        feats,
        action,
        reward,
        torch.zeros(1, 1),
        gamma=0.5,
        bootstrap_horizon=1,
        critic_target=critic,
    )
    assert alive["q_loss"].item() == pytest.approx((1.0 + 0.5 * 5.0) ** 2)


def test_iql_losses_v_target_uses_a_q_subset_all_heads_still_regress():
    torch.manual_seed(0)
    batch, tokens, feat_dim, horizon, action_dim = 4, 3, 8, 2, 3
    critic = TokenCriticEnsemble(
        feat_dim, action_dim, horizon, num_q=5, n_heads=2, hidden=8
    )
    feats_cur = torch.randn(batch, tokens, feat_dim)
    feats_next = torch.randn(batch, tokens, feat_dim)
    action = torch.randn(batch, horizon, action_dim)
    reward = torch.zeros(batch, horizon)
    done = torch.zeros(batch, horizon)
    torch.manual_seed(7)
    q_indices = torch.randperm(5)[:2]
    torch.manual_seed(7)
    losses = iql_losses(
        critic,
        feats_cur,
        feats_next,
        action,
        reward,
        done,
        n_q_to_sample=2,
        critic_target=critic,
    )
    q_all = critic.q_values(feats_cur, action)
    expected_target = reduce_q(q_all, "min", indices=q_indices)
    v_s = critic.v_value(feats_cur)
    assert losses["v_loss"].item() == pytest.approx(
        expectile_loss(expected_target - v_s, 0.7).item()
    )
    # Q loss still averages all five heads against the TD target, not the subset.
    with torch.no_grad():
        v_next = critic.v_value(feats_next)
        q_target = discounted_reward_sum(reward, 0.99, done) + 0.99 * v_next
    expected_q = (
        sum(torch.nn.functional.mse_loss(q_all[i], q_target) for i in range(5)) / 5
    )
    assert losses["q_loss"].item() == pytest.approx(expected_q.item())


def test_padding_mask_hides_tokens_from_v_and_q_and_shapes_ignore_length():
    torch.manual_seed(0)
    feat_dim, horizon, action_dim = 8, 2, 3
    critic = TokenCriticEnsemble(
        feat_dim, action_dim, horizon, num_q=2, n_heads=2, hidden=8
    )
    feats = torch.randn(2, 5, feat_dim)
    action = torch.randn(2, horizon, action_dim)
    reward = torch.zeros(2, horizon)
    done = torch.zeros(2, horizon)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, -2:] = True
    first = iql_losses(
        critic,
        feats,
        feats,
        action,
        reward,
        done,
        key_padding_mask=mask,
        next_key_padding_mask=mask,
    )
    tweaked = feats.clone()
    tweaked[mask] = 9
    second = iql_losses(
        critic,
        tweaked,
        tweaked,
        action,
        reward,
        done,
        key_padding_mask=mask,
        next_key_padding_mask=mask,
    )
    assert torch.equal(first["v_loss"], second["v_loss"])
    assert torch.equal(first["q_loss"], second["q_loss"])
    unmasked = iql_losses(
        critic, tweaked, tweaked, action=action, reward=reward, done=done
    )
    assert not torch.allclose(unmasked["v_loss"], first["v_loss"])

    def shapes(length: int) -> list[tuple[int, ...]]:
        head = TokenValueHead(
            feat_dim, n_heads=2, hidden=8, action_dim=action_dim, action_horizon=horizon
        )
        head(torch.randn(2, length, feat_dim), action)
        return [tuple(parameter.shape) for parameter in head.parameters()]

    assert shapes(1) == shapes(17)
