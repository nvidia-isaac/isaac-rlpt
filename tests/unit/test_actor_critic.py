# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan §3 file 3 — transformer actor/critic network tests.

Exercises ``isaac_rl_posttraining.modules.transformer`` at the canonical DSRL
latent geometry (chunk_size=40 tokens x 35 latent dims, bounds [-2, 2]) with a
CPU-sized transformer: one observation token is repeated per chunk step,
positionally encoded, and a single shared head decodes every token; the critic
consumes [obs_token, action_tokens] and pools the first (observation) token.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
    TransformerCritic,
    TransformerCriticConfig,
)

# Canonical latent geometry (see test_config.py: chunk_size=40, noise_dim=35, bound 2.0).
CHUNK_SIZE = 40
ACTION_DIM = 35
ACTION_BOUND = 2.0
OBS_DIM = 7
D_MODEL = 16


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _spaces():
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
    )
    act_space = gym.spaces.Box(
        -ACTION_BOUND, ACTION_BOUND, (ACTION_DIM,), dtype=np.float32
    )
    return obs_space, act_space


def _make_actor(
    chunk_size: int = CHUNK_SIZE,
    log_std_init: float = 0.0,
    *,
    official_policy_init: bool = False,
    use_tanh_output: bool = True,
):
    obs_space, act_space = _spaces()
    config = TransformerActorConfig(
        observation_space=obs_space,
        action_space=act_space,
        d_model=D_MODEL,
        nhead=2,
        num_encoder_layers=1,
        chunk_size=chunk_size,
        feature_hidden_dims=[],
        output_hidden_dims=[D_MODEL],
        featurizer={"state": [D_MODEL]},
        use_language_embeddings=False,  # unit tests must not construct MiniLM
        use_tanh_output=use_tanh_output,
        log_std_init=log_std_init,
        official_policy_init=official_policy_init,
        transformer_dropout=0.0,
        dropout_rate=0.0,
        positional_dropout=0.0,
    )
    return TransformerActor(config)


def _make_critic(use_action: bool = True, pooling_strategy: str = "first"):
    obs_space, act_space = _spaces()
    config = TransformerCriticConfig(
        observation_space=obs_space,
        action_space=act_space,
        use_action=use_action,
        d_model=D_MODEL,
        nhead=2,
        num_encoder_layers=1,
        chunk_size=CHUNK_SIZE,
        feature_hidden_dims=[],
        output_hidden_dims=[],
        featurizer={"state": [D_MODEL]},
        use_language_embeddings=False,
        pooling_strategy=pooling_strategy,
        transformer_dropout=0.0,
        dropout_rate=0.0,
        positional_dropout=0.0,
    )
    return TransformerCritic(config)


def _obs(batch: int) -> dict:
    return {"state": torch.randn(batch, OBS_DIM)}


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


def test_actor_output_shape_b_k40_35():
    actor = _make_actor()
    mean, log_std, extra = actor.get_action_dist_params(_obs(3))
    assert mean.shape == (3, CHUNK_SIZE, ACTION_DIM)
    assert log_std.shape == (3, CHUNK_SIZE, ACTION_DIM)
    assert extra == {}
    actions, _ = actor.act(_obs(3))
    assert actions.shape == (3, CHUNK_SIZE, ACTION_DIM)


def test_actor_bounds_pm2():
    actor = _make_actor()
    for deterministic in (False, True):
        actions, _ = actor.act(_obs(16), deterministic=deterministic)
        assert torch.all(actions <= ACTION_BOUND + 1e-6)
        assert torch.all(actions >= -ACTION_BOUND - 1e-6)


def test_actor_head_shared_across_tokens():
    """One head module decodes every chunk token (row-wise weight sharing)."""
    actor = _make_actor()
    head = actor.mean_processor
    assert isinstance(head, nn.Module)
    assert not isinstance(head, (nn.ModuleList, nn.ModuleDict)), (
        "per-token head modules would break weight sharing across chunk steps"
    )
    assert head(torch.randn(6, D_MODEL)).shape[-1] == ACTION_DIM


def test_token_positional_diversity():
    """Identical observation, different chunk positions -> different outputs."""
    actor = _make_actor()
    mean, _, _ = actor.get_action_dist_params({"state": torch.zeros(2, OBS_DIM)})
    step0 = mean[:, 0, :]
    diverse = [
        h
        for h in range(1, CHUNK_SIZE)
        if not torch.allclose(mean[:, h, :], step0, atol=1e-5)
    ]
    assert diverse, (
        "positional encoding failed: all chunk steps produced identical outputs"
    )


def test_actor_stochastic_diversity():
    actor = _make_actor()
    obs = _obs(4)
    first, _ = actor.act({k: v.clone() for k, v in obs.items()}, deterministic=False)
    second, _ = actor.act({k: v.clone() for k, v in obs.items()}, deterministic=False)
    assert not torch.allclose(first, second)


def test_actor_deterministic_repeatable():
    actor = _make_actor()
    obs = _obs(4)
    first, _ = actor.act({k: v.clone() for k, v in obs.items()}, deterministic=True)
    second, _ = actor.act({k: v.clone() for k, v in obs.items()}, deterministic=True)
    torch.testing.assert_close(first, second)


def test_official_policy_init_matches_full_horizon_noise_prior():
    actor = _make_actor(
        log_std_init=0.0,
        official_policy_init=True,
        use_tanh_output=False,
    )
    mean, log_std, _ = actor.get_action_dist_params(_obs(3))

    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(log_std, torch.zeros_like(log_std))


def test_action_log_prob_keeps_graph():
    """SAC's actor update differentiates through sampled actions and log-probs."""
    actor = _make_actor()
    actions, log_prob = actor.action_log_prob(_obs(3))
    assert actions.shape == (3, CHUNK_SIZE, ACTION_DIM)
    assert log_prob.shape == (3, CHUNK_SIZE)
    assert actions.requires_grad
    assert log_prob.requires_grad
    log_prob.sum().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and any(torch.any(g != 0) for g in grads)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


def test_critic_q_shape_per_sample():
    critic = _make_critic()
    q = critic(_obs(5), torch.randn(5, CHUNK_SIZE, ACTION_DIM))
    assert q.shape == (5, 1)
    assert torch.isfinite(q).all()


def test_critic_reacts_to_actions():
    critic = _make_critic()
    critic.eval()
    obs = _obs(4)
    q_zero = critic(
        {k: v.clone() for k, v in obs.items()}, torch.zeros(4, CHUNK_SIZE, ACTION_DIM)
    )
    q_ones = critic(
        {k: v.clone() for k, v in obs.items()}, torch.ones(4, CHUNK_SIZE, ACTION_DIM)
    )
    assert not torch.allclose(q_zero, q_ones), "Q-function ignores its action input"


def test_critic_pooling_first():
    """'first' pooling returns exactly the observation/CLS token."""
    critic = _make_critic(pooling_strategy="first")
    assert critic.pooling_strategy == "first"
    transformer_output = torch.randn(3, 1 + CHUNK_SIZE, D_MODEL)
    pooled = critic._pool_transformer_output(transformer_output)
    torch.testing.assert_close(pooled, transformer_output[:, 0, :])


def test_value_net_variant_no_action():
    vnet = _make_critic(use_action=False)
    v = vnet(_obs(6))
    assert v.shape == (6, 1)
    assert torch.isfinite(v).all()


def test_one_optimizer_step_all_finite():
    actor = _make_actor()
    critic = _make_critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=0)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=0)

    _, log_prob = actor.action_log_prob(_obs(4))
    actor_loss = -log_prob.mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    q = critic(_obs(4), torch.randn(4, CHUNK_SIZE, ACTION_DIM))
    critic_loss = (q - 1.0).pow(2).mean()
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    for module in (actor, critic):
        for name, param in module.named_parameters():
            assert torch.isfinite(param).all(), (
                f"non-finite parameter after one step: {name}"
            )
