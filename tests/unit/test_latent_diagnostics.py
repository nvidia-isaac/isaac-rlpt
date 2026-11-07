# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SAC latent diagnostics describe the latent the frozen decoder receives.

A bounded steering actor samples in normalized ``[-1, 1]`` coordinates and
maps the sample onto ``[-bound, bound]`` before the frozen policy decodes it.
The ``noise/*`` metrics exist to show whether that decoded latent stays on the
decoder's N(0, 1) scale, so they must be computed on the executed latent, not
on the normalized sample (where every bound reads the same and nothing can
ever exceed 1).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
)

OBS_DIM, ACT_DIM, CHUNK = 7, 5, 4


class _StubCritic(nn.Module):
    """Q(obs, action) == one constant per ensemble member."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.v = nn.Parameter(torch.tensor(float(value)))

    def forward(self, obs, action):
        batch = action.shape[0]
        return self.v.expand(batch, 1) + 0.0 * action.reshape(batch, -1).sum(
            -1, keepdim=True
        )

    def reset_parameters(self):
        # Keep the test-set weights; nothing to re-initialize.
        pass


class _StubSampler:
    def can_sample(self, batch_size) -> bool:
        return True

    def sample(self, batch_size):
        b = int(batch_size)
        return {
            "obs": {"state": torch.randn(b, OBS_DIM)},
            "action": torch.randn(b, CHUNK, ACT_DIM).clamp(-1, 1),
            "reward": torch.zeros(b),
            "next_obs": {"state": torch.randn(b, OBS_DIM)},
            "done": torch.zeros(b),
            "truncated": torch.zeros(b),
        }


def _make_sac(bound: float) -> SAC:
    torch.manual_seed(0)
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
    )
    actor = TransformerActor(
        TransformerActorConfig(
            observation_space=obs_space,
            action_space=gym.spaces.Box(-bound, bound, (ACT_DIM,), dtype=np.float32),
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            chunk_size=CHUNK,
            feature_hidden_dims=[],
            output_hidden_dims=[16],
            featurizer={"state": [16]},
            use_language_embeddings=False,
            use_tanh_output=True,
            log_std_init=0.0,
            transformer_dropout=0.0,
            dropout_rate=0.0,
            positional_dropout=0.0,
        )
    )
    return SAC(
        SACConfig(
            actor=actor,
            critic=_StubCritic(1.0),
            sampler=_StubSampler(),
            learning_starts=0,
            ent_coef="auto",
            target_entropy=0,
            gamma=0.9,
            batch_size=64,
            n_critics_to_sample=2,
            num_critics=2,
            num_critic_updates_per_actor_update=1,
            num_updates_per_train_step=1,
            pooled_critic_features=False,
            train_actor_with_kl_divergence=False,
            use_amp=False,
        )
    )


def test_latent_diagnostics_are_in_executed_latent_units():
    """With a [-2, 2] latent box, the reported latent statistics exceed the normalized [-1, 1] range."""
    sac = _make_sac(bound=2.0)
    metrics = sac.train_step(rollout_step=1)
    # tanh(N(0, 1)) has standard deviation ~0.63; the decoder receives 2 * tanh(u).
    assert metrics["noise/w_absmax"] > 1.0
    assert metrics["noise/w_std"] == pytest.approx(1.26, abs=0.12)
    # |2 tanh(u)| > 1.9 for |u| > 1.83: a few percent of the entries, never zero.
    assert metrics["noise/w_frac_saturated"] > 0.0


def test_latent_diagnostics_track_the_bound():
    """The same actor weights report a latent spread proportional to the bound."""
    stds = {}
    for bound in (1.0, 3.0):
        sac = _make_sac(bound=bound)
        stds[bound] = sac.train_step(rollout_step=1)["noise/w_std"]
    assert stds[3.0] == pytest.approx(3.0 * stds[1.0], rel=0.05)
