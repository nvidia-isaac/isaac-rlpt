# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restoring a learner state under a configuration that trains different components.

A run with a fixed entropy coefficient has no temperature to save, so its
checkpoint carries no ``log_ent_coef``. Loading it under ``ent_coef: auto``
must say so instead of failing on a bare key lookup; the reverse (a tuned
temperature loaded into a fixed-coefficient configuration) stays loadable and
leaves the fixed coefficient alone.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.constants import (
    ACTOR,
    ACTOR_OPTIMIZER,
    CRITIC,
    CRITIC_OPTIMIZER,
    CRITIC_TARGET,
)
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
)

MANAGED = (ACTOR, CRITIC, CRITIC_TARGET, ACTOR_OPTIMIZER, CRITIC_OPTIMIZER)


class _StubCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs, action):
        return self.v.expand(action.shape[0], 1) + 0.0 * action.reshape(
            action.shape[0], -1
        ).sum(-1, keepdim=True)

    def reset_parameters(self):
        # Keep the test-set weights; nothing to re-initialize.
        pass


def _sac(ent_coef):
    torch.manual_seed(0)
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)}
    )
    actor = TransformerActor(
        TransformerActorConfig(
            observation_space=obs_space,
            action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            chunk_size=2,
            feature_hidden_dims=[],
            output_hidden_dims=[8],
            featurizer={"state": [8]},
            use_language_embeddings=False,
            transformer_dropout=0.0,
            dropout_rate=0.0,
        )
    )
    return SAC(
        SACConfig(
            actor=actor,
            critic=_StubCritic(),
            learning_starts=0,
            ent_coef=ent_coef,
            target_entropy=0,
            num_critics=2,
            n_critics_to_sample=2,
            num_critic_updates_per_actor_update=1,
            num_updates_per_train_step=1,
            pooled_critic_features=False,
            train_actor_with_kl_divergence=False,
        )
    )


def test_fixed_temperature_checkpoint_names_the_mismatch_under_auto_temperature():
    fixed = _sac(0.05)
    state = fixed.learner_state(managed=MANAGED)
    assert "log_ent_coef" not in state  # nothing to save for a fixed coefficient

    tuned = _sac("auto")
    with pytest.raises(ValueError, match="log_ent_coef.*algorithm.ent_coef"):
        tuned.load_learner_state(state, managed=MANAGED)


def test_tuned_temperature_checkpoint_loads_under_a_fixed_coefficient():
    tuned = _sac("auto")
    with torch.no_grad():
        tuned.log_ent_coef.fill_(-2.0)
    tuned.step_counter = 7
    state = tuned.learner_state(managed=MANAGED)
    assert "log_ent_coef" in state

    fixed = _sac(0.05)
    fixed.load_learner_state(state, managed=MANAGED)
    assert fixed.log_ent_coef is None
    assert float(fixed.ent_coef_tensor) == pytest.approx(0.05)
    assert fixed.step_counter == 7
