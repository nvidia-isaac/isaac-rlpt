# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SAC learner speedups keep training bit-identical.

Covers the batched target-network update and the frozen critic during the actor update.
"""

from __future__ import annotations

import contextlib
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

import isaac_rl_posttraining.algorithms.sac.sac as sac_module
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
    TransformerCritic,
    TransformerCriticConfig,
)
from isaac_rl_posttraining.utils.networks import (
    TargetUpdater,
    frozen_parameters,
    polyak_update,
)

OBS_DIM, ACT_DIM, CHUNK, WIDTH = 6, 3, 4, 16


def _per_tensor_polyak(params, target_params, tau):
    """The update as it was written before batching: one mul_ and add_ per tensor."""
    with torch.no_grad():
        for p, tp in zip(params, target_params):
            if tp.data.data_ptr() == p.data.data_ptr() or tau >= 1.0:
                tp.data.copy_(p.data)
            else:
                tp.data.mul_(1.0 - tau)
                tp.data.add_(p.data, alpha=tau)


# ---------------------------------------------------------------------------
# Target-network update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.005, 0.1, 0.5, 1.0])
def test_batched_polyak_matches_the_per_tensor_update_bit_for_bit(tau):
    generator = torch.Generator().manual_seed(0)
    shapes = [(64, 33), (33,), (7, 5, 3), ()]
    online = [torch.randn(shape, generator=generator) for shape in shapes]
    target = [torch.randn(shape, generator=generator) for shape in shapes]
    expected = [t.clone() for t in target]
    for _ in range(5):
        _per_tensor_polyak(online, expected, tau)
        polyak_update(online, target, tau)
    for got, want in zip(target, expected):
        assert torch.equal(got, want)


def test_polyak_leaves_storage_shared_with_the_online_network_alone():
    shared = torch.full((3,), 2.0)
    online = [shared, torch.ones(2)]
    target = [shared, torch.zeros(2)]
    polyak_update(online, target, 0.5)
    assert torch.equal(shared, torch.full((3,), 2.0))
    assert torch.equal(target[1], torch.full((2,), 0.5))


def test_target_updater_lists_parameters_once_per_module_pair():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    updater = TargetUpdater()
    calls = []
    original = nn.Module.parameters

    def counting(module, recurse=True):
        calls.append(module)
        return original(module, recurse)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nn.Module, "parameters", counting)
        for _ in range(3):
            updater.soft_update(online, target, 0.5)
        assert len(calls) == 2
        other = nn.Linear(3, 2)
        updater.soft_update(other, target, 1.0)
        assert len(calls) == 4
    assert torch.equal(target.weight, other.weight)


def test_a_pickled_target_updater_relists_the_copied_parameters():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    updater = TargetUpdater()
    updater.soft_update(online, target, 0.5)
    online_copy, target_copy, updater_copy = pickle.loads(
        pickle.dumps((online, target, updater))
    )
    before = target.weight.detach().clone()
    updater_copy.soft_update(online_copy, target_copy, 1.0)
    assert torch.equal(target_copy.weight, online_copy.weight)
    assert torch.equal(target.weight, before)


# ---------------------------------------------------------------------------
# Frozen critic during the actor update
# ---------------------------------------------------------------------------


def test_frozen_parameters_keeps_input_gradients_and_restores_the_flags():
    module = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    module[2].bias.requires_grad_(False)
    inputs = torch.randn(5, 4, requires_grad=True)
    (expected,) = torch.autograd.grad(module(inputs).sum(), inputs)
    with frozen_parameters(module):
        assert not any(p.requires_grad for p in module.parameters())
        module(inputs).sum().backward()
    assert torch.equal(inputs.grad, expected)
    assert all(p.grad is None for p in module.parameters())
    assert [p.requires_grad for p in module.parameters()] == [True, True, True, False]


# ---------------------------------------------------------------------------
# SAC: the full update stays bit-identical
# ---------------------------------------------------------------------------


class _Sampler:
    def can_sample(self, batch_size):
        return True

    def sample(self, batch_size):
        b = int(batch_size)
        return {
            "obs": {"state": torch.randn(b, OBS_DIM)},
            "action": torch.randn(b, CHUNK, ACT_DIM).clamp(-1, 1),
            "reward": -torch.rand(b),
            "next_obs": {"state": torch.randn(b, OBS_DIM)},
            "done": (torch.rand(b) < 0.2).float(),
            "truncated": torch.zeros(b),
        }


def _spaces():
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
    )
    return obs_space, gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)


def _make_sac() -> SAC:
    torch.manual_seed(0)
    np.random.seed(0)
    obs_space, act_space = _spaces()
    common = {
        "observation_space": obs_space,
        "action_space": act_space,
        "d_model": WIDTH,
        "nhead": 2,
        "num_encoder_layers": 1,
        "chunk_size": CHUNK,
        "feature_hidden_dims": [],
        "featurizer": {"state": [WIDTH]},
        "use_language_embeddings": False,
        "transformer_dropout": 0.0,
        "dropout_rate": 0.0,
        "positional_dropout": 0.0,
    }
    actor = TransformerActor(
        TransformerActorConfig(
            output_hidden_dims=[WIDTH], use_tanh_output=True, **common
        )
    )
    critic = TransformerCritic(
        TransformerCriticConfig(use_action=True, output_hidden_dims=[], **common)
    )
    return SAC(
        SACConfig(
            actor=actor,
            critic=critic,
            sampler=_Sampler(),
            learning_starts=0,
            batch_size=8,
            num_critics=4,
            n_critics_to_sample=2,
            num_critic_updates_per_actor_update=4,
            num_updates_per_train_step=3,
            pooled_critic_features=True,
            train_actor_with_kl_divergence=False,
            use_amp=False,
            tau=0.05,
            diagnostics_interval=2,
        )
    )


def _train(reference: bool) -> tuple[SAC, list[dict]]:
    with pytest.MonkeyPatch.context() as patch:
        if reference:
            patch.setattr(
                sac_module, "frozen_parameters", lambda module: contextlib.nullcontext()
            )
            patch.setattr(
                TargetUpdater,
                "soft_update",
                lambda self, online, target, tau: _per_tensor_polyak(
                    online.parameters(), target.parameters(), tau
                ),
            )
        sac = _make_sac()
        torch.manual_seed(1)
        np.random.seed(1)
        metrics = [sac.train_step(rollout_step=None) for _ in range(4)]
    return sac, metrics


def _state(sac: SAC) -> dict[str, torch.Tensor]:
    state = {}
    for name in ("actor", "critic", "critic_target"):
        state.update(
            {f"{name}.{k}": v for k, v in getattr(sac, name).state_dict().items()}
        )
    for name in ("actor_optimizer", "critic_optimizer", "ent_coef_optimizer"):
        for index, entry in getattr(sac, name).state_dict()["state"].items():
            state.update(
                {
                    f"{name}.{index}.{k}": v
                    for k, v in entry.items()
                    if torch.is_tensor(v)
                }
            )
    state["log_ent_coef"] = sac.log_ent_coef.detach()
    return state


def test_sac_updates_are_bit_identical_to_the_unbatched_unfrozen_update():
    fast, fast_metrics = _train(reference=False)
    slow, slow_metrics = _train(reference=True)
    fast_state, slow_state = _state(fast), _state(slow)
    assert fast_state.keys() == slow_state.keys()
    for key in fast_state:
        assert torch.equal(fast_state[key], slow_state[key]), key
    for got, want in zip(fast_metrics, slow_metrics):
        assert _comparable(got) == _comparable(want)


def _comparable(metrics: dict) -> dict:
    """Metrics without timings, histograms as their raw values."""
    return {
        key: tuple(np.asarray(getattr(value, "values", value)).ravel().tolist())
        for key, value in metrics.items()
        if key != "train_step_total_time_s"
    }


def test_the_actor_update_computes_no_critic_weight_gradients():
    sac = _make_sac()
    accumulations = []
    weight = next(sac.critic.parameters())
    weight.register_post_accumulate_grad_hook(lambda p: accumulations.append(1))
    sac.train_step(rollout_step=None)
    critic_steps = (
        sac.config.num_updates_per_train_step
        * sac.config.num_critic_updates_per_actor_update
    )
    assert len(accumulations) == critic_steps
    assert all(p.requires_grad for p in sac.critic.parameters())


def test_critic_step_metrics_are_plain_floats():
    sac = _make_sac()
    metrics = sac.train_step(rollout_step=None)
    for key in (
        "critic_loss",
        "q_values_mean",
        "target_q_mean",
        "reward_mean",
        "reward_min",
        "reward_max",
    ):
        assert isinstance(metrics[key], float) and np.isfinite(metrics[key]), key
