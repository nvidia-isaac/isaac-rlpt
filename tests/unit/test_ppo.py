# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for provider-neutral PPO math and learner behavior."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.algorithms.ppo import (
    GaussianActorCritic,
    PPOConfig,
    PPOLearner,
    generalized_advantage_estimate,
)
from isaac_rl_posttraining.runtimes import LearnerBatch, LearnerContext
from scripts.bench_rlinf_ppo_fsdp import _batch, _global_batch_sha256


class _LocalModules:
    def __init__(self) -> None:
        self.module = None
        self.optimizer = None

    def prepare(self, module, *, name, publish=False):
        assert name == "actor_critic"
        assert publish is True
        self.module = module
        return module

    def register_optimizer(self, name, optimizer, *, module):
        assert name == "actor_critic"
        assert module is self.module
        self.optimizer = optimizer


def _ppo_batch(model: GaussianActorCritic, samples: int = 16) -> LearnerBatch:
    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(samples, 5, generator=generator)
    with torch.no_grad():
        actions, old_log_probs, old_values = model.act(observations)
    advantages = torch.linspace(-1.0, 1.0, samples)
    returns = old_values + advantages
    return LearnerBatch(
        tensors={
            "observations": observations.numpy(),
            "actions": actions.numpy(),
            "old_log_probs": old_log_probs.numpy(),
            "advantages": advantages.numpy(),
            "returns": returns.numpy(),
            "old_values": old_values.numpy(),
        },
        samples=samples,
    )


def test_gae_bootstraps_timeouts_but_stops_episode_trace():
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)
    next_values = np.array([[0.6], [4.0], [9.0]], dtype=np.float32)
    terminated = np.array([[False], [False], [True]])
    truncated = np.array([[False], [True], [False]])
    steps = np.array([[1], [2], [1]], dtype=np.int64)

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        steps,
        gamma=0.5,
        gae_lambda=0.8,
    )

    second_delta = 2.0 + 0.5**2 * 4.0 - 0.6
    first_delta = 1.0 + 0.5 * 0.6 - 0.5
    expected = np.array(
        [
            [first_delta + 0.5 * 0.8 * second_delta],
            [second_delta],
            [3.0 - 0.7],
        ],
        dtype=np.float32,
    )
    assert np.allclose(advantages, expected)
    assert np.allclose(returns, expected + values)


def test_gae_rejects_invalid_shapes_and_durations():
    valid = np.ones((2, 1), dtype=np.float32)
    flags = np.zeros((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="share shape"):
        generalized_advantage_estimate(
            valid,
            valid[:1],
            valid,
            flags,
            flags,
            np.ones((2, 1), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="positive"):
        generalized_advantage_estimate(
            valid,
            valid,
            valid,
            flags,
            flags,
            np.zeros((2, 1), dtype=np.int64),
        )


def test_squashed_policy_recomputes_behavior_log_probability():
    torch.manual_seed(3)
    model = GaussianActorCritic(5, 3, hidden_sizes=(16, 16))
    observations = torch.randn(8, 5)

    actions, behavior_log_prob, behavior_values = model.act(observations)
    log_prob, entropy, values = model(observations, actions)

    assert actions.shape == (8, 3)
    assert torch.all(actions.abs() < 1.0)
    assert torch.allclose(log_prob, behavior_log_prob, atol=1e-5, rtol=1e-5)
    assert torch.equal(values, behavior_values)
    assert entropy.shape == (8,)


def test_ppo_learner_updates_and_restores_algorithm_state():
    torch.manual_seed(11)
    original = GaussianActorCritic(5, 2, hidden_sizes=(32, 32))
    batch = _ppo_batch(original)
    modules = _LocalModules()
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        modules,
        lambda: original,
        PPOConfig(epochs=2, minibatch_size=8, seed=19),
    )
    before = {
        name: value.detach().clone() for name, value in original.state_dict().items()
    }

    metrics = learner.train_step(batch)

    assert learner.updates == 1
    assert metrics["epochs_completed"] == pytest.approx(2.0)
    assert metrics["minibatches"] == pytest.approx(4.0)
    assert metrics["optimizer_seconds"] > 0.0
    assert metrics["learner_rank_samples_per_second"] > 0.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], value)
        for name, value in original.state_dict().items()
    )

    state = learner.checkpoint_state()
    learner.updates = 9
    learner.load_checkpoint_state(state)
    assert learner.updates == 1


def test_ppo_update_returns_its_cached_gpu_memory(monkeypatch):
    # Kept by the allocator, an update's activations starve simulators rendering on the same GPU.
    torch.manual_seed(13)
    model = GaussianActorCritic(5, 2, hidden_sizes=(32, 32))
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _LocalModules(),
        lambda: model,
        PPOConfig(epochs=1, minibatch_size=8, seed=3),
    )
    released = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Adam asks CUDA whether it is capturing a graph once is_available() is true; a CUDA wheel without a GPU then raises.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: released.append(learner.updates)
    )

    learner.train_step(_ppo_batch(model))

    assert released == [1]


def test_target_kl_stops_before_an_out_of_region_minibatch_update():
    torch.manual_seed(23)
    model = GaussianActorCritic(5, 2, hidden_sizes=(32, 32))
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _LocalModules(),
        lambda: model,
        PPOConfig(
            learning_rate=1.0,
            epochs=4,
            minibatch_size=4,
            target_kl=1e-8,
            seed=29,
        ),
    )

    metrics = learner.train_step(_ppo_batch(model, samples=16))

    assert metrics["early_stopped"] == pytest.approx(1.0)
    assert metrics["minibatches"] < 4.0


def test_ppo_batch_contract_rejects_missing_and_uneven_minibatches():
    model = GaussianActorCritic(5, 2, hidden_sizes=(8,))
    modules = _LocalModules()
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        modules,
        lambda: model,
        PPOConfig(epochs=1, minibatch_size=8),
    )
    batch = _ppo_batch(model, samples=16)
    missing = LearnerBatch(
        tensors={
            name: value for name, value in batch.tensors.items() if name != "returns"
        },
        samples=16,
    )
    with pytest.raises(ValueError, match="missing=.*returns"):
        learner.train_step(missing)

    uneven = LearnerBatch(
        tensors={name: np.asarray(value)[:10] for name, value in batch.tensors.items()},
        samples=10,
    )
    with pytest.raises(ValueError, match="divisible"):
        learner.train_step(uneven)


def test_ppo_batch_contract_rejects_nonfinite_and_nonscalar_targets():
    model = GaussianActorCritic(5, 2, hidden_sizes=(8,))
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _LocalModules(),
        lambda: model,
        PPOConfig(epochs=1, minibatch_size=8),
    )
    batch = _ppo_batch(model, samples=16)
    nonfinite_tensors = dict(batch.tensors)
    nonfinite_advantages = np.asarray(nonfinite_tensors["advantages"]).copy()
    nonfinite_advantages[3] = np.nan
    nonfinite_tensors["advantages"] = nonfinite_advantages
    with pytest.raises(ValueError, match="advantages.*finite"):
        learner.train_step(LearnerBatch(nonfinite_tensors, samples=16))

    nonscalar_tensors = dict(batch.tensors)
    nonscalar_tensors["returns"] = np.zeros((16, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="one scalar per sample"):
        learner.train_step(LearnerBatch(nonscalar_tensors, samples=16))


def test_fsdp_benchmark_shards_one_fixed_global_batch():
    base = argparse.Namespace(
        global_batch=16,
        num_learner_ranks=1,
        data_seed=31,
        sequence_length=3,
        observation_dim=5,
        action_dim=2,
    )
    one_rank = _batch(base, 0)
    one_rank_digest = _global_batch_sha256(base)
    base.num_learner_ranks = 4
    four_rank_batches = [_batch(base, learner_rank) for learner_rank in range(4)]

    for name, value in one_rank.tensors.items():
        combined = np.concatenate(
            [np.asarray(batch.tensors[name]) for batch in four_rank_batches]
        )
        assert np.array_equal(combined, value)
    assert _global_batch_sha256(base) == one_rank_digest


def test_ppo_accepts_readonly_arrays_from_remote_object_stores():
    model = GaussianActorCritic(5, 2, hidden_sizes=(8,))
    batch = _ppo_batch(model, samples=16)
    for value in batch.tensors.values():
        assert isinstance(value, np.ndarray)
        value.flags.writeable = False
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _LocalModules(),
        lambda: model,
        PPOConfig(epochs=1, minibatch_size=8),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.train_step(batch)

    assert not any("not writable" in str(item.message) for item in caught)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip_ratio", 0.0),
        ("epochs", 0),
        ("entropy_coef", -1.0),
        ("target_kl", 0.0),
        ("seed", -1),
        ("discount_per", "chunk"),
    ],
)
def test_ppo_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(PPOConfig(), **{field: value})
