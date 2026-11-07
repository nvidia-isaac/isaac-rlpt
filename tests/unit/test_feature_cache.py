# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached features travel from decision to learner: transition field, replay export, PPO batches."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_transition

from isaac_rl_posttraining.algorithms.ppo import (
    PPOConfig,
    PPOLearner,
    build_rank_batches,
)
from isaac_rl_posttraining.algorithms.ppo.model import GaussianActorCritic
from isaac_rl_posttraining.buffers import ReplayBuffer, save_buffer
from isaac_rl_posttraining.policies.feature_cache import (
    BF16_SUFFIX,
    CACHED_FEATURES_KEY,
    pack_features,
    stack_features,
    unpack_features,
)
from isaac_rl_posttraining.rollouts import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    LearnerBatch,
    LearnerContext,
    RolloutResult,
)
from isaac_rl_posttraining.training import trajectories_from_results

WINDOW = 2


def test_pack_stack_unpack_round_trip_keeps_bf16_bits():
    features = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.bfloat16)
    mask = torch.ones(4, 3, dtype=torch.long)
    records = pack_features({"tokens": features, "mask": mask}, 4)
    assert len(records) == 4
    assert records[0]["tokens" + BF16_SUFFIX].dtype == np.int16
    stacked = stack_features(records)
    assert stacked["tokens" + BF16_SUFFIX].shape == (4, 2)
    restored = unpack_features(stacked, "cpu")
    assert restored["tokens"].dtype == torch.bfloat16
    assert torch.equal(restored["tokens"], features)
    assert torch.equal(restored["mask"], mask)
    with pytest.raises(ValueError, match="cannot be split"):
        pack_features({"bad": torch.zeros(3, 2)}, 2)
    with pytest.raises(ValueError, match="disagree on keys"):
        stack_features([{"a": np.zeros(1)}, {"b": np.zeros(1)}])


def test_replay_buffer_stores_and_exports_cached_features(tmp_path):
    buffer = ReplayBuffer(capacity=4)
    for step in range(2):
        buffer.add(
            make_transition(
                step_in_episode=step,
                obs={"state": np.full(3, step, dtype=np.float32)},
                action=np.zeros(2, dtype=np.float32),
                reward=float(step),
                next_obs={"state": np.full(3, step + 1, dtype=np.float32)},
                done=step == 1,
                info={"success": True},
                features={
                    "tokens" + BF16_SUFFIX: np.full((4, 2), step, dtype=np.int16)
                },
            )
        )
    stored = buffer.transitions()
    assert stored[1].policy_outputs.features["tokens" + BF16_SUFFIX][0, 0] == 1
    path = tmp_path / "buffer"
    save_buffer(buffer, path)
    archive = np.load(path / "episodes.npz")
    assert archive[
        "policy_outputs/features/tokens" + BF16_SUFFIX + "/values"
    ].shape == (2, 4, 2)
    # A buffer without cached features exports none.
    plain = ReplayBuffer(capacity=2)
    plain.add(
        make_transition(
            episode_id=1, obs={"state": np.zeros(3)}, action=np.zeros(2), done=True
        )
    )
    assert plain.transitions()[0].policy_outputs.features is None
    save_buffer(plain, tmp_path / "plain")
    assert not any(
        key.startswith("policy_outputs/features")
        for key in np.load(tmp_path / "plain" / "episodes.npz")
    )


def _decision(episode_id: int, step: int, *, cached: dict | None, done: bool = False):
    transition = make_transition(
        episode_id=episode_id,
        step_in_episode=step * WINDOW,
        obs={
            "state": np.full(3, step, dtype=np.float32),
            "stochastic_step": np.asarray(1),
        },
        action=np.zeros((3, 2, 2), dtype=np.float32),
        action_kind="chain",
        rewards=np.asarray([0.0, 1.0 if done else 0.0], dtype=np.float32),
        done=done,
        info={"success": done},
        log_prob=-1.0,
        value=0.5,
        next_value=0.0 if done else 0.5,
        features=cached,
    )
    chunk = ActionChunk.from_arrays(
        learner_action=np.zeros((1, 3, 2, 2)),
        environment_actions=np.zeros((1, WINDOW, 2)),
        execution_length=WINDOW,
        learner_action_kind="chain",
        behavior_log_prob=np.asarray([-1.0]),
        behavior_value=np.asarray([0.5]),
    )
    return CollectedDecision(
        environment_index=0, transition=transition, action_chunk=chunk
    )


def _result(decisions) -> RolloutResult:
    return RolloutResult(
        worker_index=0,
        decisions=tuple(decisions),
        env_steps=WINDOW * len(decisions),
        episodes_completed=1,
        policy_version=0,
    )


def test_on_policy_batches_carry_stacked_cached_features():
    features = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.bfloat16)
    records = pack_features(
        {"tokens": features, "mask": torch.ones(4, 3, dtype=torch.long)}, 4
    )
    decisions = [
        _decision(0, step, cached=records[step], done=step == 3) for step in range(4)
    ]
    trajectories = trajectories_from_results([_result(decisions)], gamma=0.99)
    assert trajectories[0].cached_features is not None
    assert len(trajectories[0].cached_features) == 4
    batches, _ = build_rank_batches(
        trajectories,
        num_learner_ranks=1,
        minibatch_size=2,
        gamma=0.99,
        gae_lambda=0.95,
        rng=np.random.default_rng(0),
    )
    cached = batches[0].tensors[CACHED_FEATURES_KEY]
    assert set(cached) == {"tokens" + BF16_SUFFIX, "mask"}
    assert cached["tokens" + BF16_SUFFIX].dtype == np.int16
    assert cached["tokens" + BF16_SUFFIX].shape == (4, 2)
    assert cached["mask"].shape == (4, 3)
    # Without cached features the batch stays as before.
    plain = [_decision(0, step, cached=None, done=step == 3) for step in range(4)]
    batches, _ = build_rank_batches(
        trajectories_from_results([_result(plain)], gamma=0.99),
        num_learner_ranks=1,
        minibatch_size=2,
        gamma=0.99,
        gae_lambda=0.95,
        rng=np.random.default_rng(0),
    )
    assert CACHED_FEATURES_KEY not in batches[0].tensors
    mixed = [decisions[0], _decision(0, 1, cached=None, done=True)]
    with pytest.raises(ValueError, match="mixes decisions"):
        trajectories_from_results([_result(mixed)], gamma=0.99)


class _CachedGaussian(GaussianActorCritic):
    """A learner model that insists on receiving cached features, sliced like the observations."""

    def __init__(self) -> None:
        super().__init__(5, 2, hidden_sizes=(16,))
        self.seen: list[dict[str, torch.Tensor]] = []

    def forward(self, observations, actions, cached_features=None):  # type: ignore[override]
        assert cached_features is not None
        assert cached_features["tokens" + BF16_SUFFIX].shape[0] == observations.shape[0]
        assert cached_features["tokens" + BF16_SUFFIX].dtype == torch.int16
        self.seen.append(cached_features)
        return super().forward(observations, actions)


class _LocalModules:
    def prepare(self, module, *, name, publish=False, frozen_submodules=()):
        return module

    def register_optimizer(self, name, optimizer, *, module):
        # Local modules need no optimizer bookkeeping.
        pass


def test_ppo_learner_passes_cached_features_to_the_model():
    torch.manual_seed(0)
    model = _CachedGaussian()
    observations = torch.randn(8, 5)
    with torch.no_grad():
        actions, old_log_probs, old_values = model.act(observations)
    batch = LearnerBatch(
        tensors={
            "observations": observations.numpy(),
            "actions": actions.numpy(),
            "old_log_probs": old_log_probs.numpy(),
            "advantages": np.linspace(-1, 1, 8, dtype=np.float32),
            "returns": (old_values + 0.1).numpy(),
            "old_values": old_values.numpy(),
            CACHED_FEATURES_KEY: {
                "tokens" + BF16_SUFFIX: np.arange(16, dtype=np.int16).reshape(8, 2)
            },
        },
        samples=8,
    )
    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _LocalModules(),
        lambda: model,
        PPOConfig(epochs=1, minibatch_size=4, seed=1),
    )
    metrics = learner.train_step(batch)
    assert metrics["minibatches"] == pytest.approx(2.0)
    assert len(model.seen) >= 2
