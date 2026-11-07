# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the provider-neutral steering policy used by rollout runtimes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.runtimes.steering_policy import (
    SteeringActionChunkPolicy,
)


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def act(self, observation, actor_state, deterministic):
        del actor_state, deterministic
        batch_size = observation["state"].shape[0]
        action = torch.full((batch_size, 2), 0.5) + self.weight
        return action, None


class _Sampler:
    def sample_action(self) -> np.ndarray:
        return np.zeros((1, 2), dtype=np.float32)


class _VLA:
    supports_batched_infer = True

    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0
        self.batch_sizes = []

    def infer(self, observations, noise):
        self.single_calls += 1
        assert np.asarray(noise).shape[0] == 1
        return {"actions": np.asarray(noise)[:, None, :]}

    def infer_batch(self, observations, noise):
        self.batch_calls += 1
        self.batch_sizes.append(len(noise))
        return {"actions": np.asarray(noise)[:, None, :]}


class _RandomVLA(_VLA):
    def infer(self, observations, noise):
        del observations
        self.single_calls += 1
        random_offset = torch.rand(1).item()
        return {"actions": np.asarray(noise)[:, None, :] + random_offset}


class _PreparedRandomVLA(_RandomVLA):
    def prepare_seeded_batch_noise(self, noise):
        return np.asarray(noise) + torch.rand(1).item()

    def infer_prepared_batch(self, observations, initial_noise):
        del observations
        self.batch_calls += 1
        self.batch_sizes.append(len(initial_noise))
        return {"actions": np.asarray(initial_noise)[:, None, :]}


class _NativePreparedRandomVLA(_VLA):
    def infer(self, observations, noise):
        del observations
        assert noise is None
        self.single_calls += 1
        sample = torch.rand(1, 1, 2).numpy()
        return {"actions": sample}

    def prepare_seeded_native_noise(self):
        return torch.rand(1, 1, 2).numpy()

    def infer_prepared_batch(self, observations, initial_noise):
        del observations
        self.batch_calls += 1
        self.batch_sizes.append(len(initial_noise))
        return {"actions": np.asarray(initial_noise)}


class _PPOActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.deterministic_calls = []

    def sample_action_chunk(self, observation, *, deterministic):
        self.deterministic_calls.append(bool(deterministic))
        batch_size = observation["state"].shape[0]
        action = torch.zeros(batch_size, 2) + self.weight
        metadata = torch.zeros(batch_size)
        return action, action, metadata, metadata


class _RandomPPOActor(_PPOActor):
    def sample_action_chunk(self, observation, *, deterministic):
        self.deterministic_calls.append(bool(deterministic))
        batch_size = observation["state"].shape[0]
        action = torch.rand(batch_size, 2)
        metadata = torch.zeros(batch_size)
        return action, action, metadata, metadata


def _policy(base_model: _VLA) -> SteeringActionChunkPolicy:
    return SteeringActionChunkPolicy(
        actor=_Actor(),
        base_model=base_model,
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )


class _ValuedActor(_Actor):
    def values(self, observation):
        return torch.full((observation["state"].shape[0],), 0.25)


def test_successors_are_prepared_with_or_without_a_value_head() -> None:
    """A SAC-style actor contributes the prepared successor only; a valued actor also scores it."""
    observations = (
        {"state": np.asarray([1.0], dtype=np.float32)},
        {"state": np.asarray([2.0], dtype=np.float32)},
    )
    prepared = _policy(_VLA()).prepare_successors(observations, session_ids=(3, 7))
    assert len(prepared) == 2
    for observation, value in prepared:
        assert "state" in observation
        assert value.shape == (1,) and np.isnan(value).all()

    valued_policy = SteeringActionChunkPolicy(
        actor=_ValuedActor(),
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )
    valued = valued_policy.prepare_successors(observations, session_ids=(3, 7))
    assert [float(value[0]) for _, value in valued] == [0.25, 0.25]


def test_shared_steering_policy_uses_one_batched_decode() -> None:
    base_model = _VLA()
    policy = _policy(base_model)

    chunks = policy.action_chunks(
        (
            {"state": np.asarray([1.0], dtype=np.float32)},
            {"state": np.asarray([2.0], dtype=np.float32)},
        ),
        mode=("train", "train"),
        session_ids=(3, 7),
    )

    assert len(chunks) == 2
    assert base_model.batch_calls == 1
    assert base_model.single_calls == 0
    assert all(chunk.batch_size == 1 for chunk in chunks)


def test_shared_steering_policy_bounds_decode_microbatches() -> None:
    base_model = _VLA()
    policy = SteeringActionChunkPolicy(
        actor=_Actor(),
        base_model=base_model,
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        max_decode_batch_size=2,
    )

    chunks = policy.action_chunks(
        tuple({"state": np.asarray([index], dtype=np.float32)} for index in range(5)),
        mode=("train",) * 5,
        session_ids=tuple(range(5)),
    )

    assert len(chunks) == 5
    assert base_model.batch_sizes == [2, 2, 1]
    assert base_model.single_calls == 0


def test_shared_steering_policy_falls_back_to_independent_decodes() -> None:
    base_model = _VLA()
    base_model.supports_batched_infer = False
    policy = _policy(base_model)

    policy.action_chunks(
        (
            {"state": np.asarray([1.0], dtype=np.float32)},
            {"state": np.asarray([2.0], dtype=np.float32)},
        ),
        mode=("warmup", "warmup"),
        session_ids=(0, 1),
    )

    assert base_model.batch_calls == 0
    assert base_model.single_calls == 2


def test_evaluation_randomness_is_stable_per_environment_seed() -> None:
    policy = _policy(_RandomVLA())
    observations = {
        3: {"state": np.asarray([1.0], dtype=np.float32)},
        7: {"state": np.asarray([2.0], dtype=np.float32)},
    }

    def evaluate(order):
        for session_id in order:
            policy.reset_episode(session_id=session_id, seed=100 + session_id)
        chunks = policy.action_chunks(
            tuple(observations[session_id] for session_id in order),
            mode=("eval",) * len(order),
            session_ids=order,
        )
        return {
            session_id: chunk.execution_actions.copy()
            for session_id, chunk in zip(order, chunks, strict=True)
        }

    forward = evaluate((3, 7))
    reverse = evaluate((7, 3))

    assert np.array_equal(forward[3], reverse[3])
    assert np.array_equal(forward[7], reverse[7])


def test_seeded_evaluation_batches_without_coupling_environment_rngs() -> None:
    base_model = _PreparedRandomVLA()
    policy = _policy(base_model)
    observations = {
        3: {"state": np.asarray([1.0], dtype=np.float32)},
        7: {"state": np.asarray([2.0], dtype=np.float32)},
    }

    def evaluate(order):
        for session_id in order:
            policy.reset_episode(session_id=session_id, seed=100 + session_id)
        chunks = policy.action_chunks(
            tuple(observations[session_id] for session_id in order),
            mode=("eval",) * len(order),
            session_ids=order,
        )
        return {
            session_id: chunk.execution_actions.copy()
            for session_id, chunk in zip(order, chunks, strict=True)
        }

    forward = evaluate((3, 7))
    reverse = evaluate((7, 3))

    assert np.array_equal(forward[3], reverse[3])
    assert np.array_equal(forward[7], reverse[7])
    assert base_model.batch_calls == 2
    assert base_model.single_calls == 0


def test_seeded_evaluation_decodes_per_session_when_the_policy_cannot_batch_it() -> (
    None
):
    """A GR00T N1.5 policy has no prepared-noise decode; a batched evaluation falls back to one decode per session."""

    class _UnpreparedVLA(_PreparedRandomVLA):
        supports_seeded_batch_decode = False

        def prepare_seeded_batch_noise(self, noise):
            raise AssertionError("the policy cannot prepare seeded batch noise")

    base_model = _UnpreparedVLA()
    policy = _policy(base_model)
    order = (3, 7)
    for session_id in order:
        policy.reset_episode(session_id=session_id, seed=100 + session_id)
    chunks = policy.action_chunks(
        tuple({"state": np.asarray([float(s)], dtype=np.float32)} for s in order),
        mode=("eval",) * len(order),
        session_ids=order,
    )

    assert len(chunks) == 2
    assert base_model.single_calls == 2 and base_model.batch_calls == 0


def test_seeded_native_evaluation_matches_scalar_noise_in_one_batched_decode() -> None:
    observations = {
        3: {"state": np.asarray([1.0], dtype=np.float32)},
        7: {"state": np.asarray([2.0], dtype=np.float32)},
    }

    scalar_base_model = _NativePreparedRandomVLA()
    scalar = SteeringActionChunkPolicy(
        actor=_Actor(),
        base_model=scalar_base_model,
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        native=True,
    )
    scalar_actions = {}
    for session_id, observation in observations.items():
        scalar.reset_episode(session_id=session_id, seed=100 + session_id)
        scalar_actions[session_id] = scalar.action_chunks(
            (observation,), mode=("eval",), session_ids=(session_id,)
        )[0].execution_actions

    batched_base_model = _NativePreparedRandomVLA()
    batched = SteeringActionChunkPolicy(
        actor=_Actor(),
        base_model=batched_base_model,
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        native=True,
    )
    for session_id in observations:
        batched.reset_episode(session_id=session_id, seed=100 + session_id)
    global_state = torch.get_rng_state().clone()
    chunks = batched.action_chunks(
        tuple(observations.values()),
        mode=("eval", "eval"),
        session_ids=tuple(observations),
    )

    for session_id, chunk in zip(observations, chunks, strict=True):
        np.testing.assert_allclose(
            chunk.execution_actions,
            scalar_actions[session_id],
            rtol=1e-6,
            atol=1e-6,
        )
    assert batched_base_model.batch_sizes == [2]
    assert batched_base_model.single_calls == 0
    assert torch.equal(torch.get_rng_state(), global_state)


def test_ppo_evaluation_omits_unused_behavior_metadata() -> None:
    actor = _PPOActor()
    policy = SteeringActionChunkPolicy(
        actor=actor,
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )

    evaluation_chunk = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)},
        mode="eval",
    )
    training_chunk = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)},
        mode="train",
    )

    assert evaluation_chunk.behavior_log_prob is None
    assert evaluation_chunk.behavior_value is None
    assert training_chunk.behavior_log_prob is not None
    assert training_chunk.behavior_value is not None
    assert actor.deterministic_calls == [True, False]


def test_ppo_evaluation_can_sample_from_seeded_policy() -> None:
    actor = _RandomPPOActor()
    policy = SteeringActionChunkPolicy(
        actor=actor,
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        sample_evaluation_actions=True,
    )

    policy.reset_episode(seed=123)
    first = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)},
        mode="eval",
    )
    policy.reset_episode(seed=123)
    replay = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)},
        mode="eval",
    )

    assert actor.deterministic_calls == [False, False]
    assert np.array_equal(first.execution_actions, replay.execution_actions)
    assert first.behavior_log_prob is None
    assert first.behavior_value is None


def test_set_policy_state_decodes_bfloat16_raw_bits() -> None:
    from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors

    policy = _policy(_VLA())
    published = {"weight": torch.tensor(1.5).to(torch.bfloat16)}

    version = policy.set_policy_state(policy_state_from_tensors(published, version=1))

    assert version == 1
    assert policy.actor.weight.dtype == torch.float32
    assert policy.actor.weight.item() == pytest.approx(1.5)


class _RecordingActor(_Actor):
    def __init__(self) -> None:
        super().__init__()
        self.deterministic_calls = []

    def act(self, observation, actor_state, deterministic):
        self.deterministic_calls.append(bool(deterministic))
        return super().act(observation, actor_state, deterministic)


class _WideSampler:
    def sample_action(self) -> np.ndarray:
        return np.asarray([[5.0, -5.0]], dtype=np.float32)


def test_evaluation_samples_the_actor_unless_deterministic() -> None:
    observation = {"state": np.asarray([1.0], dtype=np.float32)}
    sampling_actor = _RecordingActor()
    sampling = SteeringActionChunkPolicy(
        actor=sampling_actor,
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )
    mean_actor = _RecordingActor()
    deterministic = SteeringActionChunkPolicy(
        actor=mean_actor,
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        deterministic_evaluation=True,
    )

    sampled = sampling.action_chunk(observation, mode="eval")
    sampling.action_chunk(observation, mode="train")
    deterministic.action_chunk(observation, mode="eval")

    assert sampling_actor.deterministic_calls == [False, False]
    assert mean_actor.deterministic_calls == [True]
    assert sampled.source == "policy"
    assert sampled.learner_action is not None


def test_warmup_latents_are_clipped_to_the_base_model_injection_bound() -> None:
    from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP

    base_model = _VLA()
    policy = SteeringActionChunkPolicy(
        actor=_Actor(),
        base_model=base_model,
        latent_sampler=_WideSampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )

    chunk = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)}, mode="warmup"
    )

    assert chunk.source == "warmup"
    np.testing.assert_array_equal(
        chunk.learner_action,
        np.asarray([[DEFAULT_LATENT_CLIP, -DEFAULT_LATENT_CLIP]], dtype=np.float32),
    )
    assert chunk.learner_action.dtype == np.float32


class _NormalizingActor(_Actor):
    """An actor whose latents live in [-2, 2] and are replayed in [-1, 1]."""

    normalize_actions = True

    def normalize_action(
        self, action: torch.Tensor, *, clamp: bool = True
    ) -> torch.Tensor:
        normalized = action / 2.0
        return torch.clamp(normalized, -1.0, 1.0) if clamp else normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return normalized_action * 2.0


def test_warmup_latents_are_replayed_in_the_actors_normalized_coordinates() -> None:
    policy = SteeringActionChunkPolicy(
        actor=_NormalizingActor(),
        base_model=_VLA(),
        latent_sampler=_WideSampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )

    chunk = policy.action_chunk(
        {"state": np.asarray([1.0], dtype=np.float32)}, mode="warmup"
    )

    assert chunk.source == "warmup"
    # The replay coordinates round-trip to the exact latent that executed,
    # including exploration outside the learned actor's reachable range.
    np.testing.assert_array_equal(
        np.asarray(chunk.environment_actions).reshape(-1),
        np.asarray([3.0, -3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        chunk.learner_action, np.asarray([[1.5, -1.5]], dtype=np.float32)
    )
    reconstructed = policy.actor.unnormalize_action(
        torch.as_tensor(chunk.learner_action)
    ).numpy()
    np.testing.assert_array_equal(
        reconstructed,
        np.asarray([[3.0, -3.0]], dtype=np.float32),
    )


def test_native_policy_evaluates_the_frozen_base_model_without_latents() -> None:
    class _NativeVLA(_VLA):
        def __init__(self) -> None:
            super().__init__()
            self.noises = []

        def infer(self, observations, noise):
            self.noises.append(noise)
            return {"actions": np.ones((1, 1, 2), dtype=np.float32)}

    base_model = _NativeVLA()
    actor = _RecordingActor()
    policy = SteeringActionChunkPolicy(
        actor=actor,
        base_model=base_model,
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        native=True,
    )
    observation = {"state": np.asarray([1.0], dtype=np.float32)}

    chunk = policy.action_chunk(observation, mode="eval")
    prepared = policy.prepare_observation(observation)

    assert base_model.noises == [None]
    assert actor.deterministic_calls == []
    assert chunk.source == "native"
    assert chunk.learner_action is None
    assert set(chunk.learner_observation) == {"state"}
    assert set(prepared) == {"state"}
    with pytest.raises(ValueError, match="native"):
        policy.action_chunk(observation, mode="train")
    with pytest.raises(ValueError, match="native"):
        policy.action_chunk(observation, mode="warmup")


def test_ppo_actor_rejects_warmup_decisions() -> None:
    policy = SteeringActionChunkPolicy(
        actor=_PPOActor(),
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )
    with pytest.raises(ValueError, match="warm-up"):
        policy.action_chunk(
            {"state": np.asarray([1.0], dtype=np.float32)}, mode="warmup"
        )
