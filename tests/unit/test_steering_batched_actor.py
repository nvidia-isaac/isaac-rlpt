# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One actor call for all sessions decides as one call per session does."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaac_rl_posttraining.modules.mlp import MLPActor, MLPActorConfig
from isaac_rl_posttraining.runtimes.steering_policy import SteeringActionChunkPolicy

STATE_DIM, LATENT_DIM, SESSIONS = 3, 2, (0, 1, 2, 3)


class _VLA:
    """Decodes the latent itself, so the chunk shows exactly what the actor chose."""

    supports_batched_infer = True

    def infer(self, observations, noise):
        return {"actions": np.asarray(noise)[:, None, :]}

    def infer_batch(self, observations, noise):
        return {"actions": np.asarray(noise)[:, None, :]}


class _Sampler:
    def sample_action(self) -> np.ndarray:
        return np.zeros((1, LATENT_DIM), dtype=np.float32)


def _actor() -> MLPActor:
    torch.manual_seed(7)
    return MLPActor(
        MLPActorConfig(
            observation_space=gym.spaces.Dict(
                {"state": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)}
            ),
            action_space=gym.spaces.Box(-2.0, 2.0, (LATENT_DIM,), np.float32),
            hidden_dims=(16, 16),
            use_tanh_output=True,
            log_std_init=-0.5,
        )
    ).eval()


def _policy(actor: MLPActor, *, batched: bool, deterministic: bool = True):
    return SteeringActionChunkPolicy(
        actor=actor,
        base_model=_VLA(),
        latent_sampler=_Sampler(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
        deterministic_evaluation=deterministic,
        batched_actor=batched,
    )


def _observations():
    return tuple(
        {"state": np.asarray([0.3 * s, -0.2 * s, 1.0 + s], dtype=np.float32)}
        for s in SESSIONS
    )


def _latents(policy, mode: str, rounds: int = 1) -> np.ndarray:
    observations = _observations()
    draws = []
    for _ in range(rounds):
        chunks = policy.action_chunks(
            observations, mode=(mode,) * len(SESSIONS), session_ids=SESSIONS
        )
        draws.append(np.stack([chunk.environment_actions[0, 0] for chunk in chunks]))
    return np.stack(draws)


def test_mean_evaluation_actions_are_identical_batched_and_unbatched() -> None:
    actor = _actor()
    unbatched = _latents(_policy(actor, batched=False), "eval")
    batched = _latents(_policy(actor, batched=True), "eval")
    np.testing.assert_allclose(batched, unbatched, rtol=0.0, atol=1e-6)


def test_seeded_evaluation_sessions_keep_their_own_noise_streams() -> None:
    actor = _actor()
    results = []
    for batched in (False, True):
        policy = _policy(actor, batched=batched, deterministic=False)
        for session_id in SESSIONS:
            policy.reset_episode(session_id, seed=100 + session_id)
        results.append(_latents(policy, "eval", rounds=3))
    np.testing.assert_allclose(results[1], results[0], rtol=0.0, atol=1e-6)
    # Each session's stream is its own: no two sessions share draws.
    assert len({tuple(row) for row in results[1][0]}) == len(SESSIONS)


def test_training_samples_follow_the_same_distribution() -> None:
    actor = _actor()
    samples = []
    for batched in (False, True):
        torch.manual_seed(0)
        samples.append(_latents(_policy(actor, batched=batched), "train", rounds=400))
    # Same per-session draw order from the process RNG, so the samples agree too.
    np.testing.assert_allclose(samples[1], samples[0], rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(
        samples[1].mean(axis=0), samples[0].mean(axis=0), atol=0.05
    )
    np.testing.assert_allclose(
        samples[1].std(axis=0), samples[0].std(axis=0), atol=0.05
    )
    assert samples[1].std(axis=0).min() > 0.01


def test_batched_actor_rejects_actors_with_per_session_state() -> None:
    class _Recurrent(MLPActor):
        def get_initial_state(self):
            return {"hidden": torch.zeros(1)}

    actor = _actor()
    actor.__class__ = _Recurrent
    with pytest.raises(ValueError, match="stateless"):
        _policy(actor, batched=True)

    class _PPOActor(torch.nn.Module):
        def sample_action_chunk(self, observation, *, deterministic):
            raise AssertionError("not reached")

    with pytest.raises(TypeError, match="PPO"):
        _policy(_PPOActor(), batched=True)
