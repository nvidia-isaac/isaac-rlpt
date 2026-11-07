# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DSRL latent is one quantity in one scale from the actor to the critic.

A steering actor samples ``tanh(u)`` in normalized ``[-1, 1]`` coordinates,
the rollout policy executes ``bound * tanh(u)`` through the frozen decoder and
replays ``tanh(u)``; the SAC core then scores the replayed latent, the actor's
fresh sample and the TD-target sample in those same normalized coordinates.
These tests pin that identity end to end with the real transformer actor and
the real rollout policy, and check that the critic's gradient reaches the
actor's mean head (the temperature and Q terms are the only forces on it).
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.envs import LatentSpaceEnv
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
)
from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP
from isaac_rl_posttraining.runtimes.steering_policy import SteeringActionChunkPolicy

OBS_DIM, ACT_DIM, CHUNK, BOUND = 7, 5, 4, 2.0


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _spaces(bound: float = BOUND):
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
    )
    act_space = gym.spaces.Box(-bound, bound, (ACT_DIM,), dtype=np.float32)
    return obs_space, act_space


def _make_actor(bound: float = BOUND, log_std_init: float = 0.0) -> TransformerActor:
    obs_space, act_space = _spaces(bound)
    return TransformerActor(
        TransformerActorConfig(
            observation_space=obs_space,
            action_space=act_space,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            chunk_size=CHUNK,
            feature_hidden_dims=[],
            output_hidden_dims=[16],
            featurizer={"state": [16]},
            use_language_embeddings=False,
            use_tanh_output=True,
            log_std_init=log_std_init,
            transformer_dropout=0.0,
            dropout_rate=0.0,
            positional_dropout=0.0,
        )
    )


class _LatentVLA:
    """A frozen decoder stand-in that returns the latent it was given as the action chunk."""

    supports_batched_infer = False

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def infer(self, observations, noise):
        latent = np.asarray(noise, dtype=np.float32)
        self.received.append(latent.copy())
        return {"actions": latent}


def _policy(
    actor: TransformerActor, warmup_exploration: str = "uniform"
) -> tuple[SteeringActionChunkPolicy, _LatentVLA]:
    obs_space, act_space = _spaces(float(actor.action_space.high[0]))
    sampler = LatentSpaceEnv(
        obs_space,
        act_space,
        base_model=None,
        noise_dim=ACT_DIM,
        chunk_size=CHUNK,
        action_bound=float(actor.action_space.high[0]),
        warmup_exploration=warmup_exploration,
        use_vlm_features=False,
    )
    base_model = _LatentVLA()
    policy = SteeringActionChunkPolicy(
        actor=actor,
        base_model=base_model,
        latent_sampler=sampler,
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=1,
    )
    return policy, base_model


def _observation() -> dict[str, np.ndarray]:
    return {"state": np.arange(OBS_DIM, dtype=np.float32)[None, :] / OBS_DIM}


# ---------------------------------------------------------------------------
# A. One latent, one scale: actor sample -> executed latent -> replayed latent
# ---------------------------------------------------------------------------


def test_policy_latent_executes_at_the_bound_and_replays_normalized():
    """Executed latent == bound * tanh(u); replay == tanh(u), the actor's own sample."""
    actor = _make_actor()
    policy, base_model = _policy(actor)
    obs = _observation()

    torch.manual_seed(123)
    chunk = policy.action_chunk(obs, mode="train")
    executed = np.asarray(base_model.received[-1])
    replay = np.asarray(chunk.learner_action)

    assert chunk.source == "policy" and chunk.learner_action_kind == "latent"
    assert executed.shape == (1, CHUNK, ACT_DIM) and replay.shape == (1, CHUNK, ACT_DIM)
    # The decoder received the sample mapped onto the [-bound, bound] box ...
    np.testing.assert_allclose(executed, BOUND * replay, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(chunk.environment_actions), executed)
    assert np.abs(replay).max() <= 1.0 and np.abs(executed).max() <= BOUND
    # ... and the replayed latent is exactly the actor's tanh sample for the same draw.
    torch.manual_seed(123)
    tensor_obs = {"state": torch.as_tensor(obs["state"])}
    with torch.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(tensor_obs)
        sample = actor.action_dist.actions_from_params(mean, log_std)
    np.testing.assert_allclose(replay, sample.numpy(), rtol=0, atol=1e-6)
    # The density SAC assigns to the replayed latent is the density of that sample.
    with torch.no_grad():
        dist = actor.action_dist.proba_distribution(mean, log_std)
        from_replay = dist.log_prob(torch.as_tensor(replay))
        from_sample = dist.log_prob(sample, actor.action_dist.gaussian_actions)
    assert torch.isfinite(from_replay).all()
    torch.testing.assert_close(from_replay, from_sample, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("warmup_exploration", ["uniform", "normal"])
def test_warmup_latent_replays_as_executed_over_bound(warmup_exploration):
    """Warm-up latents replay as executed / bound, unclamped, so they round-trip exactly."""
    actor = _make_actor()
    policy, base_model = _policy(actor, warmup_exploration=warmup_exploration)
    executed_all = []
    replay_all = []
    for _ in range(20):
        chunk = policy.action_chunk(_observation(), mode="warmup")
        assert chunk.source == "warmup"
        executed_all.append(np.asarray(base_model.received[-1]))
        replay_all.append(np.asarray(chunk.learner_action))
    executed = np.concatenate(executed_all)
    replay = np.concatenate(replay_all)
    np.testing.assert_allclose(replay * BOUND, executed, rtol=0, atol=1e-6)
    assert np.abs(executed).max() <= DEFAULT_LATENT_CLIP
    reconstructed = actor.unnormalize_action(torch.as_tensor(replay)).numpy()
    np.testing.assert_allclose(reconstructed, executed, rtol=0, atol=1e-6)
    if warmup_exploration == "uniform":
        assert np.abs(replay).max() <= 1.0
    else:
        # N(0, 1) exploration leaves the actor's reachable box; the replay keeps it.
        assert np.abs(replay).max() > 1.0


class _RecordingCritic(nn.Module):
    """Q(obs, action) == w . mean(action); records every action batch it scores."""

    def __init__(self, slope: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(slope)))
        self.seen: list[torch.Tensor] = []

    def forward(self, obs, action):
        self.seen.append(action.detach().clone())
        return self.w * action.reshape(action.shape[0], -1).mean(dim=-1, keepdim=True)

    def reset_parameters(self):
        # Keep the test-set weights; nothing to re-initialize.
        pass


class _ReplaySampler:
    """A scripted replay of latents the rollout policy produced (normalized coordinates)."""

    def __init__(self, actions: torch.Tensor):
        self.actions = actions

    def can_sample(self, batch_size) -> bool:
        return True

    def sample(self, batch_size):
        b = int(batch_size)
        idx = torch.arange(b) % self.actions.shape[0]
        return {
            "obs": {"state": torch.randn(b, OBS_DIM)},
            "action": self.actions[idx].clone(),
            "reward": torch.zeros(b),
            "next_obs": {"state": torch.randn(b, OBS_DIM)},
            "done": torch.zeros(b),
            "truncated": torch.zeros(b),
        }


def _make_sac(actor, critic, sampler, **overrides) -> SAC:
    kwargs = {
        "actor": actor,
        "critic": critic,
        "sampler": sampler,
        "learning_starts": 0,
        "ent_coef": "auto",
        "target_entropy": 0,
        "target_update_interval": 1,
        "gamma": 0.9,
        "compute_chunked_gamma": False,
        "tau": 0.005,
        "batch_size": 8,
        "n_critics_to_sample": 2,
        "num_critics": 2,
        "critic_reduction": "min",
        "num_critic_updates_per_actor_update": 1,
        "num_updates_per_train_step": 1,
        "train_critic_with_entropy": False,
        "train_actor_with_entropy": True,
        "train_actor_with_kl_divergence": False,
        "pooled_critic_features": False,
        "actor_optimizer_lr": 1e-3,
        "critic_optimizer_lr": 0.0,
        "ent_coef_lr": 1e-2,
    }
    kwargs.update(overrides)
    return SAC(SACConfig(**kwargs))


def test_critic_scores_replayed_actor_and_target_latents_in_one_coordinate_system():
    """Current Q reads the replayed tanh latent; actor and TD-target Q read fresh tanh samples of the same actor."""
    actor = _make_actor()
    policy, _ = _policy(actor)
    replayed = torch.cat(
        [
            torch.as_tensor(
                policy.action_chunk(_observation(), mode="train").learner_action
            )
            for _ in range(8)
        ]
    )
    critic = _RecordingCritic()
    sac = _make_sac(actor, critic, _ReplaySampler(replayed))
    online = sac.critic.critics
    target = sac.critic_target.critics
    for member in (*online, *target):
        member.seen.clear()

    sac.train_step(rollout_step=10)

    # TD target: the target ensemble scored the next-state action sampled by
    # the same actor (a tanh sample, hence inside the normalized box).
    target_actions = [a for member in target for a in member.seen]
    assert target_actions, "target critics were not evaluated"
    for action in target_actions:
        assert action.shape[1:] == (CHUNK, ACT_DIM)
        assert action.abs().max() <= 1.0
    # Online ensemble: first the replayed latent (current Q), then the actor's
    # fresh sample (actor objective) -- both in normalized coordinates.
    for member in online:
        assert len(member.seen) == 2
        # The core shuffles the draw, so compare rows as a set: every scored
        # latent is one of the replayed ones, bit for bit.
        distances = (
            (member.seen[0][:, None] - replayed[None, :]).abs().flatten(2).amax(dim=2)
        )
        assert torch.equal(distances.amin(dim=1), torch.zeros(replayed.shape[0]))
        assert member.seen[1].abs().max() <= 1.0
        assert member.seen[1].shape == replayed.shape
    # No scale between the three: unnormalizing any of them lands on the decoder's box.
    for action in (*target_actions, *(m.seen[1] for m in online)):
        assert actor.unnormalize_action(action).abs().max() <= BOUND


# ---------------------------------------------------------------------------
# D. The critic's gradient reaches the actor; the temperature is the other force
# ---------------------------------------------------------------------------


def test_q_gradient_moves_the_actor_mean_when_the_temperature_is_zero():
    """With alpha == 0 and Q increasing in the action, the actor's mean latent climbs."""
    actor = _make_actor()
    critic = _RecordingCritic(slope=1.0)
    sac = _make_sac(
        actor,
        critic,
        _ReplaySampler(torch.zeros(8, CHUNK, ACT_DIM)),
        ent_coef=0.0,
        actor_optimizer_lr=1e-2,
    )
    obs = {"state": torch.randn(16, OBS_DIM)}
    with torch.no_grad():
        before = torch.tanh(actor.get_action_dist_params(obs)[0]).mean().item()
    grads_seen = False
    for _ in range(20):
        metrics = sac.train_step(rollout_step=10)
        grad = actor.mean_processor[-1].weight.grad
        if grad is not None and grad.abs().sum() > 0:
            grads_seen = True
    with torch.no_grad():
        after = torch.tanh(actor.get_action_dist_params(obs)[0]).mean().item()
    assert grads_seen, "the actor objective produced no gradient on the mean head"
    assert after > before + 0.05, (before, after)
    assert float(metrics["actor_chosen_q_mean"]) > 0.0


def test_unreachable_entropy_target_turns_the_temperature_into_a_clock():
    """While the actor's entropy stays above the target, log alpha falls by ent_coef_lr per update.

    A 5-dim tanh-Gaussian at sigma == 1 has ~3.35 nats per flow step; with
    target_entropy == 0 the temperature loss has one sign, so Adam's
    normalized step lowers log alpha by exactly the learning rate each update
    whatever the size of the gap (the same mechanism at 35 dims: alpha ==
    exp(-ent_coef_lr * updates) in the G1 campaign).
    """
    actor = _make_actor()
    critic = _RecordingCritic(slope=0.0)
    lr = 1e-2
    sac = _make_sac(
        actor,
        critic,
        _ReplaySampler(torch.zeros(8, CHUNK, ACT_DIM)),
        ent_coef_lr=lr,
        actor_optimizer_lr=0.0,
    )
    log_alphas = []
    for _ in range(20):
        metrics = sac.train_step(rollout_step=10)
        log_alphas.append(float(sac.log_ent_coef.detach()))
    assert float(metrics["actor_log_pis_mean"]) < -2.0  # entropy far above the target
    steps = np.diff([0.0, *log_alphas])
    np.testing.assert_allclose(steps, -lr, rtol=0.05)
    assert float(metrics["ent_coef"]) == pytest.approx(np.exp(-19 * lr), rel=0.05)


def test_actor_diagnostics_report_gradient_scales_without_changing_the_update():
    """diagnostics_interval records the Q/entropy gradient scales and critic responses, and leaves the update as is."""

    def build(interval: int) -> SAC:
        torch.manual_seed(7)
        actor = _make_actor()
        critic = _RecordingCritic(slope=1.0)
        return _make_sac(
            actor,
            critic,
            _ReplaySampler(torch.zeros(8, CHUNK, ACT_DIM)),
            diagnostics_interval=interval,
        )

    plain = build(0)
    instrumented = build(1)
    # The core shuffles its draw with NumPy's stream and samples with torch's.
    torch.manual_seed(11)
    np.random.seed(11)
    plain_metrics = plain.train_step(rollout_step=10)
    torch.manual_seed(11)
    np.random.seed(11)
    metrics = instrumented.train_step(rollout_step=10)

    assert not any(key.startswith("diag/") for key in plain_metrics)
    for name in (
        "dq_da_rms",
        "actor_grad_norm_q",
        "actor_grad_norm_entropy",
        "q_actor",
        "q_replay",
        "q_actor_minus_replay",
        "q_prior_latent",
        "q_uniform_latent",
        "q_std_over_latents",
        "q_std_over_obs",
    ):
        assert f"diag/{name}" in metrics, name
    # Q == mean(action): dQ/da is 1 / (CHUNK * ACT_DIM) per element, the same on every sample.
    assert float(metrics["diag/dq_da_rms"]) == pytest.approx(
        1.0 / (CHUNK * ACT_DIM), rel=1e-4
    )
    assert float(metrics["diag/actor_grad_norm_q"]) > 0.0
    assert float(metrics["diag/actor_grad_norm_entropy"]) > 0.0
    assert float(metrics["diag/q_replay"]) == pytest.approx(
        0.0, abs=1e-6
    )  # replayed latents are zeros
    assert float(metrics["diag/q_std_over_latents"]) > 0.0
    assert float(metrics["diag/q_std_over_obs"]) == pytest.approx(
        0.0, abs=1e-6
    )  # Q ignores the state
    # The instrumented update produced the same actor as the plain one.
    for (name, a), (_, b) in zip(
        plain.actor.state_dict().items(),
        instrumented.actor.state_dict().items(),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(("interval", "expected_calls"), [(1, 20), (10, 2), (7, 3)])
def test_actor_diagnostics_fire_every_interval_actor_updates(
    monkeypatch, interval, expected_calls
):
    """With the shipped cadence (4 critic slices per actor update, 10 gradient steps per call) the
    diagnostics fire once per ``interval`` actor updates: 20 actor updates over two calls give
    20 / 2 / 3 firings for intervals 1 / 10 / 7 (updates 0 and 10; 0, 7 and 14)."""
    import isaac_rl_posttraining.algorithms.sac.sac as sac_module

    calls = []
    original = sac_module.record_actor_diagnostics

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sac_module, "record_actor_diagnostics", counting)
    actor = _make_actor()
    sac = _make_sac(
        actor,
        _RecordingCritic(slope=1.0),
        _ReplaySampler(torch.zeros(8, CHUNK, ACT_DIM)),
        diagnostics_interval=interval,
        num_critic_updates_per_actor_update=4,
        num_updates_per_train_step=10,
        num_critics=4,
    )
    for _ in range(2):
        metrics = sac.train_step(rollout_step=10)
    assert len(calls) == expected_calls
    assert "diag/dq_da_rms" in metrics
