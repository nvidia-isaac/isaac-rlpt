# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan §3 file 4 — SAC core update-loop tests.

Pins the generic SAC machinery in ``isaac_rl_posttraining.algorithms.sac``
against the canonical DSRL hyperparameters (see
``test_config.py::test_canonical_sac_hyperparameters``): target-net Polyak
updates, entropy-coefficient autotuning, randomized-ensemble (REDQ-style)
target sampling with min-reduction actor pessimism, the 10x(4-critic:1-actor)
update cadence, learning-starts gating, batched 64x4 sampling, and the
chunk-discounted TD target ``r + gamma**K * Q'``.

Networks are CPU-sized: a tiny real TransformerActor plus constant-valued stub
critics and a scripted stub buffer, so every TD quantity is exact.
"""

from __future__ import annotations

from unittest.mock import Mock

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
from isaac_rl_posttraining.utils.networks import polyak_update

OBS_DIM, ACT_DIM, CHUNK = 7, 5, 4
# Canonical values pinned in test_config.py::test_canonical_sac_hyperparameters.
CANONICAL_LEARNING_STARTS = 1200
CANONICAL_BATCH_SIZE = 64
CANONICAL_NUM_CRITICS = 4
CANONICAL_N_SAMPLED = 2
CANONICAL_CRITIC_PER_ACTOR = 4
CANONICAL_UPDATES_PER_TRAIN_STEP = 10


class _StubCritic(nn.Module):
    """Q(obs, action) == a constant per-critic parameter; records batch sizes."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.v = nn.Parameter(torch.tensor(float(value)))
        self.forward_batches: list[int] = []

    def forward(self, obs, action):
        batch = action.shape[0]
        self.forward_batches.append(int(batch))
        # Zero-weight action term keeps the graph without changing the value.
        return self.v.expand(batch, 1) + 0.0 * action.reshape(batch, -1).sum(
            -1, keepdim=True
        )

    def reset_parameters(self):  # ensemble copies keep their assigned value
        pass


class _StubSampler:
    """Scripted replay sampler returning a fixed-distribution batch."""

    def __init__(self, reward: float = 1.0, done: float = 0.0, truncated: float = 0.0):
        self.reward, self.done, self.truncated = reward, done, truncated
        self.requested: list[int] = []

    def can_sample(self, batch_size) -> bool:
        return True

    def sample(self, batch_size):
        self.requested.append(int(batch_size))
        b = int(batch_size)
        return {
            "obs": {"state": torch.randn(b, OBS_DIM)},
            "action": torch.randn(b, CHUNK, ACT_DIM).clamp(-1, 1),
            "reward": torch.full((b,), self.reward),
            "next_obs": {"state": torch.randn(b, OBS_DIM)},
            "done": torch.full((b,), self.done),
            "truncated": torch.full((b,), self.truncated),
        }


def _make_actor(log_std_init: float = 0.0):
    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
    )
    act_space = gym.spaces.Box(-2.0, 2.0, (ACT_DIM,), dtype=np.float32)
    config = TransformerActorConfig(
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
    return TransformerActor(config)


def _make_sac(**overrides):
    torch.manual_seed(0)
    kwargs = {
        "actor": _make_actor(overrides.pop("log_std_init", 0.0)),
        "critic": _StubCritic(1.0),
        "sampler": _StubSampler(),
        "learning_starts": 0,
        "ent_coef": "auto",
        "target_entropy": 0,
        "target_update_interval": 1,
        "gamma": 0.9,
        "compute_chunked_gamma": True,
        "tau": 0.005,
        "batch_size": 8,
        "n_critics_to_sample": CANONICAL_N_SAMPLED,
        "num_critics": CANONICAL_NUM_CRITICS,
        "critic_reduction": "min",
        "num_critic_updates_per_actor_update": CANONICAL_CRITIC_PER_ACTOR,
        "num_updates_per_train_step": 1,
        "train_critic_with_entropy": False,
        "train_actor_with_entropy": True,
        "train_actor_with_kl_divergence": False,
        "pooled_critic_features": False,  # exercise the explicit ensemble path
        "actor_optimizer_lr": 1e-4,
        "critic_optimizer_lr": 1e-4,
        "ent_coef_lr": 1e-2,
    }
    kwargs.update(overrides)
    return SAC(SACConfig(**kwargs))


class _TargetRecorder:
    """Records every TD target passed to mse_loss during train_step."""

    def __init__(self, monkeypatch):
        self.targets: list[torch.Tensor] = []
        original = torch.nn.functional.mse_loss

        def _recording(pred, target, *args, **kw):
            self.targets.append(target.detach().clone())
            return original(pred, target, *args, **kw)

        monkeypatch.setattr(torch.nn.functional, "mse_loss", _recording)

    def unique_values(self) -> set[float]:
        return {round(float(t.mean()), 5) for t in self.targets}


# ---------------------------------------------------------------------------
# Target networks
# ---------------------------------------------------------------------------


def test_target_soft_update_tau_exact():
    """theta' <- tau * theta + (1 - tau) * theta', verified numerically."""
    tau = 0.005
    params = [torch.full((3,), 2.0), torch.full((2, 2), -4.0)]
    targets = [torch.zeros(3), torch.full((2, 2), 6.0)]
    expected = [tau * p + (1.0 - tau) * t for p, t in zip(params, targets)]
    polyak_update(params, targets, tau)
    for got, want in zip(targets, expected):
        torch.testing.assert_close(got, want, atol=1e-8, rtol=0)
    # tau=1.0 is a hard copy
    hard_target = [torch.zeros(3)]
    polyak_update([torch.full((3,), 5.0)], hard_target, 1.0)
    torch.testing.assert_close(hard_target[0], torch.full((3,), 5.0))


def test_target_update_interval_respected():
    """Polyak fires only when the global update step hits the interval."""
    sac = _make_sac(
        num_critic_updates_per_actor_update=1, target_update_interval=3, tau=0.5
    )

    def snapshot():
        return [p.detach().clone() for p in sac.critic_target.parameters()]

    changed = []
    for _ in range(6):
        before = snapshot()
        sac.train_step(rollout_step=10_000)
        after = snapshot()
        changed.append(any(not torch.equal(a, b) for a, b in zip(before, after)))
    # update steps 0..5 with interval 3 -> polyak at steps 0 and 3 only
    assert changed == [True, False, False, True, False, False]


# ---------------------------------------------------------------------------
# Entropy autotuning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "log_std_init", [0.0, -6.0], ids=["high_entropy", "low_entropy"]
)
def test_ent_coef_autotune_direction(log_std_init):
    """With target_entropy=0, sign(d log_alpha) == sign(mean log_prob).

    High-entropy policy (log_prob < 0) -> alpha decreases; a near-deterministic
    policy (log_prob > 0) -> alpha increases to push entropy back up.
    """
    sac = _make_sac(log_std_init=log_std_init, target_entropy=0, ent_coef_lr=1e-2)
    assert float(sac.log_ent_coef.detach()) == pytest.approx(0.0)
    metrics = sac.train_step(rollout_step=10_000)
    delta = float(sac.log_ent_coef.detach())
    mean_log_prob = float(metrics["actor_log_pis_mean"])
    assert abs(mean_log_prob) > 0
    assert abs(delta) > 0
    assert np.sign(delta) == np.sign(mean_log_prob)


# ---------------------------------------------------------------------------
# REDQ-style randomized ensemble
# ---------------------------------------------------------------------------


def test_redq_samples_exactly_2_of_4():
    """Each critic update evaluates exactly n_critics_to_sample=2 of the 4 targets."""
    sac = _make_sac()
    for critic in sac.critic_target.critics:
        critic.forward_batches.clear()
    sac.train_step(rollout_step=10_000)
    per_critic = [len(c.forward_batches) for c in sac.critic_target.critics]
    assert len(per_critic) == CANONICAL_NUM_CRITICS
    # 1 gradient step x 4 critic updates x 2 sampled targets each = 8 evaluations
    assert sum(per_critic) == CANONICAL_CRITIC_PER_ACTOR * CANONICAL_N_SAMPLED
    # randperm indexing never evaluates one target critic more than once per update
    assert max(per_critic) <= CANONICAL_CRITIC_PER_ACTOR


@pytest.mark.parametrize(
    "reduction,expected", [("min", 0.0), ("mean", 1.5)], ids=["min", "mean"]
)
def test_redq_min_reduction(reduction, expected):
    """critic_reduction controls the actor's pessimism over the full ensemble."""
    # lr=0 keeps the stub critic values 0,1,2,3 exact through the critic updates.
    sac = _make_sac(critic_reduction=reduction, critic_optimizer_lr=0.0)
    for i, critic in enumerate(sac.critic.critics):
        with torch.no_grad():
            critic.v.fill_(float(i))
    metrics = sac.train_step(rollout_step=10_000)
    assert float(metrics["actor_chosen_q_mean"]) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Update cadence
# ---------------------------------------------------------------------------


def test_update_cadence_is_10_actor_updates_each_after_4_critic_updates():
    """One train_step performs num_updates_per_train_step actor updates, each preceded by 4 critic updates."""
    sac = _make_sac(num_updates_per_train_step=CANONICAL_UPDATES_PER_TRAIN_STEP)
    real_actor_step = sac.actor_optimizer.step
    real_critic_step = sac.critic_optimizer.step
    sac.actor_optimizer.step = Mock(side_effect=real_actor_step)
    sac.critic_optimizer.step = Mock(side_effect=real_critic_step)
    sac.train_step(rollout_step=10_000)
    actor_steps = sac.actor_optimizer.step.call_count
    critic_steps = sac.critic_optimizer.step.call_count
    assert actor_steps == CANONICAL_UPDATES_PER_TRAIN_STEP
    assert critic_steps == CANONICAL_CRITIC_PER_ACTOR * actor_steps


def test_learning_starts_blocks_updates():
    """rollout_step < learning_starts (1200) is a strict no-op."""
    sac = _make_sac(learning_starts=CANONICAL_LEARNING_STARTS)
    real_step = sac.actor_optimizer.step
    sac.actor_optimizer.step = Mock(side_effect=real_step)
    result = sac.train_step(rollout_step=CANONICAL_LEARNING_STARTS - 1)
    assert not result  # empty metrics: nothing trained
    assert sac.sampler.requested == []
    assert sac.actor_optimizer.step.call_count == 0
    # The boundary step trains.
    result = sac.train_step(rollout_step=CANONICAL_LEARNING_STARTS)
    assert result
    assert sac.actor_optimizer.step.call_count == 1


def test_batch_sampling_64x4_chunking():
    """The buffer is sampled once per gradient step at batch_size x 4, then
    sliced into 4 critic-update chunks of batch_size."""
    sac = _make_sac(batch_size=CANONICAL_BATCH_SIZE)
    for critic in sac.critic.critics:
        critic.forward_batches.clear()
    sac.train_step(rollout_step=10_000)
    assert sac.sampler.requested == [CANONICAL_BATCH_SIZE * CANONICAL_CRITIC_PER_ACTOR]
    # Current-Q evaluations (online ensemble) see exactly batch_size rows each.
    current_q_batches = [b for c in sac.critic.critics for b in c.forward_batches]
    assert current_q_batches, "online critics were never evaluated"
    assert set(current_q_batches) == {CANONICAL_BATCH_SIZE}


# ---------------------------------------------------------------------------
# Discounting and bootstrap masking
# ---------------------------------------------------------------------------


def test_gamma_chunk_discount_gamma_pow_exec_len(monkeypatch):
    """Chunked next-actions of length K discount the bootstrap by gamma**K.

    The canonical entry point equivalently folds the executed-chunk length into
    an effective gamma (gamma**action_exec_len) with per-chunk rewards.
    """
    gamma = 0.9
    recorder = _TargetRecorder(monkeypatch)
    sac = _make_sac(gamma=gamma, compute_chunked_gamma=True, critic_optimizer_lr=0.0)
    sac.train_step(rollout_step=10_000)
    # Stub target critics all output 1.0, reward=1, no terminals:
    # target = r + gamma**K * Q' = 1 + 0.9**CHUNK * 1.
    expected = 1.0 + gamma**CHUNK * 1.0
    assert recorder.targets, "no critic loss computed"
    assert recorder.unique_values() == {round(expected, 5)}


@pytest.mark.parametrize(
    "done,truncated", [(1.0, 0.0), (0.0, 1.0)], ids=["terminated", "truncated"]
)
def test_bootstrap_masked_on_terminal(monkeypatch, done, truncated):
    """Gap case: done OR truncated masks the bootstrap (no value through caps),
    so the TD target collapses to the reward exactly."""
    recorder = _TargetRecorder(monkeypatch)
    sac = _make_sac(sampler=_StubSampler(reward=1.0, done=done, truncated=truncated))
    sac.train_step(rollout_step=10_000)
    assert recorder.targets, "no critic loss computed"
    assert recorder.unique_values() == {1.0}


def test_softplus_ent_coef_starts_at_one_and_decays():
    """Softplus parameterization: alpha starts at 1 and moves toward the target."""
    # Only the temperature matters here: one critic update per actor update, frozen critics, a larger temperature lr.
    sac = _make_sac(
        ent_coef_parameterization="softplus",
        temperature_loss_type="alpha",
        target_entropy=-16,
        num_critic_updates_per_actor_update=1,
        critic_optimizer_lr=0.0,
        ent_coef_lr=5e-2,
    )
    alpha0 = torch.nn.functional.softplus(sac.log_ent_coef.detach()).item()
    assert abs(alpha0 - 1.0) < 1e-6
    for _ in range(5):
        sac.train_step()
    alpha1 = torch.nn.functional.softplus(sac.log_ent_coef.detach()).item()
    # The stub policy's entropy sits above the -16 target, so the temperature
    # must decrease (and softplus keeps it positive).
    assert 0.0 < alpha1 < alpha0


def test_ent_coef_parameterization_is_validated():
    """softplus needs the linear alpha loss; unknown parameterizations are refused."""
    with pytest.raises(ValueError, match="softplus"):
        _make_sac(
            ent_coef_parameterization="softplus",
            temperature_loss_type="log_alpha",
        )
    with pytest.raises(ValueError, match="parameterization"):
        _make_sac(ent_coef_parameterization="linear")


def test_use_amp_off_disables_autocast():
    """use_amp=False forces full-precision updates: no autocast dtype is resolved."""
    assert _make_sac(use_amp=False).amp_dtype is None


@pytest.mark.parametrize(("bf16", "expected"), [(True, torch.bfloat16), (False, None)])
def test_amp_dtype_is_bf16_or_fp32(monkeypatch, bf16, expected):
    """Autocast uses BF16 where supported; FP16-only GPUs train in FP32 (no unscaled FP16)."""
    from isaac_rl_posttraining.utils.torch_utils import resolve_amp_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda including_emulation=True: bf16
    )
    assert resolve_amp_dtype() is expected


def test_separate_actor_critic_grad_clips_applied(monkeypatch):
    """Per-network clips reach clip_grad_norm_ with their own max_norm."""
    sac = _make_sac(
        clip_grad_norm=None,
        actor_clip_grad_norm=3.5,
        critic_clip_grad_norm=10.0,
    )
    seen: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def _recording(params, max_norm, **kwargs):
        seen.append(float(max_norm))
        return original(params, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _recording)
    sac.train_step()
    assert 3.5 in seen and 10.0 in seen


def test_auto_target_entropy_comes_from_the_actor_not_an_environment():
    """target_entropy='auto' is -dim(action) of the actor's own distribution; no env object is consulted."""
    sac = _make_sac(target_entropy="auto")
    action_dim = int(sac.actor.action_dist.action_dim)
    assert action_dim > 1
    assert sac.target_entropy == -float(action_dim)
