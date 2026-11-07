# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-drawn replay batches feed the learner without changing SAC's losses."""

from __future__ import annotations

import functools
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from conftest import make_transition

from isaac_rl_posttraining.algorithms.replay_learner import (
    ReplayLearner,
    pack_batches,
    unpack_batches,
)
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler
from isaac_rl_posttraining.modules.mlp import (
    MLPActor,
    MLPActorConfig,
    MLPCritic,
    MLPCriticConfig,
)
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LocalLearnerGroup,
)

STATE_DIM, DINO_DIM, LATENT_DIM = 3, 4, 2


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _spaces() -> tuple[gym.spaces.Dict, gym.spaces.Box]:
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "dino_embedding": gym.spaces.Box(-np.inf, np.inf, (DINO_DIM,), np.float32),
        }
    )
    return observation_space, gym.spaces.Box(-1.0, 1.0, (LATENT_DIM,), np.float32)


def _build_sac(
    buffer: ReplayBuffer, demonstrations: ReplayBuffer | None = None, **overrides
) -> SAC:
    observation_space, action_space = _spaces()
    actor = MLPActor(
        MLPActorConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(8, 8),
            use_tanh_output=True,
            log_std_init=-1.0,
        )
    )
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(8, 8),
            use_action=True,
        )
    )
    values = {
        "learning_starts": 0,
        "batch_size": 4,
        "num_critics": 2,
        "n_critics_to_sample": 2,
        "num_critic_updates_per_actor_update": 2,
        "num_updates_per_train_step": 2,
        "pooled_critic_features": False,
        "train_actor_with_kl_divergence": False,
        "use_amp": False,
        "ent_coef": "auto",
    }
    values.update(overrides)
    return SAC(
        SACConfig(
            env=SimpleNamespace(action_space=action_space),
            action_space=action_space,
            actor=actor,
            critic=critic,
            buffer=buffer,
            sampler=UniformSampler(buffer),
            demonstrations=demonstrations,
            **values,
        )
    )


def _filled_buffer(
    transitions: int = 24, *, reward: float | None = None, source: str = "policy"
) -> ReplayBuffer:
    """A replay buffer with a fixed, generator-seeded content independent of the global RNGs."""
    generator = np.random.default_rng(123)
    buffer = ReplayBuffer(capacity=64)
    for index in range(transitions):
        buffer.add(
            make_transition(
                episode_id=index // 6,
                step_in_episode=index % 6,
                obs={
                    "state": generator.standard_normal(STATE_DIM).astype(np.float32),
                    "dino_embedding": generator.standard_normal(DINO_DIM).astype(
                        np.float32
                    ),
                },
                action=generator.uniform(-1.0, 1.0, LATENT_DIM).astype(np.float32),
                reward=float(generator.uniform(-1.0, 0.0))
                if reward is None
                else reward,
                next_obs={
                    "state": generator.standard_normal(STATE_DIM).astype(np.float32),
                    "dino_embedding": generator.standard_normal(DINO_DIM).astype(
                        np.float32
                    ),
                },
                done=index % 6 == 5,
                info={"is_success": False},
                behavior_source=source,
            )
        )
    return buffer


class _CountingSampler:
    """Wraps a sampler and counts its draws, so a learner that samples for itself is caught."""

    def __init__(self, inner):
        self.inner = inner
        self.draws = 0

    def can_sample(self, batch_size):
        return self.inner.can_sample(batch_size)

    def sample(self, batch_size):
        self.draws += 1
        return self.inner.sample(batch_size)


def test_learner_batch_packs_one_replay_draw_per_gradient_step() -> None:
    _seed(0)
    algorithm = _build_sac(_filled_buffer())

    batch = algorithm.learner_batch()

    fields = (
        "obs",
        "action",
        "reward",
        "next_obs",
        "done",
        "truncated",
        "bootstrap_steps",
    )
    assert set(batch.tensors) == {
        f"{step}/replay/{field}" for step in (0, 1) for field in fields
    }
    # One oversized draw per step, split into critic slices by the algorithm.
    assert batch.tensors["0/replay/reward"].shape == (4 * 2,) and batch.samples == 16
    assert batch.tensors["0/replay/obs"]["state"].shape == (8, STATE_DIM)
    assert batch.tensors["1/replay/bootstrap_steps"].dtype == np.int64
    per_step = unpack_batches(batch, torch.device("cpu"))
    assert [set(step) for step in per_step] == [{"replay"}, {"replay"}]
    assert per_step[0]["replay"]["obs"]["state"].shape == (8, STATE_DIM)
    # Pre-drawn batches must match the gradient steps one for one.
    with pytest.raises(ValueError, match="pre-drawn batches"):
        algorithm.train_step(batches=[{}])


def test_learner_batch_adds_the_behavior_cloning_draws_and_skips_an_empty_success_draw() -> (
    None
):
    _seed(0)
    online = _filled_buffer(reward=-1.0)
    demonstrations = _filled_buffer(12, reward=9.0, source="expert")
    algorithm = _build_sac(
        online,
        demonstrations,
        expert_bc_coef=0.5,
        online_success_bc_coef=0.5,
        num_updates_per_train_step=1,
    )

    batch = algorithm.learner_batch()

    # Online episodes without a success end are failures: the success draw is left out.
    assert {key.split("/")[1] for key in batch.tensors} == {"replay", "expert"}
    assert batch.tensors["0/replay/reward"].tolist() == [-1.0] * 8
    assert batch.tensors["0/expert/reward"].tolist() == [9.0] * 4
    assert batch.samples == 12
    (step,) = unpack_batches(batch, torch.device("cpu"))
    assert set(step) == {"replay", "expert"}
    # The terms fail loudly when their data is missing.
    with pytest.raises(ValueError, match="demonstrations"):
        _build_sac(online, expert_bc_coef=0.5)


def test_pack_and_unpack_validate_their_shape() -> None:
    with pytest.raises(ValueError, match="at least one gradient step"):
        pack_batches([])
    with pytest.raises(ValueError, match="no batches"):
        pack_batches([{}])
    row = {
        "obs": {"state": np.zeros((1, 1), np.float32)},
        "action": np.zeros((1, 1), np.float32),
        "reward": np.zeros(1, np.float32),
        "next_obs": {"state": np.zeros((1, 1), np.float32)},
        "done": np.zeros(1, np.float32),
        "truncated": np.zeros(1, np.float32),
        "bootstrap_steps": np.ones(1, np.int64),
    }
    with pytest.raises(ValueError, match="no '/'"):
        pack_batches([{"a/b": row}])
    packed = pack_batches([{"replay": row}, {"replay": row}])
    second_step_only = LearnerBatch(
        tensors={
            key: value for key, value in packed.tensors.items() if key.startswith("1/")
        },
        samples=1,
    )
    with pytest.raises(ValueError, match="cover 0..N-1"):
        unpack_batches(second_step_only, torch.device("cpu"))


def _run_in_process(calls: int) -> list[dict[str, float]]:
    _seed(7)
    algorithm = _build_sac(_filled_buffer())
    return [
        {
            key: float(value)
            for key, value in algorithm.train_step().items()
            if np.isscalar(value)
        }
        for _ in range(calls)
    ]


def _run_pre_drawn(
    calls: int,
) -> tuple[list[dict[str, float]], LocalLearnerGroup, SAC, _CountingSampler]:
    _seed(7)
    algorithm = _build_sac(_filled_buffer())
    counting = _CountingSampler(algorithm.sampler)
    algorithm.sampler = counting
    group = LocalLearnerGroup(
        functools.partial(ReplayLearner, algorithm=algorithm), device="cpu"
    )
    metrics = []
    for _ in range(calls):
        batch = algorithm.learner_batch()
        draws_before = counting.draws
        (result,) = group.train_step([batch])
        assert counting.draws == draws_before, (
            "the learner must consume the pre-drawn batches, not sample"
        )
        metrics.append(dict(result.metrics))
    return metrics, group, algorithm, counting


def test_pre_drawn_batches_reproduce_in_process_losses_exactly() -> None:
    """Same seeds, same buffer: the learner path must produce the same metrics per call."""
    in_process = _run_in_process(calls=3)
    pre_drawn, group, _, counting = _run_pre_drawn(calls=3)

    assert len(in_process) == len(pre_drawn) == 3
    assert counting.draws == 3 * 2
    for call_index, (expected, actual) in enumerate(
        zip(in_process, pre_drawn, strict=True)
    ):
        shared = set(expected) & set(actual)
        assert {"critic_loss", "actor_loss", "ent_coef", "q_values_mean"} <= shared
        for key in sorted(shared):
            if key == "train_step_total_time_s":
                continue
            assert np.allclose(expected[key], actual[key], rtol=0.0, atol=0.0), (
                f"call {call_index}: {key} in-process={expected[key]} pre-drawn={actual[key]}"
            )
    assert group.hooks().gradient_steps() == 6


def test_replay_learner_checkpoint_round_trip_restores_algorithm_state(
    tmp_path,
) -> None:
    _, group, algorithm, _ = _run_pre_drawn(calls=2)
    assert group.hooks().gradient_steps() == 4

    group.save_checkpoint(str(tmp_path))
    saved_actor = {k: v.clone() for k, v in algorithm.actor.state_dict().items()}
    saved_temperature = algorithm.log_ent_coef.detach().clone()
    saved_moments = {
        k: v.clone()
        for k, v in algorithm.ent_coef_optimizer.state_dict()["state"][0].items()
    }

    group.train_step([algorithm.learner_batch()])
    assert not torch.equal(algorithm.log_ent_coef.detach(), saved_temperature)

    group.load_checkpoint(str(tmp_path))

    for key, value in saved_actor.items():
        assert torch.equal(algorithm.actor.state_dict()[key], value)
    assert torch.equal(algorithm.log_ent_coef.detach(), saved_temperature)
    restored_moments = algorithm.ent_coef_optimizer.state_dict()["state"][0]
    for key, value in saved_moments.items():
        assert torch.equal(restored_moments[key], value)
    assert algorithm.step_counter == 4
    assert group.hooks().gradient_steps() == 4
    assert (tmp_path / "learner.pt").is_file()


def test_replay_learner_checkpoint_round_trip_with_a_fixed_entropy_coefficient(
    tmp_path,
) -> None:
    """A fixed ``ent_coef`` has no temperature or temperature optimizer to carry."""
    _seed(7)
    algorithm = _build_sac(_filled_buffer(), ent_coef=0.1)
    group = LocalLearnerGroup(
        functools.partial(ReplayLearner, algorithm=algorithm), device="cpu"
    )
    assert algorithm.log_ent_coef is None and algorithm.ent_coef_optimizer is None
    assert set(group._worker.checkpoint_state()) == {"step_counter"}

    group.train_step([algorithm.learner_batch()])
    group.save_checkpoint(str(tmp_path))
    saved_actor = {k: v.clone() for k, v in algorithm.actor.state_dict().items()}

    group.train_step([algorithm.learner_batch()])
    assert any(
        not torch.equal(algorithm.actor.state_dict()[k], v)
        for k, v in saved_actor.items()
    )

    group.load_checkpoint(str(tmp_path))

    for key, value in saved_actor.items():
        assert torch.equal(algorithm.actor.state_dict()[key], value)
    assert algorithm.step_counter == 2
    assert algorithm.log_ent_coef is None and algorithm.ent_coef_optimizer is None


def test_sac_declares_the_modules_and_optimizers_the_learner_manages(tmp_path) -> None:
    _seed(0)
    algorithm = _build_sac(_filled_buffer())
    modules = algorithm.learner_modules()
    assert list(modules) == ["actor", "critic", "critic_target"]
    assert [name for name, entry in modules.items() if entry.publish] == ["actor"]
    assert {
        name: entry.module for name, entry in algorithm.learner_optimizers().items()
    } == {
        "actor_optimizer": "actor",
        "critic_optimizer": "critic",
    }
    assert algorithm.replicated_parameters() == (algorithm.log_ent_coef,)
    group = LocalLearnerGroup(
        functools.partial(ReplayLearner, algorithm=algorithm), device="cpu"
    )
    group.save_checkpoint(str(tmp_path))
    payload = torch.load(tmp_path / "learner.pt", weights_only=False)
    assert list(payload["modules"]) == ["actor", "critic", "critic_target"]
    assert list(payload["optimizers"]) == ["actor_optimizer", "critic_optimizer"]
    assert {"step_counter", "log_ent_coef"} <= set(payload["algorithm"])


def test_step_scheduled_settings_need_respect_step_schedules() -> None:
    with pytest.raises(ValueError, match="respect_step_schedules=true") as error:
        _build_sac(
            _filled_buffer(),
            actor_learning_starts=100,
            actor_behavior_kl_coef=0.1,
            actor_behavior_anchor_step=5,
        )
    message = str(error.value)
    assert (
        "algorithm.actor_behavior_anchor_step, algorithm.actor_behavior_kl_coef, "
        in message
    )
    assert (
        "algorithm.actor_learning_starts are scheduled by environment step" in message
    )


@pytest.mark.parametrize(
    ("respect", "env_steps", "actor_moves"), [(True, 10, False), (True, 200, True)]
)
def test_learner_passes_env_steps_only_when_schedules_are_respected(
    respect, env_steps, actor_moves
) -> None:
    """With the flag on, the actor window opens at the batch's environment step."""
    _seed(0)
    algorithm = _build_sac(
        _filled_buffer(), respect_step_schedules=respect, actor_learning_starts=100
    )
    group = LocalLearnerGroup(
        functools.partial(ReplayLearner, algorithm=algorithm), device="cpu"
    )
    before = {key: value.clone() for key, value in algorithm.actor.state_dict().items()}
    group.train_step([algorithm.learner_batch().at_env_steps(env_steps)])
    moved = any(
        not torch.equal(before[key], value)
        for key, value in algorithm.actor.state_dict().items()
    )
    assert moved is actor_moves


def test_learner_ignores_env_steps_by_default() -> None:
    """Today's behaviour: without the flag the update never sees the step count."""
    seen = []
    _seed(0)
    algorithm = _build_sac(_filled_buffer())
    original = algorithm.train_step

    def recording(*args, **kwargs):
        seen.append(kwargs.get("rollout_step"))
        return original(*args, **kwargs)

    algorithm.train_step = recording
    group = LocalLearnerGroup(
        functools.partial(ReplayLearner, algorithm=algorithm), device="cpu"
    )
    group.train_step([algorithm.learner_batch().at_env_steps(5000)])
    assert seen == [None]


def test_learner_replica_leaves_the_controller_state_behind() -> None:
    _seed(0)
    demonstrations = _filled_buffer(12, reward=9.0, source="expert")
    algorithm = _build_sac(_filled_buffer(), demonstrations, expert_bc_coef=0.5)
    replica = algorithm.learner_replica()
    for name in ("buffer", "sampler", "logger", "expert_sampler", "success_sampler"):
        assert getattr(replica, name) is None
    assert replica.config.buffer is None and replica.config.demonstrations is None
    assert replica.config.env is None and replica.config.sampler is None
    assert replica.actor is algorithm.actor
    # The controller's own object keeps its replay access.
    assert algorithm.buffer is not None and algorithm.expert_sampler is not None
