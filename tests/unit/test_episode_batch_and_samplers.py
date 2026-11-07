# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sampler library over the episode buffer, and complete-episode batching.

Covers uniform draws (over everything or a filtered subset), the fixed
success share, mixing several buffers at fixed shares (with the legacy share
arithmetic: floors, the last member takes the remainder, empty members hand
their share to the others), consecutive-decision windows, and the padded
complete-episode batch recurrent learners consume.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_transition

from isaac_rl_posttraining.buffers import (
    MixedSampler,
    ReplayBuffer,
    Sampler,
    SequenceSampler,
    SuccessFractionSampler,
    UniformSampler,
    allocate_shares,
    sample_episode_batch,
)
from isaac_rl_posttraining.buffers.samplers import renormalized


def _add_episode(
    buffer: ReplayBuffer,
    episode_id,
    *,
    reward: float,
    length: int = 2,
    success: bool = True,
    source="policy",
):
    """Add a ``length``-decision episode whose transitions all carry ``reward`` on their last step."""
    for step in range(length):
        buffer.add(
            make_transition(
                episode_id=episode_id,
                step_in_episode=step,
                obs={"state": np.asarray([step], np.float32)},
                action=np.full((3, 4), float(step), np.float32),
                reward=reward,
                next_obs={"state": np.asarray([step + 1], np.float32)},
                done=step == length - 1,
                info={"is_success": success},
                behavior_source=source,
            )
        )


def _outcome_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(100)
    _add_episode(buffer, "s0", reward=5.0, success=True)
    _add_episode(buffer, "s1", reward=6.0, success=True)
    _add_episode(buffer, "f0", reward=-1.0, success=False)
    return buffer


def test_uniform_sampler_draws_over_everything_or_a_filter():
    buffer = _outcome_buffer()
    buffer.add(
        make_transition(
            episode_id="open",
            obs={"state": np.zeros(1, np.float32)},
            action=np.zeros((3, 4)),
            reward=9.0,
        )
    )
    rng = np.random.default_rng(0)
    everything = UniformSampler(buffer, rng=rng).sample(64)["reward"].tolist()
    assert set(everything) == {5.0, 6.0, -1.0, 9.0}, (
        "an episode still running is sampleable"
    )
    successes = (
        UniformSampler(buffer, rng=rng, outcome="success").sample(16)["reward"].tolist()
    )
    assert set(successes) <= {5.0, 6.0}
    assert isinstance(UniformSampler(buffer), Sampler)
    empty = UniformSampler(ReplayBuffer(4))
    assert not empty.can_sample(1) and empty.sample(4) == {}
    with pytest.raises(ValueError, match="batch_size"):
        UniformSampler(buffer).sample(0)


def test_success_fraction_sampler_splits_every_batch_and_renormalizes():
    buffer = _outcome_buffer()
    sampler = SuccessFractionSampler(buffer, 0.5, rng=np.random.default_rng(1))
    rewards = sampler.sample(8)["reward"].tolist()
    assert (
        sum(1 for r in rewards if r > 0) == 4 and sum(1 for r in rewards if r < 0) == 4
    )
    assert sampler.allocate(8) == {"success": 4, "other": 4}
    # Without a failure yet, the successes supply the whole batch, and vice versa.
    only_success = ReplayBuffer(100)
    _add_episode(only_success, "s", reward=5.0, success=True)
    assert (
        SuccessFractionSampler(only_success, 0.25).sample(6)["reward"].tolist()
        == [5.0] * 6
    )
    only_failure = ReplayBuffer(100)
    _add_episode(only_failure, "f", reward=-1.0, success=False)
    assert (
        SuccessFractionSampler(only_failure, 0.9).sample(6)["reward"].tolist()
        == [-1.0] * 6
    )
    # Episodes still running take no part: their outcome is unknown.
    running = ReplayBuffer(100)
    running.add(
        make_transition(
            episode_id="r",
            obs={"state": np.zeros(1, np.float32)},
            action=np.zeros((3, 4)),
        )
    )
    assert (
        not SuccessFractionSampler(running, 0.5).can_sample(1)
        and SuccessFractionSampler(running, 0.5).sample(4) == {}
    )
    with pytest.raises(ValueError, match="fraction"):
        SuccessFractionSampler(buffer, 1.5)


def test_shares_floor_every_member_and_give_the_last_one_the_remainder():
    assert allocate_shares(100, {"a": 0.29, "b": 0.71}) == {"a": 29, "b": 71}
    assert allocate_shares(7, {"a": 0.5, "b": 0.5}) == {"a": 3, "b": 4}
    assert allocate_shares(5, {"a": 0.34, "b": 0.33, "c": 0.33}) == {
        "a": 1,
        "b": 1,
        "c": 3,
    }
    assert allocate_shares(3, {}) == {}
    assert renormalized({"a": 0.2, "b": 0.8}, ["a"]) == {"a": 1.0}
    assert renormalized({"a": 0.0, "b": 0.0}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert renormalized({"a": 0.2, "b": 0.8}, []) == {}


def test_mixed_sampler_draws_fixed_shares_from_several_buffers():
    online = ReplayBuffer(100)
    _add_episode(online, "o", reward=-1.0, success=False)
    demos = ReplayBuffer(100)
    _add_episode(demos, "d", reward=9.0, source="expert")
    sampler = MixedSampler(
        {"online": online, "demos": demos},
        {"online": 0.75, "demos": 0.25},
        rng=np.random.default_rng(2),
    )

    assert sampler.allocate(8) == {"online": 6, "demos": 2}
    rewards = sampler.sample(8)["reward"].tolist()
    assert rewards.count(-1.0) == 6 and rewards.count(9.0) == 2
    batch = sampler.sample(8)
    assert batch["obs"]["state"].shape == (8, 1) and batch["action"].shape == (8, 3, 4)
    assert len(batch["info"]) == 8 and set(batch["behavior_source"]) == {
        "policy",
        "expert",
    }
    # An empty member hands its share to the others; equal shares by default.
    assert MixedSampler(
        {"online": online, "demos": ReplayBuffer(4)}, {"online": 0.5, "demos": 0.5}
    ).allocate(8) == {"online": 8}
    assert MixedSampler({"online": online, "demos": demos}).shares() == {
        "online": 0.5,
        "demos": 0.5,
    }
    assert not MixedSampler({"a": ReplayBuffer(4), "b": ReplayBuffer(4)}).can_sample(1)


def test_mixed_sampler_validates_its_members_and_fractions():
    buffer = _outcome_buffer()
    with pytest.raises(ValueError, match="at least one member"):
        MixedSampler({})
    with pytest.raises(ValueError, match="name exactly"):
        MixedSampler({"a": buffer}, {"b": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        MixedSampler({"a": buffer, "b": buffer}, {"a": 0.5, "b": 0.6})
    with pytest.raises(ValueError, match="non-negative"):
        MixedSampler({"a": buffer, "b": buffer}, {"a": -0.5, "b": 1.5})
    with pytest.raises(TypeError, match="sampler or a replay buffer"):
        MixedSampler({"a": object()})


def test_sequence_sampler_returns_consecutive_decisions_of_one_episode():
    buffer = ReplayBuffer(100)
    _add_episode(buffer, "long", reward=1.0, length=4)
    _add_episode(buffer, "short", reward=2.0, length=1)
    sampler = SequenceSampler(buffer, 3, rng=np.random.default_rng(3))
    assert sampler.starts().tolist() == [0, 1]
    batch = sampler.sample(5)
    assert batch["obs"]["state"].shape == (5, 3, 1) and batch["action"].shape == (
        5,
        3,
        3,
        4,
    )
    assert batch["reward"].shape == (5, 3)
    for window in batch["obs"]["state"]:
        steps = window[:, 0].tolist()
        assert steps == [steps[0] + offset for offset in range(3)], (
            "windows are consecutive decisions"
        )
    assert (
        not SequenceSampler(buffer, 5).can_sample(1)
        and SequenceSampler(buffer, 5).sample(2) == {}
    )
    with pytest.raises(ValueError, match="length"):
        SequenceSampler(buffer, 0)


def test_sample_episode_batch_pads_complete_episodes():
    buffer = ReplayBuffer(100)
    _add_episode(buffer, "a", reward=5.0, length=2)
    _add_episode(buffer, "b", reward=-1.0, length=3, success=False)
    buffer.add(
        make_transition(
            episode_id="open",
            obs={"state": np.zeros(1, np.float32)},
            action=np.zeros((3, 4)),
        )
    )

    batch = sample_episode_batch(
        buffer, batch_size=4, device=torch.device("cpu"), rng=np.random.default_rng(4)
    )

    assert batch["num_complete_episodes"] == 2
    assert batch["actions"].shape == (4, 3, 3, 4) and batch["rewards"].shape == (
        4,
        3,
        1,
    )
    # States carry one extra timestep (obs_0 .. next_obs_T); previous_actions align to it.
    assert batch["states"]["state"].shape[1] == batch["actions"].shape[1] + 1
    assert batch["previous_actions"].shape[1] == batch["actions"].shape[1] + 1
    for row in range(4):
        length = int(batch["lengths"][row])
        assert length in (2, 3) and int(batch["mask"][row].sum().item()) == length
        assert bool(batch["dones"][row, length - 1, 0])
        assert set(batch["rewards"][row, :length, 0].tolist()) <= {5.0, -1.0}
    assert (
        sample_episode_batch(ReplayBuffer(4), batch_size=2, device=torch.device("cpu"))
        == {}
    )
    with pytest.raises(ValueError, match="batch_size"):
        sample_episode_batch(buffer, batch_size=0, device=torch.device("cpu"))
