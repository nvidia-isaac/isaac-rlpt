# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rejection retries are deterministic and preserve accepted environments."""

import pytest
import torch

from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)
from example_workflows.v2d_box_lift_posttraining.reset_validation import (
    Penetration,
    ResetSampler,
)


def sampler(fixed=True, attempts=8):
    return ResetSampler(
        3,
        seed=2,
        fixed=fixed,
        device=torch.device("cpu"),
        config=ResetValidationConfig(True, attempts, 0.001, ()),
    )


def test_retries_select_new_candidates_and_fixed_partial_resets_repeat():
    resetter = sampler()
    poses = torch.zeros(3, 8)
    draws = [[] for _ in range(3)]
    calls = [0, 0, 0]

    def sample(ids):
        poses[ids] = torch.rand(len(ids), 8)
        for i in ids.tolist():
            draws[i].append(poses[i].clone())
            calls[i] += 1

    def check(i):
        if i == 1 and calls[i] <= 2:
            return [Penetration("hand", "box", 0.01)]
        return []

    before = torch.get_rng_state()
    resetter.reset(torch.arange(3), sample=sample, check=check)
    assert torch.equal(torch.get_rng_state(), before)
    assert calls == [1, 3, 1]
    assert not torch.equal(draws[1][0], draws[1][1])
    accepted = poses.clone()
    resetter.reset(torch.tensor([2, 1]), sample=sample, check=check)
    assert torch.equal(poses, accepted)
    assert calls == [1, 4, 2]
    assert torch.equal(torch.get_rng_state(), before)


def test_exhaustion_reports_env_and_collision_pair():
    resetter = sampler(attempts=2)
    with pytest.raises(RuntimeError, match="environment 1 after 2 attempts.*hand"):
        resetter.reset(
            torch.tensor([1]),
            sample=lambda ids: torch.rand(5),
            check=lambda i: [Penetration("hand", "box", 0.02)],
        )
    assert resetter.accepted_seeds == [None, None, None]
    assert resetter.rejections.tolist() == [0, 2, 0]


def test_random_resets_change_and_are_order_independent():
    def run(order):
        resetter = sampler(fixed=False)
        poses = torch.zeros(3, 8)

        def sample(ids):
            poses[ids] = torch.rand(len(ids), 8)

        resetter.reset(torch.tensor(order), sample=sample, check=lambda i: [])
        first = poses.clone()
        resetter.reset(torch.tensor(order), sample=sample, check=lambda i: [])
        assert not torch.equal(first, poses)
        return poses

    assert torch.equal(run([0, 1, 2]), run([2, 0, 1]))


def test_explicit_random_reset_seed_is_reproducible_and_local():
    resetter = sampler(fixed=False)
    poses = torch.zeros(3, 8)

    def sample(ids):
        poses[ids] = torch.rand(len(ids), 8)

    resetter.reseed(None, 123)
    resetter.reset(torch.arange(3), sample=sample, check=lambda i: [])
    first = poses.clone()
    counters = list(resetter.next_candidates)
    resetter.reseed(torch.tensor([1]), 123)
    resetter.reset(torch.tensor([1]), sample=sample, check=lambda i: [])
    assert torch.equal(first, poses)
    assert resetter.next_candidates == counters
    resetter.reseed(torch.tensor([1]), 321)
    resetter.reset(torch.tensor([1]), sample=sample, check=lambda i: [])
    assert not torch.equal(first[1], poses[1])
    assert torch.equal(first[[0, 2]], poses[[0, 2]])


def test_cached_fixed_candidate_cannot_silently_change_when_invalid():
    resetter = sampler()
    resetter.reset(torch.tensor([0]), sample=lambda ids: None, check=lambda i: [])
    accepted = resetter.accepted_seeds[0]
    resetter.reseed(None, 999)
    with pytest.raises(RuntimeError, match="fixed_cached=True"):
        resetter.reset(
            torch.tensor([0]),
            sample=lambda ids: None,
            check=lambda i: [Penetration("hand", "box", 0.1)],
        )
    assert resetter.accepted_seeds[0] == accepted
