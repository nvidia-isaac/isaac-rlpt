# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional TD-target bounds preserve the remaining learning behavior."""

import pytest
import torch

from isaac_rl_posttraining.algorithms.sac import SACConfig
from tests.unit.test_sac_core import _make_sac, _StubCritic, _TargetRecorder


@pytest.mark.parametrize("estimate,expected", [(3.0, 1.0), (-2.0, 0.0)])
def test_clip_bounds_nonterminal_targets_to_declared_range(
    monkeypatch, estimate, expected
):
    sac = _make_sac(
        critic=_StubCritic(estimate),
        gamma=1.0,
        td_target_clip=(0.0, 1.0),
        critic_optimizer_lr=0.0,
    )
    sac.sampler.reward = 0.0
    sac.sampler.done = 0.0
    recorder = _TargetRecorder(monkeypatch)
    metrics = sac.train_step(rollout_step=10000)
    assert recorder.unique_values() == {expected}
    assert metrics["td_target_max"] == expected
    assert metrics["target_q_mean"] == pytest.approx(estimate)


def test_clip_leaves_terminal_targets_alone(monkeypatch):
    sac = _make_sac(critic=_StubCritic(3.0), gamma=1.0, td_target_clip=(0.0, 1.0))
    sac.sampler.reward = 1.0
    sac.sampler.done = 1.0
    recorder = _TargetRecorder(monkeypatch)
    sac.train_step(rollout_step=10000)
    assert recorder.unique_values() == {1.0}


def test_no_clip_is_bit_identical_to_the_default(monkeypatch):
    torch.manual_seed(0)
    sac_default = _make_sac(critic=_StubCritic(3.0), gamma=1.0)
    torch.manual_seed(0)
    sac_none = _make_sac(critic=_StubCritic(3.0), gamma=1.0, td_target_clip=None)
    torch.manual_seed(1)
    metrics_default = sac_default.train_step(rollout_step=10000)
    torch.manual_seed(1)
    metrics_none = sac_none.train_step(rollout_step=10000)
    for key in ("critic_loss", "target_q_mean", "td_target_max"):
        assert metrics_default[key] == metrics_none[key]
    for (name, a), (_, b) in zip(
        sac_default.critic.state_dict().items(),
        sac_none.critic.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("bounds", [(1.0, 0.0), (0.0, float("inf")), (0.5, 0.5)])
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="td_target_clip"):
        _make_sac(td_target_clip=bounds)


def test_config_default_is_unbounded():
    assert SACConfig.__dataclass_fields__["td_target_clip"].default is None
