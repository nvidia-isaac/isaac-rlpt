# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretrained N1.7-3B has no baked G1 embodiment; IQL registers it from demos."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaac_rl_posttraining.envs.embodiments import (
    get_dataset_adapter,
    resolve_adapter_option,
)
from isaac_rl_posttraining.policies.gr00t.flow_sde_n17 import (
    Gr00tN17FlowSDEActorCritic,
    ensure_n17_embodiment,
)
from simulators.g1_arena.embodiment import G1_DECODE_HORIZON


class _Policy:
    def __init__(self) -> None:
        self.processor = SimpleNamespace(modality_configs={})
        self.modality_json = None
        self.baked = False

    def register_baked_embodiment(self) -> None:
        self.baked = True
        self.modality_json = {"state": {"arm": {"start": 0, "end": 1}}}


def test_spec_options_forward_the_dataset_used_to_register_a_raw_base():
    backend = SimpleNamespace(
        embodiment_tag="new_embodiment",
        action_horizon=40,
        backbone_model_name_or_path=None,
        dataset_dir="/demos/lerobot",
        dataset_adapter="g1_wbc",
    )
    options = Gr00tN17FlowSDEActorCritic.spec_options(backend)
    assert options["dataset_dir"] == "/demos/lerobot"
    assert options["dataset_adapter"] == "g1_wbc"
    assert options["action_window"] == 40
    assert (
        resolve_adapter_option(options)["dataset_adapter"].decode_horizon
        == G1_DECODE_HORIZON
    )


def test_spec_options_omit_the_dataset_for_a_baked_checkpoint():
    backend = SimpleNamespace(
        embodiment_tag="new_embodiment",
        action_horizon=16,
        backbone_model_name_or_path=None,
        dataset_dir=None,
        dataset_adapter="g1_wbc",
    )
    options = Gr00tN17FlowSDEActorCritic.spec_options(backend)
    assert "dataset_dir" not in options and "dataset_adapter" not in options


def test_baked_embodiment_is_left_alone(monkeypatch: pytest.MonkeyPatch):
    policy = _Policy()
    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.has_baked_embodiment",
        lambda processor, tag: True,
    )
    called = {}

    def _register(*args, **kwargs):
        called["yes"] = True

    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.register_embodiment", _register
    )
    ensure_n17_embodiment(
        policy,
        embodiment_tag="new_embodiment",
        action_window=40,
        dataset_dir="/demos",
        dataset_adapter=get_dataset_adapter("g1_wbc"),
    )
    assert policy.baked
    assert "yes" not in called


def test_raw_base_registers_the_g1_decode_window_from_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
):
    policy = _Policy()
    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.has_baked_embodiment",
        lambda processor, tag: False,
    )
    seen: dict[str, object] = {}

    def _register(
        processor, dataset_dir, tag, horizon, state_groups=None, action_groups=None
    ):
        seen.update(
            dataset_dir=dataset_dir,
            tag=tag,
            horizon=horizon,
            state_groups=list(state_groups or []),
            action_groups=list(action_groups or []),
        )

    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.register_embodiment", _register
    )
    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.modality_json_from_processor",
        lambda processor, tag: {"state": {"left_arm": {"start": 0, "end": 7}}},
    )
    ensure_n17_embodiment(
        policy,
        embodiment_tag="new_embodiment",
        action_window=40,
        dataset_dir="/demos/lerobot",
        dataset_adapter=get_dataset_adapter("g1_wbc"),
    )
    assert seen["horizon"] == G1_DECODE_HORIZON == 16
    assert seen["dataset_dir"] == "/demos/lerobot"
    assert "left_arm" in seen["state_groups"]
    assert policy.modality_json == {"state": {"left_arm": {"start": 0, "end": 7}}}
    assert not policy.baked


def test_raw_base_without_a_dataset_is_an_error(monkeypatch: pytest.MonkeyPatch):
    policy = _Policy()
    monkeypatch.setattr(
        "isaac_rl_posttraining.policies.gr00t.data.has_baked_embodiment",
        lambda processor, tag: False,
    )
    with pytest.raises(ValueError, match="dataset_dir"):
        ensure_n17_embodiment(
            policy,
            embodiment_tag="new_embodiment",
            action_window=40,
            dataset_dir=None,
            dataset_adapter=get_dataset_adapter("g1_wbc"),
        )
