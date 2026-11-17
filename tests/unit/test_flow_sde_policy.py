# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flow-SDE actor-critic as a batched action-chunk policy, on a fake model."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.policies.feature_cache import BF16_SUFFIX
from isaac_rl_posttraining.policies.gr00t.flow_sde import (
    BF16_PIXEL_KEY,
    STOCHASTIC_STEP_KEY,
    FlowSDEConfig,
    FlowSDESample,
    Gr00tFlowSDEActorCritic,
    pack_model_inputs,
    unpack_model_inputs,
)
from isaac_rl_posttraining.policies.gr00t.flow_sde_policy import (
    FlowSDEActionChunkPolicy,
    FlowSDEPolicySpec,
)
from isaac_rl_posttraining.policies.trainable import TrainablePolicySpec
from isaac_rl_posttraining.runtimes import (
    PolicyContext,
    policy_state_from_tensors,
)

WINDOW, STEPS, ACTION_DIM = 4, 4, 2
INSTRUCTION = "stack the cubes"


class _FakePolicy(nn.Module):
    """The surface of a trainable policy the rollout policy touches, with a frozen backbone."""

    frozen_submodules = ("model.backbone",)
    cached_feature_names = ("backbone_features",)
    requires_cached_features = False
    pack_inputs = staticmethod(pack_model_inputs)
    unpack_inputs = staticmethod(unpack_model_inputs)

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 2)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Linear(2, 2)
        self.video_keys = ["video.cam"]
        self.state_keys = ["state.q"]
        self.action_window = WINDOW
        self.denoising_steps = STEPS
        self.calls = 0
        self.stochastic_steps: list[int | None] = []

    def build_inputs(self, frames, states, instruction):
        assert instruction == INSTRUCTION
        batch = frames["video.cam"].shape[0]
        return {
            "state": torch.as_tensor(states["state.q"]),
            "eagle_pixel_values": torch.zeros(batch, 3, 2, 2),
        }

    def learner_inputs(self, inputs):
        return {
            key: value for key, value in inputs.items() if not key.startswith("eagle_")
        }

    def act(self, inputs, *, stochastic_step):
        batch = inputs["state"].shape[0]
        self.calls += 1
        self.stochastic_steps.append(stochastic_step)
        return FlowSDESample(
            actions=torch.zeros(batch, WINDOW, ACTION_DIM),
            chains=torch.zeros(batch, STEPS + 1, WINDOW, ACTION_DIM),
            log_prob=torch.full((batch,), -1.0),
            value=torch.full((batch,), float(self.calls)),
            stochastic_step=stochastic_step,
            cached_features={
                "backbone_features": torch.zeros(batch, 3, 2).to(torch.bfloat16)
            },
        )

    def decode(self, normalized, *, inputs, states):
        assert "state" in inputs and "state.q" in states
        return np.zeros((normalized.shape[0], WINDOW, ACTION_DIM), dtype=np.float32)


def _observations(count: int) -> tuple[dict[str, np.ndarray], ...]:
    return tuple(
        {
            "video.cam": np.zeros((4, 4, 3), dtype=np.uint8),
            "state.q": np.full(2, float(index), dtype=np.float32),
        }
        for index in range(count)
    )


def _policy(
    fake: _FakePolicy, *, cache_features: bool = True, seed: int = 0
) -> FlowSDEActionChunkPolicy:
    return FlowSDEActionChunkPolicy(
        fake,
        instruction=INSTRUCTION,
        rng=np.random.default_rng(seed),
        cache_features=cache_features,
    )


def _spec() -> TrainablePolicySpec:
    return TrainablePolicySpec(
        base_model="gr00t_n15",
        checkpoint="/ckpt",
        flow_sde=FlowSDEConfig(),
        options={"data_config": "m:C", "embodiment_tag": "tag"},
    )


def test_training_decisions_noise_one_step_and_carry_the_chain_and_cached_features() -> (
    None
):
    fake = _FakePolicy()
    policy = _policy(fake)

    chunks = policy.action_chunks(
        _observations(3), mode=("train",) * 3, session_ids=(0, 1, 2)
    )

    assert fake.calls == 1 and len(chunks) == 3
    (stochastic_step,) = fake.stochastic_steps
    assert 0 <= stochastic_step < STEPS
    for index, chunk in enumerate(chunks):
        assert chunk.learner_action.shape == (1, STEPS + 1, WINDOW, ACTION_DIM)
        assert chunk.environment_actions.shape == (1, WINDOW, ACTION_DIM)
        assert chunk.execution_length == WINDOW and chunk.source == "policy"
        assert float(chunk.behavior_log_prob[0]) == pytest.approx(-1.0)
        assert float(chunk.behavior_value[0]) == pytest.approx(1.0)
        assert set(chunk.learner_observation) == {"state", STOCHASTIC_STEP_KEY}
        np.testing.assert_array_equal(
            chunk.learner_observation["state"], [[index, index]]
        )
        assert chunk.learner_observation[STOCHASTIC_STEP_KEY].tolist() == [
            stochastic_step
        ]
        assert set(chunk.cached_features) == {"backbone_features" + BF16_SUFFIX}
        assert chunk.cached_features["backbone_features" + BF16_SUFFIX].shape == (
            1,
            3,
            2,
        )


def test_evaluation_decodes_deterministically_and_ships_pixels_without_caching() -> (
    None
):
    fake = _FakePolicy()
    policy = _policy(fake, cache_features=False)

    (chunk,) = policy.action_chunks(_observations(1), mode=("eval",), session_ids=(0,))

    assert fake.stochastic_steps == [None]
    assert chunk.learner_observation[STOCHASTIC_STEP_KEY].tolist() == [-1]
    assert BF16_PIXEL_KEY in chunk.learner_observation and chunk.cached_features is None
    assert chunk.behavior_value is not None


def test_stochastic_steps_follow_the_policy_rng() -> None:
    first, second = _FakePolicy(), _FakePolicy()
    for _ in range(5):
        _policy(first, seed=11).action_chunks(
            _observations(1), mode=("train",), session_ids=(0,)
        )
    policy = _policy(second, seed=11)
    for _ in range(5):
        policy.action_chunks(_observations(1), mode=("train",), session_ids=(0,))

    expected = [int(step) for step in np.random.default_rng(11).integers(STEPS, size=5)]
    assert second.stochastic_steps == expected
    assert first.stochastic_steps == [expected[0]] * 5


def test_batches_share_one_mode_and_have_no_warm_up() -> None:
    policy = _policy(_FakePolicy())
    with pytest.raises(ValueError, match="one decision mode"):
        policy.action_chunks(
            _observations(2), mode=("train", "eval"), session_ids=(0, 1)
        )
    with pytest.raises(ValueError, match="warm-up"):
        policy.action_chunks(_observations(1), mode=("warmup",), session_ids=(0,))
    assert policy.action_chunks((), mode=(), session_ids=()) == ()


def test_published_weights_exclude_frozen_keys() -> None:
    fake = _FakePolicy()
    policy = _policy(fake)
    new_head = {name: value + 1.0 for name, value in fake.head.state_dict().items()}

    state = policy_state_from_tensors(
        {f"head.{name}": value.to(torch.bfloat16) for name, value in new_head.items()},
        version=1,
    )
    assert policy.set_policy_state(state) == 1
    assert torch.allclose(
        fake.head.weight.detach(),
        new_head["weight"].to(torch.bfloat16).float(),
        atol=1e-2,
    )
    with pytest.raises(ValueError, match="must advance"):
        policy.set_policy_state(state)
    with pytest.raises(ValueError, match="unexpected"):
        policy.set_policy_state(policy_state_from_tensors(fake.state_dict(), version=2))


def test_factory_loads_per_policy_seeded_and_checks_the_action_window(
    monkeypatch,
) -> None:
    fakes: list[_FakePolicy] = []
    loads: list[dict] = []

    def load(cls, checkpoint, **kwargs):
        loads.append({"checkpoint": checkpoint, **kwargs})
        fakes.append(_FakePolicy())
        return fakes[-1]

    monkeypatch.setattr(Gr00tFlowSDEActorCritic, "load", classmethod(load))
    factory = FlowSDEPolicySpec(
        spec=_spec(), action_window=WINDOW, instruction=INSTRUCTION, base_seed=3
    )

    policy = factory(
        PolicyContext(
            policy_index=1, num_policies=2, device="cpu", placement="rollout_worker"
        )
    )
    policy.action_chunks(_observations(1), mode=("train",), session_ids=(0,))

    assert loads == [
        {
            "checkpoint": "/ckpt",
            "device": "cpu",
            "flow_sde": FlowSDEConfig(),
            "value_head_seed": 0,
            "data_config": "m:C",
            "embodiment_tag": "tag",
        }
    ]
    assert fakes[0].stochastic_steps == [
        int(np.random.default_rng(3 + 1 + 7919).integers(STEPS))
    ]
    with pytest.raises(ValueError, match="action_window"):
        FlowSDEPolicySpec(
            spec=_spec(), action_window=WINDOW + 1, instruction=INSTRUCTION
        )(
            PolicyContext(
                policy_index=0, num_policies=1, device="cpu", placement="rollout_worker"
            )
        )


def test_factory_checks_feature_caching_against_the_base_model(monkeypatch) -> None:
    class _Uncacheable(_FakePolicy):
        cached_feature_names = ()

    class _CacheOnly(_FakePolicy):
        requires_cached_features = True

    spec = _spec()
    monkeypatch.setattr(
        TrainablePolicySpec, "policy_class", property(lambda self: _Uncacheable)
    )
    with pytest.raises(ValueError, match="no cacheable features"):
        FlowSDEPolicySpec(
            spec=spec,
            action_window=WINDOW,
            instruction=INSTRUCTION,
            cache_features=True,
        )
    monkeypatch.setattr(
        TrainablePolicySpec, "policy_class", property(lambda self: _CacheOnly)
    )
    with pytest.raises(ValueError, match="cache_features=true"):
        FlowSDEPolicySpec(
            spec=spec,
            action_window=WINDOW,
            instruction=INSTRUCTION,
            cache_features=False,
        )
