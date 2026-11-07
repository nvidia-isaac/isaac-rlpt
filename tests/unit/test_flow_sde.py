# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-SDE sampling and log-density for GR00T PPO, exercised on a tiny fake head.

The fake mirrors the surface of the N1.5 model that
:class:`~isaac_rl_posttraining.policies.gr00t.flow_sde.Gr00tFlowSDEActorCritic`
touches (backbone, action-head submodules, ``prepare_input``), so the tests pin
the sampling math without the GR00T stack: the deterministic step is Euler
integration, the learner's ``forward`` reproduces the behavior log-probability
from the stored chain, a deterministic decode has zero log-probability, and the
input packing round-trips.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.policies.gr00t.flow_sde import (
    BF16_PIXEL_KEY,
    PIXEL_KEY,
    STOCHASTIC_STEP_KEY,
    FlowSDEConfig,
    Gr00tFlowSDEActorCritic,
    flow_sde_step_moments,
    gaussian_log_prob,
    pack_model_inputs,
    unpack_model_inputs,
)

H, D, EMB, FEAT, STATE = (
    4,
    3,
    6,
    8,
    5,
)  # flow horizon, flow width, head embedding, backbone width, state width


class _CategoryLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(D, EMB)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(actions) + (timesteps.float() / 1000.0)[:, None, None]


class _DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(EMB, EMB)
        self.cross = nn.Linear(FEAT, EMB)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        context = self.cross(encoder_hidden_states).mean(dim=1, keepdim=True)
        return torch.tanh(self.linear(hidden_states) + context)


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            add_pos_embed=True, backbone_embedding_dim=FEAT, input_embedding_dim=EMB
        )
        self.action_horizon = H
        self.action_dim = D
        self.num_inference_timesteps = 4
        self.num_timestep_buckets = 1000
        self.state_encoder = _CategoryLinear(STATE, EMB)
        self.action_encoder = _ActionEncoder()
        self.action_decoder = _CategoryLinear(EMB, D)
        self.future_tokens = nn.Embedding(2, EMB)
        self.position_embedding = nn.Embedding(16, EMB)
        self.model = _DiT()
        self.norm = nn.LayerNorm(FEAT)

    def process_backbone_output(self, backbone_output):
        backbone_output["backbone_features"] = self.norm(
            backbone_output["backbone_features"]
        )
        return backbone_output


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, FEAT)

    def forward(self, inputs):
        if PIXEL_KEY in inputs:
            # The learner path must hand the backbone the bf16 pixels with a flat image axis.
            assert (
                inputs[PIXEL_KEY].dtype == torch.bfloat16
                and inputs[PIXEL_KEY].ndim == 4
            )
        tokens = self.proj(inputs["tokens"])
        return {"backbone_features": tokens, "backbone_attention_mask": inputs["mask"]}


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.action_head = _FakeHead()

    def prepare_input(self, inputs):
        action_inputs = SimpleNamespace(
            state=inputs["state"], embodiment_id=inputs["embodiment_id"]
        )
        return inputs, action_inputs


class _FakeTransform:
    def __init__(self) -> None:
        self.applied = []

    def apply(self, observation):
        self.applied.append(observation)
        batch = len(observation["prompt"])
        return {
            "tokens": torch.zeros(batch, 5, 2),
            "mask": torch.ones(batch, 5, dtype=torch.long),
        }

    def unapply(self, data):
        return {
            "action.a": data["action"][..., :2].numpy(),
            "action.b": data["action"][..., 2:3].numpy(),
        }


def _inputs(batch: int, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "tokens": torch.randn(batch, 5, 2, generator=generator),
        "mask": torch.cat(
            [
                torch.ones(batch, 3, dtype=torch.long),
                torch.zeros(batch, 2, dtype=torch.long),
            ],
            dim=1,
        ),
        "state": torch.randn(batch, 1, STATE, generator=generator),
        "embodiment_id": torch.zeros(batch, dtype=torch.long),
    }


def _actor_critic(noise_level: float = 0.5) -> Gr00tFlowSDEActorCritic:
    torch.manual_seed(0)
    model = _FakeModel()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    return Gr00tFlowSDEActorCritic(
        model,
        transform=_FakeTransform(),
        video_keys=["video.image"],
        state_keys=["state.x"],
        action_keys=["action.a", "action.b"],
        language_key="prompt",
        action_window=3,
        valid_action_dim=2,
        config=FlowSDEConfig(
            noise_level=noise_level, denoising_steps=4, value_hidden_sizes=(8,)
        ),
    )


def test_deterministic_step_is_euler_integration():
    x_t = torch.randn(2, H, D)
    velocity = torch.randn(2, H, D)
    index = torch.tensor([1, 2])
    mean, std = flow_sde_step_moments(
        x_t, velocity, index, num_steps=4, noise_level=0.5, stochastic=False
    )
    assert torch.allclose(mean, x_t + 0.25 * velocity, atol=1e-6)
    assert torch.all(std == 0)


def test_stochastic_step_matches_closed_form():
    x_t = torch.randn(1, H, D)
    velocity = torch.randn(1, H, D)
    k, num_steps, noise_level = 2, 4, 0.5
    mean, std = flow_sde_step_moments(
        x_t,
        velocity,
        torch.tensor([k]),
        num_steps=num_steps,
        noise_level=noise_level,
        stochastic=True,
    )
    t, dt = k / num_steps, 1.0 / num_steps
    sigma = noise_level * np.sqrt((1 - t) / t)
    x0 = x_t - velocity * t
    x1 = x_t + velocity * (1 - t)
    expected_mean = x0 * (1 - (t + dt) - sigma**2 * dt / (2 * (1 - t))) + x1 * (t + dt)
    assert torch.allclose(mean, expected_mean, atol=1e-5)
    assert torch.allclose(
        std, torch.full_like(std, float(np.sqrt(dt) * sigma)), atol=1e-6
    )


def test_gaussian_log_prob_matches_torch_normal():
    sample, mean, std = torch.randn(4), torch.randn(4), torch.rand(4) + 0.1
    expected = torch.distributions.Normal(mean, std).log_prob(sample)
    assert torch.allclose(gaussian_log_prob(sample, mean, std), expected, atol=1e-6)


@pytest.mark.parametrize("stochastic_step", [0, 1, 3])
def test_forward_reproduces_sampled_chain_log_prob(stochastic_step):
    actor_critic = _actor_critic()
    inputs = _inputs(batch=3)
    sample = actor_critic.act(inputs, stochastic_step=stochastic_step)
    assert sample.chains.shape == (3, 5, H, D)
    assert torch.equal(sample.actions, sample.chains[:, -1])
    assert sample.log_prob.shape == (3,) and sample.value.shape == (3,)
    assert torch.all(sample.log_prob != 0)

    observations = dict(inputs)
    observations[STOCHASTIC_STEP_KEY] = torch.full(
        (3,), stochastic_step, dtype=torch.long
    )
    log_prob, entropy, value = actor_critic(observations, sample.chains)
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-4)
    assert torch.allclose(value, sample.value, atol=1e-5)
    assert torch.all(entropy == 0)
    # Only the noised step contributes: chains differ from the deterministic decode there.
    deterministic = actor_critic.act(
        inputs, stochastic_step=None, noise=sample.chains[:, 0]
    )
    assert torch.allclose(
        deterministic.chains[:, : stochastic_step + 1],
        sample.chains[:, : stochastic_step + 1],
        atol=1e-6,
    )
    assert not torch.allclose(
        deterministic.chains[:, stochastic_step + 1],
        sample.chains[:, stochastic_step + 1],
    )


def test_deterministic_decode_has_zero_log_prob_and_is_repeatable():
    actor_critic = _actor_critic()
    inputs = _inputs(batch=2)
    noise = torch.randn(2, H, D)
    first = actor_critic.act(inputs, stochastic_step=None, noise=noise)
    second = actor_critic.act(inputs, stochastic_step=None, noise=noise)
    assert torch.all(first.log_prob == 0)
    assert torch.equal(first.actions, second.actions)
    assert first.stochastic_step is None
    with pytest.raises(ValueError, match="stochastic_step"):
        actor_critic.act(inputs, stochastic_step=4)


def test_learner_gradients_reach_head_and_value_but_not_backbone():
    actor_critic = _actor_critic()
    inputs = _inputs(batch=2)
    sample = actor_critic.act(inputs, stochastic_step=1)
    observations = dict(inputs)
    observations[STOCHASTIC_STEP_KEY] = torch.tensor([1, 1])
    log_prob, _, value = actor_critic(observations, sample.chains)
    (log_prob.sum() + value.sum()).backward()
    assert all(
        parameter.grad is None for parameter in actor_critic.model.backbone.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in actor_critic.model.action_head.parameters()
    )
    assert all(
        parameter.grad is not None for parameter in actor_critic.value_head.parameters()
    )
    trainable = actor_critic.trainable_parameter_names()
    assert trainable and not any(
        name.startswith("model.backbone") for name in trainable
    )


def test_build_inputs_uses_transform_batched_layout_and_decode_concatenates_keys():
    actor_critic = _actor_critic()
    frames = {"video.image": np.zeros((2, 4, 4, 3), dtype=np.uint8)}
    states = {"state.x": np.ones((2, STATE), dtype=np.float32)}
    actor_critic.build_inputs(frames, states, "stack the cubes")
    observation = actor_critic.transform.applied[-1]
    assert observation["video.image"].shape == (2, 1, 4, 4, 3)
    assert observation["state.x"].shape == (2, 1, STATE)
    assert list(observation["prompt"]) == ["stack the cubes"] * 2
    with pytest.raises(KeyError, match="modality keys"):
        actor_critic.build_inputs({}, states, "x")
    decoded = actor_critic.decode(
        torch.arange(2 * H * D, dtype=torch.float32).reshape(2, H, D),
        inputs={},
        states={},
    )
    assert decoded.shape == (2, H, 3)


def test_pack_unpack_round_trip_keeps_bf16_bits_and_image_axis():
    batch, images = 2, 2
    pixels = torch.randn(batch * images, 3, 4, 4)
    inputs = {
        PIXEL_KEY: pixels,
        "eagle_image_sizes": torch.tensor([[4, 4]] * (batch * images)),
        "eagle_input_ids": torch.arange(batch * 7).reshape(batch, 7),
        "state": torch.randn(batch, 1, STATE),
        "embodiment_id": torch.tensor([31] * batch),
    }
    records = pack_model_inputs(inputs, batch)
    assert len(records) == batch
    assert records[0][BF16_PIXEL_KEY].dtype == np.int16
    assert records[0][BF16_PIXEL_KEY].shape == (images, 3, 4, 4)
    assert records[0]["eagle_image_sizes"].shape == (images, 2)
    stacked = {
        key: torch.as_tensor(np.stack([record[key] for record in records]))
        for key in records[0]
    }
    restored = unpack_model_inputs(stacked, "cpu")
    assert restored[PIXEL_KEY].dtype == torch.bfloat16
    assert restored[PIXEL_KEY].shape == pixels.shape
    assert torch.equal(restored[PIXEL_KEY], pixels.to(torch.bfloat16))
    assert restored["eagle_image_sizes"].shape == (batch * images, 2)
    assert torch.equal(restored["eagle_input_ids"], inputs["eagle_input_ids"])
    # Per-sample scalars stay 0-d, so the stacked batch is 1-D like the transform's own output.
    assert records[0]["embodiment_id"].shape == ()
    assert torch.equal(restored["embodiment_id"], inputs["embodiment_id"])
    with pytest.raises(ValueError, match="cannot be split"):
        pack_model_inputs({"odd": torch.zeros(3)}, batch)


def test_flow_sde_config_validation():
    with pytest.raises(ValueError):
        FlowSDEConfig(noise_level=0.0)
    with pytest.raises(ValueError):
        FlowSDEConfig(denoising_steps=0)
    with pytest.raises(ValueError):
        FlowSDEConfig(value_hidden_sizes=())


def test_forward_unpacks_transported_observations():
    actor_critic = _actor_critic()
    inputs = _inputs(batch=2)
    sample = actor_critic.act(inputs, stochastic_step=2)
    transported = dict(inputs)
    transported[PIXEL_KEY] = torch.randn(2 * 2, 3, 4, 4)
    transported["eagle_image_sizes"] = torch.tensor([[4, 4]] * 4)
    records = pack_model_inputs(transported, 2)
    observations = {
        key: torch.as_tensor(np.stack([record[key] for record in records]))
        for key in records[0]
    }
    observations[STOCHASTIC_STEP_KEY] = torch.tensor([2, 2])
    log_prob, _, value = actor_critic(observations, sample.chains)
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-4)
    assert torch.allclose(value, sample.value, atol=1e-5)


def test_cached_features_replace_the_backbone_pass_exactly():
    from isaac_rl_posttraining.policies.feature_cache import (
        FeatureCaching,
        pack_features,
        stack_features,
    )

    actor_critic = _actor_critic()
    assert isinstance(actor_critic, FeatureCaching)
    inputs = _inputs(batch=3)
    sample = actor_critic.act(inputs, stochastic_step=1)
    assert set(sample.cached_features) == set(actor_critic.cached_feature_names)
    # Raw backbone outputs, before the head's trainable normalization.
    assert torch.equal(
        sample.cached_features["backbone_features"],
        actor_critic.model.backbone(inputs)["backbone_features"],
    )

    with_backbone_inputs = dict(inputs)
    with_backbone_inputs[PIXEL_KEY] = torch.randn(3 * 2, 3, 4, 4)
    with_backbone_inputs["eagle_image_sizes"] = torch.tensor([[4, 4]] * 6)
    kept = actor_critic.learner_inputs(with_backbone_inputs)
    assert not any(key.startswith("eagle_") for key in kept) and "state" in kept
    records = pack_model_inputs(kept, 3)
    observations = {
        key: torch.as_tensor(np.stack([record[key] for record in records]))
        for key in records[0]
    }
    observations[STOCHASTIC_STEP_KEY] = torch.tensor([1, 1, 1])
    cached = stack_features(pack_features(sample.cached_features, 3))

    log_prob, _, value = actor_critic(
        observations, sample.chains, cached_features=cached
    )
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-5)
    assert torch.allclose(value, sample.value, atol=1e-6)
    # The head's processing of the cached tokens stays in the gradient path.
    (log_prob.sum() + value.sum()).backward()
    assert actor_critic.model.action_head.norm.weight.grad is not None
    with pytest.raises(KeyError, match="cached features lack"):
        actor_critic(
            observations,
            sample.chains,
            cached_features={"backbone_features": cached["backbone_features"]},
        )
