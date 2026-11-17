# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trainable-policy contract behind flow-SDE PPO: registry, loading spec, and the N1.7 and pi0 adapters.

The adapters run on stand-ins that mirror the surface of the real models (the
N1.7 head's submodules; openpi's PyTorch pi0), so the tests pin the shared
math on each base model without the model stacks: the learner's ``forward``
reproduces the behavior log-probability from the stored chain, gradients reach
the trainable part and not the frozen one, and each base model's own conventions
(no future tokens and a flattened state token for N1.7; reversed flow time and
a negated velocity for pi0) hold.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.algorithms.iql.critics import TokenValueHead
from isaac_rl_posttraining.policies import available_base_models, get_base_model
from isaac_rl_posttraining.policies.flow_sde import (
    FlowConditioning,
    FlowSDEActorCritic,
    masked_mean,
)
from isaac_rl_posttraining.policies.gr00t.flow_sde import Gr00tFlowSDEActorCritic
from isaac_rl_posttraining.policies.gr00t.flow_sde_n17 import Gr00tN17FlowSDEActorCritic
from isaac_rl_posttraining.policies.pi0.flow_sde import (
    IMAGE_SLOTS,
    Pi0FlowSDEActorCritic,
    attention_2d_masks,
    sample_openpi_time,
)
from isaac_rl_posttraining.policies.trainable import (
    STOCHASTIC_STEP_KEY,
    FlowSDEConfig,
    TrainablePolicy,
    TrainablePolicySpec,
    available_trainable_policies,
    get_trainable_policy,
    register_trainable_policy,
    resolve_trainable_policy_name,
)

# ---------------------------------------------------------------------------
# Registry and loading spec
# ---------------------------------------------------------------------------


def test_registry_resolves_adapters_lazily_and_names_the_generation():
    assert set(available_trainable_policies()) >= {"gr00t_n15", "gr00t_n17"}
    assert get_trainable_policy("gr00t_n15") is Gr00tFlowSDEActorCritic
    assert get_trainable_policy("gr00t_n17") is Gr00tN17FlowSDEActorCritic
    with pytest.raises(KeyError, match="unknown trainable policy"):
        get_trainable_policy("mock")
    assert resolve_trainable_policy_name("gr00t_n17", "/nowhere") == "gr00t_n17"
    with pytest.raises(KeyError, match="no trainable adapter"):
        resolve_trainable_policy_name("mock", "/nowhere")
    # The base-model names the fine-tuning ones alias resolve to the one GR00T base model.
    assert (
        get_base_model("gr00t_n15")
        is get_base_model("gr00t")
        is get_base_model("gr00t_n17")
    )
    assert {"gr00t_n15", "gr00t_n17"} <= set(available_base_models())


def test_gr00t_family_name_resolves_from_the_checkpoint(tmp_path):
    with pytest.raises(
        ValueError, match="base_model=gr00t_n15 or base_model=gr00t_n17"
    ):
        resolve_trainable_policy_name("gr00t", str(tmp_path / "missing"))
    n15 = tmp_path / "n15"
    n15.mkdir()
    (n15 / "config.json").write_text(json.dumps({"model_type": "gr00t_n1_5"}))
    n17 = tmp_path / "n17"
    n17.mkdir()
    (n17 / "config.json").write_text(json.dumps({"model_type": "gr00t_n1d7"}))
    assert resolve_trainable_policy_name("gr00t", str(n15)) == "gr00t_n15"
    assert resolve_trainable_policy_name("gr00t", str(n17)) == "gr00t_n17"


def test_register_trainable_policy_mirrors_the_base_model_registry():
    register_trainable_policy(
        "unit_test_policy",
        "isaac_rl_posttraining.policies.gr00t.flow_sde:Gr00tFlowSDEActorCritic",
    )
    try:
        assert get_trainable_policy("unit_test_policy") is Gr00tFlowSDEActorCritic
        with pytest.raises(ValueError, match="already registered"):
            register_trainable_policy("unit_test_policy", "a:B")
        with pytest.raises(ValueError, match="module:Class"):
            register_trainable_policy("bad_target", "no_colon", override=True)
        register_trainable_policy(
            "unit_test_policy",
            "a:B",
            requires=("surely_not_installed_pkg",),
            override=True,
        )
        assert "unit_test_policy" not in available_trainable_policies()
        with pytest.raises(ImportError, match="optional dependencies"):
            get_trainable_policy("unit_test_policy")
    finally:
        from isaac_rl_posttraining.policies import trainable

        trainable._TRAINABLE_POLICIES.pop("unit_test_policy", None)
        trainable._OPTIONAL_REQUIREMENTS.pop("unit_test_policy", None)


def test_spec_builds_through_the_adapter_class(monkeypatch):
    calls = []

    def load(cls, checkpoint, **kwargs):
        calls.append((cls, checkpoint, kwargs))
        return "built"

    monkeypatch.setattr(Gr00tN17FlowSDEActorCritic, "load", classmethod(load))
    spec = TrainablePolicySpec(
        base_model="gr00t_n17",
        checkpoint="/ckpt",
        flow_sde=FlowSDEConfig(denoising_steps=2),
        value_head_seed=5,
        options={"embodiment_tag": "gr1", "action_window": 8},
    )
    assert spec.frozen_submodules == ("model.backbone",)
    assert spec.build("cpu") == "built"
    assert calls == [
        (
            Gr00tN17FlowSDEActorCritic,
            "/ckpt",
            {
                "device": "cpu",
                "flow_sde": FlowSDEConfig(denoising_steps=2),
                "value_head_seed": 5,
                "embodiment_tag": "gr1",
                "action_window": 8,
            },
        )
    ]
    # Without a device the spec loads onto the process's current device; pin it so any box reads the same.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec.build()
    assert calls[-1][2]["device"] == "cpu"
    with pytest.raises(KeyError, match="unknown trainable policy"):
        TrainablePolicySpec(
            base_model="nope", checkpoint="/ckpt", flow_sde=FlowSDEConfig()
        )
    with pytest.raises(ValueError, match="checkpoint"):
        TrainablePolicySpec(
            base_model="gr00t_n15", checkpoint="", flow_sde=FlowSDEConfig()
        )


# ---------------------------------------------------------------------------
# GR00T N1.7 adapter on a stand-in head
# ---------------------------------------------------------------------------

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


class _N17DiT(nn.Module):
    """N1.7 conditions on state + action tokens only (no future tokens)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(EMB, EMB)
        self.cross = nn.Linear(FEAT, EMB)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        assert hidden_states.shape[1] == 1 + H, "one state token plus the action tokens"
        context = self.cross(encoder_hidden_states).mean(dim=1, keepdim=True)
        return torch.tanh(self.linear(hidden_states) + context)


class _N17Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            add_pos_embed=True,
            backbone_embedding_dim=FEAT,
            input_embedding_dim=EMB,
            use_alternate_vl_dit=False,
        )
        self.action_horizon = H
        self.action_dim = D
        self.num_inference_timesteps = 4
        self.num_timestep_buckets = 1000
        self.state_encoder = _CategoryLinear(
            2 * STATE, EMB
        )  # two history steps flattened into one token
        self.action_encoder = _ActionEncoder()
        self.action_decoder = _CategoryLinear(EMB, D)
        self.position_embedding = nn.Embedding(16, EMB)
        self.model = _N17DiT()
        self.vlln = nn.LayerNorm(FEAT)

    def process_backbone_output(self, backbone_output):
        backbone_output["backbone_features"] = self.vlln(
            backbone_output["backbone_features"]
        )
        return backbone_output


class _N17Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, FEAT)

    def forward(self, inputs):
        tokens = self.proj(inputs["tokens"])
        return {
            "backbone_features": tokens,
            "backbone_attention_mask": inputs["mask"].bool(),
            "image_mask": torch.zeros_like(inputs["mask"], dtype=torch.bool),
        }


class _N17Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _N17Backbone()
        self.action_head = _N17Head()

    def prepare_input(self, inputs):
        return inputs, SimpleNamespace(
            state=inputs["state"], embodiment_id=inputs["embodiment_id"]
        )


MODALITY_JSON = {
    "state": {"arm": {"start": 0, "end": 3}, "hand": {"start": 3, "end": 5}},
    "action": {"arm": {"start": 0, "end": 2}},
}


def _n17_actor_critic(noise_level: float = 0.5) -> Gr00tN17FlowSDEActorCritic:
    torch.manual_seed(0)
    model = _N17Model()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    return Gr00tN17FlowSDEActorCritic(
        model,
        processor=object(),
        embodiment_tag="gr1",
        modality_json=MODALITY_JSON,
        video_keys=["video.ego_view"],
        action_keys=["action.arm"],
        action_window=3,
        config=FlowSDEConfig(
            noise_level=noise_level, denoising_steps=4, value_hidden_sizes=(8,)
        ),
    )


def _n17_inputs(batch: int, seed: int = 0) -> dict[str, torch.Tensor]:
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
        "state": torch.randn(batch, 2, STATE, generator=generator),  # [B, history, S]
        "embodiment_id": torch.zeros(batch, dtype=torch.long),
    }


def test_n17_adapter_declares_its_contract():
    actor_critic = _n17_actor_critic()
    assert isinstance(actor_critic, TrainablePolicy)
    assert actor_critic.frozen_submodules == ("model.backbone",)
    assert (
        actor_critic.requires_cached_features
        and actor_critic.cached_feature_names
        == (
            "backbone_features",
            "backbone_attention_mask",
        )
    )
    # State groups in layout order, the raw action width, and the head's chunk geometry.
    assert actor_critic.state_keys == [
        "state.arm",
        "state.hand",
    ] and actor_critic.video_keys == ["video.ego_view"]
    assert (
        actor_critic.valid_action_dim,
        actor_critic.chunk_shape,
        actor_critic.denoising_steps,
    ) == (2, (H, D), 4)
    assert actor_critic.learner_inputs(
        _n17_inputs(1) | {"input_ids": torch.zeros(1, 3)}
    ).keys() == {
        "state",
        "embodiment_id",
    }


@pytest.mark.parametrize("stochastic_step", [0, 2, 3])
def test_n17_forward_reproduces_sampled_chain_log_prob(stochastic_step):
    actor_critic = _n17_actor_critic()
    inputs = _n17_inputs(batch=3)
    sample = actor_critic.act(inputs, stochastic_step=stochastic_step)
    assert sample.chains.shape == (3, 5, H, D) and torch.all(sample.log_prob != 0)
    assert set(sample.cached_features) == set(actor_critic.cached_feature_names)

    observations = dict(inputs)
    observations[STOCHASTIC_STEP_KEY] = torch.full(
        (3,), stochastic_step, dtype=torch.long
    )
    log_prob, entropy, value = actor_critic(observations, sample.chains)
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-4)
    assert torch.allclose(value, sample.value, atol=1e-5)
    assert torch.all(entropy == 0)
    deterministic = actor_critic.act(
        inputs, stochastic_step=None, noise=sample.chains[:, 0]
    )
    assert torch.all(deterministic.log_prob == 0)
    assert torch.allclose(
        deterministic.chains[:, : stochastic_step + 1],
        sample.chains[:, : stochastic_step + 1],
        atol=1e-6,
    )


def test_n17_cached_features_replace_the_backbone_and_gradients_stay_in_the_head():
    from isaac_rl_posttraining.policies.feature_cache import (
        pack_features,
        stack_features,
    )

    actor_critic = _n17_actor_critic()
    inputs = _n17_inputs(batch=2)
    sample = actor_critic.act(inputs, stochastic_step=1)
    observations = dict(actor_critic.learner_inputs(inputs))
    observations[STOCHASTIC_STEP_KEY] = torch.tensor([1, 1])
    cached = stack_features(pack_features(sample.cached_features, 2))
    log_prob, _, value = actor_critic(
        observations, sample.chains, cached_features=cached
    )
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-5)
    assert torch.allclose(value, sample.value, atol=1e-6)
    (log_prob.sum() + value.sum()).backward()
    assert all(
        parameter.grad is None for parameter in actor_critic.model.backbone.parameters()
    )
    assert actor_critic.model.action_head.vlln.weight.grad is not None
    assert all(
        parameter.grad is not None for parameter in actor_critic.value_head.parameters()
    )
    with pytest.raises(KeyError, match="cached features lack"):
        actor_critic(
            observations,
            sample.chains,
            cached_features={"backbone_features": cached["backbone_features"]},
        )


def test_n17_build_inputs_flattens_state_groups_and_decode_uses_the_processor(
    monkeypatch,
):
    from isaac_rl_posttraining.policies.gr00t import data as gr00t_data

    actor_critic = _n17_actor_critic()
    seen = {}

    def build_inputs(processor, obs_batch, embodiment_tag, modality_json, task, device):
        seen.update(obs_batch=obs_batch, embodiment_tag=embodiment_tag, task=task)
        return {
            "state": torch.zeros(2, 1, STATE),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
        }

    def decode_action_chunks(
        processor, normalized, embodiment_tag, modality_json, states
    ):
        seen.update(decoded_states=states, normalized=normalized)
        return np.ones((normalized.shape[0], H, 2), dtype=np.float32)

    monkeypatch.setattr(gr00t_data, "build_inputs", build_inputs)
    monkeypatch.setattr(gr00t_data, "decode_action_chunks", decode_action_chunks)
    frames = {"video.ego_view": np.zeros((2, 4, 4, 3), dtype=np.uint8)}
    states = {
        "state.arm": np.ones((2, 3), dtype=np.float32),
        "state.hand": 2 * np.ones((2, 2), dtype=np.float32),
    }
    inputs = actor_critic.build_inputs(frames, states, "pick up the drill")
    np.testing.assert_array_equal(seen["obs_batch"]["state"], [[1, 1, 1, 2, 2]] * 2)
    assert (
        seen["obs_batch"]["video.ego_view"].shape == (2, 4, 4, 3)
        and seen["task"] == "pick up the drill"
    )
    decoded = actor_critic.decode(torch.zeros(2, H, D), inputs=inputs, states=states)
    assert decoded.shape == (2, 3, 2), "sliced to the executed window"
    np.testing.assert_array_equal(seen["decoded_states"], [[1, 1, 1, 2, 2]] * 2)
    with pytest.raises(ValueError, match="state.hand"):
        actor_critic.build_inputs(
            frames, {**states, "state.hand": np.ones((2, 3), dtype=np.float32)}, "x"
        )
    with pytest.raises(KeyError, match="modality keys"):
        actor_critic.build_inputs({}, states, "x")


def test_n17_spec_options_require_the_baked_embodiment_and_the_window():
    base_model = SimpleNamespace(
        embodiment_tag="gr1", action_horizon=16, backbone_model_name_or_path=None
    )
    assert Gr00tN17FlowSDEActorCritic.spec_options(base_model) == {
        "embodiment_tag": "gr1",
        "action_window": 16,
        "backbone_model_name_or_path": None,
    }
    with pytest.raises(ValueError, match="embodiment_tag"):
        Gr00tN17FlowSDEActorCritic.spec_options(
            SimpleNamespace(embodiment_tag=None, action_horizon=16)
        )
    with pytest.raises(ValueError, match="data_config"):
        Gr00tFlowSDEActorCritic.spec_options(
            SimpleNamespace(data_config=None, embodiment_tag="x")
        )


# ---------------------------------------------------------------------------
# pi0 adapter on a stand-in of openpi's PyTorch model
# ---------------------------------------------------------------------------

PI0_H, PI0_D, PI0_WIDTH, TOKENS = 4, 3, 6, 5


class _PaliGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(PI0_WIDTH, PI0_WIDTH)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=PI0_WIDTH)
        )


class _PaliGemmaWithExpert(nn.Module):
    """Prefix pass returns hidden states and a cache; the cache is a function of the prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _PaliGemma()
        self.gemma_expert = nn.Linear(PI0_WIDTH, PI0_WIDTH)
        self.calls = []

    def forward(
        self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache
    ):
        self.calls.append(attention_mask.shape)
        prefix = self.paligemma.proj(inputs_embeds[0])
        return (prefix, None), prefix.mean(dim=1)


class _FakePi0(nn.Module):
    """openpi's ``PI0Pytorch`` surface: ``embed_prefix``, ``paligemma_with_expert.forward``, ``denoise_step``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_horizon=PI0_H, action_dim=PI0_D)
        self.paligemma_with_expert = _PaliGemmaWithExpert()
        self.image_proj = nn.Linear(3 * 2 * 2, PI0_WIDTH)
        self.token_embedding = nn.Embedding(10, PI0_WIDTH)
        self.state_proj = nn.Linear(PI0_D, PI0_WIDTH)
        self.action_in_proj = nn.Linear(PI0_D, PI0_WIDTH)
        self.action_out_proj = nn.Linear(PI0_WIDTH, PI0_D)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = [
            self.image_proj(image.reshape(image.shape[0], -1))[:, None]
            for image in images
        ]
        pads = [mask[:, None] for mask in img_masks]
        embs.append(self.token_embedding(lang_tokens))
        pads.append(lang_masks)
        prefix = torch.cat(embs, dim=1)
        pad = torch.cat(pads, dim=1)
        return prefix, pad, torch.zeros_like(pad, dtype=torch.long)

    def denoise_step(self, state, prefix_pkv, prefix_pad_masks, x_t, timestep):
        # Velocity depends on the prefix cache, the state, the chain state and the time, like the real expert.
        conditioning = self.paligemma_with_expert.gemma_expert(
            prefix_pkv + self.state_proj(state)
        )
        hidden = torch.tanh(
            self.action_in_proj(x_t)
            + conditioning[:, None]
            + timestep[:, None, None].to(x_t.dtype)
        )
        return self.action_out_proj(hidden)

    def sample_actions(self, noise, inputs, num_steps):
        """openpi's own Euler decode: from noise at t = 1 to actions at t = 0."""
        images = [inputs[f"image/{slot}"] for slot in IMAGE_SLOTS]
        masks = [inputs[f"image_mask/{slot}"] for slot in IMAGE_SLOTS]
        embs, pad, att = self.embed_prefix(
            images, masks, inputs["tokenized_prompt"], inputs["tokenized_prompt_mask"]
        )
        (_, _), cache = self.paligemma_with_expert.forward(
            attention_mask=attention_2d_masks(pad, att),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        dt = -1.0 / num_steps
        x_t, time = noise, 1.0
        while time >= -dt / 2:
            v_t = self.denoise_step(
                inputs["state"], cache, pad, x_t, torch.full((x_t.shape[0],), time)
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t


def _pi0_transform(sample):
    """A stand-in for openpi's input pipeline: 2x2 images in three slots, a padded state, tokenized prompt."""
    image = np.asarray(sample["observation/image"], dtype=np.float32) / 255.0
    state = np.zeros(PI0_D, dtype=np.float32)
    state[: sample["observation/state"].shape[0]] = sample["observation/state"]
    tokens = np.array([len(sample["prompt"]) % 10] * 3 + [0, 0])
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": np.zeros_like(image),
            "right_wrist_0_rgb": np.zeros_like(image),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": False,
            "right_wrist_0_rgb": False,
        },
        "state": state,
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": tokens > 0,
    }


def _pi0_output_transform(outputs):
    return {
        "actions": np.asarray(outputs["actions"])[:, :2]
        + np.asarray(outputs["state"])[None, :2]
    }


def _pi0_actor_critic() -> Pi0FlowSDEActorCritic:
    torch.manual_seed(0)
    model = _FakePi0()
    for parameter in model.paligemma_with_expert.paligemma.parameters():
        parameter.requires_grad_(False)
    return Pi0FlowSDEActorCritic(
        model,
        input_transform=_pi0_transform,
        output_transform=_pi0_output_transform,
        observation_keys={
            "video.image": "observation/image",
            "state.q": "observation/state",
        },
        action_window=3,
        valid_action_dim=2,
        prefix_width=PI0_WIDTH,
        config=FlowSDEConfig(
            noise_level=0.5, denoising_steps=4, value_hidden_sizes=(8,)
        ),
    )


def _pi0_observations(batch: int):
    frames = {
        "video.image": (np.arange(batch * 2 * 2 * 3) % 255)
        .astype(np.uint8)
        .reshape(batch, 2, 2, 3)
    }
    states = {"state.q": np.ones((batch, 2), dtype=np.float32)}
    return frames, states


def test_pi0_adapter_declares_its_contract_and_builds_channels_first_inputs():
    actor_critic = _pi0_actor_critic()
    assert isinstance(actor_critic, TrainablePolicy)
    assert actor_critic.frozen_submodules == ("model.paligemma_with_expert.paligemma",)
    assert (
        actor_critic.cached_feature_names == ()
        and not actor_critic.requires_cached_features
    )
    assert (actor_critic.video_keys, actor_critic.state_keys) == (
        ["video.image"],
        ["state.q"],
    )
    frames, states = _pi0_observations(2)
    inputs = actor_critic.build_inputs(frames, states, "stack")
    assert inputs["image/base_0_rgb"].shape == (2, 3, 2, 2) and inputs[
        "state"
    ].shape == (2, PI0_D)
    assert inputs["image_mask/left_wrist_0_rgb"].tolist() == [False, False]
    assert (
        inputs["tokenized_prompt"].shape == (2, TOKENS)
        and inputs["tokenized_prompt_mask"].dtype == torch.bool
    )
    with pytest.raises(ValueError, match="video."):
        Pi0FlowSDEActorCritic(
            _FakePi0(),
            input_transform=_pi0_transform,
            output_transform=_pi0_output_transform,
            observation_keys={
                "image": "observation/image",
                "state.q": "observation/state",
            },
            action_window=3,
            valid_action_dim=2,
            prefix_width=PI0_WIDTH,
            config=FlowSDEConfig(),
        )


def test_pi0_deterministic_decode_matches_openpi_euler_integration():
    actor_critic = _pi0_actor_critic()
    frames, states = _pi0_observations(2)
    inputs = actor_critic.build_inputs(frames, states, "stack")
    noise = torch.randn(2, PI0_H, PI0_D)
    sample = actor_critic.act(inputs, stochastic_step=None, noise=noise)
    reference = actor_critic.model.sample_actions(noise, inputs, num_steps=4)
    assert torch.allclose(sample.actions, reference, atol=1e-5)
    assert torch.all(sample.log_prob == 0) and sample.cached_features == {}
    decoded = actor_critic.decode(sample.actions, inputs=inputs, states=states)
    assert decoded.shape == (2, 3, 2)
    expected = sample.actions[:, :3, :2].numpy() + inputs["state"][:, None, :2].numpy()
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


@pytest.mark.parametrize("stochastic_step", [0, 1, 3])
def test_pi0_forward_reproduces_sampled_chain_log_prob(stochastic_step):
    actor_critic = _pi0_actor_critic()
    frames, states = _pi0_observations(3)
    inputs = actor_critic.build_inputs(frames, states, "stack")
    sample = actor_critic.act(inputs, stochastic_step=stochastic_step)
    observations = dict(inputs)
    observations[STOCHASTIC_STEP_KEY] = torch.full(
        (3,), stochastic_step, dtype=torch.long
    )
    log_prob, entropy, value = actor_critic(observations, sample.chains)
    assert torch.allclose(log_prob, sample.log_prob, atol=1e-4)
    assert torch.allclose(value, sample.value, atol=1e-5)
    assert torch.all(entropy == 0)
    (log_prob.sum() + value.sum()).backward()
    assert all(
        parameter.grad is None
        for parameter in actor_critic.model.paligemma_with_expert.paligemma.parameters()
    )
    assert actor_critic.model.action_out_proj.weight.grad is not None
    assert actor_critic.model.paligemma_with_expert.gemma_expert.weight.grad is not None
    with pytest.raises(ValueError, match="no cacheable features"):
        actor_critic(
            observations, sample.chains, cached_features={"x": torch.zeros(3, 1)}
        )


def test_pi0_value_pools_the_prefix_and_spec_options_validate():
    actor_critic = _pi0_actor_critic()
    frames, states = _pi0_observations(2)
    conditioning = actor_critic.condition(
        actor_critic.build_inputs(frames, states, "stack")
    )
    features = actor_critic.value_features(conditioning)
    assert features.shape == (2, PI0_WIDTH + PI0_D)
    pooled = masked_mean(
        conditioning.context.prefix_hidden, conditioning.context.prefix_pad_mask
    )
    assert torch.allclose(features[:, :PI0_WIDTH], pooled)
    base_model = SimpleNamespace(
        openpi_config="pi05_libero",
        observation_keys={"video.image": "observation/image"},
        action_horizon=10,
        action_dim=7,
    )
    assert Pi0FlowSDEActorCritic.spec_options(base_model) == {
        "openpi_config": "pi05_libero",
        "observation_keys": {"video.image": "observation/image"},
        "action_window": 10,
        "action_dim": 7,
    }
    with pytest.raises(ValueError, match="openpi_config"):
        Pi0FlowSDEActorCritic.spec_options(
            SimpleNamespace(
                openpi_config=None, observation_keys={}, action_horizon=1, action_dim=1
            )
        )


def test_attention_masks_follow_openpi_block_causality():
    pad = torch.tensor([[True, True, True, False]])
    att = torch.tensor([[0, 0, 1, 0]])
    masks = attention_2d_masks(pad, att)
    # Block 0 (tokens 0, 1) attends within itself; token 2 (block 1) sees everything valid; padding sees nothing.
    assert masks[0].tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [False, False, False, False],
    ]


def test_every_adapter_is_a_flow_sde_actor_critic():
    for name in available_trainable_policies():
        assert issubclass(get_trainable_policy(name), FlowSDEActorCritic), name


class _ToyFlowSDE(FlowSDEActorCritic):
    """Identity-decode flow with zero velocity, for the offline contract tests."""

    def __init__(
        self,
        *,
        action_window: int = 4,
        valid_action_dim: int = 3,
        flow_horizon: int = 4,
        flow_width: int = 3,
    ):
        super().__init__(
            video_keys=["cam"],
            state_keys=["state.arm"],
            action_window=action_window,
            valid_action_dim=valid_action_dim,
            chunk_shape=(flow_horizon, flow_width),
            value_input_dim=8,
            config=FlowSDEConfig(denoising_steps=2, value_hidden_sizes=(8,)),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    @property
    def device(self) -> torch.device:
        return next(self.value_head.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.value_head.parameters()).dtype

    def build_inputs(self, frames, states, instruction):
        del states, instruction
        batch = int(np.asarray(frames["cam"]).shape[0])
        return {"token": torch.zeros(batch, 1)}

    def decode(self, normalized, *, inputs, states):
        del inputs, states
        return (
            normalized[:, : self.action_window, : self.valid_action_dim]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    def condition(self, inputs, cached_features=None):
        del cached_features
        batch = int(next(iter(inputs.values())).shape[0])
        tokens = torch.zeros(batch, 3, 8, device=self.device, dtype=self.dtype)
        return FlowConditioning(context=tokens, cached={}, batch_size=batch)

    def velocity(self, conditioning, x_t, step_index):
        del conditioning, step_index
        return torch.zeros_like(x_t)

    def value_features(self, conditioning):
        return conditioning.context.mean(dim=1)

    @classmethod
    def spec_options(cls, base_model):
        del base_model
        return {}

    @classmethod
    def load(cls, checkpoint, *, device, flow_sde, value_head_seed=0, **options):
        del checkpoint, device, flow_sde, value_head_seed, options
        return cls()


def test_flow_sde_infer_decodes_fresh_noise_per_call():
    policy = _ToyFlowSDE()
    obs = {
        "cam": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "state.arm": np.zeros((2, 2), dtype=np.float32),
    }
    actions = policy.infer(obs)
    assert actions.shape == (2, 4, 3)
    assert not np.allclose(actions, policy.infer(obs))


def test_actor_bc_loss_builds_x_t_at_the_grid_time_it_states():
    """The shared velocity is asked at grid step k for an x_t built at t = k / K.

    pi0 replaces this grid with openpi's continuous Beta time; see
    ``test_pi0_bc_loss_samples_openpi_time``.
    """
    policy = _ToyFlowSDE()
    seen = []

    def velocity(conditioning, x_t, step_index):
        del conditioning
        seen.append((x_t.clone(), step_index.clone()))
        return torch.zeros_like(x_t)

    policy.velocity = velocity
    action = torch.ones(64, 4, 3)
    torch.manual_seed(0)
    noise = torch.randn(64, 4, 3)
    torch.manual_seed(0)
    policy.actor_bc_loss({"token": torch.zeros(64, 1), "action": action})
    x_t, step = seen[0]
    time = (step.float() / policy.denoising_steps).reshape(64, 1, 1)
    torch.testing.assert_close(x_t, (1 - time) * noise + time * action)


def test_pi0_bc_loss_samples_openpi_time():
    """Continuous t ~ Beta(1.5, 1) * 0.999 + 0.001, and the shared-time flip.

    openpi builds ``x_t = t * noise + (1 - t) * action`` and regresses
    ``noise - action``. Shared time is ``1 - t``, so the same mixture is
    ``(1 - t_shared) * noise + t_shared * action`` and the target is
    ``action - noise``, which is what ``velocity`` returns after negating
    pi0's prediction.
    """
    torch.manual_seed(0)
    draw = sample_openpi_time(4096, torch.device("cpu"))
    assert float(draw.min()) >= 0.001 - 1e-6
    assert float(draw.max()) <= 1.0 + 1e-6
    assert draw.unique().numel() > 100
    assert abs(float(draw.mean()) - (1.5 / 2.5 * 0.999 + 0.001)) < 0.02
    torch.manual_seed(0)
    assert torch.equal(draw, sample_openpi_time(4096, torch.device("cpu")))

    policy = _pi0_actor_critic()
    frames, states = _pi0_observations(32)
    action = torch.randn(32, PI0_H, PI0_D)
    batch = {**policy.build_inputs(frames, states, "stack"), "action": action}
    torch.manual_seed(123)
    noise = torch.randn(32, PI0_H, PI0_D)
    pi0_time = sample_openpi_time(32, torch.device("cpu"))
    shared = (1.0 - pi0_time).reshape(32, 1, 1)
    expected = (1.0 - shared) * noise + shared * action
    seen: dict[str, torch.Tensor] = {}

    def denoise_step(state, prefix_pkv, prefix_pad_masks, x_t, timestep):
        del state, prefix_pkv, prefix_pad_masks
        seen["x_t"] = x_t.detach().clone()
        seen["time"] = timestep.detach().clone()
        return (noise - action).to(dtype=x_t.dtype)

    policy.model.denoise_step = denoise_step
    torch.manual_seed(123)
    loss = policy.actor_bc_loss(batch)
    torch.testing.assert_close(seen["time"].float(), pi0_time)
    torch.testing.assert_close(seen["x_t"].float(), expected)
    assert torch.all(loss.abs() < 1e-5)
    grid = (
        1.0
        - torch.arange(policy.denoising_steps, dtype=torch.float32)
        / policy.denoising_steps
    )
    distance = (pi0_time[:, None] - grid[None, :]).abs().min(dim=1).values
    assert not bool((distance < 1e-4).all())


def test_flow_sde_feature_tokens_and_actor_bc_loss():
    policy = _ToyFlowSDE()
    inputs = {"token": torch.zeros(3, 1), "action": torch.zeros(3, 4, 3)}
    tokens, mask = policy.feature_tokens({"token": torch.zeros(3, 1)})
    assert mask is None and tokens.shape == (3, 3, 8)
    loss = policy.actor_bc_loss(inputs)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()


def _same_pool_different_sequences(batch: int, length: int, dim: int):
    """Two sequences with the same masked mean and different padded values."""
    valid = torch.zeros(batch, length, dtype=torch.bool)
    valid[:, : length - 1] = True
    base = torch.randn(batch, length, dim)
    alt = base.clone()
    alt[:, 0] = base[:, 0] + base[:, 1]
    alt[:, 1] = 0
    alt[:, -1] = base[:, -1] + 4
    return base, alt, valid


def _v_of_sequences(
    policy, contexts
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def condition(inputs, cached_features=None):
        del cached_features
        ctx = contexts[int(inputs["which"])]
        return FlowConditioning(context=ctx, cached={}, batch_size=2)

    policy.condition = condition
    tokens_a, mask_a = policy.feature_tokens({"which": 0})
    tokens_b, mask_b = policy.feature_tokens({"which": 1})
    assert torch.allclose(
        policy.value_features(
            FlowConditioning(context=contexts[0], cached={}, batch_size=2)
        ),
        policy.value_features(
            FlowConditioning(context=contexts[1], cached={}, batch_size=2)
        ),
    )
    head = TokenValueHead(int(tokens_a.shape[-1]), n_heads=2, hidden=16)
    value_a = head(tokens_a.float(), key_padding_mask=mask_a)
    value_b = head(tokens_b.float(), key_padding_mask=mask_b)
    tweaked = tokens_a.clone()
    tweaked[mask_a] = 17
    assert torch.allclose(head(tweaked, key_padding_mask=mask_a), value_a)
    assert not torch.allclose(head(tweaked), value_a)
    assert not tokens_a.requires_grad and not tokens_b.requires_grad
    return value_a, value_b, tokens_a, mask_a


def test_n17_critic_reads_vl_tokens_not_the_mean_pool():
    torch.manual_seed(0)
    policy = _n17_actor_critic()
    seq_a, seq_b, valid = _same_pool_different_sequences(2, 4, FEAT)
    state = torch.randn(2, 1, EMB, requires_grad=True)
    seq_a.requires_grad_(True)
    contexts = [
        SimpleNamespace(vl_embs=seq, mask=valid, state_features=state)
        for seq in (seq_a, seq_b)
    ]
    value_a, value_b, tokens, mask = _v_of_sequences(policy, contexts)
    assert not torch.allclose(value_a, value_b)
    assert tokens.shape == (2, 5, FEAT) and mask[:, -1].eq(False).all()
    # State is narrower than the backbone tokens, so it occupies the leading coordinates.
    assert torch.allclose(tokens[:, -1, :EMB], state.detach().squeeze(1))
    assert torch.count_nonzero(tokens[:, -1, EMB:]) == 0


def test_pi0_critic_reads_prefix_tokens_not_the_mean_pool():
    torch.manual_seed(0)
    policy = _pi0_actor_critic()
    seq_a, seq_b, valid = _same_pool_different_sequences(2, 4, PI0_WIDTH)
    state = torch.randn(2, PI0_D)
    contexts = [
        SimpleNamespace(prefix_hidden=seq, prefix_pad_mask=valid, state=state)
        for seq in (seq_a, seq_b)
    ]
    value_a, value_b, tokens, mask = _v_of_sequences(policy, contexts)
    assert not torch.allclose(value_a, value_b)
    assert tokens.shape == (2, 5, PI0_WIDTH) and not mask[:, -1].any()
    assert torch.allclose(tokens[:, -1], policy.model.state_proj(state))


def test_feature_tokens_do_not_train_the_backbone_or_the_action_head():
    torch.manual_seed(0)
    n17 = _n17_actor_critic()
    tokens, mask = n17.feature_tokens(
        {
            "tokens": torch.randn(2, 3, 2),
            "mask": torch.tensor([[True, True, False], [True, False, False]]),
            "state": torch.randn(2, 2, STATE),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
        }
    )
    assert tokens.shape == (2, 4, FEAT)
    assert mask.tolist() == [[False, False, True, False], [False, True, True, False]]
    TokenValueHead(FEAT, n_heads=2, hidden=8)(
        tokens.float(), key_padding_mask=mask
    ).sum().backward()
    assert all(parameter.grad is None for parameter in n17.parameters())

    pi0 = _pi0_actor_critic()
    frames, states = _pi0_observations(2)
    tokens, mask = pi0.feature_tokens(pi0.build_inputs(frames, states, "stack"))
    assert tokens.shape[-1] == PI0_WIDTH and tokens.shape[1] == mask.shape[1] > 1
    assert mask.dtype == torch.bool and mask.any() and not mask[:, -1].any()
    TokenValueHead(PI0_WIDTH, n_heads=2, hidden=8)(
        tokens.float(), key_padding_mask=mask
    ).sum().backward()
    assert all(parameter.grad is None for parameter in pi0.parameters())
