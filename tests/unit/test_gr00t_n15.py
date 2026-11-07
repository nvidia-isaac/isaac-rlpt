# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.5 support, exercised with fakes (no GPU, no ``gr00t``, no network).

A tiny fake N1.5 model (config, Eagle-style backbone output with an
attention mask, a flow head drawing ``torch.randn`` like the real one), a
fake modality transform (``apply`` / ``unapply`` / ``set_metadata``), and a
fake data config stand in for the real stack; the checkpoint loader is
monkeypatched to hand them to :class:`Gr00tN15Policy`. Pinned here:

* the latent-steered decode substitutes the injected noise for the head's
  ``torch.randn`` draw of shape ``(B, H, 32)``,
* backbone features pool over valid tokens only, to ``[B, 2048]`` float32,
* our observation dict maps to the transform's batched layout,
* ``act`` decodes to ``[16, 28]`` in action-key order through ``unapply``,
* the N1.5 path is selected by ``config.json``'s ``model_type`` alone,
* ``denoising_steps`` reaches the head, and the config fields reach the wrapper.
"""

from __future__ import annotations

import enum
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from conftest import REPO_ROOT
from torch import nn

from isaac_rl_posttraining.policies.gr00t import base_model as base_model_mod
from isaac_rl_posttraining.policies.gr00t import n15
from isaac_rl_posttraining.policies.gr00t import wrapper as wrapper_mod
from isaac_rl_posttraining.policies.gr00t.model import (
    LatentInjectionConfig,
    build_base_noise,
)
from isaac_rl_posttraining.policies.gr00t.n15 import (
    Gr00tN15Model,
    Gr00tN15Policy,
    LoadedN15Checkpoint,
    is_n15_checkpoint,
    masked_max,
    masked_mean,
    read_model_type,
    resolve_data_config,
)
from isaac_rl_posttraining.policies.gr00t.vla import get_vla_policy_cls

H, D, FEAT = 16, 32, 2048  # flow horizon, padded flow width, backbone feature width
VIDEO_KEYS = ["video.left_wrist_view", "video.right_wrist_view", "video.room_view"]
STATE_KEYS = [
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
]
ACTION_KEYS = [
    "action.left_arm",
    "action.right_arm",
    "action.left_hand",
    "action.right_hand",
]
LANGUAGE_KEY = "annotation.human.task_description"
WIDTHS = dict.fromkeys(ACTION_KEYS, 7)
A_RAW = sum(WIDTHS.values())  # 28
PROMPT = "assemble the trocar"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeActionHead(nn.Module):
    """Flow head stand-in: draws the initial noise exactly like the real head.

    Each Euler step adds a constant velocity, so ``num_inference_timesteps``
    steps shift the noise by ``0.5`` in total (zero steps return the noise).
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=H, action_dim=D, backbone_embedding_dim=FEAT
        )
        self.num_inference_timesteps = 0

    def get_action(
        self, backbone_output: dict[str, Any], action_input: Any
    ) -> dict[str, Any]:
        vl_embs = backbone_output["backbone_features"]
        actions = torch.randn(
            size=(vl_embs.shape[0], self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=vl_embs.device,
        )
        for _ in range(self.num_inference_timesteps):
            actions = actions + (1.0 / self.num_inference_timesteps) * 0.5
        return {"action_pred": actions}


class _FakeBackbone(nn.Module):
    def __init__(self, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        super().__init__()
        self.tokens = tokens
        self.mask = mask

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        # Batch size follows the processed inputs (the fake transform wraps them
        # under "processed"); a single sample keeps the recorded tokens as is.
        batch = 1
        processed = inputs.get("processed") if isinstance(inputs, dict) else None
        if isinstance(processed, dict):
            for value in processed.values():
                if hasattr(value, "shape") and len(value.shape) >= 1:
                    batch = int(value.shape[0])
                    break
        tokens, mask = self.tokens.clone(), self.mask.clone()
        if batch > 1 and tokens.shape[0] == 1:
            tokens = tokens.expand(batch, -1, -1).clone()
            mask = mask.expand(batch, -1).clone()
        return {"backbone_features": tokens, "backbone_attention_mask": mask}


class _FakeN15Model(nn.Module):
    """Duck-typed ``GR00T_N1_5``: config, backbone, action_head, prepare_input, get_action."""

    def __init__(
        self, tokens: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        if tokens is None:
            # Five tokens carrying the values 0..4: mean 2.0, max 4.0.
            tokens = (
                torch.arange(5, dtype=torch.float32)[None, :, None]
                .expand(1, 5, FEAT)
                .clone()
            )
            mask = torch.ones(1, 5, dtype=torch.long)
        self.config = SimpleNamespace(action_horizon=H, action_dim=D)
        self.backbone = _FakeBackbone(tokens, mask)
        self.action_head = _FakeActionHead()
        self.anchor = nn.Parameter(
            torch.zeros(1)
        )  # device/dtype anchor, like a real model
        self.prepared: list[Any] = []

    def prepare_input(self, inputs: Any) -> tuple[Any, Any]:
        self.prepared.append(inputs)
        return inputs, inputs

    def get_action(self, inputs: Any) -> dict[str, Any]:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head.get_action(
            self.backbone(backbone_inputs), action_inputs
        )


class _FakeTransform:
    """Modality transform stand-in with a per-key affine decode.

    ``unapply`` splits the padded action width into the action keys (dropping
    the padding) and un-normalizes each key as ``x * scale + shift``, like the
    ConcatTransform -> StateActionTransform -> StateActionToTensor chain.
    """

    def __init__(self, widths: dict[str, int]) -> None:
        self.widths = dict(widths)
        self.scale = {key: float(index + 2) for index, key in enumerate(widths)}
        self.shift = {key: 0.1 * (index + 1) for index, key in enumerate(widths)}
        self.metadata: Any = None
        self.training = True
        self.seen: dict[str, Any] | None = None

    def set_metadata(self, metadata: Any) -> None:
        self.metadata = metadata

    def eval(self) -> None:
        self.training = False

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        self.seen = data
        return {"processed": data}

    def unapply(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        action = data["action"]
        assert torch.is_tensor(action), "unapply consumes the normalized action tensor"
        decoded, start = {}, 0
        for key, width in self.widths.items():
            block = action[..., start : start + width]
            start += width
            decoded[key] = (block * self.scale[key] + self.shift[key]).numpy()
        return decoded


class _FakeDataConfig:
    video_keys = VIDEO_KEYS
    state_keys = STATE_KEYS
    action_keys = ACTION_KEYS
    language_keys = [LANGUAGE_KEY]  # noqa: RUF012
    observation_indices = [0]  # noqa: RUF012
    action_indices = list(range(H))  # noqa: RUF012

    def modality_config(self) -> dict[str, SimpleNamespace]:
        return {
            "video": SimpleNamespace(
                delta_indices=[0], modality_keys=list(self.video_keys)
            ),
            "state": SimpleNamespace(
                delta_indices=[0], modality_keys=list(self.state_keys)
            ),
            "action": SimpleNamespace(
                delta_indices=list(self.action_indices),
                modality_keys=list(self.action_keys),
            ),
            "language": SimpleNamespace(
                delta_indices=[0], modality_keys=list(self.language_keys)
            ),
        }

    def transform(self) -> _FakeTransform:
        return _FakeTransform(WIDTHS)


def _make_policy(
    monkeypatch: pytest.MonkeyPatch,
    tokens: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    **policy_kwargs: Any,
) -> tuple[Gr00tN15Policy, _FakeN15Model, _FakeTransform, dict[str, Any]]:
    """Build a policy over the fakes by monkeypatching the checkpoint loader."""
    model = _FakeN15Model(tokens, mask)
    config = _FakeDataConfig()
    transform = config.transform()
    loader_calls: dict[str, Any] = {}

    def fake_load(base_ckpt: str, data_config: Any, embodiment_tag: str, device: str):
        loader_calls.update(
            base_ckpt=base_ckpt,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=device,
        )
        return LoadedN15Checkpoint(
            model=model,
            transform=transform,
            modality_config=config.modality_config(),
            action_widths=dict(WIDTHS),
        )

    monkeypatch.setattr(n15, "load_n15_checkpoint", fake_load)
    policy = Gr00tN15Policy(
        base_ckpt="/fake/ckpt", data_config=config, device="cpu", **policy_kwargs
    )
    return policy, model, transform, loader_calls


def _obs(prompt: str | None = PROMPT) -> dict[str, Any]:
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    obs: dict[str, Any] = {
        "video.left_wrist_view": frame,  # [H, W, C]
        "video.right_wrist_view": frame[None],  # [1, H, W, C]
        "video.room_view": torch.from_numpy(frame.copy()),  # torch uint8
        "state.left_arm": np.arange(7, dtype=np.float64),  # [d] float64
        "state.right_arm": torch.ones(1, 7),  # torch [1, d]
        "state.left_hand": np.zeros((1, 1, 7), dtype=np.float32),  # already [1, 1, d]
        "state.right_hand": np.full(7, 0.5, dtype=np.float32),
        "dino_embedding": np.zeros(4, dtype=np.float32),  # learner-side entry, ignored
    }
    if prompt is not None:
        obs["prompt"] = prompt
    return obs


def _write_config(directory: Path, config: dict[str, Any]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


def test_registered_and_shaped_from_config():
    assert get_vla_policy_cls("gr00t_n1_5") is Gr00tN15Policy
    model = Gr00tN15Model(_FakeN15Model())
    assert (model.action_horizon, model.action_dim, model.feat_dim) == (H, D, FEAT)
    assert model.denoising_steps == 0
    model.denoising_steps = 3
    assert model.base_model.action_head.num_inference_timesteps == 3
    native = model.get_action({"processed": None})
    assert torch.is_tensor(native) and native.shape == (1, H, D)


def test_get_action_with_noise_substitutes_latent_for_randn():
    """(a) The randn shim feeds the latent-derived base noise to the head; with
    zero denoising steps the head output IS that noise, and randn is restored."""
    original_randn = torch.randn
    config = LatentInjectionConfig(
        fresh_base=False
    )  # deterministic base -> exact comparison
    model = Gr00tN15Model(_FakeN15Model(), latent_injection=config)
    latent = torch.linspace(
        -2.0, 2.0, A_RAW
    )  # [latent_dim]: broadcast over the horizon
    out = model.get_action_with_noise({"processed": None}, latent)
    expected = build_base_noise(latent, H, D, config)
    assert out.shape == (1, H, D)
    torch.testing.assert_close(out, expected)
    assert torch.randn is original_randn, (
        "the randn shim must be restored after the call"
    )

    # Default knobs: the steered block is the (clipped) latent on every step,
    # the padding dims are fresh noise; per-step latents map 1:1.
    model = Gr00tN15Model(_FakeN15Model())
    out = model.get_action_with_noise({}, torch.full((1, A_RAW), 10.0))
    torch.testing.assert_close(out[:, :, :A_RAW], torch.full((1, H, A_RAW), 3.0))
    assert 0.3 < float(out[:, :, A_RAW:].std()) < 1.7
    per_step = torch.randn(1, H, A_RAW).clamp(-2, 2)
    torch.testing.assert_close(
        model.get_action_with_noise({}, per_step)[:, :, :A_RAW], per_step
    )


def test_extract_features_pools_over_valid_tokens():
    """(b) Masked mean/max over the attention mask, float32 [B, 2048]."""
    tokens = torch.zeros(2, 3, FEAT, dtype=torch.bfloat16)
    for step in range(3):
        tokens[:, step, :] = float(step + 1)  # token values 1, 2, 3
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    model = Gr00tN15Model(_FakeN15Model(tokens, mask))
    features = model.extract_features({})
    assert features.shape == (2, FEAT) and features.dtype == torch.float32
    torch.testing.assert_close(features[0], torch.full((FEAT,), 1.5))  # mean of {1, 2}
    torch.testing.assert_close(features[1], torch.full((FEAT,), 1.0))  # mean of {1}
    maxed = masked_max(*model.backbone_tokens({}))
    torch.testing.assert_close(
        maxed[0], torch.full((FEAT,), 2.0)
    )  # masked-out 3 excluded
    torch.testing.assert_close(maxed[1], torch.full((FEAT,), 1.0))
    # All-valid mask: plain mean.
    torch.testing.assert_close(
        masked_mean(tokens, torch.ones(2, 3)), torch.full((2, FEAT), 2.0)
    )


# ---------------------------------------------------------------------------
# Policy: observation mapping and decoding
# ---------------------------------------------------------------------------


def test_policy_construction_binds_layout(monkeypatch):
    policy, model, transform, loader_calls = _make_policy(
        monkeypatch, embodiment_tag="new_embodiment"
    )
    assert loader_calls["base_ckpt"] == "/fake/ckpt"
    assert loader_calls["embodiment_tag"] == "new_embodiment"
    assert loader_calls["device"] == "cpu"
    assert policy.model.base_model is model and policy.transform is transform
    assert policy.decoded_action_horizon == H
    assert policy.action_horizon == H  # flow window
    assert policy.raw_action_dim == A_RAW  # env side: sum of the action-key widths
    assert policy.model.action_dim == D  # flow/latent width
    assert policy.model.feat_dim == FEAT
    assert policy.action_keys == ACTION_KEYS
    assert policy.language_key == LANGUAGE_KEY
    assert policy.compute_dtype == torch.bfloat16


def test_build_model_inputs_maps_observation_to_batched_layout(monkeypatch):
    """(c) Frames -> [1, 1, H, W, C] uint8, states -> [1, 1, d] float32, prompt
    -> one-element string array under the language key; nothing else."""
    policy, _, transform, _ = _make_policy(monkeypatch)
    policy.build_model_inputs(_obs())
    seen = transform.seen
    assert set(seen) == set(VIDEO_KEYS + STATE_KEYS + [LANGUAGE_KEY])
    for key in VIDEO_KEYS:
        assert seen[key].shape == (1, 1, 8, 8, 3) and seen[key].dtype == np.uint8
        assert isinstance(seen[key], np.ndarray)
        assert int(seen[key][0, 0, 0, 0, 0]) == 7
    for key in STATE_KEYS:
        assert seen[key].shape == (1, 1, 7) and seen[key].dtype == np.float32
        assert isinstance(seen[key], np.ndarray)
    np.testing.assert_allclose(seen["state.left_arm"][0, 0], np.arange(7))
    np.testing.assert_allclose(seen["state.right_arm"][0, 0], np.ones(7))
    language = seen[LANGUAGE_KEY]
    assert isinstance(language, np.ndarray) and language.shape == (1,)
    assert language.dtype.kind == "U" and str(language[0]) == PROMPT


def test_build_model_inputs_language_sources(monkeypatch):
    policy, _, transform, _ = _make_policy(monkeypatch)
    # A list prompt (vectorized env convention) yields its first entry.
    policy.build_model_inputs(_obs(prompt=None) | {"prompt": [PROMPT, "other"]})
    assert str(transform.seen[LANGUAGE_KEY][0]) == PROMPT
    # No prompt and no registered default is a hard error, not a silent blank.
    with pytest.raises(ValueError, match="task instruction"):
        policy.build_model_inputs(_obs(prompt=None))
    policy.register_baked_embodiment(task="default instruction")
    policy.build_model_inputs(_obs(prompt=None))
    assert str(transform.seen[LANGUAGE_KEY][0]) == "default instruction"
    # An observation prompt still wins over the default.
    policy.build_model_inputs(_obs())
    assert str(transform.seen[LANGUAGE_KEY][0]) == PROMPT


def test_build_model_inputs_rejects_bad_frames_and_missing_keys(monkeypatch):
    policy, _, _, _ = _make_policy(monkeypatch)
    missing = _obs()
    del missing["video.room_view"]
    with pytest.raises(KeyError, match="video.room_view"):
        policy.build_model_inputs(missing)
    missing_state = _obs()
    del missing_state["state.left_hand"]
    with pytest.raises(KeyError, match="state.left_hand"):
        policy.build_model_inputs(missing_state)
    float_frame = _obs() | {"video.room_view": np.zeros((8, 8, 3), dtype=np.float32)}
    with pytest.raises(TypeError, match="uint8"):
        policy.build_model_inputs(float_frame)
    stacked = _obs() | {"video.room_view": np.zeros((2, 8, 8, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="one \\[H, W, C\\] image"):
        policy.build_model_inputs(stacked)


def test_act_decodes_chunk_in_action_key_order(monkeypatch):
    """(d) act -> [16, 28]; the steered decode unapplies per key, in order."""
    policy, _, transform, _ = _make_policy(
        monkeypatch, latent_injection=LatentInjectionConfig(tile_dims=True)
    )
    action, actor_state = policy.act(_obs(), actor_state="carried")
    assert torch.is_tensor(action) and action.shape == (H, A_RAW)
    assert action.dtype == torch.float32 and actor_state == "carried"

    # With tile_dims the whole initial noise is a function of the latent, and
    # with zero denoising steps the head returns it, so the decoded chunk is
    # exactly the per-key affine of the latent in action-key order.
    latent = torch.linspace(-1.0, 1.0, A_RAW)[None]  # [1, 28]
    decoded = policy.act_with_noise(_obs(), latent)
    assert isinstance(decoded, np.ndarray) and decoded.shape == (H, A_RAW)
    start = 0
    for key in ACTION_KEYS:
        width = WIDTHS[key]
        block = latent[0, start : start + width].numpy()
        expected = block * transform.scale[key] + transform.shift[key]
        np.testing.assert_allclose(
            decoded[:, start : start + width], np.tile(expected, (H, 1)), rtol=1e-6
        )
        start += width

    zeros = policy.decode_action(torch.zeros(1, H, D))
    assert zeros.shape == (H, A_RAW)
    start = 0
    for key in ACTION_KEYS:
        np.testing.assert_allclose(
            zeros[:, start : start + WIDTHS[key]], transform.shift[key], rtol=1e-6
        )
        start += WIDTHS[key]


def test_infer_reference_returns_pooled_features_and_native_chunk(monkeypatch):
    policy, _, _, _ = _make_policy(monkeypatch)
    features, reference = policy.infer_reference(_obs())
    assert features.shape == (1, 2 * FEAT) and features.dtype == torch.float32
    torch.testing.assert_close(
        features[0, :FEAT], torch.full((FEAT,), 2.0)
    )  # masked mean of 0..4
    torch.testing.assert_close(
        features[0, FEAT:], torch.full((FEAT,), 4.0)
    )  # masked max
    assert isinstance(reference, np.ndarray) and reference.shape == (1, H, A_RAW)
    features_train, _ = policy.infer_reference(_obs(), image_transform_mode="train")
    torch.testing.assert_close(features_train, features)
    with pytest.raises(ValueError):
        policy.infer_reference(_obs(), image_transform_mode="augment")


def test_denoising_steps_applied_to_head(monkeypatch):
    """(f) denoising_steps reaches action_head.num_inference_timesteps and is consumed."""
    policy, model, _, _ = _make_policy(monkeypatch)
    assert (
        model.action_head.num_inference_timesteps == 0
    )  # None keeps the checkpoint's value
    policy, model, _, _ = _make_policy(
        monkeypatch,
        denoising_steps=4,
        latent_injection=LatentInjectionConfig(tile_dims=True),
    )
    assert model.action_head.num_inference_timesteps == 4
    assert policy.model.denoising_steps == 4
    latent = torch.zeros(1, A_RAW)
    out = policy.model.get_action_with_noise({}, latent)
    torch.testing.assert_close(
        out, torch.full((1, H, D), 0.5)
    )  # four steps shift the noise by 0.5
    with pytest.raises(ValueError, match="denoising_steps"):
        _make_policy(monkeypatch, denoising_steps=0)


# ---------------------------------------------------------------------------
# Dispatch on the checkpoint's declared model type
# ---------------------------------------------------------------------------


def test_read_model_type_and_dispatch_predicate(tmp_path):
    """(e) The decision reads config.json only: N1.5 iff model_type == gr00t_n1_5."""
    n15_dir = _write_config(
        tmp_path / "n15", {"model_type": "gr00t_n1_5", "architectures": ["GR00T_N1_5"]}
    )
    n17_dir = _write_config(tmp_path / "n17", {"model_type": "gr00t_n1d7"})
    untyped_dir = _write_config(tmp_path / "untyped", {"architectures": ["Something"]})
    assert read_model_type(n15_dir) == "gr00t_n1_5"
    assert read_model_type(n17_dir) == "gr00t_n1d7"
    assert read_model_type(untyped_dir) is None
    assert is_n15_checkpoint(n15_dir) is True
    assert is_n15_checkpoint(n17_dir) is False
    assert is_n15_checkpoint(untyped_dir) is False
    with pytest.raises(ValueError, match="base_ckpt"):
        read_model_type("")


class _StubPolicy(nn.Module):
    """What the wrapper touches on a constructed policy after building it."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.task: str | None = None

    @property
    def decoded_action_horizon(self) -> int:
        return H

    def register_baked_embodiment(self, task: str | None = None) -> None:
        self.task = task


def _record_make_vla_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_make(name: str, **kwargs: Any) -> _StubPolicy:
        calls.append((name, kwargs))
        return _StubPolicy()

    monkeypatch.setattr(wrapper_mod, "make_vla_policy", fake_make)
    return calls


def test_wrapper_selects_n15_path_from_model_type(monkeypatch, tmp_path):
    calls = _record_make_vla_policy(monkeypatch)
    n15_dir = _write_config(tmp_path / "n15", {"model_type": "gr00t_n1_5"})
    wrapper = wrapper_mod.SteeredGr00tWrapper(
        base_ckpt=n15_dir,
        device="cpu",
        data_config="some.module:SomeDataConfig",
        denoising_steps=4,
        task_instruction=PROMPT,
    )
    assert [name for name, _ in calls] == ["gr00t_n1_5"]
    kwargs = calls[0][1]
    assert kwargs["base_ckpt"] == n15_dir
    assert kwargs["data_config"] == "some.module:SomeDataConfig"
    assert kwargs["denoising_steps"] == 4
    assert kwargs["embodiment_tag"] == "new_embodiment"
    assert "action_horizon" not in kwargs
    assert wrapper.policy.task == PROMPT  # baked path registers the default instruction
    assert wrapper.action_horizon == H
    assert all(not p.requires_grad for p in wrapper.policy.model.parameters())


def test_wrapper_keeps_n17_call_unchanged(monkeypatch, tmp_path):
    calls = _record_make_vla_policy(monkeypatch)
    n17_dir = _write_config(tmp_path / "n17", {"model_type": "gr00t_n1d7"})
    wrapper_mod.SteeredGr00tWrapper(
        base_ckpt=n17_dir, device="cpu", task_instruction=PROMPT
    )
    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "gr00t"
    assert kwargs == {
        "base_ckpt": n17_dir,
        "backbone_model_name_or_path": None,
        "device": "cpu",
        "embodiment_tag": "new_embodiment",
        "action_horizon": None,
        "latent_injection": None,
    }


def test_wrapper_rejects_fields_of_the_other_generation(monkeypatch, tmp_path):
    calls = _record_make_vla_policy(monkeypatch)
    n15_dir = _write_config(tmp_path / "n15", {"model_type": "gr00t_n1_5"})
    n17_dir = _write_config(tmp_path / "n17", {"model_type": "gr00t_n1d7"})
    with pytest.raises(ValueError, match="N1.5"):
        wrapper_mod.SteeredGr00tWrapper(
            base_ckpt=n17_dir, device="cpu", data_config="m:C"
        )
    with pytest.raises(ValueError, match="N1.5"):
        wrapper_mod.SteeredGr00tWrapper(
            base_ckpt=n17_dir, device="cpu", denoising_steps=4
        )
    with pytest.raises(ValueError, match="data_config"):
        wrapper_mod.SteeredGr00tWrapper(base_ckpt=n15_dir, device="cpu")
    for field_name in ("bc_ckpt", "dataset_dir"):
        with pytest.raises(ValueError, match=field_name):
            wrapper_mod.SteeredGr00tWrapper(
                base_ckpt=n15_dir,
                device="cpu",
                data_config="m:C",
                **{field_name: "/some/path"},
            )
    with pytest.raises(ValueError, match="multitask_dirs"):
        wrapper_mod.SteeredGr00tWrapper(
            base_ckpt=n15_dir,
            device="cpu",
            data_config="m:C",
            multitask_dirs=["/a", "/b"],
        )
    with pytest.raises(ValueError, match="action_horizon"):
        wrapper_mod.SteeredGr00tWrapper(
            base_ckpt=n15_dir, device="cpu", data_config="m:C", action_horizon=8
        )
    assert calls == [], "validation must fail before any model load"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_from_dsrl_config_forwards_n15_fields(monkeypatch):
    omegaconf = pytest.importorskip("omegaconf")
    received: dict[str, Any] = {}

    class _RecordingWrapper:
        def __init__(self, **kwargs: Any) -> None:
            received.update(kwargs)

    monkeypatch.setattr(base_model_mod, "SteeredGr00tWrapper", _RecordingWrapper)
    cfg = omegaconf.OmegaConf.create(
        {
            "base_model": {
                "ckpt": "/ckpt",
                "embodiment_tag": "new_embodiment",
                "data_config": "pkg.mod:Cfg",
                "denoising_steps": 4,
            },
            "dsrl": {"noise_dim": None, "noise_action_bound": 2.0},
        }
    )
    base_model_mod.Gr00tBaseModelPolicy.from_dsrl_config(cfg, "cpu")
    assert received["data_config"] == "pkg.mod:Cfg"
    assert received["denoising_steps"] == 4 and isinstance(
        received["denoising_steps"], int
    )
    received.clear()
    cfg.base_model.data_config = None
    cfg.base_model.denoising_steps = None
    base_model_mod.Gr00tBaseModelPolicy.from_dsrl_config(cfg, "cpu")
    assert received["data_config"] is None and received["denoising_steps"] is None


def test_base_model_schema_and_yaml_carry_n15_fields():
    yaml = pytest.importorskip("yaml")
    with open(
        os.path.join(
            REPO_ROOT, "isaac_rl_posttraining", "configs", "base_model", "gr00t.yaml"
        ),
        encoding="utf-8",
    ) as handle:
        doc = yaml.safe_load(handle)
    assert "data_config" in doc and doc["data_config"] is None
    assert "denoising_steps" in doc and doc["denoising_steps"] is None


# ---------------------------------------------------------------------------
# Lazy gr00t seams, exercised against stub modules
# ---------------------------------------------------------------------------


def _install_gr00t_stubs(
    monkeypatch: pytest.MonkeyPatch, **modules: types.ModuleType
) -> None:
    """Install stub ``gr00t.*`` leaf modules (and stub parent packages) for the test."""
    for dotted, module in modules.items():
        parts = dotted.split(".")
        for depth in range(1, len(parts)):
            package = types.ModuleType(".".join(parts[:depth]))
            package.__path__ = []  # package-like
            monkeypatch.setitem(sys.modules, package.__name__, package)
        monkeypatch.setitem(sys.modules, dotted, module)


def _stub_data_config_module() -> types.ModuleType:
    module = types.ModuleType("gr00t.experiment.data_config")

    class BaseDataConfig:
        pass

    def load_data_config(spec: str) -> str:
        return f"loaded:{spec}"

    module.BaseDataConfig = BaseDataConfig
    module.load_data_config = load_data_config
    return module


def test_resolve_data_config_accepts_string_class_and_instance(monkeypatch):
    module = _stub_data_config_module()
    _install_gr00t_stubs(monkeypatch, **{"gr00t.experiment.data_config": module})

    class _Custom(module.BaseDataConfig):
        pass

    assert resolve_data_config("so100") == "loaded:so100"
    assert resolve_data_config("pkg.mod:Cfg") == "loaded:pkg.mod:Cfg"
    assert isinstance(resolve_data_config(_Custom), _Custom)
    instance = _Custom()
    assert resolve_data_config(instance) is instance
    with pytest.raises(TypeError, match="data_config"):
        resolve_data_config(42)
    with pytest.raises(TypeError, match="data_config"):
        resolve_data_config(dict)  # a class, but not a data config


def test_load_n15_checkpoint_delegates_to_native_policy(monkeypatch):
    """The loader runs the pinned policy's load sequence and keeps model,
    transform, and per-key action widths; the model comes back frozen."""
    data_config_module = _stub_data_config_module()
    policy_module = types.ModuleType("gr00t.model.policy")
    constructed: dict[str, Any] = {}
    fake_model = _FakeN15Model()
    fake_transform = _FakeTransform(WIDTHS)
    metadata = SimpleNamespace(
        modalities=SimpleNamespace(
            action={
                key.split(".", 1)[1]: SimpleNamespace(shape=(7,)) for key in ACTION_KEYS
            }
        )
    )

    class NativeGr00tPolicy:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)
            self.model = fake_model
            self.modality_transform = kwargs["modality_transform"]
            self.metadata = metadata

    policy_module.Gr00tPolicy = NativeGr00tPolicy
    tags_module = types.ModuleType("gr00t.data.embodiment_tags")
    tags_module.EmbodimentTag = enum.Enum(
        "EmbodimentTag", {"NEW_EMBODIMENT": "new_embodiment"}
    )
    _install_gr00t_stubs(
        monkeypatch,
        **{
            "gr00t.experiment.data_config": data_config_module,
            "gr00t.model.policy": policy_module,
            "gr00t.data.embodiment_tags": tags_module,
        },
    )

    class _Config(data_config_module.BaseDataConfig):
        def modality_config(self):
            return _FakeDataConfig().modality_config()

        def transform(self):
            return fake_transform

    fake_model.train()
    for param in fake_model.parameters():
        param.requires_grad_(True)
    loaded = n15.load_n15_checkpoint(
        "/ckpt/n15", _Config, embodiment_tag="new_embodiment", device="cpu"
    )
    assert constructed["model_path"] == "/ckpt/n15"
    assert constructed["embodiment_tag"] == "new_embodiment"
    assert constructed["device"] == "cpu"
    assert constructed["modality_transform"] is fake_transform
    assert list(constructed["modality_config"]) == [
        "video",
        "state",
        "action",
        "language",
    ]
    assert loaded.model is fake_model and loaded.transform is fake_transform
    assert loaded.action_widths == WIDTHS
    assert not fake_model.training
    assert all(not p.requires_grad for p in fake_model.parameters())


# ---------------------------------------------------------------------------
# Embodiment-tag aliases (RLinf fine-tunes under tags the pinned release lacks)
# ---------------------------------------------------------------------------


def _install_fake_gr00t(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Stub the four ``gr00t`` modules the alias path touches; return the policy constructions."""
    constructed: list[dict[str, Any]] = []

    class EmbodimentTag(enum.Enum):
        GR1 = "gr1"
        NEW_EMBODIMENT = "new_embodiment"

    class DatasetMetadata:
        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> SimpleNamespace:
            EmbodimentTag(
                data["embodiment_tag"]
            )  # the pinned schema validates against the enum
            return SimpleNamespace(**data)

    class Gr00tPolicy:
        def __init__(
            self,
            model_path,
            embodiment_tag,
            modality_config,
            modality_transform,
            device,
            denoising_steps=None,
        ):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
            self._modality_transform = modality_transform
            constructed.append(
                {"model_path": model_path, "embodiment_tag": embodiment_tag}
            )
            self._load_metadata(Path(model_path) / "experiment_cfg")

        def _load_metadata(self, exp_cfg_dir):
            metadatas = json.loads((Path(exp_cfg_dir) / "metadata.json").read_text())
            self.metadata = DatasetMetadata.model_validate(
                metadatas[self.embodiment_tag.value]
            )
            self._modality_transform.set_metadata(self.metadata)

    for name, attrs in {
        "gr00t": {},
        "gr00t.data": {},
        "gr00t.data.embodiment_tags": {"EmbodimentTag": EmbodimentTag},
        "gr00t.data.schema": {"DatasetMetadata": DatasetMetadata},
        "gr00t.model": {},
        "gr00t.model.policy": {"Gr00tPolicy": Gr00tPolicy},
    }.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    return constructed


def _checkpoint_with_metadata(tmp_path: Path, metadata: dict[str, Any]) -> str:
    (tmp_path / "experiment_cfg").mkdir(parents=True)
    (tmp_path / "experiment_cfg" / "metadata.json").write_text(json.dumps(metadata))
    return str(tmp_path)


def test_resolve_model_embodiment_tag(monkeypatch):
    _install_fake_gr00t(monkeypatch, {})
    assert n15.resolve_model_embodiment_tag("new_embodiment") == "new_embodiment"
    assert n15.resolve_model_embodiment_tag("gr1") == "gr1"
    assert n15.resolve_model_embodiment_tag("isaaclab_franka") == "new_embodiment"
    with pytest.raises(ValueError, match="has no alias"):
        n15.resolve_model_embodiment_tag("martian_arm")


def test_aliased_tag_loads_through_the_pinned_slot_with_its_own_statistics(
    monkeypatch, tmp_path
):
    metadata = {
        "isaaclab_franka": {
            "statistics": {"state": {"x": {"min": [0.3], "max": [0.7]}}},
            "modalities": {},
            "embodiment_tag": "isaaclab_franka",
        }
    }
    constructed = _install_fake_gr00t(monkeypatch, metadata)
    ckpt = _checkpoint_with_metadata(tmp_path, metadata)
    transform = _FakeTransform(WIDTHS)
    policy = n15._make_native_policy(ckpt, "isaaclab_franka", {}, transform, "cpu")
    assert constructed == [{"model_path": ckpt, "embodiment_tag": "new_embodiment"}]
    assert policy.metadata.statistics == metadata["isaaclab_franka"]["statistics"]
    assert (
        policy.metadata.embodiment_tag == "new_embodiment"
    )  # validated under the pinned tag
    assert transform.metadata is policy.metadata


def test_aliased_tag_missing_from_metadata_is_loud(monkeypatch, tmp_path):
    metadata = {
        "new_embodiment": {
            "statistics": {},
            "modalities": {},
            "embodiment_tag": "new_embodiment",
        }
    }
    _install_fake_gr00t(monkeypatch, metadata)
    ckpt = _checkpoint_with_metadata(tmp_path, metadata)
    with pytest.raises(ValueError, match="isaaclab_franka"):
        n15._make_native_policy(
            ckpt, "isaaclab_franka", {}, _FakeTransform(WIDTHS), "cpu"
        )


def test_known_tag_uses_the_native_policy_unchanged(monkeypatch, tmp_path):
    metadata = {
        "new_embodiment": {
            "statistics": {},
            "modalities": {},
            "embodiment_tag": "new_embodiment",
        }
    }
    constructed = _install_fake_gr00t(monkeypatch, metadata)
    ckpt = _checkpoint_with_metadata(tmp_path, metadata)
    policy = n15._make_native_policy(
        ckpt, "new_embodiment", {}, _FakeTransform(WIDTHS), "cpu"
    )
    assert (
        type(policy).__name__ == "Gr00tPolicy"
        and constructed[0]["embodiment_tag"] == "new_embodiment"
    )


# ---------------------------------------------------------------------------
# Batched decode (one forward for N envs)
# ---------------------------------------------------------------------------


def test_act_batch_stacks_samples_and_decodes_each(monkeypatch):
    policy, model, transform, _ = _make_policy(monkeypatch)
    model.action_head.num_inference_timesteps = 2
    obs_list = [_obs(), _obs()]
    obs_list[1]["state.left_arm"] = np.arange(7, dtype=np.float32) + 100.0
    latents = np.zeros((2, 3, D), dtype=np.float32)
    actions = policy.act_batch(obs_list, latents)
    assert actions.shape == (2, H, A_RAW)
    seen = transform.seen
    assert (
        seen["video.room_view"].shape == (2, 1, 8, 8, 3)
        and seen["video.room_view"].dtype == np.uint8
    )
    assert seen["state.left_arm"].shape == (2, 1, 7)
    np.testing.assert_array_equal(seen["state.left_arm"][1, 0], np.arange(7) + 100.0)
    assert list(seen[LANGUAGE_KEY]) == [PROMPT, PROMPT]
    # Native (unsteered) batch and shape checks.
    assert policy.act_batch(obs_list).shape == (2, H, A_RAW)
    with pytest.raises(ValueError, match="carry 3 samples"):
        policy.act_batch(obs_list, np.zeros((3, D), dtype=np.float32))
    with pytest.raises(ValueError, match="one shared instruction"):
        policy.act_batch([_obs(), _obs(prompt="stack the cubes")])
    with pytest.raises(ValueError, match="at least one"):
        policy.act_batch([])


def test_steered_wrapper_batched_path_reaches_the_n15_policy(monkeypatch):
    from isaac_rl_posttraining.policies.gr00t.wrapper import SteeredGr00tWrapper

    policy, _model, _transform, _ = _make_policy(monkeypatch)
    wrapper = object.__new__(SteeredGr00tWrapper)
    wrapper.policy = policy
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    obs = {
        "video.left_wrist_view": np.stack([frame, frame]),
        "video.right_wrist_view": np.stack([frame, frame]),
        "video.room_view": np.stack([frame, frame]),
        "state.left_arm": np.zeros((2, 7), dtype=np.float32),
        "state.right_arm": np.ones((2, 7), dtype=np.float32),
        "state.left_hand": np.zeros((2, 7), dtype=np.float32),
        "state.right_hand": np.zeros((2, 7), dtype=np.float32),
        "state": np.zeros((2, 28), dtype=np.float32),
        "prompt": PROMPT,
    }
    result = wrapper.infer_batch(obs, np.zeros((2, D), dtype=np.float32))
    assert result["actions"].shape == (2, H, A_RAW)
    assert SteeredGr00tWrapper.supports_batched_infer is True
