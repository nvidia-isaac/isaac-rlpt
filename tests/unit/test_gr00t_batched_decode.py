# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched GR00T decode: one backbone forward steers all N envs.

The vectorized DSRL collector used to decode the frozen GR00T policy one env
at a time (N sequential forwards) because the wrapper's ``infer`` only
accepted a single sample. GR00T's model (``get_action_with_noise``) and data
processor (``build_inputs``) already batch, so ``act_batch`` / ``infer_batch``
run all N in one forward. These tests pin the plumbing without the real model:
the observation split, the noise-shape contract, and that the batched input
builder stacks a leading ``[B]`` axis while decoding each sample against its
own state.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaac_rl_posttraining.policies.gr00t.wrapper import SteeredGr00tWrapper


class _RecordingPolicy:
    """Stand-in policy whose act_batch encodes each sample's state[0]."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], np.ndarray | None]] = []
        self.prepared_calls: list[tuple[list[dict], np.ndarray]] = []

    def act_batch(self, obs_list, latents):
        self.calls.append((obs_list, latents))
        horizon, action_dim = 4, 3
        out = np.zeros((len(obs_list), horizon, action_dim), dtype=np.float32)
        for b, obs in enumerate(obs_list):
            out[b] = float(np.asarray(obs["state"]).reshape(-1)[0])
        return out

    def prepare_initial_noise(self, latent):
        return np.asarray(latent)[:, None, :] + 1.0

    def act_batch_with_initial_noise(self, obs_list, initial_noise):
        self.prepared_calls.append((obs_list, initial_noise))
        return np.repeat(np.asarray(initial_noise)[:, :, :3], 4, axis=1)


def _wrapper_with(policy) -> SteeredGr00tWrapper:
    wrapper = object.__new__(
        SteeredGr00tWrapper
    )  # skip the heavy model-loading __init__
    wrapper.policy = policy
    return wrapper


def test_as_obs_list_splits_batched_dict():
    obs = {
        "state": np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32),
        "ego_view_image": np.zeros((3, 2, 2, 3), dtype=np.uint8),
        "prompt": "lift the drill",  # shared scalar — passes through
    }
    out = SteeredGr00tWrapper._as_obs_list(obs)
    assert len(out) == 3
    assert [float(o["state"][0]) for o in out] == [10.0, 20.0, 30.0]
    assert all(o["prompt"] == "lift the drill" for o in out)
    assert all(o["ego_view_image"].shape == (2, 2, 3) for o in out)
    # An already-split list passes through as a copy.
    dicts = [{"state": np.zeros(2)}, {"state": np.ones(2)}]
    assert SteeredGr00tWrapper._as_obs_list(dicts) is not dicts
    assert SteeredGr00tWrapper._as_obs_list(dicts) == dicts


def test_infer_batch_routes_per_sample_and_shapes():
    assert SteeredGr00tWrapper.supports_batched_infer is True
    policy = _RecordingPolicy()
    wrapper = _wrapper_with(policy)
    obs = {
        "state": np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32),
        "ego_view_image": np.zeros((3, 2, 2, 3), dtype=np.uint8),
        "prompt": "lift",
    }
    noise = np.zeros((3, 5), dtype=np.float32)
    result = wrapper.infer_batch(obs, noise)
    assert result["actions"].shape == (3, 4, 3)
    # Per-env state routed to the matching output row, not mixed across the batch.
    np.testing.assert_allclose(result["actions"][:, 0, 0], [10.0, 20.0, 30.0])
    obs_list, forwarded = policy.calls[0]
    assert len(obs_list) == 3 and obs_list[1]["prompt"] == "lift"
    np.testing.assert_allclose(forwarded, noise)


def test_infer_batch_rejects_noise_batch_mismatch():
    wrapper = _wrapper_with(_RecordingPolicy())
    obs = {"state": np.zeros((3, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="B == 3"):
        wrapper.infer_batch(obs, np.zeros((2, 5), dtype=np.float32))


def test_prepared_batch_preserves_per_sample_noise() -> None:
    policy = _RecordingPolicy()
    wrapper = _wrapper_with(policy)
    observations = {
        "state": np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
    }
    first = wrapper.prepare_seeded_batch_noise(np.zeros((1, 3), dtype=np.float32))
    second = wrapper.prepare_seeded_batch_noise(np.ones((1, 3), dtype=np.float32))
    prepared = np.concatenate([first, second], axis=0)

    result = wrapper.infer_prepared_batch(observations, prepared)

    assert result["actions"].shape == (2, 4, 3)
    np.testing.assert_allclose(result["actions"][:, 0], [[1, 1, 1], [2, 2, 2]])
    obs_list, forwarded = policy.prepared_calls[0]
    assert len(obs_list) == 2
    np.testing.assert_allclose(forwarded, prepared)


def test_act_batch_stacks_inputs_and_decodes_per_state(monkeypatch):
    """policy.act_batch builds a [B]-batched model input and decodes per env."""
    from isaac_rl_posttraining.policies.gr00t import data as gr00t_data
    from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

    policy = object.__new__(Gr00tPolicy)  # skip model loading
    policy.device = "cpu"
    policy.embodiment_tag = "tag"
    policy.modality_json = {}
    policy.processor = SimpleNamespace(shortest_image_edge=1)
    policy.task = "default task"

    captured: dict = {}

    def _fake_build_inputs(processor, obs_batch, tag, modality_json, task, device):
        captured["obs_batch"] = obs_batch
        captured["task"] = task
        return "MODEL_INPUTS"

    monkeypatch.setattr(gr00t_data, "build_inputs", _fake_build_inputs)

    class _FakeModel:
        def get_action_with_noise(self, inputs, latent):
            assert inputs == "MODEL_INPUTS"
            batch = latent.shape[0]
            return torch.arange(batch * 4 * 3, dtype=torch.float32).reshape(batch, 4, 3)

    policy.model = _FakeModel()
    # _unapply is exercised separately; here return the normalized chunk so the
    # per-sample decode loop and state routing are observable.
    policy._unapply = lambda normalized, state: np.asarray(normalized) + float(state[0])

    obs_list = [
        {
            "state": np.array([1.0, 9.0], dtype=np.float32),
            "ego_view_image": np.zeros((2, 2, 3), np.uint8),
        },
        {
            "state": np.array([100.0, 9.0], dtype=np.float32),
            "ego_view_image": np.ones((2, 2, 3), np.uint8),
        },
    ]
    out = policy.act_batch(obs_list, np.zeros((2, 5), dtype=np.float32))

    assert out.shape == (2, 4, 3)
    assert captured["obs_batch"]["state"].shape == (2, 2)
    assert captured["obs_batch"]["ego_view_image"].shape == (2, 2, 2, 3)
    # One shared instruction is broadcast over the batch (GR00T's multi-env mode).
    assert captured["task"] == "default task"
    # Row 0 decoded against state[0]=1, row 1 against state[0]=100.
    np.testing.assert_allclose(out[0], np.arange(0, 12).reshape(4, 3) + 1.0)
    np.testing.assert_allclose(out[1], np.arange(12, 24).reshape(4, 3) + 100.0)


def _bare_policy():
    from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

    policy = object.__new__(Gr00tPolicy)
    policy.device = "cpu"
    policy.embodiment_tag = "tag"
    policy.modality_json = {}
    policy.processor = SimpleNamespace(shortest_image_edge=1)
    policy.task = "default task"
    return policy


def test_act_batch_refuses_heterogeneous_instructions():
    """Per-sample instructions would pad to the batch max — refused, not batched."""
    policy = _bare_policy()
    obs_list = [
        {"state": np.zeros(2, np.float32), "prompt": "lift the drill"},
        {"state": np.zeros(2, np.float32), "prompt": "push the button"},
    ]
    with pytest.raises(ValueError, match="one shared instruction"):
        policy.act_batch(obs_list, np.zeros((2, 5), np.float32))


def test_resolve_processor_dir_matches_gr00t_layouts(tmp_path):
    """The processor lives at the root (older) or under processor/ (N1.7 fine-tunes)."""
    from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

    resolve = Gr00tPolicy._resolve_processor_dir
    # N1.7 fine-tune layout: root has config.json only, processor/ holds the config.
    ckpt = tmp_path / "n17"
    (ckpt / "processor").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "processor" / "processor_config.json").write_text("{}")
    assert resolve(str(ckpt)) == str(ckpt / "processor")
    # Root ships its own processor: use the root even if a processor/ dir exists.
    rooted = tmp_path / "rooted"
    (rooted / "processor").mkdir(parents=True)
    (rooted / "processor_config.json").write_text("{}")
    assert resolve(str(rooted)) == str(rooted)
    # No processor dir at all (or a hub model id): pass through unchanged.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert resolve(str(bare)) == str(bare)
    assert resolve("nvidia/GR00T-N1.7-3B") == "nvidia/GR00T-N1.7-3B"


def test_online_modality_uses_baked_layout_not_buffer_layout(monkeypatch):
    """The env's flat state follows the checkpoint's baked group order, not the
    offline buffer's dataset layout (legs-included, arms at [15:22]/[29:36])."""
    from isaac_rl_posttraining.policies.gr00t import data as gr00t_data

    policy = _bare_policy()
    buffer_layout = {  # the G1 WBC adapter's dataset layout: 43-dim, legs first
        "state": {
            "left_leg": {"start": 0, "end": 6},
            "left_arm": {"start": 15, "end": 22},
            "right_arm": {"start": 29, "end": 36},
        },
        "action": {"left_arm": {"start": 0, "end": 7}},
    }
    baked_layout = {  # contiguous upper-body order the env sends
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "waist": {"start": 28, "end": 31},
        },
        "action": {"left_arm": {"start": 0, "end": 7}},
    }
    monkeypatch.setattr(gr00t_data, "has_baked_embodiment", lambda proc, tag: True)
    monkeypatch.setattr(
        gr00t_data, "modality_json_from_processor", lambda proc, tag: baked_layout
    )
    assert policy._online_modality_json(buffer_layout) is baked_layout

    # A raw base model bakes nothing: the buffer's layout is all there is.
    monkeypatch.setattr(gr00t_data, "has_baked_embodiment", lambda proc, tag: False)
    modality_json = policy._online_modality_json(buffer_layout)
    assert modality_json is buffer_layout  # NOSONAR - must not copy


def test_act_batch_refuses_heterogeneous_camera_keys():
    """An env missing a camera key the others carry fails loud, not with a raw KeyError."""
    policy = _bare_policy()
    obs_list = [
        {
            "state": np.zeros(2, np.float32),
            "ego_view_image": np.zeros((2, 2, 3), np.uint8),
        },
        {"state": np.zeros(2, np.float32)},  # no ego frame
    ]
    with pytest.raises(ValueError, match="same camera keys"):
        policy.act_batch(obs_list, np.zeros((2, 5), np.float32))


def test_act_batch_native_noise_none(monkeypatch):
    """latents=None takes the native get_action path, still batched."""
    from isaac_rl_posttraining.policies.gr00t import data as gr00t_data
    from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

    policy = object.__new__(Gr00tPolicy)
    policy.device = "cpu"
    policy.embodiment_tag = "tag"
    policy.modality_json = {}
    policy.processor = SimpleNamespace(shortest_image_edge=1)
    policy.task = "t"
    monkeypatch.setattr(gr00t_data, "build_inputs", lambda *a, **k: "IN")

    class _FakeModel:
        def get_action(self, inputs):
            return {"action_pred": torch.zeros(2, 4, 3)}

    policy.model = _FakeModel()
    policy._unapply = lambda normalized, state: np.asarray(normalized)
    obs_list = [{"state": np.zeros(2, np.float32)}, {"state": np.zeros(2, np.float32)}]
    out = policy.act_batch(obs_list, None)
    assert out.shape == (2, 4, 3)


@pytest.mark.parametrize("shape", [(240, 320, 3), (256, 341, 3), (480, 640, 3)])
def test_native_frames_reach_the_checkpoint_image_transform(monkeypatch, shape):
    """The checkpoint processor owns resizing, including upscaling native camera frames."""
    from isaac_rl_posttraining.policies.gr00t import data as gr00t_data
    from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

    policy = object.__new__(Gr00tPolicy)
    policy.device = "cpu"
    policy.embodiment_tag = "tag"
    policy.modality_json = {}
    policy.processor = SimpleNamespace(
        shortest_image_edge=256,
        modality_configs={"tag": {"video": SimpleNamespace(modality_keys=[])}},
    )
    policy.task = "default task"
    received = []

    def build_inputs(processor, observations, *args):
        assert processor is policy.processor
        received.append(observations["ego_view_image"])
        return "MODEL_INPUTS"

    monkeypatch.setattr(gr00t_data, "build_inputs", build_inputs)
    state = np.zeros(2, dtype=np.float32)
    frame = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    observation = {"state": state, "ego_view_image": frame}
    assert policy._build_model_inputs(observation, state) == "MODEL_INPUTS"
    assert (
        policy._build_model_inputs_batch([observation, observation], [state, state])
        == "MODEL_INPUTS"
    )
    np.testing.assert_array_equal(received[0], frame[None])
    np.testing.assert_array_equal(received[1], np.stack([frame, frame]))
