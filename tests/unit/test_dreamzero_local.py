# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process DreamZero base model building blocks (no groot, no model stack).

Covers the pieces that must hold without the heavy model installed:

1. the DSRL noise patch semantics — 3-D action-draw-only interception, the
   consumed flag, exact-shape validation, and the untouched seeded video
   draw;
2. the embodiment layouts — env-obs conversion shapes/dtypes for both
   embodiments, the object-ndarray language contract (a plain list would be
   split into *characters* downstream), blank-filled missing views, and the
   pinned multi-view concat / prompt templates the checkpoints were trained
   with;
3. the build-time geometry validation matrix.
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch needed by the DSRL noise patch")
import torch

from isaac_rl_posttraining.policies.dreamzero.layouts import (
    LIBERO_SIM_SPEC,
    convert_env_obs,
    get_embodiment_spec,
    normalize_instruction_text,
)
from isaac_rl_posttraining.policies.dreamzero.patch import (
    _dsrl_generate_noise,
)
from isaac_rl_posttraining.policies.dreamzero.validation import (
    validate_dreamzero_dsrl_geometry,
)

# ---------------------------------------------------------------------------
# The DSRL noise patch (pure torch; the fake head stands in for WANPolicyHead)
# ---------------------------------------------------------------------------


class _FakeHead:
    """The two side-channel attributes the patch reads/writes."""

    def __init__(self, noise=None):
        self._dsrl_external_action_noise = noise
        self._dsrl_noise_consumed = False

    generate_noise = _dsrl_generate_noise


def test_patch_intercepts_only_the_3d_action_draw():
    z = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 32)
    head = _FakeHead(noise=z)

    out = head.generate_noise((2, 16, 32), seed=1140, device="cpu", dtype=torch.float32)
    assert head._dsrl_noise_consumed is True
    torch.testing.assert_close(out, z)

    # The 5-D video draw keeps the seeded behavior and does not consume.
    head_video = _FakeHead(noise=z)
    video = head_video.generate_noise((1, 16, 1, 8, 8), seed=1140, dtype=torch.float32)
    assert head_video._dsrl_noise_consumed is False
    reference = torch.randn(
        (1, 16, 1, 8, 8), generator=torch.Generator("cpu").manual_seed(1140)
    )
    torch.testing.assert_close(video, reference)


def test_patch_casts_injected_noise_to_requested_dtype():
    z = torch.zeros(1, 16, 32, dtype=torch.float32)
    head = _FakeHead(noise=z)
    out = head.generate_noise((1, 16, 32), dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16


def test_patch_rejects_shape_mismatch():
    head = _FakeHead(noise=torch.zeros(1, 16, 8))
    with pytest.raises(ValueError, match="does not match the.*action-latent shape"):
        head.generate_noise((1, 16, 32))
    assert head._dsrl_noise_consumed is False


def test_patch_is_inert_without_side_channel():
    head = _FakeHead(noise=None)
    first = head.generate_noise((1, 16, 32), seed=7, dtype=torch.float32)
    second = head.generate_noise((1, 16, 32), seed=7, dtype=torch.float32)
    torch.testing.assert_close(first, second)  # seeded draw, deterministic
    assert head._dsrl_noise_consumed is False


# ---------------------------------------------------------------------------
# Embodiment layouts
# ---------------------------------------------------------------------------


def _libero_obs(batch_size=2):
    rng = np.random.default_rng(0)
    return {
        "main_image": rng.integers(
            0, 256, size=(batch_size, 32, 32, 3), dtype=np.uint8
        ),
        "wrist_image": rng.integers(
            0, 256, size=(batch_size, 32, 32, 3), dtype=np.uint8
        ),
        "state": rng.standard_normal((batch_size, 8)).astype(np.float32),
        "prompt": ["pick up the bowl"] * batch_size,
    }


def test_libero_conversion_shapes_and_keys():
    obs = _libero_obs()
    converted = convert_env_obs(obs, LIBERO_SIM_SPEC)
    assert set(converted) == {
        "video.image",
        "video.wrist_image",
        "state.state",
        "annotation.task",
    }
    assert converted["video.image"].shape == (2, 1, 32, 32, 3)
    assert converted["video.image"].dtype == np.uint8
    assert converted["state.state"].shape == (2, 1, 8)
    assert converted["state.state"].dtype == np.float32


def test_language_is_an_object_ndarray_of_full_strings():
    """A plain list of strings gets split into characters by the per-sample
    tree split downstream; the object ndarray is treated as one leaf."""
    converted = convert_env_obs(_libero_obs(), LIBERO_SIM_SPEC)
    language = converted["annotation.task"]
    assert isinstance(language, np.ndarray) and language.dtype == object
    assert language[1] == "pick up the bowl"


def test_float_images_in_unit_range_are_scaled():
    obs = _libero_obs()
    obs["main_image"] = obs["main_image"].astype(np.float32) / 255.0
    converted = convert_env_obs(obs, LIBERO_SIM_SPEC)
    assert converted["video.image"].dtype == np.uint8
    assert converted["video.image"].max() > 1


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda obs: obs.pop("prompt"),
        lambda obs: obs.pop("state"),
        lambda obs: obs.pop("main_image"),
        lambda obs: obs.__setitem__("main_image", np.zeros((2, 32, 32), np.uint8)),
        lambda obs: obs.__setitem__("wrist_image", np.zeros((5, 32, 32, 3), np.uint8)),
        lambda obs: obs.__setitem__("state", np.zeros((2, 8, 1), np.float32)),
    ],
)
def test_malformed_observations_raise(corrupt):
    obs = _libero_obs()
    corrupt(obs)
    with pytest.raises((KeyError, ValueError)):
        convert_env_obs(obs, LIBERO_SIM_SPEC)


def test_libero_concat_is_horizontal_exterior_then_wrist():
    images = np.zeros((2, 1, 3, 4, 5), dtype=np.uint8)
    images[0] = 1  # exterior
    images[1] = 2  # wrist
    tiled = LIBERO_SIM_SPEC.concat_views(images)
    assert tiled.shape == (1, 1, 3, 4, 10)
    assert (tiled[0, :, :, :, :5] == 1).all() and (tiled[0, :, :, :, 5:] == 2).all()


def test_prompt_templates_pinned():
    """The templates are what the checkpoints were SFT-trained on."""
    libero = LIBERO_SIM_SPEC.format_prompt("pick up the bowl")
    assert libero == (
        "A multi-view video shows that a robot pick up the bowl The video is "
        "split into two horizontal views: the left view shows the exterior "
        "camera and the right view shows the wrist camera. The robot "
        "pick up the bowl"
    )


def test_spec_registry_and_projector_ids():
    assert get_embodiment_spec("libero_sim") is LIBERO_SIM_SPEC
    assert LIBERO_SIM_SPEC.tag_mapping == {"libero_sim": 21}
    with pytest.raises(ValueError, match="Unsupported DreamZero embodiment_tag"):
        get_embodiment_spec("g1_wbc")


def test_normalize_instruction_text():
    assert normalize_instruction_text("Pick UP the Bowl") == "pick up the bowl"
    assert normalize_instruction_text("['Close the drawer']") == "close the drawer"


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

_VALID_GEOMETRY = {
    "head_action_dim": 32,
    "head_action_horizon": 16,
    "env_action_dim": 7,
    "noise_dim": 32,
    "chunk_size": None,
    "action_exec_len": 16,
    "base_model_action_horizon": None,
    "base_model_action_dim": None,
}


def test_geometry_validation_accepts_the_validated_recipe():
    validate_dreamzero_dsrl_geometry(**_VALID_GEOMETRY)
    validate_dreamzero_dsrl_geometry(
        **{
            **_VALID_GEOMETRY,
            "base_model_action_horizon": 16,
            "base_model_action_dim": 7,
        }
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"noise_dim": 8}, "dsrl.noise_dim"),
        ({"chunk_size": 40}, "training.chunk_size"),
        ({"action_exec_len": 8}, "dsrl.action_exec_len"),
        ({"base_model_action_horizon": 24}, "base_model.action_horizon"),
        ({"base_model_action_dim": 8}, "base_model.action_dim"),
    ],
)
def test_geometry_validation_rejects_each_mismatch(override, match):
    with pytest.raises(ValueError, match=match):
        validate_dreamzero_dsrl_geometry(**{**_VALID_GEOMETRY, **override})


# ---------------------------------------------------------------------------
# Base model noise handling (scripted policy; no model stack)
# ---------------------------------------------------------------------------


class _RecordingPolicy:
    """Duck-typed DreamZeroLocalPolicy recording what reaches predict_chunk."""

    action_horizon = 16
    noise_dim = 32
    env_action_dim = 7
    task = None

    def __init__(self):
        self.last_noise = None

    def predict_chunk(self, env_obs, noise):
        self.last_noise = noise
        batch = int(np.asarray(env_obs["main_image"]).shape[0])
        return np.zeros((batch, self.action_horizon, self.env_action_dim), np.float32)


def _base_model_obs(batch_size=1):
    rng = np.random.default_rng(0)
    return {
        "main_image": rng.integers(
            0, 256, size=(batch_size, 16, 16, 3), dtype=np.uint8
        ),
        "wrist_image": rng.integers(
            0, 256, size=(batch_size, 16, 16, 3), dtype=np.uint8
        ),
        "state": np.zeros((batch_size, 8), dtype=np.float32),
        "prompt": ["stack the cube"] * batch_size,
    }


def test_base_model_tiles_single_latent_and_clips_at_injection():
    from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP
    from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
        DreamZeroBaseModelPolicy,
    )

    policy = _RecordingPolicy()
    base_model = DreamZeroBaseModelPolicy(policy)
    assert base_model.supports_steering is True
    assert (
        base_model.latent_dim,
        base_model.action_dim,
        base_model.action_horizon,
    ) == (
        32,
        7,
        16,
    )

    z = np.full((1, 32), 1e3, dtype=np.float32)  # one latent, extreme values
    out = base_model.infer(_base_model_obs(), z)
    assert out["actions"].shape == (1, 16, 7)
    # The single latent was clipped, then tiled across the full horizon.
    assert policy.last_noise.shape == (1, 16, 32)
    np.testing.assert_allclose(policy.last_noise, DEFAULT_LATENT_CLIP)

    base_model.infer(_base_model_obs(), None)
    assert policy.last_noise is None  # native mode: the sampler's own draw


def test_base_model_rejects_bad_noise_shapes_and_wrong_latent_dim():
    from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
        DreamZeroBaseModelPolicy,
    )

    base_model = DreamZeroBaseModelPolicy(_RecordingPolicy())
    for bad in (np.zeros((1, 8)), np.zeros((1, 17, 32)), np.zeros((32,))):
        with pytest.raises(ValueError, match="noise must have shape"):
            base_model.infer(_base_model_obs(), bad)
    with pytest.raises(ValueError, match="latent_dim"):
        DreamZeroBaseModelPolicy(_RecordingPolicy(), latent_dim=8)


def test_from_dsrl_config_required_fields_fail_before_model_load():
    from omegaconf import OmegaConf

    from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
        DreamZeroBaseModelPolicy,
    )

    def cfg_with(**base_model_overrides):
        return OmegaConf.create(
            {
                "base_model": {
                    "ckpt": "/checkpoints/dreamzero",
                    "embodiment_tag": "libero_sim",
                    "tokenizer_path": "google/umt5-xxl",
                    "target_video_height": None,
                    "target_video_width": None,
                    **base_model_overrides,
                },
                "dsrl": {"noise_dim": 32, "action_exec_len": 16},
                "training": {"chunk_size": None},
            }
        )

    with pytest.raises(ValueError, match="base_model.ckpt"):
        DreamZeroBaseModelPolicy.from_dsrl_config(cfg_with(ckpt=None), "cpu")
    with pytest.raises(ValueError, match="base_model.embodiment_tag"):
        DreamZeroBaseModelPolicy.from_dsrl_config(cfg_with(embodiment_tag=None), "cpu")
    with pytest.raises(ValueError, match="base_model.tokenizer_path"):
        DreamZeroBaseModelPolicy.from_dsrl_config(cfg_with(tokenizer_path=None), "cpu")
    with pytest.raises(ValueError, match="must be set together"):
        DreamZeroBaseModelPolicy.from_dsrl_config(
            cfg_with(target_video_height=160), "cpu"
        )


def test_ensure_batched_env_obs_lifts_single_env_slices():
    """The vectorized collector decodes per env with unbatched observations."""
    from isaac_rl_posttraining.policies.dreamzero.layouts import ensure_batched_env_obs

    single = {
        "main_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the bowl",
    }
    lifted = ensure_batched_env_obs(single)
    assert lifted["main_image"].shape == (1, 256, 256, 3)
    assert lifted["wrist_image"].shape == (1, 256, 256, 3)
    assert lifted["state"].shape == (1, 8)
    assert lifted["prompt"] == "pick up the bowl"
    converted = convert_env_obs(lifted, LIBERO_SIM_SPEC)
    assert converted["video.image"].shape == (1, 1, 256, 256, 3)

    batched = {
        "main_image": np.zeros((2, 256, 256, 3), dtype=np.uint8),
        "state": np.zeros((2, 8), dtype=np.float32),
    }
    assert ensure_batched_env_obs(batched) is batched  # NOSONAR - must not copy
