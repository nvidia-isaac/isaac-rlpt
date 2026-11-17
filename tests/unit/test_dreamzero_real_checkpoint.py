# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process DreamZero base model against a real bundled checkpoint.

The scripted-fake enrollment in ``test_base_model_policy.py`` covers the
contract mechanics; this module covers what only the real model can show —
checkpoint-derived geometry, the deterministic seeded native mode, and that
injected latents actually reach and steer the sampler.

Gated like the gr00t on-checkpoint suite: set

    ISAAC_RL_TEST_DREAMZERO_CKPT=<bundled SFT checkpoint dir>
    ISAAC_RL_TEST_DREAMZERO_TAG=<embodiment tag; default libero_sim>

and run on a CUDA machine with the groot package installed (the RLinf venv
recipe in ``simulators/libero/README.md`` Part 4). Loading the 5B bundle
takes ~26 GiB of VRAM and ~2 minutes; the base model is built once per module.
"""

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.timeout(0),
]  # loads a real DreamZero checkpoint

_CKPT_ENV = "ISAAC_RL_TEST_DREAMZERO_CKPT"
_TAG_ENV = "ISAAC_RL_TEST_DREAMZERO_TAG"

if not os.environ.get(_CKPT_ENV):
    pytest.skip(
        f"set {_CKPT_ENV} (and optionally {_TAG_ENV}) to run the DreamZero "
        "real-checkpoint suite",
        allow_module_level=True,
    )

pytest.importorskip("torch", reason="torch needed to load the model")
pytest.importorskip(
    "groot.vla.data.schema",
    reason="groot (the DreamZero repo) not installed; see docs/base_models.md",
)

#: The released checkpoints' geometry, keyed by embodiment tag.
_EXPECTED_GEOMETRY = {
    "libero_sim": {"action_horizon": 16, "env_action_dim": 7},
}


@pytest.fixture(scope="module")
def real_base_model():
    import torch
    from omegaconf import OmegaConf

    from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
        DreamZeroBaseModelPolicy,
    )

    if not torch.cuda.is_available():
        pytest.skip("the DreamZero real-checkpoint suite needs a CUDA device")
    tag = os.environ.get(_TAG_ENV, "libero_sim")
    expected = _EXPECTED_GEOMETRY[tag]
    cfg = OmegaConf.create(
        {
            "base_model": {
                "ckpt": os.environ[_CKPT_ENV],
                "embodiment_tag": tag,
                "metadata_json_path": None,
                "tokenizer_path": "google/umt5-xxl",
                "text_encoder_cpu_offload": False,
                # Uncompiled: the compile path is covered by the launcher
                # ladder, and warmup would dominate the suite's runtime.
                "torch_compile": False,
                "num_dit_steps": None,
                "predict_micro_batch_size": 1,
                "target_video_height": None,
                "target_video_width": None,
                "task_instruction": None,
                "action_horizon": None,
                "action_dim": None,
            },
            "dsrl": {"noise_dim": 32, "action_exec_len": expected["action_horizon"]},
            "training": {"chunk_size": None},
        }
    )
    return DreamZeroBaseModelPolicy.from_dsrl_config(cfg, "cuda"), expected


def _obs(rng_seed: int = 0) -> dict:
    rng = np.random.default_rng(rng_seed)
    return {
        "main_image": rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8),
        "state": rng.standard_normal((1, 8)).astype(np.float32),
        "prompt": ["pick up the object and place it on the plate"],
    }


def test_checkpoint_geometry(real_base_model):
    base_model, expected = real_base_model
    assert base_model.supports_steering is True
    assert base_model.action_horizon == expected["action_horizon"]
    assert base_model.action_dim == expected["env_action_dim"]
    assert base_model.latent_dim == 32  # the WAN head's padded action-latent width


def test_native_mode_is_deterministic(real_base_model):
    """The model draws its action latent with a fixed seed: the same
    observation must decode to the same chunk (the base policy's own
    evaluation protocol, and what made the parity sweeps reproducible)."""
    base_model, expected = real_base_model
    obs = _obs()
    first = base_model.infer(obs, None)["actions"]
    second = base_model.infer(obs, None)["actions"]
    assert first.shape == (1, expected["action_horizon"], expected["env_action_dim"])
    np.testing.assert_allclose(first, second, atol=1e-3)


def test_injected_latents_reach_and_steer_the_sampler(real_base_model):
    """Distinct latents must decode to distinct chunks — with the
    consumed-flag guard active, this is end-to-end proof the injection
    steers the real flow-matching denoise rather than being ignored."""
    base_model, _ = real_base_model
    obs = _obs()
    rng = np.random.default_rng(7)
    z_a = rng.standard_normal((1, 32)).astype(np.float32)
    z_b = rng.standard_normal((1, 32)).astype(np.float32)

    actions_a = base_model.infer(obs, z_a)["actions"]
    actions_a_again = base_model.infer(obs, z_a)["actions"]
    actions_b = base_model.infer(obs, z_b)["actions"]

    np.testing.assert_allclose(actions_a, actions_a_again, atol=1e-3)
    assert np.abs(actions_a - actions_b).max() > 1e-3
    assert np.isfinite(actions_a).all() and np.isfinite(actions_b).all()


def test_single_latent_equals_pretiled_chunk(real_base_model):
    """A ``[B, 32]`` latent and its explicit ``[B, H, 32]`` tiling must
    decode identically — pins the tile-one-z convention the validated
    recipe trains with."""
    base_model, expected = real_base_model
    obs = _obs()
    z = np.random.default_rng(11).standard_normal((1, 32)).astype(np.float32)
    tiled = np.repeat(z[:, None, :], expected["action_horizon"], axis=1)
    np.testing.assert_allclose(
        base_model.infer(obs, z)["actions"],
        base_model.infer(obs, tiled)["actions"],
        atol=1e-3,
    )
