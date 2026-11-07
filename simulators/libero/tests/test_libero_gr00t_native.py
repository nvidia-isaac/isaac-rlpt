# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-checkpoint native inference through the LIBERO-embodiment base model.

Gated on a downloaded GR00T-N1.7-LIBERO checkpoint (per-suite subfolder) and
a CUDA GPU:

    ISAAC_RL_TEST_GR00T_LIBERO_CKPT=/path/to/GR00T-N1.7-LIBERO/libero_10 \
      pytest tests/integration/test_libero_gr00t_native.py -m gpu

Pins the full baked-statistics path end to end: processor restoration
(``libero_sim`` modality configs plus statistics), the derived flat layout,
and one native decode from a synthetic two-camera observation into the
16-step, 7-dim raw action chunk (gripper in the training data's ``[0, 1]``
convention).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_CKPT_ENV = "ISAAC_RL_TEST_GR00T_LIBERO_CKPT"

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(0)]  # loads a real GR00T checkpoint


@pytest.fixture(scope="module")
def base_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip(
        "gr00t.data.embodiment_tags", reason="external gr00t package not installed"
    )
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    checkpoint = os.environ.get(_CKPT_ENV)
    if not checkpoint or not os.path.isdir(checkpoint):
        pytest.skip(
            f"set {_CKPT_ENV} to a GR00T-N1.7-LIBERO suite checkpoint directory"
        )

    from isaac_rl_posttraining.policies.gr00t import Gr00tBaseModelPolicy

    return Gr00tBaseModelPolicy.from_checkpoint(
        base_ckpt=checkpoint, embodiment_tag="libero_sim", device="cuda"
    )


def _libero_observation() -> dict:
    rng = np.random.default_rng(0)
    frame = lambda: rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return {
        "video.image": frame(),
        "video.wrist_image": frame(),
        "state": np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.02, -0.02], np.float32),
        "prompt": "put both moka pots on the stove",
    }


def test_baked_registration_shapes(base_model):
    """The checkpoint self-describes: 16-step decode of 7-dim actions from a
    40-step flow window (no dataset directory involved)."""
    assert base_model.action_horizon == 16
    assert base_model.action_dim == 7
    assert base_model.latent_horizon == 40
    assert base_model.latent_dim == 7


def test_native_infer_decodes_a_raw_chunk(base_model):
    result = base_model.infer(_libero_observation(), noise=None)
    actions = result["actions"]
    assert actions.shape == (1, 16, 7)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()
    # Un-normalized dataset ranges: OSC deltas within [-1, 1] (the data's
    # min/max), gripper in the [0, 1] dataset convention.
    assert np.abs(actions[..., :6]).max() <= 1.0 + 1e-5
    assert actions[..., 6].min() >= -1e-5 and actions[..., 6].max() <= 1.0 + 1e-5


def test_steered_infer_accepts_flow_window_latents(base_model):
    """A [1, 40, 7] per-step latent (the DSRL chunk geometry) decodes."""
    noise = np.zeros((1, 40, 7), dtype=np.float32)
    result = base_model.infer(_libero_observation(), noise=noise)
    assert result["actions"].shape == (1, 16, 7)
    assert np.isfinite(result["actions"]).all()
