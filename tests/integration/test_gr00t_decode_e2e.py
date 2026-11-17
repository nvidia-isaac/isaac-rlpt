# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T base model decode end-to-end on a real checkpoint (marked gpu, no sim).

Loads the real GR00T N1.7 base model from a checkpoint directory
(``POSTTRAIN_BASE_CKPT``) with single-task normalization stats
(``POSTTRAIN_DATA_ROOT/<task>/lerobot`` or ``GR00T_DATASET_DIR``) and pins the
inference contract on real weights:

* ``infer(obs, noise=[1, 40, 35])`` decodes a finite ``[1, 40, 35]`` chunk
  (chunk-40 flow horizon, 35-dim WBC joint actions),
* native mode (``noise=None``) runs through the same path,
* the base model reports the shape properties RL code sizes itself with.

No HF downloads happen here: the checkpoint and dataset must already exist
locally (skip otherwise).
"""

import inspect
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(0)]  # loads a real GR00T checkpoint

DECODE_HORIZON = 16  # the checkpoint's baked action window (what infer returns)
LATENT_HORIZON = 40  # the flow head's noise window (steering latents accepted)
ACTION_DIM = 35
LATENT_DIM = 35
STATE_DIM = 43
TASK = os.environ.get("ISAAC_RL_POSTTRAINING_TEST_TASK", "LMDrillLift")


def _checkpoint_dir():
    ckpt = os.environ.get("POSTTRAIN_BASE_CKPT") or os.environ.get(
        "ISAAC_RL_POSTTRAINING_TEST_GR00T_CKPT"
    )
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip(
            "POSTTRAIN_BASE_CKPT not set / not a directory (real GR00T checkpoint required)"
        )
    return ckpt


def _dataset_dir():
    explicit = os.environ.get("GR00T_DATASET_DIR")
    if explicit and os.path.isdir(explicit):
        return explicit
    root = os.environ.get("POSTTRAIN_DATA_ROOT")
    if root:
        candidate = os.path.join(root, TASK, "lerobot")
        if os.path.isdir(candidate):
            return candidate
    pytest.skip(
        "no dataset dir (set GR00T_DATASET_DIR or POSTTRAIN_DATA_ROOT/<task>/lerobot)"
    )


def _find_base_model_cls():
    """The GR00T BaseModelPolicy implementation from the policies subpackage."""
    pytest.importorskip("torch")
    pytest.importorskip("gr00t")
    from isaac_rl_posttraining.policies.gr00t import Gr00tBaseModelPolicy

    return Gr00tBaseModelPolicy


@pytest.fixture(scope="module")
def base_model():
    cls = _find_base_model_cls()
    ckpt, dataset = _checkpoint_dir(), _dataset_dir()
    factory = getattr(cls, "from_checkpoint", None)
    if factory is None:
        pytest.fail(
            f"{cls.__name__} must expose from_checkpoint(...) — the canonical construction "
            "path pinned by the unit contract suite",
            pytrace=False,
        )
    params = inspect.signature(factory).parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    kwargs = {}
    for value, names in (
        (
            ckpt,
            (
                "base_ckpt",
                "checkpoint",
                "checkpoint_dir",
                "ckpt",
                "model_path",
                "gr00t_base_ckpt",
            ),
        ),
        (dataset, ("dataset_dir", "data_dir", "dataset_path", "gr00t_dataset_dir")),
        ("cuda", ("device",)),
    ):
        target = next((n for n in names if n in params), None)
        if target is None and accepts_kwargs:
            target = names[0]
        if target is not None:
            kwargs[target] = value
    try:
        return factory(**kwargs)
    except TypeError as exc:
        pytest.fail(
            f"API mismatch calling {cls.__name__}.from_checkpoint (signature {sorted(params)}): {exc}",
            pytrace=False,
        )


def _obs():
    return {
        "ego_view_image": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        "state": np.zeros((1, STATE_DIM), dtype=np.float32),
    }


def test_real_checkpoint_reports_contract_properties(base_model):
    """The loaded base model exposes the shape contract RL code sizes itself with."""
    assert base_model.action_horizon == DECODE_HORIZON
    assert base_model.latent_horizon == LATENT_HORIZON
    assert base_model.action_dim == ACTION_DIM
    assert base_model.latent_dim == LATENT_DIM


def test_infer_with_fixed_latent_decodes_a_finite_chunk(base_model):
    """A fixed [1, LATENT_HORIZON, 35] latent decodes to a finite [1, DECODE_HORIZON, 35] action chunk."""
    noise = (
        np.random.default_rng(0)
        .standard_normal((1, LATENT_HORIZON, LATENT_DIM))
        .astype(np.float32)
    )
    result = base_model.infer(observations=_obs(), noise=noise)
    assert "actions" in result
    actions = np.asarray(result["actions"])
    assert actions.shape == (1, DECODE_HORIZON, ACTION_DIM), f"got {actions.shape}"
    assert np.all(np.isfinite(actions)), "decoded actions contain non-finite values"


def test_native_mode_runs(base_model):
    """noise=None (native, un-steered base policy) runs the same decode path."""
    result = base_model.infer(observations=_obs(), noise=None)
    actions = np.asarray(result["actions"])
    assert actions.shape == (1, DECODE_HORIZON, ACTION_DIM)
    assert np.all(np.isfinite(actions))
