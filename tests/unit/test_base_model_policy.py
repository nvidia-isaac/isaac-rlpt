# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseModelPolicy contract tests.

Two layers:

1. MockBaseModelPolicy cases, which use the mock's position-coded decode to make
   broadcast/steering/clipping semantics exactly observable.
2. A base-model-generic contract suite parameterized over every constructible
   base_model: the in-tree mock always runs; the GR00T base model runs when the
   external ``gr00t`` package plus checkpoint env vars
   (``ISAAC_RL_TEST_GR00T_BASE_CKPT`` / ``ISAAC_RL_TEST_GR00T_DATASET_DIR``
   and optional ``ISAAC_RL_TEST_GR00T_BC_CKPT``) are available and skips
   otherwise; the served DreamZero base model runs against the in-process fake
   server (no network, no model stack) whenever ``msgpack`` is installed; the
   in-process DreamZero and pi0 base models run over scripted stand-in policies.

The generic suite is capability-aware: steering-mode cases assert the pinned
``NotImplementedError`` on base models that report ``supports_steering=False``
and exercise steered behavior everywhere else, so native-only base models get
full coverage instead of a blanket skip.
"""

import inspect
import os

import numpy as np
import pytest
from conftest import FakeDreamZeroServer

from isaac_rl_posttraining.policies.base import (
    DEFAULT_LATENT_CLIP,
    BaseModelPolicy,
    MockBaseModelPolicy,
)
from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
    DreamZeroBaseModelPolicy,
)
from isaac_rl_posttraining.policies.dreamzero.server_base_model import (
    DreamZeroServerBaseModelPolicy,
)


def _position_code(base_model):
    steps = np.arange(base_model.action_horizon, dtype=np.float64)[:, None]
    dims = np.arange(base_model.action_dim, dtype=np.float64)[None, :]
    return 10.0 * steps + 1000.0 * dims


def test_native_mode_shape_and_dict_key():
    base_model = MockBaseModelPolicy()
    out = base_model.infer({"state": np.zeros((3, 5))}, noise=None)
    assert set(out) == {"actions"}
    assert out["actions"].shape == (3, base_model.action_horizon, base_model.action_dim)


def test_native_mode_noise_is_fresh_per_call():
    base_model = MockBaseModelPolicy()
    obs = {"state": np.zeros((2, 4))}
    first = base_model.infer(obs, None)["actions"]
    second = base_model.infer(obs, None)["actions"]
    assert not np.allclose(first, second)


def test_per_step_noise_steers_only_first_k_steps():
    base_model = MockBaseModelPolicy(action_horizon=4, action_dim=3)
    noise = np.zeros((1, 2, 3))
    actions = base_model.infer({}, noise)["actions"]
    residual = actions[0] - _position_code(base_model)
    np.testing.assert_allclose(residual[:2], 0.0, atol=1e-3)
    # Steps beyond K must carry fresh gaussian noise, not zeros.
    assert np.abs(residual[2:]).max() > 1e-3


def test_unsteered_action_dims_get_fresh_noise():
    base_model = MockBaseModelPolicy(action_horizon=2, action_dim=4, latent_dim=2)
    noise = np.zeros((1, 2, 2))
    actions = base_model.infer({}, noise)["actions"]
    residual = actions[0] - _position_code(base_model)
    np.testing.assert_allclose(residual[:, :2], 0.0, atol=1e-3)
    assert np.abs(residual[:, 2:]).max() > 1e-3


def test_full_noise_makes_output_deterministic():
    """A full [B, H, latent_dim] latent leaves nothing to sample: two differently seeded mocks agree."""
    noise = np.linspace(-1.0, 1.0, num=1 * 4 * 3).reshape(1, 4, 3)
    first = MockBaseModelPolicy(action_horizon=4, action_dim=3, seed=1).infer({}, noise)
    second = MockBaseModelPolicy(action_horizon=4, action_dim=3, seed=2).infer(
        {}, noise
    )
    np.testing.assert_array_equal(first["actions"], second["actions"])


@pytest.mark.parametrize(
    "bad_shape",
    [
        (2, 3),  # batch mismatch (observations say B=1)
        (1, 2, 5),  # wrong latent width
        (3,),  # missing batch dimension
    ],
)
def test_invalid_noise_shapes_raise(bad_shape):
    base_model = MockBaseModelPolicy(action_horizon=4, action_dim=3)
    with pytest.raises(ValueError):
        base_model.infer({"state": np.zeros((1, 2))}, np.zeros(bad_shape))


# ---------------------------------------------------------------------------
# Base-model-generic contract suite (mock always; gr00t on-cluster;
# dreamzero_server via the in-process fake server; pi0 via a scripted openpi
# stand-in)
# ---------------------------------------------------------------------------

_GR00T_BASE_ENV = "ISAAC_RL_TEST_GR00T_BASE_CKPT"
_GR00T_BC_ENV = "ISAAC_RL_TEST_GR00T_BC_CKPT"
_GR00T_DATA_ENV = "ISAAC_RL_TEST_GR00T_DATASET_DIR"


def _build_gr00t_base_model():
    # The local ``gr00t`` checkout directory resolves as a hollow namespace
    # package, so gate on a real leaf module, not on ``import gr00t``.
    pytest.importorskip(
        "gr00t.data.embodiment_tags",
        reason="external gr00t package not installed (contract runs on-cluster, Wave 3)",
    )
    from isaac_rl_posttraining.policies.gr00t import (
        Gr00tBaseModelPolicy as base_model_cls,
    )

    base_ckpt = os.environ.get(_GR00T_BASE_ENV)
    dataset_dir = os.environ.get(_GR00T_DATA_ENV)
    if not base_ckpt or not dataset_dir:
        pytest.skip(
            f"set {_GR00T_BASE_ENV} and {_GR00T_DATA_ENV} (and optionally {_GR00T_BC_ENV}) "
            "to run the gr00t contract on a checkpoint"
        )
    return base_model_cls.from_checkpoint(
        base_ckpt=base_ckpt,
        bc_ckpt=os.environ.get(_GR00T_BC_ENV),
        dataset_dir=dataset_dir,
    )


def _build_pi0_base_model():
    """The pi0 base model over a scripted stand-in for the openpi policy.

    The stand-in reproduces the pi0_libero geometry (50-step chunk, 32-dim
    padded flow latents, 7-dim decoded actions) so the base model's own
    validation, clipping, and shaping run the full contract offline.
    """
    pytest.importorskip("torch", reason="torch needed by the pi0 policy module")
    from isaac_rl_posttraining.policies.pi0.base_model import Pi0BaseModelPolicy

    class _ScriptedOpenpiPolicy:
        action_horizon = 50

        def infer(self, observations, noise=None):
            rng = np.random.default_rng(0)
            return {
                "actions": rng.standard_normal((1, self.action_horizon, 7)).astype(
                    np.float32
                )
            }

    return Pi0BaseModelPolicy(_ScriptedOpenpiPolicy(), action_dim=7, latent_dim=32)


def _build_dreamzero_server_base_model():
    """The served DreamZero base model driven through the scripted in-process server.

    Only ``msgpack`` (the wire codec) is needed; the fake connection stands
    in for the network, so the contract runs everywhere, deterministic and
    offline.
    """
    pytest.importorskip("msgpack", reason="msgpack needed for the DreamZero wire codec")
    from isaac_rl_posttraining.policies.dreamzero.client import DreamZeroClient

    server = FakeDreamZeroServer()
    client = DreamZeroClient("ws://fake-dreamzero:8000", open_connection=server.connect)
    return DreamZeroServerBaseModelPolicy(client)


def _build_dreamzero_base_model():
    """The in-process DreamZero base model over a scripted stand-in policy.

    The stand-in reproduces the 5B/LIBERO geometry (16-step chunk, 32-dim
    WAN action latents, 7-dim decoded actions) so the base model's own
    validation, clipping, tiling, and shaping run the full contract offline.
    """
    pytest.importorskip("torch", reason="torch needed by the DreamZero policy module")
    from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
        DreamZeroBaseModelPolicy,
    )

    class _ScriptedDreamZeroPolicy:
        action_horizon = 16
        noise_dim = 32
        env_action_dim = 7
        task = None

        def predict_chunk(self, env_obs, noise):
            batch = int(np.asarray(env_obs["main_image"]).shape[0])
            if noise is not None:
                expected = (batch, self.action_horizon, self.noise_dim)
                assert noise.shape == expected, noise.shape
            rng = np.random.default_rng(0)
            return rng.standard_normal(
                (batch, self.action_horizon, self.env_action_dim)
            ).astype(np.float32)

    return DreamZeroBaseModelPolicy(_ScriptedDreamZeroPolicy())


@pytest.fixture(
    scope="module",
    params=["mock", "gr00t", "dreamzero_local", "dreamzero_server", "pi0"],
)
def any_base_model(request):
    """Every registered, constructible BaseModelPolicy implementation."""
    if request.param == "mock":
        return MockBaseModelPolicy(action_horizon=6, action_dim=5, seed=123)
    if request.param == "dreamzero_local":
        return _build_dreamzero_base_model()
    if request.param == "dreamzero_server":
        return _build_dreamzero_server_base_model()
    if request.param == "pi0":
        return _build_pi0_base_model()
    return _build_gr00t_base_model()


def _contract_obs(base_model) -> dict:
    """A batch-1 observation dict appropriate for the base model under test."""
    if isinstance(base_model, MockBaseModelPolicy):
        return {"state": np.zeros((1, 4), dtype=np.float32)}
    rng = np.random.default_rng(0)
    if isinstance(base_model, DreamZeroBaseModelPolicy):
        return {
            "main_image": rng.integers(0, 256, size=(1, 64, 64, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, size=(1, 64, 64, 3), dtype=np.uint8),
            "state": np.zeros((1, 8), dtype=np.float32),
            "prompt": ["pick up the object"],
        }
    if isinstance(base_model, DreamZeroServerBaseModelPolicy):
        return {
            "observation/exterior_image_0_left": rng.integers(
                0, 256, size=(1, 180, 320, 3), dtype=np.uint8
            ),
            "observation/exterior_image_1_left": rng.integers(
                0, 256, size=(1, 180, 320, 3), dtype=np.uint8
            ),
            "observation/wrist_image_left": rng.integers(
                0, 256, size=(1, 180, 320, 3), dtype=np.uint8
            ),
            "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
            "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
            "prompt": "pick up the object",
        }
    return {
        "ego_view_image": rng.integers(0, 256, size=(1, 480, 640, 3), dtype=np.uint8),
        "state": rng.standard_normal((1, 43)).astype(np.float32),
    }


def _steered_or_refused(base_model, obs, noise):
    """Run a steered call, honoring the capability flag (contract note 5).

    Returns the actions array when the base model supports steering; on a
    native-only base model, asserts the pinned ``NotImplementedError`` and
    returns ``None``.
    """
    if base_model.supports_steering:
        return base_model.infer(obs, noise)["actions"]
    with pytest.raises(NotImplementedError):
        base_model.infer(obs, noise)
    return None


def test_actions_shape_b_h_araw(any_base_model):
    """noise=None runs the un-steered base policy: one 'actions' array of shape [B, H, action_dim]."""
    out = any_base_model.infer(_contract_obs(any_base_model), None)
    assert set(out) == {"actions"} and isinstance(out["actions"], np.ndarray)
    assert out["actions"].shape == (
        1,
        any_base_model.action_horizon,
        any_base_model.action_dim,
    )


def test_single_latent_broadcasts_over_horizon(any_base_model):
    # A per-dimension pattern (not a constant) so a permuted or partially applied broadcast is caught.
    noise = np.linspace(-1.0, 1.0, any_base_model.latent_dim, dtype=np.float32)[None]
    actions = _steered_or_refused(any_base_model, _contract_obs(any_base_model), noise)
    if actions is None:
        return  # native-only base model: the refusal is the contract behavior
    assert actions.shape == (
        1,
        any_base_model.action_horizon,
        any_base_model.action_dim,
    )
    if isinstance(any_base_model, MockBaseModelPolicy):
        residual = actions[0] - _position_code(any_base_model)
        np.testing.assert_allclose(
            residual[:, : any_base_model.latent_dim],
            np.tile(noise, (any_base_model.action_horizon, 1)),
            atol=1e-3,
        )


def test_per_step_latents_k_le_h(any_base_model):
    horizon = any_base_model.latent_horizon
    obs = _contract_obs(any_base_model)
    if not any_base_model.supports_steering:
        # Native-only base model: every steered shape gets the pinned refusal,
        # including the over-long K (steering is refused before shape checks).
        for k in (min(2, horizon), horizon, horizon + 1):
            with pytest.raises(NotImplementedError):
                any_base_model.infer(
                    obs, np.zeros((1, k, any_base_model.latent_dim), dtype=np.float32)
                )
        return
    for k in sorted({min(2, horizon), horizon}):
        noise = np.zeros((1, k, any_base_model.latent_dim), dtype=np.float32)
        out = any_base_model.infer(obs, noise)
        assert out["actions"].shape == (
            1,
            any_base_model.action_horizon,
            any_base_model.action_dim,
        ), f"K={k}"
    # Contract: K > H is an unsupported shape and must raise ValueError.
    with pytest.raises(ValueError):
        any_base_model.infer(
            obs,
            np.zeros((1, horizon + 1, any_base_model.latent_dim), dtype=np.float32),
        )


def test_properties_and_capability_flag(any_base_model):
    """Positive integer geometry, a steering window no shorter than the decoded chunk, and a boolean
    capability flag that is load-bearing: False means every non-None latent is refused."""
    assert (
        isinstance(any_base_model.action_horizon, int)
        and any_base_model.action_horizon > 0
    )
    assert isinstance(any_base_model.action_dim, int) and any_base_model.action_dim > 0
    assert isinstance(any_base_model.latent_dim, int) and any_base_model.latent_dim > 0
    # latent_dim is independent of action_dim: equal for gr00t (35/35), but
    # pi0 steers its padded flow space (32) over 7-dim decoded actions.
    assert any_base_model.latent_horizon >= any_base_model.action_horizon
    assert isinstance(any_base_model.supports_steering, bool)
    if not any_base_model.supports_steering:
        with pytest.raises(NotImplementedError):
            any_base_model.infer(
                _contract_obs(any_base_model),
                np.zeros((1, any_base_model.latent_dim), dtype=np.float32),
            )


def test_injection_clip(any_base_model):
    """Extreme latents are clipped at injection: the decode stays finite, and
    for the mock the steered residual sits exactly at +/-DEFAULT_LATENT_CLIP."""
    noise = np.full((1, any_base_model.latent_dim), 1e3, dtype=np.float32)
    actions = _steered_or_refused(any_base_model, _contract_obs(any_base_model), noise)
    if actions is None:
        return  # native-only base model: nothing is ever injected
    assert np.isfinite(actions).all()
    if isinstance(any_base_model, MockBaseModelPolicy):
        residual = actions[0] - _position_code(any_base_model)
        np.testing.assert_allclose(
            residual[:, : any_base_model.latent_dim], DEFAULT_LATENT_CLIP, atol=1e-3
        )


def test_actions_finite(any_base_model):
    obs = _contract_obs(any_base_model)
    rng = np.random.default_rng(7)
    native = any_base_model.infer(obs, None)["actions"]
    assert np.isfinite(native).all()
    assert np.issubdtype(native.dtype, np.floating)
    for noise in (
        rng.standard_normal((1, any_base_model.latent_dim)).astype(np.float32),
        rng.standard_normal(
            (1, min(2, any_base_model.action_horizon), any_base_model.latent_dim)
        ).astype(np.float32),
    ):
        actions = _steered_or_refused(any_base_model, obs, noise)
        if actions is None:
            continue  # native-only base model: refusal asserted instead
        assert np.isfinite(actions).all()
        assert np.issubdtype(actions.dtype, np.floating)


# ---------------------------------------------------------------------------
# Interface signature pins (type-hinted, keyword-stable API)
# ---------------------------------------------------------------------------


def test_infer_signature_pinned():
    params = list(inspect.signature(BaseModelPolicy.infer).parameters)
    assert params == ["self", "observations", "noise"]
    # The ABC documents its types (public-library requirement).
    assert BaseModelPolicy.infer.__annotations__, (
        "BaseModelPolicy.infer must be type-annotated"
    )
    # Concrete mock keeps the keyword API with native mode as the default.
    mock_params = inspect.signature(MockBaseModelPolicy.infer).parameters
    assert list(mock_params) == ["self", "observations", "noise"]
    assert mock_params["noise"].default is None
