# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan §3 file 8 — GR00T base model tests (no HF downloads, no real model).

Layers, from most to least local:

1. ``build_base_noise(latent, horizon, action_dim, config)`` — the pure
   flow-init-noise builder. All steering knobs are fields of the frozen
   ``LatentInjectionConfig`` dataclass (``latent_clip=3.0``,
   ``latent_scale=1.0``, ``additive=False``, ``full_horizon_tile=True``,
   ``fresh_base=True``, ``tile_dims=False``); the legacy ``DSRL_LATENT_*``
   env vars are gone by design and must be ignored. Runs without the
   external ``gr00t`` package.
2. Class structure — constructor pins for the config-driven statistics
   source (``multitask_dirs``) and injection knobs, plus a static source
   scan proving no env-var knob reads survive. Runs without ``gr00t``.
3. Decode / stats-registration glue with a mock processor and stubbed
   ``gr00t.*`` leaf modules — per-dimension decode only, single-task vs
   combined multitask stats selection. Runs without ``gr00t``.
4. Cluster-only constructor-branch checks on a real checkpoint (env-var
   gated, Wave 3).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GR00T_PKG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "policies" / "gr00t"

from isaac_rl_posttraining.policies.gr00t import (
    Gr00tBaseModelPolicy,
    Gr00tModel,
    Gr00tPolicy,
    LatentInjectionConfig,
    SteeredGr00tWrapper,
    build_base_noise,
)
from isaac_rl_posttraining.policies.gr00t.data import (
    _combined_stats as combined_stats_fn,
)
from isaac_rl_posttraining.policies.gr00t.data import (
    build_inputs as build_inputs_fn,
)
from isaac_rl_posttraining.policies.gr00t.data import (
    modality_json_from_processor,
    register_baked_embodiment,
    register_embodiment,
)

H, D, REAL = 8, 6, 4  # flow horizon, flow action width, steered (real) dims
KNOB_DEFAULTS = {
    "latent_clip": 3.0,
    "latent_scale": 1.0,
    "additive": False,
    "full_horizon_tile": True,
    "fresh_base": True,
    "tile_dims": False,
}
LEGACY_ENV_KNOBS = {
    "DSRL_LATENT_FULL_HORIZON": "0",
    "DSRL_LATENT_FRESH_BASE": "0",
    "DSRL_LATENT_CLIP": "99.0",
    "DSRL_LATENT_SCALE": "17.0",
    "DSRL_LATENT_ADDITIVE": "1",
    "DSRL_LATENT_TILE_DIMS": "1",
    "GR00T_MULTITASK_DIRS": "/nonexistent/multitask",
    "GR00T_TASK_DIR": "/nonexistent/task",
    "GR00T_ACTION_HORIZON": "13",
}


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _noise(latent, **knobs) -> torch.Tensor:
    config = LatentInjectionConfig(**knobs) if knobs else None
    out = build_base_noise(
        torch.as_tensor(latent, dtype=torch.float32),
        horizon=H,
        action_dim=D,
        config=config,
    )
    return torch.as_tensor(out)


# ---------------------------------------------------------------------------
# 1. Pure noise builder
# ---------------------------------------------------------------------------


def test_noise_builder_signature_and_defaults():
    """Pure function + frozen knob config with the pinned defaults, annotated."""
    params = inspect.signature(build_base_noise).parameters
    assert list(params) == ["latent", "horizon", "action_dim", "config"]
    assert params["config"].default is None

    assert dataclasses.is_dataclass(LatentInjectionConfig)
    fields = {f.name: f for f in dataclasses.fields(LatentInjectionConfig)}
    assert set(fields) == set(KNOB_DEFAULTS), (
        f"knob fields must be exactly {sorted(KNOB_DEFAULTS)}, got {sorted(fields)}"
    )
    for name, default in KNOB_DEFAULTS.items():
        assert fields[name].default == default, (
            f"knob {name} default must be {default!r}, got {fields[name].default!r}"
        )
    # Frozen: knobs are configuration, not mutable state.
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataclasses.replace(LatentInjectionConfig()).__setattr__("latent_clip", 1.0)


def test_noise_single_broadcast():
    """[B, real] steers every horizon step with the same latent (replace mode)."""
    latent = torch.full((2, REAL), 0.5)
    base = _noise(latent)
    assert base.shape == (2, H, D)
    torch.testing.assert_close(base[:, :, :REAL], torch.full((2, H, REAL), 0.5))


def test_noise_tile_when_k_lt_h():
    """[B, K, real] with K < H tiles the per-step latents across the horizon."""
    k = 3
    latent = torch.arange(2 * k * REAL, dtype=torch.float32).reshape(2, k, REAL) / 100.0
    base = _noise(latent, full_horizon_tile=True)
    for h in range(H):
        torch.testing.assert_close(base[:, h, :REAL], latent[:, h % k, :])


def test_noise_direct_when_k_eq_h():
    """K == H (the chunk-40 path: chunk_size == action_horizon) maps 1:1."""
    latent = torch.randn(2, H, REAL).clamp(-2, 2)
    base = _noise(latent)
    torch.testing.assert_close(base[:, :, :REAL], latent)


def test_full_horizon_tile_disabled_first_k_only():
    """full_horizon_tile=False reverts to steering only the first K steps."""
    k = 3
    latent = torch.zeros(1, k, REAL)
    base = _noise(latent, full_horizon_tile=False, fresh_base=True)
    torch.testing.assert_close(base[:, :k, :REAL], torch.zeros(1, k, REAL))
    assert float(base[:, k:, :REAL].abs().max()) > 1e-3, (
        "steps beyond K must carry fresh noise when tiling is disabled"
    )


def test_additive_mode_perturbs_not_replaces():
    """additive=True adds the latent to the native init-noise; latent=0 then
    recovers the native base exactly (the BC-competent RL starting point)."""
    zero = torch.zeros(1, H, REAL)
    native = _noise(zero, additive=True, fresh_base=False)
    replaced = _noise(zero, additive=False, fresh_base=False)
    # replace-with-zeros zeroes the steered block; additive keeps native noise there
    torch.testing.assert_close(replaced[:, :, :REAL], torch.zeros(1, H, REAL))
    assert float(native[:, :, :REAL].abs().max()) > 1e-3
    # and the additive shift is exact: base(l) == base(0) + l on the steered block
    latent = torch.randn(1, H, REAL).clamp(-2, 2)
    shifted = _noise(latent, additive=True, fresh_base=False)
    torch.testing.assert_close(
        shifted[:, :, :REAL], native[:, :, :REAL] + latent, atol=1e-5, rtol=0
    )
    # padding dims are untouched by the latent in either mode
    torch.testing.assert_close(shifted[:, :, REAL:], native[:, :, REAL:])


def test_fresh_base_resampled_each_call():
    latent = torch.full((1, REAL), 0.25)
    first = _noise(latent, fresh_base=True)
    second = _noise(latent, fresh_base=True)
    assert not torch.allclose(first[:, :, REAL:], second[:, :, REAL:]), (
        "padding dims must be freshly sampled on every call"
    )
    # steered block is unaffected by the resampling
    torch.testing.assert_close(first[:, :, :REAL], second[:, :, :REAL])


def test_seeded_base_when_fresh_disabled():
    """fresh_base=False gives a deterministic decode: identical noise each call."""
    latent = torch.full((2, REAL), 0.25)
    first = _noise(latent, fresh_base=False)
    second = _noise(latent, fresh_base=False)
    torch.testing.assert_close(first, second)
    # the deterministic base is still noise-shaped, not zeroed
    padding = first[:, :, REAL:]
    assert float(padding.abs().max()) > 1e-3
    assert 0.3 < float(padding.std()) < 1.7


def test_clip_applied_pm3():
    base = _noise(torch.full((1, REAL), 10.0))
    torch.testing.assert_close(base[:, :, :REAL], torch.full((1, H, REAL), 3.0))
    base = _noise(torch.full((1, REAL), -10.0))
    torch.testing.assert_close(base[:, :, :REAL], torch.full((1, H, REAL), -3.0))
    # knob is honored
    base = _noise(torch.full((1, REAL), 10.0), latent_clip=1.5)
    torch.testing.assert_close(base[:, :, :REAL], torch.full((1, H, REAL), 1.5))


def test_scale_applied_after_clip():
    """latent_scale shrinks the (already clipped) latent: clip THEN scale."""
    base = _noise(torch.full((1, REAL), 2.0), latent_scale=0.5)
    torch.testing.assert_close(base[:, :, :REAL], torch.full((1, H, REAL), 1.0))
    base = _noise(torch.full((1, REAL), 10.0), latent_clip=3.0, latent_scale=0.5)
    torch.testing.assert_close(base[:, :, :REAL], torch.full((1, H, REAL), 1.5))


def test_tile_dims_knob_covers_full_width():
    """tile_dims=True repeats the steered block across all D dims: no entry of
    the [B, H, D] init-noise is left fresh, so the output is a pure function
    of the latent."""
    latent = torch.randn(2, H, REAL).clamp(-2, 2)
    base = _noise(latent, tile_dims=True)
    assert base.shape == (2, H, D)
    for dim in range(D):
        torch.testing.assert_close(base[:, :, dim], latent[:, :, dim % REAL])
    # fully steered => deterministic even with fresh_base=True
    again = _noise(latent, tile_dims=True)
    torch.testing.assert_close(base, again)


def test_padded_dims_remain_gaussian():
    """tile_dims=False leaves dims [real:D] as fresh N(0, 1) draws."""
    latent = torch.zeros(64, REAL)
    base = _noise(latent, tile_dims=False, fresh_base=True)
    padding = base[:, :, REAL:].reshape(-1)
    assert padding.numel() == 64 * H * (D - REAL)
    assert abs(float(padding.mean())) < 0.15
    assert 0.85 < float(padding.std()) < 1.15
    # never constant / zeroed
    assert float(padding.abs().max()) > 1e-2


def test_prepared_initial_noise_decodes_without_advancing_rng():
    class _BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(
                action_horizon=H,
                max_action_dim=D,
                backbone_embedding_dim=4,
            )

        def get_action(self, inputs):
            return {
                "action_pred": torch.randn(
                    inputs["batch"], H, D, device=self.anchor.device
                )
            }

    model = Gr00tModel(_BaseModel())
    latent = torch.full((2, H, REAL), 0.25)
    prepared = model.prepare_initial_noise(latent)
    rng_state = torch.get_rng_state().clone()

    decoded = model.get_action_with_initial_noise({"batch": 2}, prepared)

    torch.testing.assert_close(decoded, prepared)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_concurrent_noise_overrides_do_not_interleave():
    """Two threads decoding at once each get their own noise, and torch.randn is restored."""
    import threading
    import time

    class _SlowBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(
                action_horizon=H, max_action_dim=D, backbone_embedding_dim=4
            )

        def get_action(self, inputs):
            time.sleep(
                0.05
            )  # widen the window in which the other thread could patch randn
            return {"action_pred": torch.randn(1, H, D)}

    original_randn = torch.randn
    outputs: dict[float, torch.Tensor] = {}

    def decode(value: float) -> None:
        model = Gr00tModel(_SlowBaseModel())
        outputs[value] = model.get_action_with_initial_noise(
            {}, torch.full((1, H, D), value)
        )

    threads = [threading.Thread(target=decode, args=(v,)) for v in (1.0, 2.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert torch.randn is original_randn
    for value, decoded in outputs.items():
        torch.testing.assert_close(decoded, torch.full((1, H, D), value))


def test_read_task_missing_file_names_path(tmp_path):
    """A missing tasks.jsonl is an error naming the path, not a silent default instruction."""
    from isaac_rl_posttraining.policies.gr00t.data import read_task

    with pytest.raises(FileNotFoundError, match=str(tmp_path / "tasks.jsonl")):
        read_task(str(tmp_path))


def test_env_vars_ignored(monkeypatch):
    """Regression (leak-bug class): the removed env-var knobs must not change
    behavior even when set. Knobs are config fields now, by design."""
    latent = torch.full((1, REAL), 10.0)
    clean = _noise(latent, fresh_base=False)
    for key, value in LEGACY_ENV_KNOBS.items():
        monkeypatch.setenv(key, value)
    dirty = _noise(latent, fresh_base=False)
    torch.testing.assert_close(clean, dirty)
    # clip stayed 3.0 (not 99), mode stayed replace (not additive)
    torch.testing.assert_close(dirty[:, :, :REAL], torch.full((1, H, REAL), 3.0))


# ---------------------------------------------------------------------------
# 2. Class structure (importable and pinned without gr00t installed)
# ---------------------------------------------------------------------------


def test_base_model_constructor_config_driven():
    """Stats selection and steering knobs are constructor fields: multitask_dirs
    (None => single-task stats from dataset_dir) plus the LatentInjectionConfig
    knob bundle. No env-var fallbacks."""
    params = inspect.signature(SteeredGr00tWrapper.__init__).parameters
    for name in (
        "base_ckpt",
        "backbone_model_name_or_path",
        "bc_ckpt",
        "dataset_dir",
        "multitask_dirs",
        "latent_injection",
    ):
        assert name in params, f"SteeredGr00tWrapper must take {name}"
    assert params["multitask_dirs"].default is None
    assert params["bc_ckpt"].default is None
    assert params["dataset_dir"].default is None
    assert params["latent_injection"].default is None  # None -> LatentInjectionConfig()
    # Embodiment selection and the baked-path task language are constructor fields too.
    assert params["embodiment_tag"].default == "new_embodiment"
    assert params["task_instruction"].default is None
    # The base model adapts the wrapper and offers checkpoint construction.
    assert inspect.ismethod(Gr00tBaseModelPolicy.from_checkpoint), (
        "Gr00tBaseModelPolicy.from_checkpoint must be a classmethod"
    )
    ctor = inspect.signature(Gr00tBaseModelPolicy.from_checkpoint).parameters
    assert "base_ckpt" in ctor


# ---------------------------------------------------------------------------
# 3. Decode / stats glue with mocked gr00t leaf modules (local)
# ---------------------------------------------------------------------------

MODALITY_JSON = {
    "state": {
        "left_arm": {"start": 0, "end": 2},
        "waist": {"start": 2, "end": 4},
        "left_leg": {"start": 4, "end": 6},  # declared but not policy-predicted
    },
    "action": {
        "left_arm": {"start": 0, "end": 2},
        "waist": {"start": 2, "end": 4},
        "left_leg": {"start": 4, "end": 6},
        "navigate_command": {"start": 6, "end": 9},
    },
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def _stub_gr00t_modules(monkeypatch):
    """Install minimal gr00t.* leaf modules so lazy imports resolve locally."""
    root = types.ModuleType("gr00t")
    root.__path__ = []  # package-like
    data = types.ModuleType("gr00t.data")
    data.__path__ = []
    tags = types.ModuleType("gr00t.data.embodiment_tags")

    class EmbodimentTag(str):
        def __new__(cls, value):
            return str.__new__(cls, value)

    tags.EmbodimentTag = EmbodimentTag
    dtypes = types.ModuleType("gr00t.data.types")

    class ModalityConfig:
        def __init__(self, delta_indices, modality_keys):
            self.delta_indices = list(delta_indices)
            self.modality_keys = list(modality_keys)

    dtypes.ModalityConfig = ModalityConfig
    data.embodiment_tags = tags
    data.types = dtypes
    root.data = data
    for name, module in {
        "gr00t": root,
        "gr00t.data": data,
        "gr00t.data.embodiment_tags": tags,
        "gr00t.data.types": dtypes,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


class _GroupCfg:
    def __init__(self, modality_keys):
        self.modality_keys = list(modality_keys)


class _PerDimDecodeProcessor:
    """Mock processor whose decode is strictly per-dimension affine per group."""

    def __init__(self, tag: str):
        self.scale = np.arange(2.0, 2.0 + 9.0)  # dims 0..8 -> scales 2..10
        self.shift = np.arange(0.0, 9.0) / 10.0
        action_groups = ["left_arm", "waist", "navigate_command"]  # trained order
        state_groups = ["left_arm", "waist"]
        self.modality_configs = {
            tag: {
                "action": _GroupCfg(action_groups),
                "state": _GroupCfg(state_groups),
                "video": _GroupCfg(["ego_view"]),
                "language": _GroupCfg(["annotation.human.task_description"]),
            }
        }
        self.seen_states = None

    def decode_action(self, action, embodiment_tag, states):
        self.seen_states = states
        # groups span the RAW width in trained order: left_arm 0:2, waist 2:4,
        # navigate 4:7 of the concatenated raw action (per-dim ops ONLY).
        spans = {"left_arm": (0, 2), "waist": (2, 4), "navigate_command": (4, 7)}
        out = {}
        for group, (start, end) in spans.items():
            out[group] = (
                action[..., start:end] * self.scale[start:end] + self.shift[start:end]
            )
        return out


def test_decode_has_no_cross_dim_mixing(monkeypatch):
    """The decode glue is slice + per-dim un-normalize + concat: probing raw
    basis vectors through a per-dim mock processor yields a diagonal response
    (output dim j depends on input dim j only, in trained group order)."""
    _stub_gr00t_modules(monkeypatch)
    tag = "new_embodiment"
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.processor = _PerDimDecodeProcessor(tag)
    policy.embodiment_tag = tag
    # decode consumes only the arm/waist/navigate groups (7 raw dims)
    policy.modality_json = {
        "state": {"left_arm": {"start": 0, "end": 2}, "waist": {"start": 2, "end": 4}},
        "action": {
            "left_arm": {"start": 0, "end": 2},
            "waist": {"start": 2, "end": 4},
            "navigate_command": {"start": 4, "end": 7},
        },
    }
    decode = policy._unapply  # per-dim decode hook: slice + un-normalize + concat

    a_raw = 7
    state = np.linspace(0.0, 1.0, 4).astype(np.float32)
    horizon = 3
    baseline = decode(np.zeros((horizon, a_raw), dtype=np.float32), state)
    assert baseline.shape == (horizon, a_raw)
    for j in range(a_raw):
        probe = np.zeros((horizon, a_raw), dtype=np.float32)
        probe[:, j] = 1.0
        response = decode(probe, state) - baseline
        support = np.nonzero(np.abs(response).max(axis=0) > 1e-9)[0]
        np.testing.assert_array_equal(
            support,
            [j],
            err_msg=f"raw action dim {j} leaked into dims {support.tolist()}",
        )
        # per-dim scaling preserved exactly (no re-mixing after the head)
        assert response[:, j] == pytest.approx(policy.processor.scale[j])
    # relative->absolute reference state was routed per group
    seen = policy.processor.seen_states
    assert seen is not None and set(seen) == {"left_arm", "waist"}
    np.testing.assert_allclose(np.asarray(seen["left_arm"]).reshape(-1), state[0:2])
    np.testing.assert_allclose(np.asarray(seen["waist"]).reshape(-1), state[2:4])


class _RecordingProcessor:
    def __init__(self):
        self.modality_configs = {}
        self.stats = None
        self.set_statistics_calls = 0

    def set_statistics(self, stats, override=False):
        self.stats = stats
        self.set_statistics_calls += 1


def _write_dataset(tmp_path: Path, name: str, action_base: float) -> Path:
    dataset = tmp_path / name
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "modality.json").write_text(json.dumps(MODALITY_JSON))
    action_dim, state_dim = 9, 6
    stats = {
        "observation.state": {
            key: [action_base + i / 10.0 for i in range(state_dim)]
            for key in ("mean", "std", "min", "max", "q01", "q99")
        },
        "action": {
            key: [action_base + i for i in range(action_dim)]
            for key in ("mean", "std", "min", "max", "q01", "q99")
        },
    }
    (meta / "stats.json").write_text(json.dumps(stats))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task": f"do the {name} task"}) + "\n"
    )
    return dataset


def test_multitask_dirs_unset_selects_single_task_stats(monkeypatch, tmp_path):
    """Single-task path (multitask_dirs=None): the dataset's own stats are
    registered, sliced to the G1 decoupled groups the adapter supplies,
    legs/eef groups excluded."""
    from simulators.g1_arena.embodiment import (
        G1_DECOUPLED_ACTION_GROUPS,
        G1_DECOUPLED_STATE_GROUPS,
    )

    _stub_gr00t_modules(monkeypatch)
    dataset = _write_dataset(tmp_path, "drill_lift_rl", action_base=1.0)
    processor = _RecordingProcessor()
    tag = "new_embodiment"
    register_embodiment(
        processor,
        str(dataset),
        tag,
        action_horizon=8,
        state_groups=G1_DECOUPLED_STATE_GROUPS,
        action_groups=G1_DECOUPLED_ACTION_GROUPS,
    )

    assert processor.set_statistics_calls == 1
    action_cfg = processor.modality_configs[tag]["action"]
    assert list(action_cfg.modality_keys) == [
        "left_arm",
        "waist",
        "navigate_command",
    ], "policy-predicted groups only, in trained order — never the dataset's legs"
    action_stats = processor.stats[tag]["action"]
    assert "left_leg" not in action_stats
    np.testing.assert_allclose(action_stats["left_arm"]["mean"], [1.0, 2.0])
    np.testing.assert_allclose(action_stats["waist"]["mean"], [3.0, 4.0])
    np.testing.assert_allclose(
        action_stats["navigate_command"]["mean"], [7.0, 8.0, 9.0]
    )
    state_stats = processor.stats[tag]["state"]
    assert "left_leg" not in state_stats
    np.testing.assert_allclose(state_stats["left_arm"]["mean"], [1.0, 1.1])


def test_multitask_dirs_set_selects_combined_stats(monkeypatch):
    """Multitask path: ONE set of stats over the union of all tasks' data —
    not any single task's stats."""
    _stub_gr00t_modules(monkeypatch)
    tag = "new_embodiment"
    state_dim, action_dim = 6, 9
    task1_states = np.full((50, state_dim), 1.0, dtype=np.float32)
    task1_actions = np.full((50, action_dim), 1.0, dtype=np.float32)
    task2_states = np.full((50, state_dim), 3.0, dtype=np.float32)
    task2_actions = np.full((50, action_dim), 3.0, dtype=np.float32)

    stats = combined_stats_fn(
        np.concatenate([task1_states, task2_states]),
        np.concatenate([task1_actions, task2_actions]),
        MODALITY_JSON,
        tag,
    )
    arm = stats[tag]["action"]["left_arm"]
    np.testing.assert_allclose(arm["mean"], [2.0, 2.0])  # union mean, not 1.0 or 3.0
    np.testing.assert_allclose(arm["min"], [1.0, 1.0])
    np.testing.assert_allclose(arm["max"], [3.0, 3.0])
    np.testing.assert_allclose(stats[tag]["state"]["waist"]["mean"], [2.0, 2.0])


# ---------------------------------------------------------------------------
# 3b. Baked-embodiment registration (datasetless, checkpoint-self-describing)
# ---------------------------------------------------------------------------

#: The libero_sim layout of the released GR00T-N1.7-LIBERO checkpoints:
#: seven 1-dim groups plus a 2-dim gripper state.
_LIBERO_STATE_GROUPS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
_LIBERO_ACTION_GROUPS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class _BakedProcessor:
    """Mock processor with a baked embodiment (modality configs + statistics)."""

    def __init__(self, tag: str):
        def _stats(width_by_group):
            return {
                group: {"mean": [0.0] * width, "std": [1.0] * width}
                for group, width in width_by_group.items()
            }

        self.modality_configs = {
            tag: {
                "video": _GroupCfg(["image", "wrist_image"]),
                "state": _GroupCfg(_LIBERO_STATE_GROUPS),
                "action": _GroupCfg(_LIBERO_ACTION_GROUPS),
                "language": _GroupCfg(["annotation.human.action.task_description"]),
            }
        }
        state_widths = {g: (2 if g == "gripper" else 1) for g in _LIBERO_STATE_GROUPS}
        action_widths = dict.fromkeys(_LIBERO_ACTION_GROUPS, 1)
        self.state_action_processor = types.SimpleNamespace(
            statistics={
                tag: {"state": _stats(state_widths), "action": _stats(action_widths)}
            },
            modality_configs={},
        )
        self.set_statistics_calls = 0

    def set_statistics(self, stats, override=False):
        self.set_statistics_calls += 1


def test_modality_json_derived_from_baked_config():
    """The flat layout comes from the baked config: group order from the
    modality keys, widths from the statistics vectors, contiguous spans."""
    tag = "libero_sim"
    layout = modality_json_from_processor(_BakedProcessor(tag), tag)
    assert list(layout["state"]) == _LIBERO_STATE_GROUPS
    assert layout["state"]["x"] == {"start": 0, "end": 1}
    assert layout["state"]["gripper"] == {"start": 6, "end": 8}  # 2-dim group
    assert list(layout["action"]) == _LIBERO_ACTION_GROUPS
    assert layout["action"]["gripper"] == {"start": 6, "end": 7}
    total_state = max(s["end"] for s in layout["state"].values())
    total_action = max(s["end"] for s in layout["action"].values())
    assert (total_state, total_action) == (8, 7)


def test_register_baked_embodiment_mirrors_config_and_keeps_stats():
    """Registration mirrors the baked config onto the sub-processors and
    NEVER rebuilds statistics (the checkpoint's train-time stats are
    authoritative)."""
    tag = "libero_sim"
    processor = _BakedProcessor(tag)
    layout = register_baked_embodiment(processor, tag)
    assert layout == modality_json_from_processor(processor, tag)
    assert tag in processor.state_action_processor.modality_configs, (
        "the baked config must be mirrored onto the state/action sub-processor"
    )
    assert processor.set_statistics_calls == 0, (
        "baked statistics must be reused verbatim, not re-registered"
    )


def test_baked_registration_unknown_tag_raises_with_available_tags():
    processor = _BakedProcessor("libero_sim")
    with pytest.raises(RuntimeError, match="libero_sim"):
        modality_json_from_processor(processor, "new_embodiment")


def test_policy_register_baked_embodiment_sets_layout(monkeypatch):
    """The policy-level hook derives modality_json and stores the task."""
    _stub_gr00t_modules(monkeypatch)
    tag = "libero_sim"
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    object.__setattr__(policy, "embodiment_tag", tag)
    object.__setattr__(policy, "processor", _BakedProcessor(tag))
    object.__setattr__(policy, "modality_json", None)
    object.__setattr__(policy, "task", None)
    Gr00tPolicy.register_baked_embodiment(policy, task="put the mug on the plate")
    assert policy.modality_json["action"]["gripper"] == {"start": 6, "end": 7}
    assert policy.task == "put the mug on the plate"
    # No processor: the precondition surfaces as RuntimeError.
    bare = Gr00tPolicy.__new__(Gr00tPolicy)
    object.__setattr__(bare, "embodiment_tag", tag)
    object.__setattr__(bare, "processor", None)
    with pytest.raises(RuntimeError):
        Gr00tPolicy.register_baked_embodiment(bare)


# ---------------------------------------------------------------------------
# 3c. Multi-camera input building and per-observation prompts
# ---------------------------------------------------------------------------


class _RecordingObservationProcessor:
    """Mock processor recording what process_observation receives."""

    def __init__(self, tag: str, video_keys):
        self.modality_configs = {
            tag: {
                "video": _GroupCfg(list(video_keys)),
                "state": _GroupCfg(["x", "gripper"]),
                "language": _GroupCfg(["annotation.human.action.task_description"]),
            }
        }
        self.seen = None

    def process_observation(self, observation, embodiment):
        self.seen = observation
        return {"processed": True}


_TWO_CAM_MODALITY_JSON = {
    "state": {"x": {"start": 0, "end": 1}, "gripper": {"start": 1, "end": 3}},
}


def test_build_inputs_routes_per_view_frames(monkeypatch):
    """'video.<key>' entries feed their views; frames are not cross-wired."""
    _stub_gr00t_modules(monkeypatch)
    tag = "libero_sim"
    processor = _RecordingObservationProcessor(tag, ["image", "wrist_image"])
    scene = np.full((1, 4, 4, 3), 10, dtype=np.uint8)
    wrist = np.full((1, 4, 4, 3), 200, dtype=np.uint8)
    build_inputs_fn(
        processor,
        {
            "video.image": scene,
            "video.wrist_image": wrist,
            "state": np.zeros((1, 3), np.float32),
        },
        tag,
        _TWO_CAM_MODALITY_JSON,
        "stack the bowls",
        device="cpu",
    )
    seen = processor.seen
    assert seen["video.image"].shape == (1, 1, 4, 4, 3)  # [B, T, H, W, C]
    np.testing.assert_array_equal(seen["video.image"][0, 0], scene[0])
    np.testing.assert_array_equal(seen["video.wrist_image"][0, 0], wrist[0])
    assert seen["annotation.human.action.task_description"] == ["stack the bowls"]
    np.testing.assert_allclose(np.asarray(seen["state.x"]).reshape(-1), [0.0])
    assert np.asarray(seen["state.gripper"]).shape == (1, 1, 2)


def test_build_inputs_shared_frame_covers_missing_views(monkeypatch):
    """A shared ego frame backfills every view without a per-view entry."""
    _stub_gr00t_modules(monkeypatch)
    tag = "libero_sim"
    processor = _RecordingObservationProcessor(tag, ["image", "wrist_image"])
    shared = np.full((1, 4, 4, 3), 42, dtype=np.uint8)
    build_inputs_fn(
        processor,
        {"ego_view_image": shared, "state": np.zeros((1, 3), np.float32)},
        tag,
        _TWO_CAM_MODALITY_JSON,
        "task",
        device="cpu",
    )
    np.testing.assert_array_equal(processor.seen["video.image"][0, 0], shared[0])
    np.testing.assert_array_equal(processor.seen["video.wrist_image"][0, 0], shared[0])


def test_build_inputs_missing_view_raises(monkeypatch):
    """No per-view entry and no shared frame is a hard error, not silence."""
    _stub_gr00t_modules(monkeypatch)
    tag = "libero_sim"
    processor = _RecordingObservationProcessor(tag, ["image", "wrist_image"])
    with pytest.raises(KeyError, match="wrist_image"):
        build_inputs_fn(
            processor,
            {
                "video.image": np.zeros((1, 4, 4, 3), np.uint8),
                "state": np.zeros((1, 3), np.float32),
            },
            tag,
            _TWO_CAM_MODALITY_JSON,
            "task",
            device="cpu",
        )


def test_obs_prompt_overrides_registered_task():
    """A prompt riding in the observation wins over the registered default;
    non-string prompts (e.g. embedded language vectors) are ignored."""
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    object.__setattr__(policy, "task", "registered default")
    assert (
        Gr00tPolicy._obs_task(policy, {"prompt": "open the drawer"})
        == "open the drawer"
    )
    assert (
        Gr00tPolicy._obs_task(policy, {"prompt": ("open the drawer",)})
        == "open the drawer"
    )
    assert Gr00tPolicy._obs_task(policy, {}) == "registered default"
    assert (
        Gr00tPolicy._obs_task(policy, {"prompt": np.zeros(3, np.float32)})
        == "registered default"
    )


def test_decoded_action_horizon_reads_registered_window():
    """The decoded horizon follows the processor's registered action window.

    Regression: the property must read ``embodiment_tag`` (the attribute the
    policy actually stores) and must never raise ``AttributeError`` from
    inside an ``nn.Module`` property, where it would be masked as a missing
    attribute at wrapper construction time.
    """
    from isaac_rl_posttraining.policies.gr00t import policy as policy_mod

    policy = policy_mod.Gr00tPolicy.__new__(policy_mod.Gr00tPolicy)
    object.__setattr__(policy, "embodiment_tag", "new_embodiment")

    class _ActionCfg:
        delta_indices = list(range(16))  # noqa: RUF012

    class _Processor:
        modality_configs = {"new_embodiment": {"action": _ActionCfg()}}  # noqa: RUF012

    object.__setattr__(policy, "processor", _Processor())
    assert policy_mod.Gr00tPolicy.decoded_action_horizon.fget(policy) == 16


def test_decoded_action_horizon_unregistered_raises_runtime_error():
    """Missing registration surfaces as RuntimeError, never AttributeError."""
    from isaac_rl_posttraining.policies.gr00t import policy as policy_mod

    policy = policy_mod.Gr00tPolicy.__new__(policy_mod.Gr00tPolicy)
    object.__setattr__(policy, "embodiment_tag", "new_embodiment")
    object.__setattr__(policy, "processor", None)
    with pytest.raises(RuntimeError):
        policy_mod.Gr00tPolicy.decoded_action_horizon.fget(policy)

    class _Processor:
        modality_configs = {}  # noqa: RUF012

    object.__setattr__(policy, "processor", _Processor())
    with pytest.raises(RuntimeError):
        policy_mod.Gr00tPolicy.decoded_action_horizon.fget(policy)


# ---------------------------------------------------------------------------
# input_spec: the load-time state-width check
# ---------------------------------------------------------------------------


class _LayoutPolicy:
    """Stands in for a loaded policy, carrying only what input_spec reads."""

    def __init__(self, modality_json=None):
        if modality_json is not None:
            self.modality_json = modality_json


def _wrapper_with(policy):
    """A minimal object exposing the real input_spec over a stub policy."""
    return type(
        "_W", (), {"policy": policy, "input_spec": SteeredGr00tWrapper.input_spec}
    )()


def test_input_spec_state_width_spans_the_layout():
    spec = _wrapper_with(_LayoutPolicy(MODALITY_JSON)).input_spec()
    assert spec is not None
    assert spec.arrays["state"].shape == (6,)  # the widest declared state group's end


def test_input_spec_is_unknown_without_a_flat_layout():
    """GR00T N1.5 keeps modality keys and delta indices, not start/end offsets,
    so its policy has no ``modality_json`` at all. The width is then unknown --
    reading the attribute unguarded used to abort every N1.5 steering run."""
    assert _wrapper_with(_LayoutPolicy()).input_spec() is None
    assert _wrapper_with(_LayoutPolicy({"state": {}})).input_spec() is None
