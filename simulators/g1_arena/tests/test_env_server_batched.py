# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched (--num-envs > 1) protocol contracts for the Arena env server.

The handler is exercised against a fake in-memory Arena environment (no
simulator): payload shapes carry the leading [N] axis, success is per-env
(never a batch-wide any()), MSG_RESET honors env_ids for partial resets,
images are skipped on intermediate chunk steps, and the whole-body action
assembly stays [N, 35] -> [N, 50]. The single-env dispatch keeps the scalar
protocol byte-identical.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.envs.transport import MSG_RESET, MSG_SPEC, MSG_STEP
from simulators.g1_arena.embodiment import (
    G1_WBC_STATE_GROUP_ORDER,
    wbc_base_height_slice,
    wbc_joint_action_slices,
    wbc_navigate_slice,
)
from simulators.g1_arena.env_server import (
    build_handler,
    configure_renderer_antialiasing,
    configure_strict_environment,
    configure_strict_launcher,
    configure_strict_renderer,
    configure_strict_torch,
    install_strict_environment_callback,
    resolve_env_spacing,
    resolve_render_once_prims,
    resolve_renderer_antialiasing,
)
from simulators.g1_arena.task_prompt import resolve_task_prompt
from simulators.g1_arena.timers import StepTimer

ACTION_SLICES = wbc_joint_action_slices()
BASE_HEIGHT_SLICE = wbc_base_height_slice()
NAVIGATE_SLICE = wbc_navigate_slice()

# The five arm/hand/waist joint groups double as the state groups; name the
# joints per group and lay them out at sim indices 0..30 (the remaining sim
# joints stand in for the whole-body-controlled legs).
POLICY_CFG = {
    group: [f"{group}_{i}" for i in range(sl.stop - sl.start)]
    for group, sl in ACTION_SLICES.items()
}
assert set(G1_WBC_STATE_GROUP_ORDER) == set(POLICY_CFG)
SIM_CFG = {
    name: idx
    for idx, name in enumerate(
        name for group in G1_WBC_STATE_GROUP_ORDER for name in POLICY_CFG[group]
    )
}
STATE_DIM = len(SIM_CFG)
SIM_JOINTS = 43
IMAGE_HW = (8, 12)


class _Remapped:
    def __init__(self, joints: torch.Tensor) -> None:
        self._joints = joints

    def get_joints_pos(self) -> torch.Tensor:
        return self._joints


def _fake_remap(named: dict[str, Any], policy_cfg, sim_cfg, device) -> _Remapped:
    """Stand-in for the Arena joint remap: checks batched group shapes."""
    num_envs = next(iter(named.values())).shape[0]
    for group, values in named.items():
        width = len(policy_cfg[group])
        assert values.shape == (num_envs, 1, width), (group, values.shape)
    # Deterministic output so the test can assert the assembled action.
    return _Remapped(torch.full((num_envs, 1, SIM_JOINTS), 7.0))


class _FakeTerminationManager:
    def __init__(self, num_envs: int) -> None:
        self.success = torch.zeros(num_envs, dtype=torch.bool)
        self.active_terms = ("success",)

    def get_term(self, name: str) -> torch.Tensor:
        assert name == "success"
        return self.success


class _FakeUnwrapped:
    def __init__(self, outer: _FakeArenaEnv, num_envs: int) -> None:
        self._outer = outer
        self.num_envs = num_envs
        self.cfg = SimpleNamespace(num_rerenders_on_reset=0)
        self.termination_manager = _FakeTerminationManager(num_envs)
        self.scene = _FakeScene(num_envs)
        self.partial_resets: list[list[int]] = []
        self.partial_reset_seeds: list[int | None] = []
        self.restored_states: list[dict[str, Any]] = []
        self.restore_seeds: list[int | None] = []
        self.restore_rerenders: list[int] = []

    def reset(
        self, env_ids: torch.Tensor, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.partial_resets.append([int(i) for i in env_ids])
        self.partial_reset_seeds.append(seed)
        return self._outer.observation(), {}

    def reset_to(
        self,
        state: dict[str, Any],
        env_ids=None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert env_ids is None
        assert is_relative
        self.restored_states.append(state)
        self.restore_seeds.append(seed)
        self.restore_rerenders.append(self.cfg.num_rerenders_on_reset)
        return self._outer.observation(), {}


class _FakeScene:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def get_state(self, *, is_relative: bool) -> dict[str, Any]:
        assert is_relative
        return {
            "articulation": {
                "robot": {
                    "root_pose": torch.arange(
                        self.num_envs * 7, dtype=torch.float32
                    ).reshape(self.num_envs, 7)
                }
            }
        }


class _FakeArenaEnv:
    """Batch-shaped Arena env double: dict obs, [N] step results."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.unwrapped = _FakeUnwrapped(self, num_envs)
        self.full_resets = 0
        self.full_reset_seeds: list[int | None] = []
        self.full_reset_rerenders: list[int] = []
        self.last_action: torch.Tensor | None = None
        self.reward = torch.arange(num_envs, dtype=torch.float32) + 0.5
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(num_envs, dtype=torch.bool)

    def observation(self) -> dict[str, Any]:
        # Joint positions encode "sim index + 100 * env" so the state
        # reindexing is verifiable per env.
        joint_pos = torch.arange(SIM_JOINTS, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        joint_pos += 100.0 * torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(
            1
        )
        rgb = torch.zeros(self.num_envs, *IMAGE_HW, 3, dtype=torch.uint8)
        return {
            "camera_obs": {"robot_head_cam_rgb": rgb},
            "policy": {"robot_joint_pos": joint_pos},
        }

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.full_resets += 1
        self.full_reset_seeds.append(seed)
        self.full_reset_rerenders.append(self.unwrapped.cfg.num_rerenders_on_reset)
        return self.observation(), {}

    def step(self, action: torch.Tensor) -> tuple[Any, ...]:
        self.last_action = action
        return self.observation(), self.reward, self.terminated, self.truncated, {}


def _make_handler(
    num_envs: int,
    *,
    reset_state_dir: str | None = None,
    reset_observation_dir: str | None = None,
):
    env = _FakeArenaEnv(num_envs)
    args = argparse.Namespace(
        num_envs=num_envs,
        task="LMDrillLift",
        embodiment="g1_factory_wbc_joint",
        camera_key="robot_head_cam",
        image_resolution=0,
        image_shortest_edge=0,
        action_scale=1.0,
        lift_height=0.16,
        reset_state_dir=reset_state_dir,
        reset_observation_dir=reset_observation_dir,
    )
    wbc = {"policy_cfg": POLICY_CFG, "sim_cfg": SIM_CFG, "remap": _fake_remap}
    handler, _step_timer = build_handler(
        env, args, torch.device("cpu"), "Lift the drill.", wbc, np, torch, cv2=None
    )
    return env, handler


def _step_message(num_envs: int, need_obs: bool = True) -> dict[str, Any]:
    actions = np.zeros((num_envs, 35), dtype=np.float32)
    actions[:, NAVIGATE_SLICE] = 0.25
    actions[:, BASE_HEIGHT_SLICE] = 0.75
    return {"type": MSG_STEP, "action": actions.tolist(), "need_obs": need_obs}


def test_step_batched_payload_shapes_and_state_reindex():
    _env, handler = _make_handler(num_envs=3)
    payload = handler(_step_message(3))

    assert payload["ego_view"].shape == (3, *IMAGE_HW, 3)
    assert payload["state"].shape == (3, STATE_DIM)
    # Sim joints were laid out at indices 0..30 in policy order, offset per env.
    expected = (
        np.arange(STATE_DIM, dtype=np.float32)[None, :] + 100.0 * np.arange(3)[:, None]
    )
    np.testing.assert_allclose(payload["state"], expected)
    np.testing.assert_allclose(payload["reward"], [0.5, 1.5, 2.5])
    assert payload["done"].shape == (3,) and payload["truncated"].shape == (3,)
    assert payload["num_steps"].tolist() == [1, 1, 1]
    assert payload["server_s"] >= 0.0  # the server echoes its own step time


def test_step_batched_success_is_per_env():
    env, handler = _make_handler(num_envs=3)
    env.terminated = torch.tensor([False, True, False])
    env.unwrapped.termination_manager.success = torch.tensor([False, True, False])
    payload = handler(_step_message(3))
    assert payload["success"].tolist() == [False, True, False]
    assert payload["done"].tolist() == [False, True, False]


def test_step_batched_assembles_wbc50():
    env, handler = _make_handler(num_envs=2)
    handler(_step_message(2))
    assert env.last_action is not None and env.last_action.shape == (2, 50)
    action = env.last_action.numpy()
    np.testing.assert_allclose(action[:, :SIM_JOINTS], 7.0)  # remapped joints
    np.testing.assert_allclose(action[:, SIM_JOINTS : SIM_JOINTS + 3], 0.25)  # navigate
    np.testing.assert_allclose(action[:, SIM_JOINTS + 3], 0.75)  # base height
    np.testing.assert_allclose(action[:, SIM_JOINTS + 4 :], 0.0)  # torso rpy


def test_step_batched_rejects_chunked_actions():
    _, handler = _make_handler(num_envs=2)
    chunk = np.zeros((2, 4, 35), dtype=np.float32)
    with pytest.raises(ValueError, match="batched action"):
        handler({"type": MSG_STEP, "action": chunk.tolist(), "need_obs": True})


def test_step_batched_skips_images_until_needed():
    env, handler = _make_handler(num_envs=2)
    quiet = handler(_step_message(2, need_obs=False))
    assert "ego_view" not in quiet and quiet["state"].shape == (2, STATE_DIM)
    # An episode end overrides need_obs: the collector consumes that frame.
    env.terminated = torch.tensor([True, False])
    final = handler(_step_message(2, need_obs=False))
    assert final["ego_view"].shape == (2, *IMAGE_HW, 3)


def test_reset_batched_env_ids_partial_reset():
    env, handler = _make_handler(num_envs=4)
    assert handler({"type": MSG_SPEC})["num_envs"] == 4
    payload = handler({"type": MSG_RESET, "seed": 456, "env_ids": [1, 3]})
    assert env.unwrapped.partial_resets == [[1, 3]]
    assert env.unwrapped.partial_reset_seeds == [456]
    assert env.full_resets == 0
    assert payload["state"].shape == (4, STATE_DIM)
    assert payload["ego_view"].shape == (4, *IMAGE_HW, 3)


def test_reset_batched_full():
    env, handler = _make_handler(num_envs=2)
    handler({"type": MSG_RESET, "seed": 123})
    assert env.full_resets == 1
    assert env.full_reset_seeds == [123]
    assert env.unwrapped.partial_resets == []


def test_seeded_reset_state_replays_across_server_processes(tmp_path):
    first_env, first_handler = _make_handler(
        num_envs=2,
        reset_state_dir=str(tmp_path),
    )
    first_handler({"type": MSG_RESET, "seed": 123})

    snapshot = tmp_path / "seed_123_envs_2.pt"
    assert snapshot.is_file()
    assert first_env.full_resets == 2
    assert first_env.full_reset_rerenders == [0, 1]
    assert first_env.unwrapped.restore_seeds == [123]
    assert first_env.unwrapped.restore_rerenders == [1]
    assert first_env.unwrapped.cfg.num_rerenders_on_reset == 0
    first_handler({"type": MSG_RESET, "seed": 123})
    assert first_env.full_resets == 3
    assert first_env.full_reset_seeds == [123, 123, 123]
    assert first_env.unwrapped.restore_seeds == [123, 123]
    assert first_env.unwrapped.restore_rerenders == [1, 1]
    assert first_env.unwrapped.cfg.num_rerenders_on_reset == 0

    retained_env, retained_handler = _make_handler(
        num_envs=2,
        reset_state_dir=str(tmp_path),
    )
    retained_handler({"type": MSG_RESET, "seed": 123})
    assert retained_env.full_resets == 1
    assert retained_env.full_reset_seeds == [123]
    assert retained_env.unwrapped.restore_seeds == [123]
    assert retained_env.unwrapped.restore_rerenders == [1]
    assert retained_env.unwrapped.cfg.num_rerenders_on_reset == 0
    root_pose = retained_env.unwrapped.restored_states[0]["articulation"]["robot"][
        "root_pose"
    ]
    torch.testing.assert_close(
        root_pose,
        torch.arange(14, dtype=torch.float32).reshape(2, 7),
    )


def test_seeded_full_resets_capture_observation_batches(tmp_path):
    capture_dir = tmp_path / "observations"
    env, handler = _make_handler(
        num_envs=2,
        reset_observation_dir=str(capture_dir),
    )

    handler({"type": MSG_RESET, "seed": 123})
    handler({"type": MSG_RESET, "seed": 123})
    handler({"type": MSG_RESET, "seed": 456, "env_ids": [1]})

    captures = sorted(capture_dir.glob("*.npz"))
    assert [path.name for path in captures] == [
        "seed_123_envs_2_reset_000.npz",
        "seed_123_envs_2_reset_001.npz",
    ]
    assert not list(capture_dir.glob("*.tmp"))
    with np.load(captures[0], allow_pickle=False) as payload:
        assert int(payload["schema_version"]) == 1
        assert int(payload["seed"]) == 123
        assert int(payload["num_envs"]) == 2
        assert int(payload["reset_index"]) == 0
        assert payload["ego_view"].shape == (2, *IMAGE_HW, 3)
        assert payload["state"].shape == (2, STATE_DIM)
    assert env.full_resets == 2


def test_single_env_dispatch_keeps_scalar_protocol():
    _env, handler = _make_handler(num_envs=1)
    payload = handler(
        {"type": MSG_STEP, "action": np.zeros(35, dtype=np.float32).tolist()}
    )
    assert isinstance(payload["reward"], float)
    assert isinstance(payload["done"], bool)
    assert payload["ego_view"].ndim == 3  # [H, W, 3], no batch axis
    assert payload["state"].shape == (STATE_DIM,)
    spec = handler({"type": MSG_SPEC})
    assert spec["num_envs"] == 1


def test_single_env_rejects_empty_action_chunk():
    _, handler = _make_handler(num_envs=1)
    with pytest.raises(ValueError, match="at least one action"):
        handler({"type": MSG_STEP, "action": []})


def test_single_env_reset_forwards_seed():
    env, handler = _make_handler(num_envs=1)
    handler({"type": MSG_RESET, "seed": 789})
    assert env.full_reset_seeds == [789]


def test_render_once_prims_defaults_to_the_dome_light_for_batched_scenes():
    """N copies of the factory's infinite dome light change every env's image;
    batched scenes render env_0's copy only by default. One env has one copy
    already; 'none' is the explicit opt-out for tasks without the factory."""
    assert resolve_render_once_prims(1, None) == []
    assert resolve_render_once_prims(256, None) == ["factory_room/Lights/AmbientFill"]
    assert resolve_render_once_prims(256, "none") == []
    assert resolve_render_once_prims(256, "a/b, c/d") == ["a/b", "c/d"]
    with pytest.raises(SystemExit):
        resolve_render_once_prims(256, " , ")


def test_env_spacing_defaults_to_200m_when_batched():
    """The factory's floor is 200 m square; closer clones overlay their floors."""
    assert resolve_env_spacing(1, None) is None  # single env: Arena's default
    assert resolve_env_spacing(8, None) == pytest.approx(200.0)
    assert resolve_env_spacing(256, None) == pytest.approx(200.0)
    assert resolve_env_spacing(256, 30.0) == pytest.approx(30.0)  # explicit wins


def test_renderer_antialiasing_names_map_to_simulation_app_values():
    assert resolve_renderer_antialiasing(None) is None
    assert resolve_renderer_antialiasing("off") == 0
    assert resolve_renderer_antialiasing("taa") == 1
    assert resolve_renderer_antialiasing("fxaa") == 2
    assert resolve_renderer_antialiasing("dlss") == 3
    assert resolve_renderer_antialiasing("rtxaa") == 4
    with pytest.raises(ValueError, match="renderer antialiasing"):
        resolve_renderer_antialiasing("unknown")


def test_renderer_antialiasing_is_attached_to_launcher_namespace():
    launcher_args = argparse.Namespace()
    assert configure_renderer_antialiasing(launcher_args, "fxaa") == 2
    assert launcher_args.anti_aliasing == 2

    default_args = argparse.Namespace()
    assert configure_renderer_antialiasing(default_args, None) is None
    assert not hasattr(default_args, "anti_aliasing")


def test_strict_launcher_enables_isaac_lab_deterministic_rendering():
    launcher_args = argparse.Namespace(deterministic=False)

    assert configure_strict_launcher(launcher_args, True) is True
    assert launcher_args.deterministic is True

    default_args = argparse.Namespace(deterministic=False)
    assert configure_strict_launcher(default_args, False) is False
    assert default_args.deterministic is False


def test_strict_renderer_disables_frame_varying_rtx_effects():
    calls = []
    settings_manager = SimpleNamespace(
        set_bool=lambda path, value: calls.append((path, value))
    )

    configure_strict_renderer(settings_manager)

    assert calls == [
        ("/rtx/directLighting/sampledLighting/stochasticLCRefresh", False),
        ("/rtx/pathtracing/temporalAccumulation", False),
        ("/rtx/post/tonemap/dither", False),
        ("/rtx/raytracing/caustics/useFrameSeed", False),
        ("/rtx/raytracing/subpixel/uniformSubpixelJittering", False),
        ("/rtx/reflections/sampledLighting/stochasticLCRefresh", False),
        ("/rtx/shadows/constantSeed", True),
    ]


def test_strict_environment_enables_physx_enhanced_determinism():
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=None))

    configured = configure_strict_environment(
        env_cfg,
        physics_cfg_factory=lambda: SimpleNamespace(enable_enhanced_determinism=False),
    )

    assert configured is env_cfg
    assert configured.sim.physics.enable_enhanced_determinism is True


def test_strict_environment_preserves_existing_arena_callback():
    callbacks = []

    def existing(env_cfg):
        callbacks.append("existing")
        env_cfg.marker = "configured"
        return env_cfg

    arena_environment = SimpleNamespace(env_cfg_callback=existing)
    install_strict_environment_callback(arena_environment)
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=SimpleNamespace(enable_enhanced_determinism=False))
    )

    configured = arena_environment.env_cfg_callback(env_cfg)

    assert callbacks == ["existing"]
    assert configured.marker == "configured"
    assert configured.sim.physics.enable_enhanced_determinism is True


def test_strict_torch_restores_flags_changed_by_seed_reset():
    deterministic_calls = []
    fake_torch = SimpleNamespace(
        use_deterministic_algorithms=deterministic_calls.append,
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                benchmark=True,
                deterministic=False,
                allow_tf32=True,
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        ),
    )

    configure_strict_torch(fake_torch)

    assert deterministic_calls == [True]
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False


def test_step_timer_reports_per_window_and_resets():
    timer = StepTimer(num_envs=64, report_every=4)
    assert timer.record(0.5, 0.01, with_images=False) is None
    assert timer.record(0.5, 0.01, with_images=False) is None
    assert timer.record(0.5, 0.01, with_images=False) is None
    summary = timer.record(0.5, 0.05, with_images=True)
    assert summary is not None
    assert "over 4 steps (through step 4)" in summary
    assert "sim 500 ms/step mean (min 500, max 500)" in summary
    assert "payload 20 ms/step mean, images on 1 steps" in summary
    assert "128.0 env-steps/s sim-only" in summary
    # The window resets; the running step count does not.
    assert timer.record(1.0, 0.0, with_images=False) is None
    assert timer.total_steps == 5 and len(timer.sim_s) == 1
    with pytest.raises(ValueError):
        StepTimer(num_envs=8, report_every=0)


def test_image_shortest_edge_is_exclusive_with_a_square_resolution(monkeypatch):
    """Letterbox-to-square and shrink-shortest-edge are different framings; both at once is refused."""
    from simulators.g1_arena import env_server

    monkeypatch.setattr(
        "sys.argv",
        [
            "env_server.py",
            "--task",
            "LMDrillLift",
            "--image-resolution",
            "224",
            "--image-shortest-edge",
            "256",
        ],
    )
    with pytest.raises(SystemExit, match="exclusive"):
        env_server.main()


def test_step_timer_reports_send_timing_when_the_serve_loop_feeds_it():
    """The serve loop's per-reply encode/send times join the sim/payload split in the summary."""
    timer = StepTimer(num_envs=8, report_every=2, log_prefix="[srv]")
    timer.record_send(0.010, 0.020)
    assert timer.record(0.5, 0.05, True) is None
    timer.record_send(0.030, 0.040)
    summary = timer.record(0.5, 0.05, True)
    assert summary is not None and summary.startswith("[srv] step timing over 2 steps")
    assert "encode 20 + send 30 ms/reply mean" in summary
    # the window resets the send accounting too
    assert timer.record(0.5, 0.05, False) is None and timer.send_s == []


def test_stall_dump_reports_the_stack_and_exits_a_hung_request():
    """A request that never returns dumps every thread's stack, then ends the server process."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import time
        from simulators.g1_arena.stall_dump import StallDump
        with StallDump(0.2, 1.0):
            time.sleep(30)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 70
    assert "Timeout" in proc.stderr
    assert "exiting with code 70" in proc.stdout


def test_stall_dump_is_silent_for_a_request_that_finishes():
    from simulators.g1_arena.stall_dump import StallDump

    with StallDump(5.0, 10.0):
        pass


def test_reset_with_index_tensors_expands_slices_only() -> None:
    from simulators.g1_arena.env_server import reset_with_index_tensors

    seen: list[Any] = []
    unwrapped = SimpleNamespace(
        num_envs=4, device="cpu", _reset_idx=lambda env_ids: seen.append(env_ids)
    )
    env = SimpleNamespace(unwrapped=unwrapped)
    reset_with_index_tensors(env)
    unwrapped._reset_idx(slice(None))
    partial = torch.tensor([1, 3])
    unwrapped._reset_idx(partial)
    assert seen[0].tolist() == [0, 1, 2, 3]
    assert seen[1] is partial


def test_run_at_50hz_chains_the_existing_callback(monkeypatch) -> None:
    import sys
    import types

    from simulators.g1_arena.env_server import run_at_50hz

    module = types.ModuleType(
        "isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg"
    )
    module.set_control_rate_50hz = lambda cfg: cfg + ["50hz"]  # type: ignore[attr-defined]
    for name in ("isaaclab_arena", "isaaclab_arena.environments", module.__name__):
        monkeypatch.setitem(
            sys.modules,
            name,
            module if name == module.__name__ else types.ModuleType(name),
        )
    arena_env = SimpleNamespace(env_cfg_callback=lambda cfg: cfg + ["strict"])
    run_at_50hz(arena_env)
    assert arena_env.env_cfg_callback([]) == ["strict", "50hz"]
    arena_env = SimpleNamespace(env_cfg_callback=None)
    run_at_50hz(arena_env)
    assert arena_env.env_cfg_callback([]) == ["50hz"]


def test_task_prompt_comes_from_the_task_config(tmp_path, monkeypatch) -> None:
    """A G1 Factory task without its own instruction is prompted from its closed-loop policy config."""
    config = tmp_path / "fake_factory" / "configs" / "gr00t" / "policy"
    config.mkdir(parents=True)
    (tmp_path / "fake_factory" / "__init__.py").write_text("")
    (config / "LMBoxLiftFloor_closedloop_config.yaml").write_text(
        'language_instruction: "Walk to the box and pick it up from the floor"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    silent = SimpleNamespace(
        unwrapped=SimpleNamespace(get_language_instruction=lambda: None)
    )
    prompt = resolve_task_prompt(silent, "LMBoxLiftFloor", "fake_factory.environments")
    assert prompt == "Walk to the box and pick it up from the floor"
    own = SimpleNamespace(
        unwrapped=SimpleNamespace(get_language_instruction=lambda: "Own words")
    )
    assert (
        resolve_task_prompt(own, "LMBoxLiftFloor", "fake_factory.environments")
        == "Own words"
    )
    with pytest.raises(SystemExit):
        resolve_task_prompt(silent, "LMUnknown", "fake_factory.environments")
