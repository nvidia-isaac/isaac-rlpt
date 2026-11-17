# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiberoEnv contract conformance against a mocked LIBERO stack.

The real LIBERO dependencies (MuJoCo/robosuite) are optional; these tests
install a scripted stand-in for the ``libero`` package into ``sys.modules``
and pin the provider's adaptation layer exactly:

* observation mapping (upright image, state-vector concatenation order,
  the prompt string riding along),
* success plumbing (``info["success"]`` / ``info["is_success"]``,
  terminated-on-success, truncation at the client-side cap),
* action pass-through (float32, single actions and sequential chunks with
  early stop),
* reset semantics (seeding, benchmark init-state injection plus settle
  steps),
* clean, idempotent close.

Real-stack counterparts: the thin cross-checks at the bottom run only where
the LIBERO benchmark package is importable, and
``tests/integration/test_libero_loading.py`` (marked ``libero``) boots the
real simulator.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO_ROOT

SIMULATOR_CONFIG_DIR = (
    Path(REPO_ROOT) / "simulators" / "libero" / "configs" / "simulator"
)

ACTION_DIM = 7
N_INIT_STATES = 3


class MockSimEnv:
    """Scripted stand-in for LIBERO's ``OffScreenRenderEnv``.

    Produces robosuite-shaped observations (a vertically flipped camera
    frame plus proprioceptive entries), records every call, and reports
    success once ``success_at_step`` raw steps have executed.
    """

    #: Row written at raw index 0 (the BOTTOM of the upright scene) and raw
    #: index -1 (the TOP); the provider must flip so that out[0] == RAW_TOP.
    RAW_BOTTOM = (1, 2, 3)
    RAW_TOP = (9, 8, 7)

    def __init__(
        self, bddl_file_name, camera_heights, camera_widths, camera_names, **kwargs
    ):
        self.bddl_file_name = bddl_file_name
        self.camera_heights = int(camera_heights)
        self.camera_widths = int(camera_widths)
        self.camera_names = list(camera_names)
        self.extra_kwargs = dict(kwargs)
        self.success_at_step: int | None = None
        self.t = 0
        self.step_actions: list[np.ndarray] = []
        self.seed_calls: list[int] = []
        self.set_init_state_calls: list[np.ndarray] = []
        self.close_calls = 0
        self.reset_calls = 0
        # LiberoEnv probes the raw robosuite env for observation widths.
        self.env = types.SimpleNamespace(_get_observations=lambda **kw: self._obs())

    #: Per-camera fill value at raw row 1: makes each camera's frame
    #: distinguishable so multi-camera routing is observable.
    CAMERA_FILL_BASE = 20

    def _obs(self):
        obs = {
            "robot0_joint_pos": np.arange(7, dtype=np.float64) + self.t,
            "robot0_gripper_qpos": np.array([100.0 + self.t, -100.0 - self.t]),
            "robot0_eef_pos": np.array([0.5, 0.5, 0.5]),
            # 90-degree rotation about x: axis-angle (pi/2, 0, 0).
            "robot0_eef_quat": np.array(
                [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]
            ),
        }
        for index, camera in enumerate(self.camera_names):
            img = np.zeros((self.camera_heights, self.camera_widths, 3), dtype=np.uint8)
            img[0, :, :] = self.RAW_BOTTOM
            img[-1, :, :] = self.RAW_TOP
            img[1, :, :] = self.CAMERA_FILL_BASE + index
            obs[f"{camera}_image"] = img
        return obs

    @property
    def _success(self):
        return self.success_at_step is not None and self.t >= self.success_at_step

    def reset(self):
        self.reset_calls += 1
        self.t = 0
        return self._obs()

    def seed(self, seed):
        self.seed_calls.append(int(seed))

    def set_init_state(self, init_state):
        self.set_init_state_calls.append(np.array(init_state, copy=True))
        return self._obs()

    def step(self, action):
        self.step_actions.append(np.array(action, copy=True))
        self.t += 1
        reward = 1.0 if self._success else 0.0
        # LIBERO overrides robosuite's done with the success predicate.
        return self._obs(), reward, self._success, {}

    def check_success(self):
        return self._success

    def close(self):
        self.close_calls += 1


class MockBenchmark:
    """Stand-in for LIBERO's per-suite ``Benchmark`` object."""

    TASKS = ("mock_task_zero", "mock_task_one")

    def __init__(self):
        self.init_state_requests: list[int] = []

    def get_task_names(self):
        return list(self.TASKS)

    def get_task(self, i):
        return types.SimpleNamespace(
            name=self.TASKS[i],
            language=f"language instruction for {self.TASKS[i]}",
            problem_folder="mock_suite",
            bddl_file=f"{self.TASKS[i]}.bddl",
            init_states_file=f"{self.TASKS[i]}.pruned_init",
        )

    def get_task_init_states(self, i):
        self.init_state_requests.append(i)
        # Distinguishable per-id states so injection is observable.
        return np.arange(N_INIT_STATES, dtype=np.float64)[:, None] + np.zeros((1, 4))


@pytest.fixture()
def mock_libero(monkeypatch, tmp_path):
    """Install a fake ``libero`` package and a valid LIBERO config location.

    Returns a handle exposing the benchmark double and the created
    ``MockSimEnv`` instances.
    """
    config_dir = tmp_path / "libero_config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("bddl_files: /mock/bddl_files\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))

    handle = types.SimpleNamespace(benchmark=MockBenchmark(), sim_envs=[])

    def make_sim_env(**kwargs):
        sim = MockSimEnv(**kwargs)
        handle.sim_envs.append(sim)
        return sim

    root = types.ModuleType("libero")
    inner = types.ModuleType("libero.libero")
    inner.get_libero_path = lambda key: f"/mock/{key}"
    benchmark_mod = types.ModuleType("libero.libero.benchmark")
    benchmark_mod.get_benchmark = lambda suite: lambda: handle.benchmark
    envs_mod = types.ModuleType("libero.libero.envs")
    envs_mod.OffScreenRenderEnv = make_sim_env
    root.libero = inner
    inner.benchmark = benchmark_mod
    inner.envs = envs_mod

    for name, mod in (
        ("libero", root),
        ("libero.libero", inner),
        ("libero.libero.benchmark", benchmark_mod),
        ("libero.libero.envs", envs_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return handle


def _make_env(mock_libero, **kwargs):
    from simulators.libero.env import LiberoEnv

    kwargs.setdefault("image_size", 8)
    kwargs.setdefault("use_init_states", False)
    return LiberoEnv(suite="libero_spatial", **kwargs)


# ---------------------------------------------------------------------------
# Construction, spaces, key mapping
# ---------------------------------------------------------------------------


def test_spaces_and_key_mapping(mock_libero):
    env = _make_env(mock_libero)
    assert env.action_space.shape == (ACTION_DIM,)
    assert env.action_space.dtype == np.float32
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)
    assert set(env.observation_space.spaces) == {"agentview_image", "state"}
    assert env.observation_space["agentview_image"].shape == (8, 8, 3)
    assert env.observation_space["state"].shape == (9,)  # joints (7) + gripper (2)
    assert env.steering_key_mapping == {
        "state": "state",
        "language": None,
        "image": ["agentview_image"],
    }
    sim = mock_libero.sim_envs[0]
    assert sim.bddl_file_name == "/mock/bddl_files/mock_suite/mock_task_zero.bddl"
    assert sim.camera_names == ["agentview"]


def test_task_selection_by_name_and_bad_inputs(mock_libero):
    env = _make_env(mock_libero, task_name="mock_task_one")
    assert env.task_id == 1 and env.task_name == "mock_task_one"
    with pytest.raises(ValueError, match="not in suite"):
        _make_env(mock_libero, task_name="no_such_task")
    with pytest.raises(ValueError, match="out of range"):
        _make_env(mock_libero, task_id=99)
    from simulators.libero.env import LiberoEnv

    with pytest.raises(ValueError, match="unknown LIBERO suite"):
        LiberoEnv(suite="libero_bogus")


def test_unconfigured_libero_raises_instead_of_hanging(
    mock_libero, monkeypatch, tmp_path
):
    """No config.yaml -> a clear error, never LIBERO's interactive prompt."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="README.md"):
        _make_env(mock_libero)


# ---------------------------------------------------------------------------
# Observation mapping
# ---------------------------------------------------------------------------


def test_reset_obs_contract_and_upright_image(mock_libero):
    env = _make_env(mock_libero)
    obs, info = env.reset(seed=0)
    assert set(obs) == {"agentview_image", "state", "prompt"}

    img = obs["agentview_image"]
    assert img.dtype == np.uint8 and img.shape == (8, 8, 3)
    # The raw buffer is bottom-up; the provider must emit the upright frame.
    assert tuple(img[0, 0]) == MockSimEnv.RAW_TOP
    assert tuple(img[-1, 0]) == MockSimEnv.RAW_BOTTOM

    state = obs["state"]
    assert state.dtype == np.float32 and state.shape == (9,)
    np.testing.assert_allclose(state[:7], np.arange(7))  # joint block first
    np.testing.assert_allclose(state[7:], [100.0, -100.0])  # then gripper

    assert obs["prompt"] == "language instruction for mock_task_zero"
    assert info["task_name"] == "mock_task_zero"
    assert info["suite"] == "libero_spatial"
    assert info["init_state_id"] is None  # init states are off by default here


def test_custom_state_keys_change_state_vector(mock_libero):
    env = _make_env(mock_libero, state_keys=("robot0_eef_pos", "robot0_gripper_qpos"))
    assert env.observation_space["state"].shape == (5,)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["state"][:3], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="state_keys"):
        _make_env(mock_libero, state_keys=("robot0_not_a_key",))


def test_multi_camera_observations(mock_libero):
    """cameras=(...) renders every camera and emits one upright frame each."""
    env = _make_env(mock_libero, cameras=("agentview", "robot0_eye_in_hand"))
    assert set(env.observation_space.spaces) == {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "state",
    }
    assert env.steering_key_mapping["image"] == [
        "agentview_image",
        "robot0_eye_in_hand_image",
    ]
    sim = mock_libero.sim_envs[0]
    assert sim.camera_names == ["agentview", "robot0_eye_in_hand"]

    obs, _ = env.reset()
    for index, key in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
        frame = obs[key]
        assert frame.shape == (8, 8, 3) and frame.dtype == np.uint8
        assert tuple(frame[0, 0]) == MockSimEnv.RAW_TOP  # upright
        # Per-camera fill row proves no cross-wiring between the views.
        assert frame[-2, 0, 0] == MockSimEnv.CAMERA_FILL_BASE + index


def test_quat_to_axis_angle_convention():
    """robosuite's quat2axisangle convention: (x,y,z,w) in, angle*axis out."""
    from simulators.libero.env import quat_to_axis_angle

    # Zero rotation maps to the zero vector (the near-singular branch).
    np.testing.assert_allclose(
        quat_to_axis_angle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3)
    )
    # 90 degrees about x: axis-angle (pi/2, 0, 0).
    quarter = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
    np.testing.assert_allclose(
        quat_to_axis_angle(quarter), [np.pi / 2, 0.0, 0.0], atol=1e-6
    )
    # 120 degrees about the (1,1,1)/sqrt(3) axis.
    axis = np.ones(3) / np.sqrt(3.0)
    angle = 2.0 * np.pi / 3.0
    quat = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
    np.testing.assert_allclose(quat_to_axis_angle(quat), axis * angle, atol=1e-6)


def test_eef_state_keys_mix_raw_and_derived(mock_libero):
    """EEF_STATE_KEYS assemble the 8-dim (pos, axis-angle, gripper) state."""
    from simulators.libero.env import EEF_STATE_KEYS

    env = _make_env(mock_libero, state_keys=EEF_STATE_KEYS)
    assert env.observation_space["state"].shape == (8,)
    obs, _ = env.reset()
    state = obs["state"]
    np.testing.assert_allclose(state[:3], [0.5, 0.5, 0.5])  # eef position
    # The mock's eef quat is a 90-degree rotation about x (see MockSimEnv).
    np.testing.assert_allclose(state[3:6], [np.pi / 2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state[6:], [100.0, -100.0])  # gripper qpos


# ---------------------------------------------------------------------------
# Initial-state protocol
# ---------------------------------------------------------------------------


def test_sequential_init_states_walk_in_order(mock_libero):
    """Episode k gets init state k modulo the file size (fixed-init protocol)."""
    env = _make_env(mock_libero, use_init_states=True, sequential_init_states=True)
    seen = []
    for _ in range(N_INIT_STATES + 2):
        _, info = env.reset()
        seen.append(info["init_state_id"])
    assert seen == [0, 1, 2, 0, 1]
    # The injected states match the requested ids (states are per-id coded).
    sim = mock_libero.sim_envs[0]
    injected = [int(call[0]) for call in sim.set_init_state_calls]
    assert injected == [0, 1, 2, 0, 1]


def test_sequential_init_states_requires_init_states(mock_libero):
    with pytest.raises(ValueError, match="use_init_states"):
        _make_env(mock_libero, use_init_states=False, sequential_init_states=True)


def test_reset_info_reports_chunking_support(mock_libero):
    """step() executes [chunk, 7] chunks natively; reset info says so (the
    rollout/eval workers key their batched-step fast path on this flag)."""
    env = _make_env(mock_libero)
    _, info = env.reset()
    assert info["supports_action_chunking"] is True


# ---------------------------------------------------------------------------
# Step semantics: pass-through, chunks, success, truncation
# ---------------------------------------------------------------------------


def test_action_passthrough_single_step(mock_libero):
    env = _make_env(mock_libero)
    env.reset()
    sim = mock_libero.sim_envs[0]
    action = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    obs, reward, terminated, truncated, info = env.step(action)
    assert len(sim.step_actions) == 1
    assert sim.step_actions[0].dtype == np.float32
    np.testing.assert_allclose(sim.step_actions[0], action, rtol=1e-6)
    assert reward == pytest.approx(0.0) and not terminated and not truncated
    assert info["success"] is False and info["is_success"] is False
    assert info["num_steps"] == 1
    assert set(obs) == {"agentview_image", "state", "prompt"}


def test_bad_action_shape_raises(mock_libero):
    env = _make_env(mock_libero)
    env.reset()
    with pytest.raises(ValueError, match="chunk"):
        env.step(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="chunk"):
        env.step(np.zeros((2, 3, ACTION_DIM), dtype=np.float32))


def test_chunk_executes_sequentially(mock_libero):
    env = _make_env(mock_libero)
    env.reset()
    sim = mock_libero.sim_envs[0]
    chunk = np.stack([np.full(ACTION_DIM, i, dtype=np.float32) for i in range(4)])
    _, _, terminated, truncated, info = env.step(chunk)
    assert len(sim.step_actions) == 4
    np.testing.assert_allclose(np.stack(sim.step_actions), chunk)
    assert info["num_steps"] == 4 and not terminated and not truncated


def test_success_terminates_and_stops_chunk_early(mock_libero):
    env = _make_env(mock_libero)
    env.reset()
    sim = mock_libero.sim_envs[0]
    sim.success_at_step = 3
    chunk = np.zeros((6, ACTION_DIM), dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(chunk)
    assert len(sim.step_actions) == 3, "the chunk must stop at the success step"
    assert terminated and not truncated
    assert info["success"] is True and info["is_success"] is True
    assert info["num_steps"] == 3
    assert reward == pytest.approx(1.0), "LIBERO's sparse success reward passes through"


def test_truncation_at_episode_cap(mock_libero):
    env = _make_env(mock_libero, max_episode_steps=4)
    env.reset()
    flags = []
    for _ in range(4):
        _, _, terminated, truncated, info = env.step(
            np.zeros(ACTION_DIM, dtype=np.float32)
        )
        flags.append((terminated, truncated))
    assert flags[:3] == [(False, False)] * 3
    assert flags[3] == (False, True), "the cap ends a failed episode as a truncation"
    assert info["success"] is False


def test_chunk_respects_episode_cap(mock_libero):
    env = _make_env(mock_libero, max_episode_steps=4)
    env.reset()
    sim = mock_libero.sim_envs[0]
    _, _, terminated, truncated, info = env.step(
        np.zeros((10, ACTION_DIM), dtype=np.float32)
    )
    assert len(sim.step_actions) == 4 and info["num_steps"] == 4
    assert truncated and not terminated


def test_cap_resets_between_episodes(mock_libero):
    env = _make_env(mock_libero, max_episode_steps=2)
    env.reset()
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    _, _, _, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert truncated
    env.reset()
    _, _, _, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert not truncated, "the step counter must reset with the episode"


# ---------------------------------------------------------------------------
# Reset semantics: seeding and init-state injection
# ---------------------------------------------------------------------------


def test_reset_seed_reaches_libero(mock_libero):
    env = _make_env(mock_libero)
    env.reset(seed=123)
    assert mock_libero.sim_envs[0].seed_calls == [123]
    env.reset()  # no seed -> no re-seeding call
    assert mock_libero.sim_envs[0].seed_calls == [123]


def test_init_state_injection_and_settle_steps(mock_libero):
    env = _make_env(mock_libero, use_init_states=True)
    sim = mock_libero.sim_envs[0]
    assert mock_libero.benchmark.init_state_requests == [0]

    _, info = env.reset(seed=0, options={"init_state_id": 1})
    assert info["init_state_id"] == 1
    assert len(sim.set_init_state_calls) == 1
    np.testing.assert_allclose(sim.set_init_state_calls[0], np.full(4, 1.0))
    # Five zero-action settle steps (LIBERO's evaluation protocol), which
    # must not consume the episode cap.
    assert len(sim.step_actions) == 5
    assert all(np.allclose(a, 0.0) for a in sim.step_actions)
    assert env._episode_steps == 0


def test_random_init_state_is_seed_deterministic(mock_libero):
    env = _make_env(mock_libero, use_init_states=True)
    _, info_a = env.reset(seed=7)
    _, info_b = env.reset(seed=7)
    assert info_a["init_state_id"] == info_b["init_state_id"]
    assert 0 <= info_a["init_state_id"] < N_INIT_STATES


def test_out_of_range_init_state_raises(mock_libero):
    env = _make_env(mock_libero, use_init_states=True)
    with pytest.raises(ValueError, match="out of range"):
        env.reset(options={"init_state_id": N_INIT_STATES})


# ---------------------------------------------------------------------------
# Close semantics
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_final(mock_libero):
    env = _make_env(mock_libero)
    env.reset()
    env.close()
    env.close()
    assert mock_libero.sim_envs[0].close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


# ---------------------------------------------------------------------------
# Real-stack cross-checks (skip wherever the LIBERO package is absent)
# ---------------------------------------------------------------------------


def _real_libero_or_skip():
    """Import the real benchmark registry, or skip -- without ever letting
    LIBERO's interactive first-run prompt hang the suite."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH") or os.path.expanduser("~/.libero")
    if not os.path.isfile(os.path.join(config_dir, "config.yaml")):
        pytest.skip("LIBERO not configured on this machine (no config.yaml)")
    return pytest.importorskip("libero.libero.benchmark")


@pytest.mark.parametrize(
    "suite", ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
)
def test_config_task_lists_match_real_benchmark(suite):
    """The task enumerations pinned in the simulator configs are the real ones."""
    benchmark_mod = _real_libero_or_skip()
    real = benchmark_mod.get_benchmark(suite)().get_task_names()
    doc = yaml.safe_load(
        (SIMULATOR_CONFIG_DIR / f"{suite}.yaml").read_text(encoding="utf-8")
    )
    assert list(doc["tasks"]) == list(real), (
        f"configs/simulator/{suite}.yaml has drifted from the LIBERO benchmark"
    )
