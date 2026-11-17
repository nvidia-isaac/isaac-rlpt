# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLabGr00tEnv contract conformance against a fake IsaacLab environment.

Isaac Sim is not needed: a scripted stand-in returns IsaacLab-shaped
observation groups (torch tensors) and pins the provider's adaptation layer:

* observation mapping (camera terms -> GR00T video keys, state slices, the
  ``state`` concatenation order, the prompt riding along),
* action padding (28 model joints -> 43 simulator joints, zeros first) and
  per-joint bounds from the articulation's soft limits,
* chunk semantics (early stop on termination, truncation at the cap),
* success plumbing (``info["is_success"]`` == terminated),
* the shared-instance cache and the app-launch guard,
* the DSRL builder's wrapper stack and the registry/Hydra wiring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from conftest import PACKAGE_NAME, REPO_ROOT
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import simulators.isaaclab.env as isaaclab_env
from simulators.isaaclab.env import (
    APP_LAUNCHED_FLAG,
    DEFAULT_JOINT_BOUND,
    DEFAULT_TASK_ID,
    FRANKA_STACK_GR00T_MAPPING,
    FRANKA_STACK_TASKS,
    TASK_MAPPINGS,
    TROCAR_GR00T_MAPPING,
    TROCAR_TASKS,
    IsaacLabGr00tEnv,
    get_or_create_isaaclab_env,
    mapping_for_task,
    quat_wxyz_to_axis_angle,
    register_task_mapping,
    release_isaaclab_env,
)

CONFIG_DIR = Path(REPO_ROOT) / PACKAGE_NAME / "configs"
SIMULATOR_DIR = Path(REPO_ROOT) / "simulators" / "isaaclab" / "configs" / "simulator"
SIMULATOR_YAML = SIMULATOR_DIR / "isaaclab_trocar.yaml"
FRANKA_YAML = SIMULATOR_DIR / "isaaclab_franka_stack.yaml"
FRANKA_TASK_ID = FRANKA_STACK_TASKS[0]
FRANKA_VIDEO_KEYS = ["video.image", "video.wrist_image"]
FRANKA_STATE_KEYS = [
    "state.x",
    "state.y",
    "state.z",
    "state.roll",
    "state.pitch",
    "state.yaw",
    "state.gripper",
]

IMG = 224
N_BODY = 29
N_HAND = 14
N_ENV_ACTION = 43
ACTION_DIM = 28
VIDEO_KEYS = ["video.room_view", "video.left_wrist_view", "video.right_wrist_view"]
STATE_KEYS = [
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
]
# Simulator joint names in the action term's order: 29 body joints, 14 hand joints.
JOINT_NAMES = [f"body_{i}" for i in range(N_BODY)] + [
    f"hand_{i}" for i in range(N_HAND)
]


class _FakeArticulation:
    """Articulation stub: joints stored in REVERSED order to make ordering observable."""

    def __init__(self, device: str) -> None:
        # Soft limits [1, 43, 2]: joint k (articulation index) has bounds -(k+1), +(k+1),
        # behind IsaacLab 3.0's ProxyArray-style ``.torch`` view.
        k = torch.arange(N_ENV_ACTION, dtype=torch.float32, device=device) + 1.0
        proxy = type("ProxyArray", (), {"torch": torch.stack([-k, k], dim=-1)[None]})()
        self.data = type("Data", (), {"soft_joint_pos_limits": proxy})()
        self._order = list(reversed(JOINT_NAMES))  # articulation index -> name

    def find_joints(self, names, joint_subset=None, preserve_order=False):
        ids = [self._order.index(name) for name in names]
        return ids, list(names)


class FakeIsaacLabEnv:
    """Scripted stand-in for an IsaacLab ``ManagerBasedRLEnv`` (one env).

    ``robot_joint_state`` carries IsaacLab's real ``[pos(29) | vel(29) |
    torque(29)]`` layout, so a slice landing outside the position block is
    visible; positions read ``100 + i``, hand joints ``400 + i``.
    """

    def __init__(
        self,
        *,
        terminate_at=None,
        truncate_at=None,
        articulation=True,
        image_size=IMG,
        device="cpu",
    ):
        self.device = device
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.image_size = image_size
        self.t = 0
        self.actions: list[torch.Tensor] = []
        self.reset_seeds: list[int | None] = []
        self.close_calls = 0
        actions_cfg = type("ActionsCfg", (), {})()
        actions_cfg.joint_pos = type(
            "JointPos", (), {"joint_names": list(JOINT_NAMES)}
        )()
        self.cfg = type("EnvCfg", (), {"actions": actions_cfg})()
        if articulation:
            self.scene = {"robot": _FakeArticulation(device)}

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        pos = 100.0 + torch.arange(N_BODY, dtype=torch.float32)
        vel = 200.0 + torch.arange(N_BODY, dtype=torch.float32)
        torque = 300.0 + torch.arange(N_BODY, dtype=torch.float32)
        body = torch.cat([pos, vel, torque])[None].to(self.device)
        hand = (400.0 + torch.arange(N_HAND, dtype=torch.float32))[None].to(self.device)
        shape = (1, self.image_size, self.image_size, 3)
        cams = {
            "front_camera": torch.full(shape, 7, dtype=torch.uint8, device=self.device),
            "left_wrist_camera": torch.full(
                shape, 8, dtype=torch.uint8, device=self.device
            ),
            "right_wrist_camera": torch.full(
                shape, 9, dtype=torch.uint8, device=self.device
            ),
        }
        return {
            "policy": {"robot_joint_state": body, "robot_dex3_joint_state": hand},
            "camera_images": cams,
        }

    def reset(self, seed=None, env_ids=None, options=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.actions.append(action.clone())
        self.t += 1
        terminated = self.terminate_at is not None and self.t >= self.terminate_at
        truncated = self.truncate_at is not None and self.t >= self.truncate_at
        reward = torch.tensor([1.0 if terminated else 0.0], device=self.device)
        done = torch.tensor([terminated], device=self.device)
        timed_out = torch.tensor([truncated], device=self.device)
        return self._obs(), reward, done, timed_out, {"log": {}}

    def close(self):
        self.close_calls += 1


class FakeFrankaStackEnv:
    """Stand-in for ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`` (one env).

    Cameras live in the ``policy`` group; the end-effector quaternion is
    (w, x, y, z) for a 90 degree rotation about z; ``subtask_terms`` carries the
    three boolean stage flags; the IK-Rel action is 7 wide (no padding).
    """

    def __init__(
        self,
        *,
        terminate_at=None,
        image_size=16,
        flags=(True, True, False),
        reset_flags=None,
        device="cpu",
        num_envs=1,
        terminate_scene=0,
    ):
        self.device = device
        self.terminate_at = terminate_at
        self.terminate_scene = terminate_scene  # scene whose episode ends at terminate_at (multi-scene fakes)
        self.image_size = image_size
        self.flags = flags
        self.reset_flags = reset_flags  # flags of the post-reset observation IsaacLab returns on a terminal step
        self.done = False
        self.num_envs = num_envs
        self.reset_env_ids: list[list[int] | None] = []
        self.step_count = np.zeros(num_envs, dtype=int)
        self.t = 0
        self.actions: list[torch.Tensor] = []
        self.close_calls = 0
        self.cfg = type("EnvCfg", (), {"actions": type("ActionsCfg", (), {})()})()
        self.scene = {}  # an articulation exists but the mapping must not read joint limits

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        n = self.num_envs
        shape = (n, self.image_size, self.image_size, 3)
        half = np.sqrt(0.5)
        # Scene i carries z = 0.2 + i and its own step count in x, so per-scene routing is observable.
        eef_pos = torch.tensor(
            [[0.5 + 0.01 * self.step_count[i], -0.1, 0.2 + i] for i in range(n)],
            device=self.device,
        )
        policy = {
            "eef_pos": eef_pos,
            "eef_quat": torch.tensor(
                [[half, 0.0, 0.0, half]] * n, device=self.device
            ),  # w, x, y, z
            "gripper_pos": torch.tensor([[0.04, -0.04]] * n, device=self.device),
            "table_cam": torch.full(shape, 3, dtype=torch.uint8, device=self.device),
            "wrist_cam": torch.full(shape, 4, dtype=torch.uint8, device=self.device),
        }
        flags = (
            self.reset_flags
            if (self.done and self.reset_flags is not None)
            else self.flags
        )
        subtask = {
            name: torch.tensor([[flag]] * n, device=self.device)
            for name, flag in zip(("grasp_1", "stack_1", "grasp_2"), flags)
        }
        return {"policy": policy, "subtask_terms": subtask}

    def reset(self, seed=None, env_ids=None, options=None):
        if env_ids is not None:
            assert torch.is_tensor(env_ids) and env_ids.dtype == torch.int32, (
                "IsaacLab indexes with an int32 tensor"
            )
            env_ids = [int(i) for i in env_ids.tolist()]
        self.reset_env_ids.append(env_ids)
        if env_ids is None:
            self.t = 0
            self.step_count[:] = 0
        else:
            self.step_count[env_ids] = 0
        self.done = False
        return self._obs(), {}

    def step(self, action):
        self.actions.append(action.clone())
        self.t += 1
        self.step_count += 1
        n = self.num_envs
        terminated = torch.zeros(n, dtype=torch.bool)
        if (
            self.terminate_at is not None
            and self.step_count[self.terminate_scene] >= self.terminate_at
        ):
            terminated[self.terminate_scene] = True
            self.step_count[self.terminate_scene] = (
                0  # IsaacLab resets the finished scene inside step
            )
        self.done = bool(terminated.any())
        reward = terminated.to(torch.float32)
        return self._obs(), reward, terminated, torch.zeros(n, dtype=torch.bool), {}

    def close(self):
        self.close_calls += 1


def _franka_env(**kw) -> IsaacLabGr00tEnv:
    fake = FakeFrankaStackEnv(
        **{k: v for k, v in kw.items() if k in ("terminate_at", "flags", "reset_flags")}
    )
    return IsaacLabGr00tEnv(
        FRANKA_TASK_ID,
        env=fake,
        mapping=FRANKA_STACK_GR00T_MAPPING,
        action_dim=7,
        image_size=16,
        max_episode_steps=kw.get("max_episode_steps", 448),
    )


@pytest.fixture(autouse=True)
def _clean_shared_cache():
    isaaclab_env._SHARED_ENVS.clear()
    yield
    isaaclab_env._SHARED_ENVS.clear()


def _model_action(scale: float = 1.0) -> np.ndarray:
    return (np.arange(ACTION_DIM, dtype=np.float32) + 1.0) * 0.01 * scale


# ---------------------------------------------------------------------------
# Observation contract
# ---------------------------------------------------------------------------


def test_observation_keys_shapes_and_state_order():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv(), seed=3)
    obs, info = env.reset()
    assert set(obs) == {*VIDEO_KEYS, *STATE_KEYS, "state", "prompt"}
    assert set(env.observation_space.spaces) == {*VIDEO_KEYS, *STATE_KEYS, "state"}
    for key, fill in zip(VIDEO_KEYS, (7, 8, 9)):
        assert obs[key].dtype == np.uint8 and obs[key].shape == (IMG, IMG, 3)
        assert (obs[key] == fill).all()
    pos = 100.0 + np.arange(N_BODY, dtype=np.float32)
    hand = 400.0 + np.arange(N_HAND, dtype=np.float32)
    expected = {
        "state.left_arm": pos[15:22],
        "state.right_arm": pos[22:29],
        "state.left_hand": hand[0:7],
        "state.right_hand": hand[7:14],
    }
    for key, value in expected.items():
        assert obs[key].dtype == np.float32 and obs[key].shape == (7,)
        np.testing.assert_array_equal(obs[key], value)
    assert obs["state"].dtype == np.float32 and obs["state"].shape == (ACTION_DIM,)
    np.testing.assert_array_equal(obs["state"], np.concatenate(list(expected.values())))
    assert obs["prompt"] == "assemble trocar from tray"
    assert info["supports_action_chunking"] is True
    assert info["prompt"] == env.language_instruction
    assert env.observation_space.contains(
        {k: v for k, v in obs.items() if k != "prompt"}
    )


def test_learner_facing_attributes():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv(), max_episode_steps=250)
    assert env.steering_key_mapping == {
        "state": "state",
        "language": None,
        "image": VIDEO_KEYS,
    }
    assert env.steering_remove_obs_keys == [*VIDEO_KEYS, *STATE_KEYS]
    assert env.language_instruction == "assemble trocar from tray"
    assert env.task_id == DEFAULT_TASK_ID
    assert env.max_episode_steps == 250


def test_mapping_is_data_driven():
    mapping = {
        "cameras": {"video.only": "front_camera"},
        "states": {"state.hand": {"term": "robot_dex3_joint_state", "slice": [0, 14]}},
        "action_term": "joint_pos",
        "prefix_pad": 29,
        "suffix_pad": 0,
        "task_description": "other task",
    }
    fake = FakeIsaacLabEnv()
    env = IsaacLabGr00tEnv(env=fake, mapping=mapping, action_dim=14)
    obs, _ = env.reset()
    assert set(obs) == {"video.only", "state.hand", "state", "prompt"}
    assert obs["state"].shape == (14,) and obs["prompt"] == "other task"
    env.step(np.ones(14, dtype=np.float32))
    executed = fake.actions[0][0]
    assert executed.shape == (N_ENV_ACTION,)
    assert (executed[:29] == 0).all() and (executed[29:] == 1).all()


def test_invalid_mapping_fails_loud():
    bad_slice = dict(
        TROCAR_GR00T_MAPPING,
        states={"state.x": {"term": "robot_joint_state", "slice": [5, 5]}},
    )
    with pytest.raises(ValueError, match="invalid slice"):
        IsaacLabGr00tEnv(env=FakeIsaacLabEnv(), mapping=bad_slice)
    bad_term = dict(TROCAR_GR00T_MAPPING, action_term="no_such_term")
    with pytest.raises(AttributeError, match="no_such_term"):
        IsaacLabGr00tEnv(env=FakeIsaacLabEnv(), mapping=bad_term)


def test_camera_resolution_mismatch_fails_loud():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv(image_size=8))
    with pytest.raises(ValueError, match="image_size=224"):
        env.reset()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_action_is_zero_padded_to_the_simulator_width():
    fake = FakeIsaacLabEnv()
    env = IsaacLabGr00tEnv(env=fake)
    env.reset()
    action = _model_action()
    env.step(action)
    assert len(fake.actions) == 1
    executed = fake.actions[0]
    assert isinstance(executed, torch.Tensor)
    assert executed.shape == (1, N_ENV_ACTION) and executed.dtype == torch.float32
    assert (executed[0, :15] == 0).all()
    np.testing.assert_allclose(executed[0, 15:].cpu().numpy(), action)


def test_action_space_uses_soft_limits_of_the_controlled_joints():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv())
    assert env.action_space.shape == (ACTION_DIM,)
    # Joint k of the action order sits at REVERSED articulation index
    # 42 - k, whose fake bounds are -(43 - k), +(43 - k).
    expected_high = np.array(
        [N_ENV_ACTION - k for k in range(15, N_ENV_ACTION)], dtype=np.float32
    )
    np.testing.assert_array_equal(env.action_space.high, expected_high)
    np.testing.assert_array_equal(env.action_space.low, -expected_high)


def test_action_space_falls_back_without_an_articulation():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv(articulation=False))
    assert (env.action_space.high == DEFAULT_JOINT_BOUND).all()
    assert (env.action_space.low == -DEFAULT_JOINT_BOUND).all()


def test_wrong_action_width_is_rejected():
    env = IsaacLabGr00tEnv(env=FakeIsaacLabEnv())
    env.reset()
    with pytest.raises(ValueError, match=r"\[28\]"):
        env.step(np.zeros(N_ENV_ACTION, dtype=np.float32))


# ---------------------------------------------------------------------------
# Chunk semantics, cap, success plumbing
# ---------------------------------------------------------------------------


def test_chunk_stops_at_termination_and_reports_success():
    fake = FakeIsaacLabEnv(terminate_at=2)
    env = IsaacLabGr00tEnv(env=fake, max_episode_steps=100)
    env.reset()
    chunk = np.tile(_model_action(), (5, 1))
    obs, reward, terminated, truncated, info = env.step(chunk)
    assert len(fake.actions) == 2
    assert terminated is True and truncated is False
    assert info["is_success"] is True and info["success"] is True
    assert info["num_steps"] == 2
    assert reward == pytest.approx(1.0)
    assert set(obs) == {*VIDEO_KEYS, *STATE_KEYS, "state", "prompt"}


def test_client_cap_truncates_without_success():
    fake = FakeIsaacLabEnv()
    env = IsaacLabGr00tEnv(env=fake, max_episode_steps=3)
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.tile(_model_action(), (5, 1)))
    assert len(fake.actions) == 3
    assert terminated is False and truncated is True
    assert info["is_success"] is False and info["num_steps"] == 3
    assert reward == pytest.approx(0.0)
    # The cap resets with the episode.
    env.reset()
    _, _, _, truncated, info = env.step(np.tile(_model_action(), (2, 1)))
    assert truncated is False and info["num_steps"] == 2


def test_simulator_truncation_propagates():
    fake = FakeIsaacLabEnv(truncate_at=2)
    env = IsaacLabGr00tEnv(env=fake, max_episode_steps=100)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.tile(_model_action(), (4, 1)))
    assert len(fake.actions) == 2
    assert terminated is False and truncated is True and info["is_success"] is False


def test_reset_seeds_first_episode_from_the_constructor_seed():
    fake = FakeIsaacLabEnv()
    env = IsaacLabGr00tEnv(env=fake, seed=11)
    env.reset()
    env.reset()
    env.reset(seed=5)
    assert fake.reset_seeds == [11, None, 5]


def test_close_is_idempotent_and_closes_an_injected_env():
    fake = FakeIsaacLabEnv()
    env = IsaacLabGr00tEnv(env=fake)
    env.close()
    env.close()
    assert fake.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


# ---------------------------------------------------------------------------
# Shared instance and app launch
# ---------------------------------------------------------------------------


def test_shared_instance_cache_creates_once_and_closes_with_the_last_handle():
    created: list[FakeIsaacLabEnv] = []

    def make_env(task_id):
        created.append(FakeIsaacLabEnv())
        return created[-1]

    first = get_or_create_isaaclab_env("Task-A", make_env=make_env)
    second = get_or_create_isaaclab_env("Task-A", make_env=make_env)
    assert first is second and len(created) == 1
    with pytest.raises(RuntimeError, match="one simulation per process"):
        get_or_create_isaaclab_env("Task-B", make_env=make_env)
    release_isaaclab_env("Task-A")
    assert created[0].close_calls == 0
    release_isaaclab_env("Task-A")
    assert created[0].close_calls == 1 and not isaaclab_env._SHARED_ENVS
    with pytest.raises(KeyError):
        release_isaaclab_env("Task-A")


def test_train_and_eval_wrappers_share_one_simulation(monkeypatch):
    created: list[FakeIsaacLabEnv] = []

    def fake_make(task_id):
        created.append(FakeIsaacLabEnv())
        return created[-1]

    monkeypatch.setattr(isaaclab_env, "_make_isaaclab_env", fake_make)
    train = IsaacLabGr00tEnv(DEFAULT_TASK_ID)
    evaluation = IsaacLabGr00tEnv(DEFAULT_TASK_ID)
    assert len(created) == 1 and train._env is evaluation._env
    train.close()
    assert created[0].close_calls == 0
    evaluation.close()
    assert created[0].close_calls == 1


def test_launch_app_honors_the_outer_launcher_flag(monkeypatch):
    monkeypatch.setattr(isaaclab_env, "_APP_LAUNCHER", None)
    monkeypatch.setenv(APP_LAUNCHED_FLAG, "1")
    isaaclab_env.launch_app()  # must not try to import Isaac Sim
    assert isaaclab_env._APP_LAUNCHER is None


@pytest.mark.skipif(
    importlib.util.find_spec("isaaclab") is not None, reason="IsaacLab installed"
)
def test_launch_app_fails_fast_without_isaaclab(monkeypatch):
    monkeypatch.setattr(isaaclab_env, "_APP_LAUNCHER", None)
    monkeypatch.delenv(APP_LAUNCHED_FLAG, raising=False)
    with pytest.raises(ImportError, match="simulators/isaaclab/README.md"):
        isaaclab_env.launch_app()


# ---------------------------------------------------------------------------
# Learner sizing and registry wiring
# ---------------------------------------------------------------------------


def _sizing_cfg(task_id: str, *, image: int, action_dim: int, num_envs: int):
    return OmegaConf.create(
        {
            "env": {
                "env_name": f"isaaclab/{task_id}",
                "max_episode_steps": 448,
                "use_full_state": False,
                "critic_image_size": None,
                "image_keys": None,
                "use_init_states": True,
                "fixed_init_state_id": None,
            },
            "simulator": {"image_resolution": image, "action_dim": action_dim},
            "training": {"seed": 0, "num_envs": num_envs, "chunk_size": 16},
            "eval": {"eval_num_envs": 1},
        }
    )


@pytest.mark.parametrize(
    "task_id,image,action_dim,make_fake,video_keys,state_keys",
    [
        (
            FRANKA_TASK_ID,
            16,
            7,
            lambda: FakeFrankaStackEnv(image_size=16),
            FRANKA_VIDEO_KEYS,
            FRANKA_STATE_KEYS,
        ),
        (DEFAULT_TASK_ID, IMG, ACTION_DIM, FakeIsaacLabEnv, VIDEO_KEYS, STATE_KEYS),
    ],
)
@pytest.mark.parametrize("phase,num_envs", [("train", 1), ("train", 8), ("eval", 1)])
def test_the_learner_is_sized_from_the_task_mapping_without_starting_isaac_sim(
    monkeypatch,
    task_id,
    image,
    action_dim,
    make_fake,
    video_keys,
    state_keys,
    phase,
    num_envs,
):
    """Sizing builds no simulation, in or out of process, and matches the spaces of a live scene."""
    import torch

    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.runners.off_policy_launch import setup_dsrl_env
    from simulators.isaaclab import runtime as isaaclab_runtime

    def no_simulator(*args, **kwargs):
        raise AssertionError("sizing the learner started Isaac Sim")

    monkeypatch.setattr(isaaclab_env, "launch_app", no_simulator)
    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", no_simulator)
    cfg = _sizing_cfg(task_id, image=image, action_dim=action_dim, num_envs=num_envs)
    sized, remove_keys = setup_dsrl_env(
        cfg,
        None,
        None,
        None,
        torch.device("cpu"),
        isaaclab_runtime.IsaacLab(),
        phase=phase,
    )

    scene = IsaacLabGr00tEnv(
        task_id,
        env=make_fake(),
        mapping=mapping_for_task(task_id),
        image_size=image,
        action_dim=action_dim,
    )
    live = VectorActionChunkingWrapper(
        gym.vector.SyncVectorEnv([lambda: scene]), chunk_size=16, n_action_steps=1
    )
    assert sized.single_observation_space == live.single_observation_space
    assert (
        sized.single_action_space.shape
        == live.single_action_space.shape
        == (action_dim,)
    )
    assert sized.single_action_space.dtype == live.single_action_space.dtype
    assert remove_keys == [*video_keys, *state_keys] == scene.steering_remove_obs_keys
    assert sized.steering_key_mapping == scene.steering_key_mapping
    with pytest.raises(RuntimeError, match="only sizes the learner"):
        sized.reset()
    sized.close()
    live.close()


def test_isaaclab_has_no_env_builder():
    from isaac_rl_posttraining.envs import providers

    assert providers.get_env_builder(f"isaaclab/{DEFAULT_TASK_ID}") is None
    assert (
        providers.get_env_builder("isaac/x")
        is providers.get_env_builder("arena/drill_lift")
        is not None
    )


def test_provider_registration_tracks_the_installed_stack():
    from isaac_rl_posttraining.envs import providers

    assert providers._PROVIDERS["isaaclab"] == (
        "simulators.isaaclab.env",
        "IsaacLabGr00tEnv",
    )
    installed = importlib.util.find_spec("isaaclab") is not None
    assert ("isaaclab" in providers.available_providers()) == installed
    if installed:
        assert providers.get_provider("isaaclab") is IsaacLabGr00tEnv
    else:
        with pytest.raises(ImportError, match="isaaclab"):
            providers.get_provider("isaaclab")


# ---------------------------------------------------------------------------
# Hydra config group
# ---------------------------------------------------------------------------

TROCAR_OVERRIDES = [
    "base_model=mock",
    "simulator=isaaclab_trocar",
    f"env.env_name=isaaclab/{DEFAULT_TASK_ID}",
    f"env.max_episode_steps=${{simulator.max_episode_steps.{DEFAULT_TASK_ID}}}",
    "env.image_keys=[video.room_view,video.left_wrist_view,video.right_wrist_view]",
    "dsrl.noise_dim=28",
    "training.num_envs=1",
]


def test_simulator_config_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["base_model=mock", "simulator=isaaclab_trocar"],
        )
    assert cfg.simulator.provider == "isaaclab"
    assert list(cfg.simulator.tasks) == list(TROCAR_TASKS)
    max_episode_steps = {k: int(v) for k, v in cfg.simulator.max_episode_steps.items()}
    assert max_episode_steps == dict.fromkeys(TROCAR_TASKS, 1000)
    assert int(cfg.simulator.action_dim) == ACTION_DIM
    assert int(cfg.simulator.image_resolution) == IMG
    assert OmegaConf.to_container(cfg.simulator.gr00t_mapping) == TROCAR_GR00T_MAPPING


def test_documented_override_set_resolves():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=TROCAR_OVERRIDES)
    assert int(cfg.env.max_episode_steps) == 1000
    assert cfg.env.env_name == f"isaaclab/{DEFAULT_TASK_ID}"
    assert list(cfg.env.image_keys) == VIDEO_KEYS
    assert int(cfg.dsrl.noise_dim) == ACTION_DIM


def test_task_stage_diagnostic_is_reported_at_episode_end():
    """A mapping ``stage_fn`` (callable or module:function) yields info['task_stage'] when the episode ends."""
    import copy

    import torch

    mapping = copy.deepcopy(TROCAR_GR00T_MAPPING)
    mapping["stage_fn"] = lambda sim: torch.tensor([3])
    env = IsaacLabGr00tEnv(
        env=FakeIsaacLabEnv(terminate_at=1), mapping=mapping, max_episode_steps=100
    )
    env.reset()
    _, _, terminated, _, info = env.step(_model_action())
    assert terminated is True
    assert info["task_stage"] == 3


# ---------------------------------------------------------------------------
# Franka stack-cube mapping (LIBERO-style layout, derived state, explicit bounds)
# ---------------------------------------------------------------------------


def test_quat_wxyz_to_axis_angle_matches_the_libero_convention():
    half = np.sqrt(0.5)
    np.testing.assert_allclose(
        quat_wxyz_to_axis_angle(np.array([half, 0, 0, half])),
        [0, 0, np.pi / 2],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        quat_wxyz_to_axis_angle(np.array([1.0, 0, 0, 0])), [0, 0, 0]
    )
    batch = quat_wxyz_to_axis_angle(np.array([[half, half, 0, 0], [half, 0, half, 0]]))
    np.testing.assert_allclose(batch, [[np.pi / 2, 0, 0], [0, np.pi / 2, 0]], atol=1e-6)
    assert batch.dtype == np.float32


def test_franka_mapping_builds_the_libero_style_state():
    env = _franka_env()
    obs, _info = env.reset()
    assert set(obs) == {*FRANKA_VIDEO_KEYS, *FRANKA_STATE_KEYS, "state", "prompt"}
    assert obs["video.image"].shape == (16, 16, 3) and (obs["video.image"] == 3).all()
    assert (
        obs["video.wrist_image"].shape == (16, 16, 3)
        and (obs["video.wrist_image"] == 4).all()
    )
    expected = np.array(
        [0.5, -0.1, 0.2, 0.0, 0.0, np.pi / 2, 0.04, -0.04], dtype=np.float32
    )
    np.testing.assert_allclose(obs["state"], expected, atol=1e-6)
    for key, width in zip(FRANKA_STATE_KEYS, (1, 1, 1, 1, 1, 1, 2)):
        assert obs[key].shape == (width,) and obs[key].dtype == np.float32
    np.testing.assert_allclose(obs["state.yaw"], [np.pi / 2], atol=1e-6)
    assert obs["prompt"].startswith("Stack the red block")
    assert env.steering_key_mapping["image"] == FRANKA_VIDEO_KEYS
    assert env.steering_remove_obs_keys == [*FRANKA_VIDEO_KEYS, *FRANKA_STATE_KEYS]
    assert env.observation_space.contains(
        {k: v for k, v in obs.items() if k != "prompt"}
    )


def test_franka_action_bounds_gripper_sign_and_no_padding():
    env = _franka_env()
    np.testing.assert_allclose(
        env.action_space.low,
        FRANKA_STACK_GR00T_MAPPING["action_bounds"]["low"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        env.action_space.high,
        FRANKA_STACK_GR00T_MAPPING["action_bounds"]["high"],
        rtol=1e-6,
    )
    env.reset()
    env.step(
        np.array(
            [
                [0.01, -0.02, 0.03, 0.0, 0.0, 0.04, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2],
            ],
            dtype=np.float32,
        )
    )
    executed = env._env.actions
    assert len(executed) == 2 and executed[0].shape == (1, 7)
    np.testing.assert_allclose(
        executed[0][0].numpy(), [0.01, -0.02, 0.03, 0.0, 0.0, 0.04, 1.0]
    )
    assert executed[1][0, 6].item() == -1.0


def test_stage_terms_count_the_flags_that_hold():
    env = _franka_env(terminate_at=1, flags=(True, True, False))
    env.reset()
    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert terminated is True and info["task_stage"] == 2


def test_stage_is_read_before_isaaclab_auto_resets_a_finished_episode():
    """IsaacLab returns the post-reset observation on a terminal step; the stage must come from before it."""
    env = _franka_env(
        terminate_at=3, flags=(True, True, True), reset_flags=(False, False, False)
    )
    env.reset()
    _, _, terminated, _, info = env.step(np.zeros((5, 7), dtype=np.float32))
    assert terminated is True and info["num_steps"] == 3 and info["task_stage"] == 3
    # A client-side cap does not reset the simulation, so the live flags are the stage.
    env = _franka_env(flags=(True, False, False), max_episode_steps=2)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros((4, 7), dtype=np.float32))
    assert terminated is False and truncated is True and info["task_stage"] == 1


def test_apply_cfg_overrides_sets_nested_paths_and_rejects_unknown_ones(monkeypatch):
    cam = type("Cam", (), {"height": 200, "width": 200})()
    robot = type("Robot", (), {"spawn": type("Spawn", (), {"usd_path": "old"})()})()
    cfg = type(
        "Cfg", (), {"scene": type("Scene", (), {"table_cam": cam, "robot": robot})()}
    )()
    monkeypatch.setattr(
        isaaclab_env,
        "_asset_roots",
        lambda: {"ISAACLAB_NUCLEUS_DIR": "https://assets/Isaac/IsaacLab"},
    )
    isaaclab_env._apply_cfg_overrides(
        cfg,
        {
            "scene.table_cam.height": 256,
            "scene.table_cam.width": 256,
            "scene.robot.spawn.usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
        },
    )
    assert (cam.height, cam.width) == (256, 256)
    assert (
        robot.spawn.usd_path
        == "https://assets/Isaac/IsaacLab/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
    )
    with pytest.raises(AttributeError, match="scene.table_cam.depth"):
        isaaclab_env._apply_cfg_overrides(cfg, {"scene.table_cam.depth": 1})


def test_env_cfg_overrides_reach_the_default_factory(monkeypatch):
    seen: dict[str, object] = {}

    def fake_make(task_id, cfg_overrides=None, num_envs=1):
        seen.update(task_id=task_id, cfg_overrides=cfg_overrides)
        return FakeFrankaStackEnv()

    monkeypatch.setattr(isaaclab_env, "_make_isaaclab_env", fake_make)
    env = IsaacLabGr00tEnv(
        FRANKA_TASK_ID, mapping=FRANKA_STACK_GR00T_MAPPING, action_dim=7, image_size=16
    )
    assert seen == {
        "task_id": FRANKA_TASK_ID,
        "cfg_overrides": FRANKA_STACK_GR00T_MAPPING["env_cfg_overrides"],
    }
    env.close()


def test_task_mapping_registry():
    assert mapping_for_task(DEFAULT_TASK_ID) is TROCAR_GR00T_MAPPING
    assert mapping_for_task(FRANKA_TASK_ID) is FRANKA_STACK_GR00T_MAPPING
    with pytest.raises(KeyError, match="register_task_mapping"):
        mapping_for_task("Isaac-Nope-v0")
    with pytest.raises(ValueError, match="unknown transform"):
        register_task_mapping(
            "Isaac-Bad-v0",
            {
                **FRANKA_STACK_GR00T_MAPPING,
                "states": {
                    "state.x": {"term": "eef_pos", "transform": "nope", "slice": [0, 1]}
                },
            },
        )
    with pytest.raises(ValueError, match="action_bounds or an action_term"):
        register_task_mapping(
            "Isaac-Bad-v0", {**FRANKA_STACK_GR00T_MAPPING, "action_bounds": None}
        )
    try:
        register_task_mapping("Isaac-New-v0", FRANKA_STACK_GR00T_MAPPING)
        assert mapping_for_task("Isaac-New-v0") == FRANKA_STACK_GR00T_MAPPING
    finally:
        TASK_MAPPINGS.pop("Isaac-New-v0", None)


def test_franka_simulator_config_composes_and_mirrors_the_code_default():
    doc = yaml.safe_load(FRANKA_YAML.read_text(encoding="utf-8"))
    assert doc["gr00t_mapping"] == FRANKA_STACK_GR00T_MAPPING and doc["tasks"] == list(
        FRANKA_STACK_TASKS
    )
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["base_model=mock", "simulator=isaaclab_franka_stack"],
        )
    assert cfg.simulator.provider == "isaaclab" and int(cfg.simulator.action_dim) == 7
    assert int(cfg.simulator.image_resolution) == 256
    assert int(cfg.simulator.max_episode_steps[FRANKA_TASK_ID]) == 448
    assert (
        OmegaConf.to_container(cfg.simulator.gr00t_mapping)
        == FRANKA_STACK_GR00T_MAPPING
    )


# ---------------------------------------------------------------------------
# Batched view: N scenes of one simulation behind the batched vector backend
# ---------------------------------------------------------------------------

from simulators.isaaclab.env import IsaacLabBatchedGr00tEnv


def _batched_env(n=3, **kw) -> tuple[IsaacLabBatchedGr00tEnv, FakeFrankaStackEnv]:
    fake = FakeFrankaStackEnv(
        num_envs=n,
        **{
            k: v
            for k, v in kw.items()
            if k in ("terminate_at", "terminate_scene", "flags")
        },
    )
    env = IsaacLabBatchedGr00tEnv(
        FRANKA_TASK_ID,
        num_envs=n,
        env=fake,
        mapping=FRANKA_STACK_GR00T_MAPPING,
        action_dim=7,
        image_size=16,
        max_episode_steps=kw.get("max_episode_steps", 448),
    )
    return env, fake


def test_batched_view_observations_and_spaces():
    env, fake = _batched_env(n=3)
    obs, info = env.reset(seed=4)
    assert env.num_envs == 3 and env.observation_space.contains(obs) and info == {}
    assert obs["video.image"].shape == (3, 16, 16, 3) and obs["state"].shape == (3, 8)
    np.testing.assert_allclose(
        obs["state"][:, 2], [0.2, 1.2, 2.2]
    )  # scene i carries z = 0.2 + i
    np.testing.assert_allclose(obs["state.yaw"][:, 0], [np.pi / 2] * 3, atol=1e-6)
    assert env.single_action_space.shape == (7,) and env.action_space.shape == (3, 7)
    assert env.steering_key_mapping["image"] == FRANKA_VIDEO_KEYS
    assert fake.reset_env_ids == [None]


def test_batched_step_routes_actions_flags_and_gripper_sign():
    env, fake = _batched_env(n=3, terminate_at=2, terminate_scene=1)
    env.reset()
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 0] = [0.01, 0.02, 0.03]
    actions[:, 6] = [0.3, -0.2, 0.0]
    obs, reward, terminated, truncated, infos = env.step(actions)
    executed = fake.actions[-1]
    assert executed.shape == (3, 7)
    np.testing.assert_allclose(executed[:, 0].numpy(), [0.01, 0.02, 0.03])
    np.testing.assert_allclose(executed[:, 6].numpy(), [1.0, -1.0, 0.0])
    assert not terminated.any() and not truncated.any() and reward.sum() == 0
    obs, reward, terminated, truncated, infos = env.step(actions)
    assert terminated.tolist() == [False, True, False] and reward.tolist() == [
        0.0,
        1.0,
        0.0,
    ]
    assert (
        infos[1]["is_success"] is True
        and infos[1]["task_stage"] == 2
        and infos[0]["is_success"] is False
    )
    assert obs["state"].shape == (3, 8)


def test_batched_client_cap_truncates_per_scene_and_masked_reset_restarts_only_those():
    env, fake = _batched_env(n=2, max_episode_steps=2)
    env.reset()
    zeros = np.zeros((2, 7), dtype=np.float32)
    env.step(zeros)
    _, _, terminated, truncated, infos = env.step(zeros)
    assert truncated.tolist() == [True, True] and not terminated.any()
    assert (
        infos[0]["task_stage"] == 2
    )  # live flags: the simulator did not reset a capped scene
    obs, _ = env.reset(options={"reset_mask": np.array([True, False])})
    assert fake.reset_env_ids[-1] == [0]
    assert obs["state"].shape == (2, 8)
    with pytest.raises(ValueError, match="actions must be"):
        env.step(np.zeros(7, dtype=np.float32))


def test_batched_view_behind_the_adapter_autoresets_finished_scenes():
    from isaac_rl_posttraining.envs.vector_backends import build_vector_env

    env, fake = _batched_env(n=2, terminate_at=3, terminate_scene=0)
    vec = build_vector_env("batched", batched_env=env, num_envs=2)
    vec.reset()
    zeros = np.zeros((2, 7), dtype=np.float32)
    for _ in range(2):
        vec.step(zeros)
    _, reward, terminated, _truncated, _ = vec.step(zeros)
    assert terminated.tolist() == [True, False]
    # The adapter's NEXT_STEP autoreset restarts scene 0 through the masked reset.
    _, reward, terminated, _truncated, _ = vec.step(zeros)
    assert (
        fake.reset_env_ids[-1] == [0]
        and not terminated.any()
        and reward[0] == pytest.approx(0.0)
    )
    vec.close()


def test_batched_view_ends_episodes_when_the_eval_view_used_the_simulation(monkeypatch):
    """The scene-0 evaluation view shares the simulation; its steps invalidate the batched episodes."""
    fake = FakeFrankaStackEnv(num_envs=2)
    monkeypatch.setattr(
        isaaclab_env,
        "_make_isaaclab_env",
        lambda task_id, cfg_overrides=None, num_envs=1: fake,
    )
    train = IsaacLabBatchedGr00tEnv(
        FRANKA_TASK_ID,
        num_envs=2,
        mapping=FRANKA_STACK_GR00T_MAPPING,
        action_dim=7,
        image_size=16,
    )
    evaluation = IsaacLabGr00tEnv(
        FRANKA_TASK_ID, mapping=FRANKA_STACK_GR00T_MAPPING, action_dim=7, image_size=16
    )
    assert train._env is evaluation._env
    train.reset()
    zeros = np.zeros((2, 7), dtype=np.float32)
    train.step(zeros)
    evaluation.reset()
    evaluation.step(
        np.zeros(7, dtype=np.float32)
    )  # scene 0 driven; scene 1 receives zeros
    assert fake.actions[-1].shape == (2, 7) and (fake.actions[-1][1] == 0).all()
    obs, _reward, terminated, truncated, infos = train.step(zeros)
    assert truncated.tolist() == [True, True] and not terminated.any()
    assert all(info["disconnected"] for info in infos) and obs["state"].shape == (2, 8)
    # Back to normal once the adapter resets: a plain step again.
    train.reset(options={"reset_mask": np.array([True, True])})
    _, _, _, truncated, infos = train.step(zeros)
    assert not truncated.any() and "disconnected" not in infos[0]
    train.close()
    evaluation.close()


def test_shared_simulation_refuses_a_larger_view_than_it_hosts(monkeypatch):
    monkeypatch.setattr(
        isaaclab_env,
        "_make_isaaclab_env",
        lambda task_id, cfg_overrides=None, num_envs=1: FakeFrankaStackEnv(
            num_envs=num_envs
        ),
    )
    view = IsaacLabGr00tEnv(
        FRANKA_TASK_ID, mapping=FRANKA_STACK_GR00T_MAPPING, action_dim=7, image_size=16
    )
    with pytest.raises(RuntimeError, match="hosts 1 scene"):
        IsaacLabBatchedGr00tEnv(
            FRANKA_TASK_ID,
            num_envs=4,
            mapping=FRANKA_STACK_GR00T_MAPPING,
            action_dim=7,
            image_size=16,
        )
    view.close()


def test_registered_task_id_follows_isaac_lab_3_0_0_contrib(monkeypatch) -> None:
    task = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
    contrib = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"
    monkeypatch.setattr(gym, "registry", {task: object()})
    assert isaaclab_env._registered_task_id(task) == task
    monkeypatch.setattr(gym, "registry", {contrib: object()})
    assert isaaclab_env._registered_task_id(task) == contrib
    monkeypatch.setattr(gym, "registry", {})
    assert isaaclab_env._registered_task_id(task) == task
