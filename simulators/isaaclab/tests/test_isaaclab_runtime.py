# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab's simulator entry without Isaac Sim: a fake scene child and the mock base model."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs.providers import get_simulator_entry
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, WorkerLayout
from isaac_rl_posttraining.envs.steering_wrappers import LatentSpaceEnv
from isaac_rl_posttraining.runtimes import (
    BatchedEpisodeRolloutWorker,
    BatchedWorkerSpec,
    GymRolloutWorker,
    GymWorkerSpec,
    RolloutRequest,
    RolloutWorkerContext,
    SparseOutcomeReward,
)
from isaac_rl_posttraining.runtimes.base import RolloutInfrastructureError
from isaac_rl_posttraining.runtimes.steering_rollouts import (
    steering_policy_spec,
    steering_worker_spec,
)
from simulators.isaaclab import runtime as isaaclab_runtime
from simulators.isaaclab.env import FRANKA_STACK_TASKS, mapping_for_task
from simulators.isaaclab.runtime import IsaacLabBatchedScenesEnv, IsaacLabSceneEnv
from simulators.isaaclab.subprocess_env import IsaacLabChildError, startup_delay_seconds

TASK_ID = FRANKA_STACK_TASKS[0]
ENV_NAME = f"isaaclab/{TASK_ID}"
IMAGE = 8
ACTION_DIM = 7
VIDEO_KEYS = ["video.image", "video.wrist_image"]
ENTRY = isaaclab_runtime.IsaacLab()
CONTEXT = RolloutWorkerContext(0, 1, (0,), "cpu")
SCENE_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "video.image": gym.spaces.Box(0, 255, (IMAGE, IMAGE, 3), np.uint8),
        "video.wrist_image": gym.spaces.Box(0, 255, (IMAGE, IMAGE, 3), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
    }
)
SCENE_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def act(self, observation, actor_state, deterministic):
        del actor_state, deterministic
        batch_size = observation["state"].shape[0]
        return torch.zeros(batch_size, ACTION_DIM) + self.weight, None


class _LatentSampler(LatentSpaceEnv):
    include_reference_action = False
    reference_action_key = "reference_action"

    def __init__(self) -> None:
        # No spaces to probe: the rollouts only sample warmup actions.
        gym.Env.__init__(self)

    def sample_action(self) -> np.ndarray:
        return np.zeros((1, ACTION_DIM), dtype=np.float32)


class _FakeChild:
    """A simulator child: batched arrays in, batched arrays out, success after three steps."""

    single_observation_space = SCENE_OBSERVATION_SPACE
    single_action_space = SCENE_ACTION_SPACE
    video_keys = VIDEO_KEYS
    state_keys: tuple[str, ...] = ()

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.seeds: list[int | None] = []
        self.actions: list[np.ndarray] = []
        self.closed = False
        self._steps = 0

    def _observation(self) -> dict[str, np.ndarray]:
        frames = np.zeros((self.num_envs, IMAGE, IMAGE, 3), dtype=np.uint8)
        return {
            "video.image": frames,
            "video.wrist_image": frames,
            "state": np.full((self.num_envs, 8), float(self._steps), dtype=np.float32),
        }

    def reset(self, seed=None):
        self.seeds.append(seed)
        self._steps = 0
        return self._observation(), {}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, ACTION_DIM)
        self.actions.append(actions.copy())
        self._steps += 1
        done = self._steps >= 3
        infos = [
            {"success": done, "is_success": done, "num_steps": 1}
            for _ in range(self.num_envs)
        ]
        return (
            self._observation(),
            np.full(self.num_envs, float(done)),
            np.full(self.num_envs, done),
            np.zeros(self.num_envs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        self.closed = True


def _cfg(runtime_name: str, base_model: str = "mock"):
    return OmegaConf.create(
        {
            "runtime": {"name": runtime_name},
            "base_model": {
                "name": base_model,
                "action_horizon": 16,
                "action_dim": ACTION_DIM,
            },
            "simulator": {"image_resolution": IMAGE, "action_dim": ACTION_DIM},
            "env": {
                "env_name": ENV_NAME,
                "max_episode_steps": 448,
                "image_keys": VIDEO_KEYS,
                "sparse_success_reward": True,
                "success_reward": 0.0,
                "failure_reward": -1.0,
                "step_reward": 0.0,
            },
            "training": {"seed": 17},
            "dsrl": {"action_exec_len": 16, "use_vlm_features": False},
            "model": {"dinov2_model": None},
        }
    )


def _spec(
    num_envs: int = 1, *, env_name: str = ENV_NAME, seed: int | None = 17
) -> EnvironmentSpec:
    return EnvironmentSpec(
        env_name=env_name,
        num_envs=num_envs,
        max_episode_steps=448,
        seed=seed,
        action_dim=ACTION_DIM,
        image_resolution=IMAGE,
    )


def _policy_spec(cfg, *, base_model=None, controller_env=None, **overrides):
    overrides.setdefault("phase", "train")
    return steering_policy_spec(
        cfg,
        actor=_Actor(),
        base_model=object() if base_model is None else base_model,
        controller_env=object() if controller_env is None else controller_env,
        latent_env=_LatentSampler(),
        action_modulator=None,
        steering_key_mapping=dict(ENTRY.steering_key_mapping(_spec())),
        dino_image_keys=VIDEO_KEYS,
        decode_batch_size=None,
        **overrides,
    )


def _worker_spec(cfg, policy, *, spec=None):
    return steering_worker_spec(
        ENTRY,
        _spec() if spec is None else spec,
        "train",
        policy=policy,
        action_exec_len=16,
        reward_transform=SparseOutcomeReward(
            success_reward=0.0, failure_reward=-1.0, step_reward=0.0
        ),
        configured_workers=None,
    )


def test_registry_resolves_the_isaaclab_entry_on_its_exact_segment() -> None:
    entry = get_simulator_entry(ENV_NAME)

    assert isinstance(entry, isaaclab_runtime.IsaacLab)
    assert get_simulator_entry("isaaclab/Other-v0") is entry
    assert entry.evaluation == "shared"
    assert entry.base_models == ("gr00t", "gr00t_n15", "gr00t_n17")
    assert entry.worker_layout(4).workers == 1 and entry.worker_layout(4).batched
    assert entry.worker_layout(1).workers == 1 and not entry.worker_layout(1).batched
    # placement.envs.workers splits the scenes over several simulations (one Isaac Sim each).
    assert entry.worker_layout(64, workers=4) == WorkerLayout(workers=4, batched=True)
    assert entry.worker_layout(4, workers=4) == WorkerLayout(workers=4, batched=False)
    assert entry.worker_layout(64, workers=1) == WorkerLayout(workers=1, batched=True)
    with pytest.raises(ValueError, match="do not split evenly"):
        entry.worker_layout(10, workers=4)
    assert entry.steering_key_mapping(_spec()) == {
        "state": "state",
        "language": None,
        "image": VIDEO_KEYS,
    }
    assert entry.instruction(_spec()) == mapping_for_task(TASK_ID)["task_description"]


def test_entry_owns_one_scene_per_worker_under_a_distributed_runtime() -> None:
    spec = _worker_spec(_cfg("rlinf"), _policy_spec(_cfg("rlinf")))

    assert isinstance(spec, GymWorkerSpec)
    assert spec.supports_batched_policy
    assert not spec.chunked_step
    assert (spec.entry, spec.environment, spec.phase) == (ENTRY, _spec(), "train")
    policy = spec.policy
    assert policy.base_model.config is not None
    assert policy.steering_key_mapping == {
        "state": "state",
        "language": None,
        "image": VIDEO_KEYS,
    }
    assert policy.action_exec_len == 16
    assert policy.dino_image_keys == tuple(VIDEO_KEYS)
    assert spec.reward_transform(
        (1.0,), terminated=True, truncated=False, info={"success": True}
    ).value == pytest.approx(0.0)
    assert spec.reward_transform(
        (0.0,), terminated=False, truncated=True, info={"success": False}
    ).value == pytest.approx(-1.0)


def test_native_policy_spec_reuses_controller_models_and_evaluation_flags() -> None:
    class _DinoWrapper:
        dinov2_model = object()

        def observation(self, observation):
            return observation

    base_model = object()
    dino = _DinoWrapper()
    evaluation = _policy_spec(
        _cfg("native"),
        base_model=base_model,
        controller_env=dino,
        deterministic_inference=True,
        sample_evaluation_actions=True,
    )

    assert evaluation.base_model.prebuilt is base_model
    assert evaluation.dino_wrapper is dino
    assert evaluation.dino_model_name is None
    assert evaluation.deterministic_inference
    assert evaluation.sample_evaluation_actions
    training = _policy_spec(_cfg("rlinf"))
    assert not training.deterministic_inference
    assert not training.sample_evaluation_actions


def test_entry_rejects_names_outside_its_form_and_unknown_tasks() -> None:
    for env_name in ("arena/drill_lift", "isaaclab/"):
        with pytest.raises(ValueError, match="isaaclab/<Task-Id>"):
            ENTRY.make_environment(_spec(env_name=env_name), "train", CONTEXT)
    with pytest.raises(ValueError, match="register_task_mapping"):
        ENTRY.steering_key_mapping(_spec(env_name="isaaclab/Isaac-Unknown-v0"))
    with pytest.raises(ValueError, match="training.seed"):
        ENTRY.make_environment(_spec(seed=None), "train", CONTEXT)


def test_scene_environments_seed_by_environment_index_and_stagger_starts(
    monkeypatch,
) -> None:
    specs = []

    class _RecordingChild:
        single_observation_space = SCENE_OBSERVATION_SPACE
        single_action_space = SCENE_ACTION_SPACE
        video_keys = VIDEO_KEYS
        state_keys: tuple[str, ...] = ()

        def __init__(self, spec) -> None:
            self.num_envs = spec.num_envs
            specs.append(spec)

    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", _RecordingChild)

    scene = ENTRY.make_environment(
        _spec(4), "train", RolloutWorkerContext(0, 4, (0,), "cpu")
    )
    assert isinstance(scene, IsaacLabSceneEnv)
    assert scene.observation_space is SCENE_OBSERVATION_SPACE
    assert scene.action_space is SCENE_ACTION_SPACE
    ENTRY.make_environment(_spec(4), "eval", RolloutWorkerContext(2, 4, (5,), "cuda:1"))
    assert [spec.seed for spec in specs] == [17, 22]
    assert {spec.num_envs for spec in specs} == {1}
    assert specs[0].task_id == TASK_ID and specs[0].max_episode_steps == 448
    assert specs[0].startup_delay_seconds == pytest.approx(0.0)
    assert specs[1].startup_delay_seconds == startup_delay_seconds(2) > 0.0
    assert specs[1].kit_log_path != specs[0].kit_log_path
    with pytest.raises(ValueError, match="one-scene"):
        IsaacLabSceneEnv(_FakeChild(num_envs=2))


def test_a_workers_scenes_share_one_simulation_with_the_worker_seed_offset(
    monkeypatch,
) -> None:
    class _RecordingChild:
        # Test double, shared on purpose.
        instances: list[_RecordingChild] = []  # noqa: RUF012

        single_observation_space = SCENE_OBSERVATION_SPACE
        single_action_space = SCENE_ACTION_SPACE
        video_keys = VIDEO_KEYS
        state_keys: tuple[str, ...] = ()

        def __init__(self, spec) -> None:
            self.spec = spec
            self.num_envs = spec.num_envs
            self.seeds: list[int | None] = []
            self.restarts = 0
            _RecordingChild.instances.append(self)

        def reset(self, seed=None):
            self.seeds.append(seed)
            return {"state": np.zeros((self.num_envs, 8), np.float32)}, {}

        def restart(self) -> None:
            self.restarts += 1

    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", _RecordingChild)

    env = ENTRY.make_scenes(
        _spec(16),
        RolloutWorkerContext(
            worker_index=2,
            num_rollout_workers=4,
            environment_indices=(8, 9, 10, 11),
            device="cuda:2",
        ),
    )

    assert isinstance(env, IsaacLabBatchedScenesEnv) and env.num_envs == 4
    child = _RecordingChild.instances[-1]
    assert (child.spec.task_id, child.spec.num_envs, child.spec.seed) == (
        TASK_ID,
        4,
        19,
    )
    assert child.spec.max_episode_steps is None, "the chunk horizon bounds the episode"
    assert (
        child.spec.startup_delay_seconds == startup_delay_seconds(2)
        and "worker2" in child.spec.kit_log_path
    )
    env.reset(seed=5)
    env.reset()
    assert child.seeds == [5 + 2 * 1_000_003, None]
    env.restart()
    assert child.restarts == 1
    assert issubclass(IsaacLabChildError, RolloutInfrastructureError)


def test_mock_base_model_collects_an_episode_through_a_fake_scene_child(
    monkeypatch,
) -> None:
    child = _FakeChild()
    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", lambda spec: child)
    spec = _worker_spec(_cfg("rlinf"), _policy_spec(_cfg("rlinf")))

    worker = spec(CONTEXT)
    assert isinstance(worker, GymRolloutWorker)
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="eval"))

    assert result.episodes_completed == 1 and result.env_steps == 3
    assert child.seeds == [None]
    assert all(action.shape == (1, ACTION_DIM) for action in child.actions)
    decision = result.decisions[0]
    assert decision.environment_index == 0
    assert decision.transition.terminated[-1] and not decision.transition.truncated[-1]
    assert decision.transition.steps == 3
    assert decision.transition.decision_reward(
        "sparse_outcome", aggregation="last"
    ) == pytest.approx(0.0)
    assert decision.transition.observation.arrays()["state"].shape == (8,)
    assert decision.action_chunk.execution_actions.shape == (1, 16, ACTION_DIM)
    worker.close()
    assert child.closed


def test_steering_scenes_share_one_batched_simulation_capped_at_the_episode_limit(
    monkeypatch,
) -> None:
    specs = []

    def _child(spec):
        specs.append(spec)
        return _FakeChild(num_envs=spec.num_envs)

    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", _child)
    env = ENTRY.make_environment(
        _spec(4), "train", RolloutWorkerContext(0, 1, (0, 1, 2, 3), "cpu")
    )

    assert isinstance(env, IsaacLabBatchedScenesEnv) and env.num_envs == 4
    assert env.single_observation_space is SCENE_OBSERVATION_SPACE
    assert env.single_action_space is SCENE_ACTION_SPACE
    (spec,) = specs
    assert (spec.num_envs, spec.seed, spec.max_episode_steps) == (4, 17, 448)
    assert "steering_train_worker0" in spec.kit_log_path


def test_mock_base_model_collects_batched_episodes_through_a_fake_multi_scene_child(
    monkeypatch,
) -> None:
    child = _FakeChild(num_envs=2)
    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", lambda spec: child)
    spec = _worker_spec(_cfg("rlinf"), _policy_spec(_cfg("rlinf")), spec=_spec(2))

    assert isinstance(spec, BatchedWorkerSpec)
    worker = spec(RolloutWorkerContext(0, 1, (0, 1), "cpu"))
    assert isinstance(worker, BatchedEpisodeRolloutWorker)
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="eval"))

    assert result.episodes_completed == 2
    assert all(action.shape == (2, ACTION_DIM) for action in child.actions)
    assert sorted(item.environment_index for item in result.decisions) == [0, 1]
    worker.close()
    assert child.closed


def test_a_scene_sizes_the_learner_as_one_gymnasium_environment() -> None:
    """An evaluation-only run sizes the learner from one scene wrapped in a vector env."""
    vector = gym.vector.SyncVectorEnv([lambda: IsaacLabSceneEnv(_FakeChild())])

    assert vector.single_observation_space == SCENE_OBSERVATION_SPACE
    assert vector.single_action_space == SCENE_ACTION_SPACE
    assert IsaacLabSceneEnv(_FakeChild()).steering_remove_obs_keys == VIDEO_KEYS
    vector.close()
