# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Factory's simulator entry, composed with steering policies, without a simulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs import LatentSpaceEnv, ResidualFlowSpaceEnv
from isaac_rl_posttraining.envs.providers import get_simulator_entry
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    GymWorkerSpec,
    RolloutWorkerContext,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.steering_rollouts import (
    steering_policy_spec,
    steering_worker_spec,
)
from simulators.g1_arena import runtime as arena_runtime
from simulators.g1_arena.env import ArenaEnv

ENTRY = arena_runtime.G1Arena()
CONTEXT = RolloutWorkerContext(
    worker_index=0, num_rollout_workers=1, environment_indices=(0,), device="cpu"
)


class _RecordedClient:
    """Stands in for the Arena clients: records the constructor arguments instead of connecting."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def act(self, observation, actor_state, deterministic):
        del actor_state, deterministic
        return torch.zeros(observation["state"].shape[0], 2), None


class _LatentSampler:
    include_reference_action = False

    def sample_action(self):
        raise AssertionError("spec construction must not sample actions")


class _DinoWrapper:
    dinov2_model = object()

    def observation(self, observation):
        return observation


def _config(num_envs: int, *, runtime: str = "native"):
    return OmegaConf.create(
        {
            "runtime": {"name": runtime, "num_rollout_workers": None},
            "base_model": {"name": "gr00t"},
            "training": {
                "num_envs": num_envs,
                "seed": 0,
                "policy_decode_batch_size": 16,
            },
            "env": {
                "env_name": "arena/drill_lift",
                "max_episode_steps": 17,
                "image_keys": ["ego_view_image"],
                "success_reward": 0.0,
                "failure_reward": -1.0,
                "step_reward": 0.0,
            },
            "simulator": {"image_resolution": 0, "action_dim": 35},
            "dsrl": {"action_exec_len": 8, "use_vlm_features": False},
            "model": {"dinov2_model": "unused"},
        }
    )


def _spec(
    num_envs: int, *, server_url: str | None = "tcp://127.0.0.1:7300"
) -> EnvironmentSpec:
    return EnvironmentSpec(
        env_name="arena/drill_lift",
        num_envs=num_envs,
        max_episode_steps=17,
        seed=0,
        server_url=server_url,
        action_dim=35,
        image_resolution=0,
    )


def _policy_spec(cfg, **overrides):
    overrides.setdefault("phase", "train")
    latent_env = overrides.pop("latent_env", _LatentSampler())
    return steering_policy_spec(
        cfg,
        actor=_Actor(),
        base_model=MockBaseModelPolicy(),
        controller_env=_DinoWrapper(),
        latent_env=latent_env,
        action_modulator=None,
        steering_key_mapping=dict(ENTRY.steering_key_mapping(_spec(1))),
        dino_image_keys=ENTRY.steering_key_mapping(_spec(1))["image"],
        decode_batch_size=(
            None
            if overrides["phase"] == "eval"
            else cfg.training.policy_decode_batch_size
        ),
        **overrides,
    )


def test_registry_resolves_the_arena_entry_lazily() -> None:
    entry = get_simulator_entry("arena/drill_lift")

    assert isinstance(entry, arena_runtime.G1Arena)
    assert get_simulator_entry("isaac/drill_lift") is entry
    assert entry.evaluation == "dedicated"
    assert entry.supports_chunked_step
    assert entry.server_url_variables == ("ARENA_SERVER_URL", "ARENA_EVAL_SERVER_URL")
    assert entry.steering_key_mapping(_spec(1)) == {
        "state": "state",
        "language": None,
        "image": ["ego_view_image"],
    }


def test_single_arena_environment_uses_a_chunk_stepping_gym_worker(monkeypatch) -> None:
    monkeypatch.setattr(arena_runtime, "ArenaEnv", _RecordedClient)
    spec = steering_worker_spec(
        ENTRY,
        _spec(1),
        "train",
        policy=_policy_spec(_config(1)),
        action_exec_len=8,
        reward_transform=None,
        configured_workers=None,
    )

    assert isinstance(spec, GymWorkerSpec)
    assert spec.chunked_step
    assert spec.supports_batched_policy
    assert (spec.entry, spec.environment, spec.phase) == (ENTRY, _spec(1), "train")
    # The worker builds the scalar client from the spec's server address and geometry.
    env = spec.make_environment(CONTEXT)
    assert env.kwargs == {
        "server_url": "tcp://127.0.0.1:7300",
        "image_resolution": 0,
        "action_dim": 35,
        "max_episode_steps": 17,
    }
    assert resolve_num_rollout_workers(spec, num_envs=1, configured=None) == 1
    assert spec.policy.base_model.prebuilt is not None
    assert spec.policy.max_decode_batch_size == 16


def test_many_arena_environments_use_one_batched_worker(monkeypatch) -> None:
    monkeypatch.setattr(arena_runtime, "ArenaBatchedEnv", _RecordedClient)
    spec = steering_worker_spec(
        ENTRY,
        _spec(32),
        "train",
        policy=_policy_spec(_config(32)),
        action_exec_len=8,
        reward_transform=None,
        configured_workers=None,
    )

    assert isinstance(spec, BatchedWorkerSpec)
    assert spec.chunks_per_episode == 3
    assert ENTRY.worker_layout(32).workers == 1 and ENTRY.worker_layout(32).batched
    env = spec.make_environment(RolloutWorkerContext(0, 1, tuple(range(32)), "cpu"))
    assert (
        env.kwargs["num_envs"] == 32
        and env.kwargs["server_url"] == "tcp://127.0.0.1:7300"
    )
    assert resolve_num_rollout_workers(spec, num_envs=32, configured=None) == 1
    with pytest.raises(ValueError, match="batched worker"):
        steering_worker_spec(
            ENTRY,
            _spec(8),
            "train",
            policy=_policy_spec(_config(8)),
            action_exec_len=8,
            reward_transform=None,
            configured_workers=2,
        )


def test_arena_environments_require_a_server_address() -> None:
    with pytest.raises(ValueError, match="ARENA_EVAL_SERVER_URL"):
        ENTRY.make_environment(_spec(1, server_url=None), "eval", CONTEXT)


TWO_SERVERS = "tcp://127.0.0.1:7300,tcp://127.0.0.1:7301"


def test_two_arena_servers_give_one_batched_worker_each(monkeypatch) -> None:
    """A phase spread over two servers: two batched workers, each connected to its own server with half
    the environments, so one server steps while the other's frames decode."""
    monkeypatch.setattr(arena_runtime, "ArenaBatchedEnv", _RecordedClient)
    two = _spec(8, server_url=TWO_SERVERS)
    assert (
        two.server_addresses() == ("tcp://127.0.0.1:7300", "tcp://127.0.0.1:7301")
        and two.num_servers == 2
    )
    assert ENTRY.worker_layout(8, num_servers=2) == arena_runtime.WorkerLayout(
        workers=2, batched=True
    )
    spec = steering_worker_spec(
        ENTRY,
        two,
        "train",
        policy=_policy_spec(_config(8)),
        action_exec_len=8,
        reward_transform=None,
        configured_workers=None,
    )
    assert isinstance(spec, BatchedWorkerSpec)
    second = spec.make_environment(RolloutWorkerContext(1, 2, (4, 5, 6, 7), "cpu"))
    assert (
        second.kwargs["server_url"] == "tcp://127.0.0.1:7301"
        and second.kwargs["num_envs"] == 4
    )
    first = spec.make_environment(RolloutWorkerContext(0, 2, (0, 1, 2, 3), "cpu"))
    assert first.kwargs["server_url"] == "tcp://127.0.0.1:7300"
    assert (
        resolve_num_rollout_workers(
            spec, num_envs=8, configured=None, batched_workers=2
        )
        == 2
    )
    assert (
        resolve_num_rollout_workers(spec, num_envs=8, configured=2, batched_workers=2)
        == 2
    )
    # The worker count is the server count: one worker for two servers has no server for its second half.
    with pytest.raises(ValueError, match="2 addresses were exported for 1 workers"):
        spec.make_environment(RolloutWorkerContext(0, 1, tuple(range(8)), "cpu"))
    with pytest.raises(
        ValueError, match="placement.envs.workers=1 does not match the 2 server"
    ):
        steering_worker_spec(
            ENTRY,
            two,
            "train",
            policy=_policy_spec(_config(8)),
            action_exec_len=8,
            reward_transform=None,
            configured_workers=1,
        )


def test_several_arena_servers_split_the_environments_equally() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        ENTRY.worker_layout(9, num_servers=2)
    with pytest.raises(ValueError, match="at least two environments"):
        ENTRY.worker_layout(2, num_servers=2)
    with pytest.raises(ValueError, match="num_servers must be a positive integer"):
        ENTRY.worker_layout(8, num_servers=0)
    with pytest.raises(ValueError, match="one server; got 2 addresses"):
        ENTRY.make_environment(_spec(1, server_url=TWO_SERVERS), "train", CONTEXT)


def test_a_worker_count_must_match_the_arena_servers() -> None:
    """One batched worker per server: a placement asking for another count fails in the plan."""
    assert ENTRY.worker_layout(8, num_servers=2, workers=2).workers == 2
    with pytest.raises(
        ValueError, match=r"placement.envs.workers=4 does not match the 1 server"
    ):
        ENTRY.worker_layout(8, workers=4)


def test_evaluation_policy_can_be_deterministic_or_native() -> None:
    steered = _policy_spec(_config(1), deterministic_evaluation=True)
    native = _policy_spec(_config(1), native=True)

    assert steered.deterministic_evaluation and not steered.native
    assert native.native and not native.deterministic_evaluation


def test_reference_decode_uses_evaluation_preprocessing_only_when_evaluating() -> None:
    assert (
        _policy_spec(_config(1), phase="train").reference_image_transform_mode
        == "train"
    )
    assert (
        _policy_spec(_config(1), phase="eval").reference_image_transform_mode == "eval"
    )


def test_non_native_runtime_rebuilds_the_frozen_policy_per_worker() -> None:
    policy = _policy_spec(_config(1, runtime="rlinf"))

    assert policy.base_model.prebuilt is None and policy.base_model.config is not None
    assert policy.dino_wrapper is None
    assert policy.dino_model_name == "unused"


class _ChunkTransport:
    """Transport double for a server that executes whole chunks."""

    def __init__(
        self, *, supports_chunking: bool, stop_after: int | None = None
    ) -> None:
        self.supports_chunking = supports_chunking
        self.stop_after = stop_after
        self.step_calls: list[tuple[np.ndarray, bool]] = []

    def connect(self) -> None:
        # Test double: there is no server to connect to.
        pass

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass

    def spec(self) -> dict[str, int]:
        return {"num_envs": 1}

    def reset(self, seed=None, env_ids=None):
        return {
            "ego_view": np.zeros((4, 6, 3), dtype=np.uint8),
            "state": np.zeros(31, dtype=np.float32),
            "info": {"supports_action_chunking": self.supports_chunking},
        }

    def step(self, action, need_obs=True):
        actions = np.asarray(action)
        self.step_calls.append((actions, need_obs))
        requested = actions.shape[0] if actions.ndim == 2 else 1
        executed = (
            requested if self.stop_after is None else min(requested, self.stop_after)
        )
        return {
            "ego_view": np.zeros((4, 6, 3), dtype=np.uint8),
            "state": np.full(31, float(len(self.step_calls)), dtype=np.float32),
            "reward": 0.5,
            "done": self.stop_after is not None and executed >= self.stop_after,
            "truncated": False,
            "success": False,
            "num_steps": executed,
            "info": {},
        }


def test_step_chunk_uses_one_round_trip_on_a_chunking_server() -> None:
    transport = _ChunkTransport(supports_chunking=True, stop_after=3)
    env = ArenaEnv(transport=transport, image_resolution=0, max_episode_steps=100)
    env.reset()

    observation, rewards, terminated, truncated, info = env.step_chunk(
        np.zeros((8, 35), np.float32)
    )

    assert len(transport.step_calls) == 1
    assert transport.step_calls[0][0].shape == (8, 35)
    assert rewards == (0.0, 0.0, 0.5)
    assert terminated and not truncated
    assert info["num_steps"] == 3
    assert observation["state"].shape == (31,)


def test_step_chunk_falls_back_to_per_action_steps_without_server_chunking() -> None:
    transport = _ChunkTransport(supports_chunking=False)
    env = ArenaEnv(transport=transport, image_resolution=0, max_episode_steps=8)
    env.reset()

    _, rewards, terminated, truncated, _ = env.step_chunk(np.zeros((8, 35), np.float32))

    # Eight per-action steps; only the last requests the camera, and the
    # eighth hits the client-side cap.
    assert len(transport.step_calls) == 8
    assert [need_obs for _, need_obs in transport.step_calls] == [False] * 7 + [True]
    assert all(actions.shape == (35,) for actions, _ in transport.step_calls)
    assert rewards == (0.5,) * 8
    assert not terminated and truncated
    assert env.need_obs

    transport = _ChunkTransport(supports_chunking=False, stop_after=1)
    env = ArenaEnv(transport=transport, image_resolution=0, max_episode_steps=100)
    env.reset()
    _, rewards, terminated, _, _ = env.step_chunk(np.zeros((8, 35), np.float32))
    assert len(transport.step_calls) == 1 and rewards == (0.5,) and terminated


def test_arena_worker_context_requires_one_environment() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ENTRY.make_environment(
            _spec(1),
            "train",
            RolloutWorkerContext(
                worker_index=0,
                num_rollout_workers=1,
                environment_indices=(0, 1),
                device="cpu",
            ),
        )


def _spaces():
    import gymnasium as gym

    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (31,), np.float32),
            "dino_embedding": gym.spaces.Box(-np.inf, np.inf, (768,), np.float32),
        }
    )
    return observation_space, gym.spaces.Box(-1.0, 1.0, (35,), np.float32)


def test_steering_policy_spec_accepts_the_real_latent_environments() -> None:
    """Plain DSRL carries no reference action; residual steering may attach one under its key."""
    observation_space, action_space = _spaces()
    base_model = MockBaseModelPolicy(action_horizon=16, action_dim=35, seed=0)
    plain = LatentSpaceEnv(
        observation_space,
        action_space,
        base_model,
        noise_dim=35,
        chunk_size=40,
        use_vlm_features=False,
    )
    assert _policy_spec(_config(1), latent_env=plain).reference_action_key is None

    class _ReferenceBaseModel(MockBaseModelPolicy):
        def infer_reference(self, observations, *, image_transform_mode="eval"):
            raise AssertionError("spec construction must not decode reference actions")

    residual = ResidualFlowSpaceEnv(
        observation_space,
        action_space,
        _ReferenceBaseModel(action_horizon=16, action_dim=35, seed=0),
        noise_dim=35,
        residual_action_dim=35,
        chunk_size=40,
        noise_action_bound=1.0,
        residual_action_bound=0.01,
        warmup_exploration="uniform",
        use_vlm_features=False,
        include_reference_action=True,
    )
    assert (
        _policy_spec(_config(1), latent_env=residual).reference_action_key
        == "reference_action"
    )


def test_groot_observation_splits_policy_state_in_modality_order() -> None:
    import numpy as np

    from simulators.g1_arena.runtime import _groot_observation

    state = np.arange(2 * 31, dtype=np.float32).reshape(2, 31)
    image = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    observation = _groot_observation({"ego_view_image": image, "state": state})
    assert observation["video.ego_view"] is image
    groups = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
    rebuilt = np.concatenate(
        [observation[f"state.{group}"] for group in groups], axis=-1
    )
    np.testing.assert_array_equal(rebuilt, state)
    assert [observation[f"state.{group}"].shape[-1] for group in groups] == [
        7,
        7,
        7,
        7,
        3,
    ]
