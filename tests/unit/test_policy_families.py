# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family dispatch: which policy, reward and update credit each algorithm family gets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.envs.reference_action import ReferenceActionEnv
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
from isaac_rl_posttraining.runtimes import (
    EditPolicySpec,
    GymWorkerSpec,
    RolloutRequest,
    RolloutWorkerContext,
    SparseOutcomeReward,
    SteeringPolicySpec,
)
from isaac_rl_posttraining.runtimes.policy_families import (
    decode_batch_size_for,
    rollout_worker_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
HORIZON, ENV_ACTION_DIM, FEAT_DIM = 3, 4, 2
ACTION_INDICES = (1, 3)
DEFAULTS = [0.5, 0.0, -0.5, 0.0]


class _RawEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "ego_view_image": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ENV_ACTION_DIM,), np.float32)
        self.steering_remove_obs_keys = ["ego_view_image"]
        self._t = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": np.asarray([self._t, 0.0, 0.0, 0.0], dtype=np.float32),
            "ego_view_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        self._t += 1
        terminated = self._t >= 4
        return self._observation(), 0.0, terminated, False, {"success": terminated}

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _FrozenReference:
    def __init__(self) -> None:
        self.policy = SimpleNamespace(model=SimpleNamespace(feat_dim=FEAT_DIM))

    def infer_reference(
        self, observation: dict[str, Any], *, image_transform_mode: str = "eval"
    ):
        del observation, image_transform_mode
        return torch.zeros(1, 2 * FEAT_DIM), np.full(
            (1, HORIZON, ENV_ACTION_DIM), 0.25, dtype=np.float32
        )


class _EditActor(torch.nn.Module):
    normalize_actions = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def act(self, observation, deterministic=False, actor_state=None):
        del deterministic, actor_state
        return observation["reference_action"] + self.weight, None


class _SteeringActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def act(self, observation, actor_state, deterministic):
        del actor_state, deterministic
        return torch.zeros(observation["state"].shape[0], 2) + self.weight, None


class _LatentSampler:
    include_reference_action = False
    reference_action_key = "reference_action"

    def sample_action(self) -> np.ndarray:
        return np.zeros((1, 2), dtype=np.float32)


class _TestSimulator(SimulatorEntry):
    name = "test"
    evaluation = "fresh_workers"

    def make_environment(self, spec, phase, context):
        del spec, phase, context
        return _RawEnv()

    def steering_key_mapping(self, spec):
        del spec
        return {"state": "state", "language": None, "image": ["ego_view_image"]}


ENTRY = _TestSimulator()
SPEC = EnvironmentSpec(env_name="test/task", num_envs=1, max_episode_steps=12, seed=0)


def _cfg(image_keys: list[str] | None = None, **direct_action: Any):
    return OmegaConf.create(
        {
            "runtime": {"name": "native"},
            "env": {
                "env_name": "test/task",
                "image_keys": image_keys,
                "sparse_success_reward": True,
                "success_reward": 0.0,
                "failure_reward": -1.0,
                "step_reward": 0.0,
            },
            "eval": {"action_selection": "sample", "eval_base": False},
            "training": {"policy_decode_batch_size": 7},
            "dsrl": {"action_exec_len": 8, "use_vlm_features": False},
            "direct_action": {
                "reference_image_transform_mode": "train",
                **direct_action,
            },
            "model": {"dinov2_model": None},
        }
    )


def _edit_components() -> dict[str, Any]:
    base_model = _FrozenReference()
    reference_env = ReferenceActionEnv(
        gym.vector.SyncVectorEnv([_RawEnv]),
        base_model,
        action_horizon=HORIZON,
        action_dim=len(ACTION_INDICES),
        action_indices=ACTION_INDICES,
        environment_action_defaults=DEFAULTS,
        reference_image_transform_mode="train",
    )
    return {
        "actor": _EditActor(),
        "base_model": base_model,
        "controller_env": reference_env,
        "latent_env": reference_env,
        "action_modulator": None,
    }


def test_decode_batch_size_precedence_and_uncapped_evaluation() -> None:
    class _Entry(SimulatorEntry):
        name = "capped"
        decode_batch_size = 16

        def make_environment(self, spec, phase, context):
            raise NotImplementedError

    entry = _Entry()
    configured = _cfg()
    assert decode_batch_size_for(configured, entry, phase="train") == 7
    configured.training.policy_decode_batch_size = None
    assert decode_batch_size_for(configured, entry, phase="train") == 16
    configured.training.policy_decode_batch_size = 7
    configured.simulator = {"decode_batch_size": 512}
    assert decode_batch_size_for(configured, entry, phase="train") == 512
    assert decode_batch_size_for(configured, entry, phase="eval") is None
    for bad in (0, -1, True, "all"):
        configured.simulator = {"decode_batch_size": bad}
        with pytest.raises(ValueError, match="simulator.decode_batch_size"):
            decode_batch_size_for(configured, entry, phase="train")


def test_edit_family_runs_a_reference_edit_policy_over_the_whole_chunk() -> None:
    components = _edit_components()

    train = rollout_worker_spec(
        "edit",
        _cfg(),
        ENTRY,
        SPEC,
        **components,
        phase="train",
        configured_workers=None,
    )
    evaluation = rollout_worker_spec(
        "edit", _cfg(), ENTRY, SPEC, **components, phase="eval", configured_workers=None
    )

    assert isinstance(train, GymWorkerSpec) and train.supports_batched_policy
    assert (train.entry, train.environment, train.phase) == (ENTRY, SPEC, "train")
    policy = train.policy
    assert isinstance(policy, EditPolicySpec)
    assert policy.base_model.prebuilt is components["base_model"]
    assert (policy.action_horizon, policy.action_indices) == (HORIZON, ACTION_INDICES)
    np.testing.assert_array_equal(policy.environment_action_defaults, DEFAULTS)
    assert policy.image_keys == ("ego_view_image",) and policy.dino_image_keys == (
        "ego_view_image",
    )
    assert policy.reference_image_transform_mode == "train"
    assert evaluation.policy.reference_image_transform_mode == "eval"
    # Both phases shape the reward: the edit family scores the shaped return it trains on.
    for spec in (train, evaluation):
        assert spec.reward_transform == SparseOutcomeReward(
            success_reward=0.0, failure_reward=-1.0, step_reward=0.0
        )

    worker = train(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        )
    )
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="warmup"))
    decisions = result.decisions
    assert [d.transition.steps for d in decisions] == [HORIZON, 1]
    assert all(d.action_chunk.execution_length == HORIZON for d in decisions)
    assert all(d.action_chunk.source == "native" for d in decisions)
    assert set(decisions[0].transition.observation.arrays()) == {
        "state",
        "vlm_features",
        "reference_action",
        "action_mask",
    }
    last = decisions[-1].transition
    assert last.terminated[-1]
    final_reward = last.decision_reward("sparse_outcome", aggregation="last")
    assert final_reward == pytest.approx(0.0)


def test_steering_family_keeps_its_policy_and_the_raw_evaluation_reward() -> None:
    components = {
        "actor": _SteeringActor(),
        "base_model": object(),
        "controller_env": object(),
        "latent_env": _LatentSampler(),
        "action_modulator": None,
    }

    train = rollout_worker_spec(
        "steering",
        _cfg(),
        ENTRY,
        SPEC,
        **components,
        phase="train",
        configured_workers=None,
    )
    evaluation = rollout_worker_spec(
        "steering",
        _cfg(),
        ENTRY,
        SPEC,
        **components,
        phase="eval",
        configured_workers=None,
    )

    assert isinstance(train.policy, SteeringPolicySpec)
    assert train.policy.action_exec_len == 8
    assert train.policy.max_decode_batch_size == 7
    assert evaluation.policy.max_decode_batch_size is None
    assert train.reward_transform == SparseOutcomeReward(
        success_reward=0.0, failure_reward=-1.0, step_reward=0.0
    )
    assert evaluation.reward_transform is None
    with pytest.raises(ValueError, match="unknown policy family"):
        rollout_worker_spec(
            "flow",
            _cfg(),
            ENTRY,
            SPEC,
            **components,
            phase="train",
            configured_workers=None,
        )


def test_both_families_embed_the_configured_cameras_over_the_entry_mapping() -> None:
    steering = {
        "actor": _SteeringActor(),
        "base_model": object(),
        "controller_env": object(),
        "latent_env": _LatentSampler(),
        "action_modulator": None,
    }
    for family, components in (("steering", steering), ("edit", _edit_components())):
        by_entry = rollout_worker_spec(
            family,
            _cfg(),
            ENTRY,
            SPEC,
            **components,
            phase="train",
            configured_workers=None,
        )
        assert by_entry.policy.dino_image_keys == ("ego_view_image",)
        configured = rollout_worker_spec(
            family,
            _cfg(image_keys=["other_cam"]),
            ENTRY,
            SPEC,
            **components,
            phase="train",
            configured_workers=None,
        )
        assert configured.policy.dino_image_keys == ("other_cam",)


def _compose(*overrides: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=["base_model=mock", *overrides])


def test_decode_batch_size_override_composes_for_g1_arena() -> None:
    from isaac_rl_posttraining.envs.providers import get_simulator_entry

    entry = get_simulator_entry("arena")
    assert (
        decode_batch_size_for(_compose("simulator=g1_arena"), entry, phase="train")
        == 16
    )
    assert (
        decode_batch_size_for(
            _compose("simulator=g1_arena", "simulator.decode_batch_size=512"),
            entry,
            phase="train",
        )
        == 512
    )


def test_sac_edit_dry_runs(monkeypatch, tmp_path) -> None:
    from scripts.train import run_from_hydra_cfg

    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_DRYRUN", "1")
    monkeypatch.chdir(tmp_path)
    run_from_hydra_cfg(_compose("algorithm=sac_edit", "training.chunk_size=null"))
