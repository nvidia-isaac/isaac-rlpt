# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO's simulator entry, composed with steering policies, without the simulator dependency."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs import LatentSpaceEnv
from isaac_rl_posttraining.envs.providers import get_simulator_entry
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy
from isaac_rl_posttraining.runtimes import (
    GymWorkerSpec,
    RolloutWorkerContext,
    SparseOutcomeReward,
)
from isaac_rl_posttraining.runtimes.steering_rollouts import (
    steering_policy_spec,
    steering_worker_spec,
)
from simulators.libero import runtime as libero_runtime

ENTRY = libero_runtime.Libero()


def _context(environment_index: int) -> RolloutWorkerContext:
    return RolloutWorkerContext(
        worker_index=environment_index,
        num_rollout_workers=8,
        environment_indices=(environment_index,),
        device="cpu",
    )


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def act(self, observation, actor_state, deterministic):
        del actor_state, deterministic
        return torch.zeros(observation["state"].shape[0], 2), None


def _latent_env() -> LatentSpaceEnv:
    return LatentSpaceEnv(
        gym.spaces.Dict({}),
        gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
        MockBaseModelPolicy(),
        noise_dim=2,
        use_vlm_features=False,
    )


class _DinoWrapper:
    dinov2_model = object()

    def observation(self, observation):
        return observation


def _config(*, runtime: str = "native", dinov2_model: str | None = "unused"):
    return OmegaConf.create(
        {
            "runtime": {"name": runtime},
            "base_model": {"name": "gr00t", "dataset_adapter": "libero_sim"},
            "env": {
                "env_name": "libero/libero_spatial/3",
                "image_keys": ["video.image", "video.wrist_image"],
            },
            "training": {"seed": 17},
            "dsrl": {"action_exec_len": 8, "use_vlm_features": False},
            "model": {"dinov2_model": dinov2_model},
        }
    )


def _spec(num_envs: int, *, seed: int | None = 17) -> EnvironmentSpec:
    return EnvironmentSpec(
        env_name="libero/libero_spatial/3",
        num_envs=num_envs,
        max_episode_steps=720,
        seed=seed,
        use_init_states=True,
        fixed_init_state_id=4,
    )


def test_registry_resolves_the_libero_entry_lazily() -> None:
    entry = get_simulator_entry("libero/libero_spatial/0")

    assert isinstance(entry, libero_runtime.Libero)
    assert get_simulator_entry("libero/libero_10/1") is entry
    assert entry.evaluation == "fresh_workers"
    assert entry.base_models == ("gr00t", "gr00t_n15", "gr00t_n17")
    assert entry.steering_key_mapping(_spec(1))["image"] == [
        "video.image",
        "video.wrist_image",
    ]
    assert not entry.supports_chunked_step


def _capture_environments(monkeypatch) -> list[dict]:
    """Record the arguments of every LIBERO environment the entry builds instead of building it."""
    built: list[dict] = []

    def fake_make_env(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr(libero_runtime, "make_libero_gr00t_env", fake_make_env)
    return built


def test_libero_worker_spec_is_runtime_independent(monkeypatch) -> None:
    built = _capture_environments(monkeypatch)
    spec = steering_worker_spec(
        ENTRY,
        _spec(4),
        "train",
        policy=steering_policy_spec(
            _config(dinov2_model=None),
            actor=_Actor(),
            base_model=MockBaseModelPolicy(),
            controller_env=object(),
            latent_env=_latent_env(),
            action_modulator=None,
            steering_key_mapping=dict(ENTRY.steering_key_mapping(_spec(1))),
            dino_image_keys=ENTRY.steering_key_mapping(_spec(1))["image"],
            phase="train",
            decode_batch_size=None,
        ),
        action_exec_len=8,
        reward_transform=SparseOutcomeReward(
            success_reward=0.0, failure_reward=-1.0, step_reward=0.0
        ),
        configured_workers=None,
    )

    assert isinstance(spec, GymWorkerSpec)
    assert spec.supports_batched_policy
    assert not spec.chunked_step
    assert ENTRY.worker_layout(4).workers == 4 and not ENTRY.worker_layout(4).batched
    assert (spec.entry, spec.environment, spec.phase) == (ENTRY, _spec(4), "train")
    # The worker builds its own task from the environment description.
    spec.make_environment(_context(0))
    assert built == [
        {
            "suite": "libero_spatial",
            "task_id": 3,
            "max_episode_steps": 720,
            "use_init_states": True,
            "fixed_init_state_id": 4,
            "seed": 17,
        }
    ]
    assert spec.reward_transform is not None
    assert spec.reward_transform(
        (1.0,), terminated=True, truncated=False, info={"success": True}
    ).value == pytest.approx(0.0)
    assert spec.reward_transform(
        (0.0,), terminated=False, truncated=True, info={"success": False}
    ).value == pytest.approx(-1.0)
    transformed = spec.reward_transform(
        (0.0, 0.0, 1.0),
        terminated=True,
        truncated=False,
        info={"success": True},
    )
    assert transformed.sequence == (0.0, 0.0, 0.0)
    transformed_failure = spec.reward_transform(
        (0.0, 0.0),
        terminated=False,
        truncated=True,
        info={"success": False},
    )
    assert transformed_failure.sequence == (0.0, -1.0)


def test_evaluation_environments_use_their_own_seed_stream(monkeypatch) -> None:
    built = _capture_environments(monkeypatch)

    ENTRY.make_environment(_spec(2), "train", _context(1))
    ENTRY.make_environment(_spec(2), "eval", _context(1))
    ENTRY.make_environment(_spec(2, seed=None), "eval", _context(1))

    assert [kwargs["seed"] for kwargs in built] == [
        17 + 1,
        17 + libero_runtime.EVALUATION_SEED_OFFSET + 1,
        None,
    ]
    with pytest.raises(ValueError, match="libero/<suite>/<task>"):
        ENTRY.make_environment(
            EnvironmentSpec(
                env_name="libero_pi0/x/0", num_envs=1, max_episode_steps=1, seed=0
            ),
            "train",
            _context(0),
        )


def test_native_libero_attachment_reuses_controller_models() -> None:
    base_model = MockBaseModelPolicy()
    dino = _DinoWrapper()
    policy = steering_policy_spec(
        _config(),
        actor=_Actor(),
        base_model=base_model,
        controller_env=dino,
        latent_env=_latent_env(),
        action_modulator=None,
        steering_key_mapping=dict(ENTRY.steering_key_mapping(_spec(1))),
        dino_image_keys=ENTRY.steering_key_mapping(_spec(1))["image"],
        decode_batch_size=None,
        phase="train",
    )

    assert policy.base_model.prebuilt is base_model
    assert policy.dino_wrapper is dino
    assert policy.dino_model_name is None


def test_libero_environment_seed_uses_global_environment_index(monkeypatch) -> None:
    built = _capture_environments(monkeypatch)
    spec = EnvironmentSpec(
        env_name="libero/libero_spatial/pick_up_the_black_bowl",
        num_envs=8,
        max_episode_steps=720,
        seed=10,
    )

    env = ENTRY.make_environment(
        spec,
        "train",
        RolloutWorkerContext(
            worker_index=1,
            num_rollout_workers=4,
            environment_indices=(6,),
            device="cpu",
        ),
    )

    assert env is not None
    assert built[0]["seed"] == 16
    assert built[0]["task_name"] == "pick_up_the_black_bowl"


def test_ppo_evaluation_policy_enables_deterministic_inference() -> None:
    policy = steering_policy_spec(
        _config(runtime="rlinf", dinov2_model=None),
        actor=_Actor(),
        base_model=MockBaseModelPolicy(),
        controller_env=object(),
        latent_env=_latent_env(),
        action_modulator=None,
        steering_key_mapping=dict(ENTRY.steering_key_mapping(_spec(1))),
        dino_image_keys=ENTRY.steering_key_mapping(_spec(1))["image"],
        decode_batch_size=None,
        phase="eval",
        deterministic_inference=True,
        sample_evaluation_actions=True,
    )
    assert policy.reference_image_transform_mode == "eval"

    assert policy.deterministic_inference
    assert policy.sample_evaluation_actions
    assert policy.base_model.prebuilt is None and policy.base_model.config is not None


@pytest.mark.parametrize(
    "env_name, entry_type, builder, base_models, image_keys, task_kwargs",
    [
        (
            "libero_pi0/libero_90/3",
            libero_runtime.LiberoPi0,
            "make_libero_pi0_env",
            ("pi0",),
            ["observation/image"],
            {"task_id": 3},
        ),
        (
            "libero_dreamzero/libero_spatial/put_the_bowl_on_the_plate",
            libero_runtime.LiberoDreamZero,
            "make_libero_dreamzero_env",
            ("dreamzero_local", "dreamzero_server"),
            ["main_image"],
            {"task_name": "put_the_bowl_on_the_plate"},
        ),
    ],
)
def test_pi0_and_dreamzero_entries_build_their_own_conventions(
    monkeypatch, env_name, entry_type, builder, base_models, image_keys, task_kwargs
) -> None:
    entry = get_simulator_entry(env_name)
    assert isinstance(entry, entry_type)
    assert entry.evaluation == "fresh_workers"
    assert entry.base_models == base_models

    spec = EnvironmentSpec(
        env_name=env_name,
        num_envs=1,
        max_episode_steps=240,
        seed=17,
        use_init_states=False,
        fixed_init_state_id=None,
    )
    assert entry.steering_key_mapping(spec)["image"] == image_keys

    built: list[dict] = []

    def fake_make_env(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr(libero_runtime, builder, fake_make_env)
    entry.make_environment(spec, "eval", _context(2))
    assert built == [
        {
            "suite": env_name.split("/")[1],
            "max_episode_steps": 240,
            "use_init_states": False,
            "fixed_init_state_id": None,
            "seed": 17 + libero_runtime.EVALUATION_SEED_OFFSET + 2,
            **task_kwargs,
        }
    ]
    with pytest.raises(ValueError, match=f"{entry.name}/<suite>/<task>"):
        entry.make_environment(
            EnvironmentSpec(
                env_name="libero/libero_10/0", num_envs=1, max_episode_steps=1, seed=0
            ),
            "train",
            _context(0),
        )


@pytest.mark.parametrize(
    "overrides, expected_entry",
    [
        (["base_model=gr00t", "env.env_name=libero/libero_10/0"], "libero"),
        (
            [
                "base_model=mock",
                "base_model.action_dim=7",
                "dsrl.noise_dim=7",
                "env.env_name=libero/libero_10/0",
            ],
            "libero",
        ),
        (
            [
                "base_model=pi0",
                "env.env_name=libero_pi0/libero_10/0",
                "algorithm.target_entropy=-16",
            ],
            "libero_pi0",
        ),
        (
            [
                "base_model=mock",
                "base_model.action_dim=7",
                "dsrl.noise_dim=7",
                "env.env_name=libero_pi0/libero_10/0",
            ],
            "libero_pi0",
        ),
        (
            [
                "base_model=dreamzero_local",
                "env.env_name=libero_dreamzero/libero_spatial/0",
            ],
            "libero_dreamzero",
        ),
    ],
)
def test_dry_run_plans_compose_for_every_libero_entry(
    tmp_path, monkeypatch, overrides, expected_entry
) -> None:
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.runners.build import train_plan
    from isaac_rl_posttraining.train import CONFIG_DIR

    register_configs()
    monkeypatch.chdir(tmp_path)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "algorithm=dsrl_sac",
                "simulator=libero_10",
                "env.max_episode_steps=400",
                *overrides,
            ],
        )

    plan = train_plan(cfg)

    assert plan.simulator == expected_entry


def test_plan_refuses_an_entry_whose_base_model_does_not_match(
    tmp_path, monkeypatch
) -> None:
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.runners.build import train_plan
    from isaac_rl_posttraining.train import CONFIG_DIR

    register_configs()
    monkeypatch.chdir(tmp_path)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        mismatched = compose(
            config_name="train",
            overrides=[
                "algorithm=dsrl_sac",
                "simulator=libero_10",
                "env.max_episode_steps=400",
                "base_model=gr00t",
                "env.env_name=libero_pi0/libero_10/0",
            ],
        )
    with pytest.raises(ValueError, match="builds environments for base model"):
        train_plan(mismatched)
