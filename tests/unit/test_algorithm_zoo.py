# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end proof that the non-recurrent algorithm zoo is runnable on CPU.

The zoo exposes three learners across three data-source paradigms:

* ``dsrl_sac`` -- latent steering (act only in the frozen flow-noise space),
* ``rfs`` -- residual flow steering (steer the latent AND add a bounded
  residual after decode),
* ``sac_edit`` -- direct action (decode a reference chunk and
  execute a bounded edit in the raw action space).

These tests build each paradigm from the same pieces the entry point wires
(``isaac_rl_posttraining/runners/off_policy_models.py``) with mock doubles,
then prove the loop runs on the runtime layer: the registry round-trips the
names, the residual/direct modulation math is correct, ``train_step`` returns
finite losses on every learner after real collection through the rollout
workers, and the full off-policy loop (collect -> train -> eval -> checkpoint)
completes for one learner of each paradigm.
"""

from __future__ import annotations

import copy
import functools
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from conftest import (
    MockEnv,
    StubBasePolicy,
    UnbatchedMockEnv,
)
from entry_doubles import scripted_worker_spec

from isaac_rl_posttraining.algorithms.registry import (
    available_algorithms,
    get_algorithm_config,
)
from isaac_rl_posttraining.algorithms.replay_learner import ReplayLearner
from isaac_rl_posttraining.algorithms.rfs import (
    RFS,
    RFSActionModulator,
    RFSConfig,
    configure_rfs_actor_distribution,
    initialize_rfs_residual_head,
)
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.algorithms.sac_edit import (
    ReferenceEditActor,
    ReferenceEditActorConfig,
    SACEdit,
    SACEditConfig,
)
from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler
from isaac_rl_posttraining.configs.registry import register_configs
from isaac_rl_posttraining.envs import (
    LatentSpaceEnv,
    ResidualFlowSpaceEnv,
)
from isaac_rl_posttraining.envs.reference_action import ReferenceActionEnv
from isaac_rl_posttraining.modules.mlp import (
    MLPActor,
    MLPActorConfig,
    MLPCritic,
    MLPCriticConfig,
)
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
    TransformerCritic,
    TransformerCriticConfig,
)
from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.runtimes import (
    EditPolicySpec,
    FrozenBaseModelSpec,
    NativeRuntime,
    NativeRuntimeConfig,
    SteeringPolicySpec,
)
from isaac_rl_posttraining.runtimes.learner import LocalLearnerGroup
from isaac_rl_posttraining.training import (
    CheckpointStore,
    OffPolicyConfig,
    OffPolicyTrainer,
    RolloutEvaluator,
    RuntimeDataSource,
)

_CPU = torch.device("cpu")
STEERING_KEY_MAPPING = {"state": "state", "language": None, "image": ["ego_view_image"]}


# ---------------------------------------------------------------------------
# Shared builders -- one per paradigm, mirroring isaac_rl_posttraining/runners/off_policy_models.py
# ---------------------------------------------------------------------------


def _steer_obs_space(state_dim: int, dino_dim: int) -> gym.spaces.Dict:
    """The {state, dino_embedding} learner observation the steering policy emits."""
    return gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32),
            "dino_embedding": gym.spaces.Box(-np.inf, np.inf, (dino_dim,), np.float32),
        }
    )


def _fresh_buffer(capacity: int = 256) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity)


def _sac_config(**overrides: Any) -> dict[str, Any]:
    """SAC hyperparameters shrunk to a fast, deterministic smoke."""
    values: dict[str, Any] = {
        "learning_starts": 0,
        "batch_size": 4,
        "num_critics": 2,
        "n_critics_to_sample": 2,
        "num_critic_updates_per_actor_update": 1,
        "num_updates_per_train_step": 1,
        "pooled_critic_features": False,
        "train_actor_with_kl_divergence": False,
        "use_amp": False,
    }
    values.update(overrides)
    return values


def _build_sac(
    state_dim: int = 4, dino_dim: int = 8, action_dim: int = 6, exec_len: int = 4
):
    """Latent-steering SAC over a frozen flow policy."""
    base_model = StubBasePolicy(horizon=8, action_dim=action_dim)
    obs_space = _steer_obs_space(state_dim, dino_dim)
    latent_env = LatentSpaceEnv(
        obs_space,
        gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        base_model,
        noise_dim=action_dim,
        chunk_size=None,
        action_bound=1.0,
        use_vlm_features=False,
    )
    actor = MLPActor(
        MLPActorConfig(
            observation_space=latent_env.observation_space,
            action_space=latent_env.action_space,
            hidden_dims=(16, 16),
            use_tanh_output=True,
            log_std_init=-1.0,
        )
    )
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=latent_env.observation_space,
            action_space=latent_env.action_space,
            hidden_dims=(16, 16),
            use_action=True,
        )
    )
    buffer = _fresh_buffer()
    algorithm = SAC(
        SACConfig(
            env=latent_env,
            action_space=latent_env.action_space,
            actor=actor,
            critic=critic,
            buffer=buffer,
            sampler=UniformSampler(buffer),
            **_sac_config(),
        )
    )

    def make_env(context) -> UnbatchedMockEnv:
        del context
        return UnbatchedMockEnv(
            MockEnv(
                episode_outcomes=("success", "fail"),
                horizon=exec_len,
                dino_dim=dino_dim,
                action_dim=action_dim,
                hard_step_limit=200,
            )
        )

    return SimpleNamespace(
        algorithm=algorithm,
        actor=actor,
        buffer=buffer,
        make_env=make_env,
        policy=SteeringPolicySpec(
            actor=actor,
            base_model=FrozenBaseModelSpec(prebuilt=base_model),
            latent_sampler=latent_env,
            steering_key_mapping=STEERING_KEY_MAPPING,
            action_exec_len=exec_len,
        ),
        action_exec_len=exec_len,
    )


def _build_rfs(
    state_dim: int = 4, dino_dim: int = 8, noise_dim: int = 2, exec_len: int = 4
):
    """Residual flow steering: joint [latent, residual] over a chunked actor."""
    residual_dim = 3
    chunk = (
        4  # RFSActionModulator requires the actor horizon to match the decode horizon.
    )
    base_model = StubBasePolicy(horizon=chunk, action_dim=residual_dim)
    obs_space = _steer_obs_space(state_dim, dino_dim)
    residual_bound = 0.05
    latent_env = ResidualFlowSpaceEnv(
        observation_space=obs_space,
        action_space=gym.spaces.Box(-1.0, 1.0, (residual_dim,), np.float32),
        base_model=base_model,
        noise_dim=noise_dim,
        residual_action_dim=residual_dim,
        chunk_size=chunk,
        noise_action_bound=1.0,
        residual_action_bound=residual_bound,
        warmup_exploration="uniform",
        use_vlm_features=False,
    )
    actor = TransformerActor(
        TransformerActorConfig(
            observation_space=latent_env.observation_space,
            action_space=latent_env.action_space,
            chunk_size=chunk,
            d_model=32,
            nhead=4,
            feature_hidden_dims=[32],
            use_language_embeddings=False,
            use_tanh_output=True,
            log_std_init=-1.0,
        )
    )
    critic = TransformerCritic(
        TransformerCriticConfig(
            observation_space=latent_env.observation_space,
            action_space=latent_env.action_space,
            chunk_size=chunk,
            d_model=32,
            nhead=4,
            feature_hidden_dims=[32],
            use_language_embeddings=False,
            use_action=True,
        )
    )
    initialize_rfs_residual_head(
        actor, latent_dim=noise_dim, initial_residual_log_std=-5.0
    )
    configure_rfs_actor_distribution(
        actor,
        latent_dim=noise_dim,
        residual_action_bound=residual_bound,
        residual_exploration_std=0.0,
    )
    buffer = _fresh_buffer()
    algorithm = RFS(
        RFSConfig(
            env=latent_env,
            action_space=latent_env.action_space,
            actor=actor,
            critic=critic,
            buffer=buffer,
            sampler=UniformSampler(buffer),
            **_sac_config(),
        )
    )
    modulator = RFSActionModulator(
        latent_dim=noise_dim,
        residual_action_dim=residual_dim,
        residual_action_bound=residual_bound,
    )

    def make_env(context) -> UnbatchedMockEnv:
        del context
        return UnbatchedMockEnv(
            MockEnv(
                episode_outcomes=("success", "fail"),
                horizon=exec_len,
                dino_dim=dino_dim,
                action_dim=residual_dim,
                hard_step_limit=200,
            )
        )

    return SimpleNamespace(
        algorithm=algorithm,
        actor=actor,
        buffer=buffer,
        make_env=make_env,
        policy=SteeringPolicySpec(
            actor=actor,
            base_model=FrozenBaseModelSpec(prebuilt=base_model),
            latent_sampler=latent_env,
            steering_key_mapping=STEERING_KEY_MAPPING,
            action_exec_len=exec_len,
            action_modulator=modulator,
        ),
        action_exec_len=exec_len,
    )


class _RefRawEnv(gym.Env):
    """A tiny direct-action raw env that terminates so episodes end."""

    def __init__(self, action_dim: int = 4, terminal_after: int = 3) -> None:
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, (action_dim,), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self.steering_remove_obs_keys: list[str] = []
        self._action_dim = int(action_dim)
        self._terminal_after = int(terminal_after)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        return {"state": np.zeros(self._action_dim, np.float32)}, {}

    def step(self, action: np.ndarray):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self._t += 1
        terminated = self._t >= self._terminal_after
        return (
            {"state": np.zeros(self._action_dim, np.float32)},
            0.0,
            terminated,
            False,
            {"is_success": bool(terminated)},
        )


class _FrozenReference:
    """Frozen policy exposing the reference-action decode + pooled features."""

    def __init__(self, action_horizon: int, action_dim: int, feat_dim: int = 2) -> None:
        self.policy = SimpleNamespace(model=SimpleNamespace(feat_dim=feat_dim))
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._feat_dim = int(feat_dim)

    def infer_reference(self, observation: Any, *, image_transform_mode: str = "eval"):
        del observation
        assert image_transform_mode in {"train", "eval"}
        features = torch.zeros((1, 2 * self._feat_dim), dtype=torch.float32)
        reference = np.full(
            (1, self._action_horizon, self._action_dim), 0.1, dtype=np.float32
        )
        return features, reference


def _reference_edit_actor_observation_space(
    observation_space, remove_obs_keys, reference_action_key
):
    """Actor view: drop images/state and the mask, keep the reference chunk."""
    actor_observation_space = copy.deepcopy(observation_space)
    for key in remove_obs_keys:
        actor_observation_space.spaces.pop(key, None)
    actor_observation_space.spaces.pop(ReferenceActionEnv.ACTION_MASK_KEY, None)
    assert reference_action_key in actor_observation_space.spaces
    return actor_observation_space


def _build_direct_action(
    algorithm_name: str, action_horizon: int = 3, action_dim: int = 4
):
    """SAC-edit over the reference-action env (bounded edits of the frozen reference)."""
    reference_action_key = ReferenceActionEnv.REFERENCE_KEY
    base_model = _FrozenReference(action_horizon, action_dim)
    # The reference-action environment sizes the models, exactly as the entry point's
    # local environment does; the rollout workers drive the raw env through the edit policy.
    env = ReferenceActionEnv(
        gym.vector.SyncVectorEnv([lambda: _RefRawEnv(action_dim=action_dim)]),
        base_model,
        action_horizon=action_horizon,
        action_dim=action_dim,
        include_sim_state=False,
        reference_action_transform="identity",
        relative_action_scale=0.05,
        reference_image_transform_mode="eval",
    )
    remove_obs_keys = list(env.learner_remove_obs_keys)
    critic_remove_obs_keys = list(remove_obs_keys) + [reference_action_key]
    # The MLP critic is sized from the keys in its observation space, so the
    # dropped keys are removed up front (as build_actor_critic_models does);
    # remove_obs_keys then strips them again from each forward observation.
    critic_obs_space = copy.deepcopy(env.single_observation_space)
    for key in critic_remove_obs_keys:
        critic_obs_space.spaces.pop(key, None)
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=critic_obs_space,
            action_space=env.single_action_space,
            remove_obs_keys=critic_remove_obs_keys,
            hidden_dims=(16, 16),
            use_action=True,
        )
    )
    actor_obs_space = _reference_edit_actor_observation_space(
        env.single_observation_space, remove_obs_keys, reference_action_key
    )
    actor = ReferenceEditActor(
        ReferenceEditActorConfig(
            observation_space=actor_obs_space,
            action_space=env.single_action_space,
            remove_obs_keys=remove_obs_keys,
            simnorm_dim=8,
            simnorm_groups=4,
            reference_action_key=reference_action_key,
            edit_scale=0.02,  # the shipped bounded-edit mode
            reference_centered_full_support=False,
            hidden_dims=(16, 16),
            use_tanh_output=True,
            deterministic=False,
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
        )
    )
    buffer = _fresh_buffer()
    assert algorithm_name == "sac_edit"
    algorithm = SACEditConfig(
        env=env,
        action_space=env.single_action_space,
        actor=actor,
        critic=critic,
        buffer=buffer,
        sampler=UniformSampler(buffer),
        **_sac_config(),
    ).create()

    def make_env(context) -> _RefRawEnv:
        del context
        return _RefRawEnv(action_dim=action_dim)

    return SimpleNamespace(
        algorithm=algorithm,
        actor=actor,
        buffer=buffer,
        env=env,
        make_env=make_env,
        policy=EditPolicySpec(
            actor=actor,
            base_model=FrozenBaseModelSpec(prebuilt=base_model),
            action_indices=tuple(env.action_indices),
            environment_action_defaults=np.array(
                env.environment_action_defaults, copy=True
            ),
            action_horizon=action_horizon,
            reference_action_transform="identity",
            relative_action_scale=0.05,
            reference_image_transform_mode="eval",
            state_key="state",
            image_keys=(),
        ),
        # The edit family earns one update per executed environment step.
        action_exec_len=1,
    )


_BUILDERS = {
    "dsrl_sac": _build_sac,
    "rfs": _build_rfs,
    "sac_edit": lambda: _build_direct_action("sac_edit"),
}


def _build(algorithm_name: str):
    return _BUILDERS[algorithm_name]()


def _open_group(built: Any, runtime: NativeRuntime, name: str):
    """One in-process rollout worker over the paradigm's environment and policy."""
    return runtime.open_rollout_group(
        scripted_worker_spec(
            built.make_env, built.policy, supports_batched_policy=True
        ),
        num_rollout_workers=1,
        num_envs=1,
        name=name,
    )


def _open_source(built: Any, runtime: NativeRuntime) -> RuntimeDataSource:
    # The workers start from a copy of the same actor weights, and the loop publishes
    # every later version itself.
    return RuntimeDataSource(
        _open_group(built, runtime, "train"),
        InProcessBufferSink(built.buffer),
        count_by="episode",
    )


def _collect_until(
    source: RuntimeDataSource,
    buffer: ReplayBuffer,
    min_transitions: int,
    max_runs: int = 40,
) -> None:
    """Collect whole episodes until the buffer holds enough transitions to train."""
    runs = 0
    while len(buffer) < min_transitions and runs < max_runs:
        source.run(can_train=True)
        runs += 1
    assert len(buffer) >= min_transitions, (
        f"collected only {len(buffer)} transitions in {runs} runs"
    )


# ---------------------------------------------------------------------------
# Registry + modulation-math unit tests
# ---------------------------------------------------------------------------


def test_registry_round_trip() -> None:
    """The zoo names resolve to their config/learner and register with Hydra."""
    expected = {
        "rfs": ("RFSConfig", RFS),
        "sac_edit": ("SACEditConfig", SACEdit),
    }
    for name, (config_name, learner_cls) in expected.items():
        config_cls = get_algorithm_config(name)
        assert config_cls.__name__ == config_name
        assert config_cls().algorithm_class is learner_cls
        assert name in available_algorithms()
    register_configs()  # idempotent; must not raise now that the zoo is registered


def test_rfs_action_modulator_splits_and_bounds() -> None:
    """The modulator splits [latent, residual] and adds the bounded residual."""
    modulator = RFSActionModulator(
        latent_dim=2, residual_action_dim=3, residual_action_bound=0.01
    )
    assert modulator.policy_action_dim == 5

    policy_actions = np.zeros((1, 4, 5), dtype=np.float32)
    policy_actions[..., :2] = 0.25
    policy_actions[..., 2:] = 1.0
    latent, residual = modulator.split(policy_actions)
    assert latent.shape == (1, 4, 2)
    assert residual.shape == (1, 4, 3)
    np.testing.assert_allclose(latent, 0.25)
    np.testing.assert_allclose(residual, 1.0)

    decoded = np.zeros((1, 4, 3), dtype=np.float32)
    modulated = modulator.apply(decoded, residual)
    # Full-magnitude (=1) normalized residual maps to exactly the physical bound.
    np.testing.assert_allclose(modulated, 0.01, atol=1e-6)
    assert np.max(np.abs(modulated - decoded)) <= 0.01 + 1e-6


def test_reference_action_env_executes_the_learner_action_within_bounds() -> None:
    """The learner's action space is the configured bound, and each executed environment
    action is exactly the learner's chunk row placed at the controlled indices."""
    built = _build_direct_action("sac_edit", action_horizon=3, action_dim=4)
    env = built.env
    assert np.allclose(env.single_action_space.low, -1.0) and np.allclose(
        env.single_action_space.high, 1.0
    )
    obs, _ = env.reset()
    reference = np.asarray(obs[ReferenceActionEnv.REFERENCE_KEY], dtype=np.float32)
    assert reference.shape == (1, 3, 4)
    assert np.all(np.abs(reference) <= 1.0 + 1e-6)

    learner_action = np.linspace(-0.9, 0.9, 4, dtype=np.float32)[
        None, :
    ]  # one chunk row
    env.step(learner_action)
    executed = env.env.envs[0].last_action
    assert executed.shape == (4,)
    assert np.allclose(executed, learner_action[0])
    with pytest.raises(ValueError, match="learner action width"):
        env.step(np.zeros((1, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# train_step returns finite losses on every learner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full loop: collect -> train -> eval -> checkpoint
# ---------------------------------------------------------------------------


def _run_loop(built: Any, tmp_path: Any, num_rollouts: int = 40) -> None:
    save_dir = os.path.join(str(tmp_path), "checkpoints")
    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    try:
        source = _open_source(built, runtime)
        learner_group = LocalLearnerGroup(
            functools.partial(ReplayLearner, algorithm=built.algorithm), device="cpu"
        )
        evaluator = RolloutEvaluator(
            group=_open_group(built, runtime, "eval"), num_episodes=1, max_decisions=50
        )
        trainer = OffPolicyTrainer(
            train_source=source,
            learner_group=learner_group,
            buffer=built.buffer,
            draw_batch=built.algorithm.learner_batch,
            evaluator=evaluator,
            config=OffPolicyConfig(
                num_rollouts=num_rollouts,
                learning_starts=0,
                eval_freq=20,
                eval_on_first_step=True,
                save_interval=20,
            ),
            checkpoint_store=CheckpointStore(save_dir),
        )
        trainer.run()
        metrics = built.algorithm.train_step(
            rollout_step=source.episode_tracker.total_steps
        )
    finally:
        runtime.close()

    checkpoints = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(save_dir)
        for name in files
        if name.endswith(".pt")
    ]
    assert checkpoints, f"no checkpoint (*.pt) written under {save_dir}"
    # A train step on the collected data reports finite losses and metrics.
    assert "critic_loss" in metrics and "actor_loss" in metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)) or (
            torch.is_tensor(value) and value.ndim == 0
        ):
            assert np.isfinite(float(value)), f"{key}={value} is not finite"


def test_e2e_loop_dsrl_sac(tmp_path) -> None:
    _run_loop(_build("dsrl_sac"), tmp_path)


def test_e2e_loop_rfs(tmp_path) -> None:
    _run_loop(_build("rfs"), tmp_path)


def test_e2e_loop_sac_edit(tmp_path) -> None:
    _run_loop(_build("sac_edit"), tmp_path)
