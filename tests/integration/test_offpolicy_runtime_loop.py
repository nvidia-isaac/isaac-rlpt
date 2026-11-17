# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The off-policy loop end to end on the mock base model: DSRL and SAC-edit train, evaluate, checkpoint.

Both runs use the runtime layer exactly as the entry point wires it (a native
rollout group per phase, the in-process learner group, the rollout evaluator)
over scripted environments, so every artifact of a real run exists here: the
training bursts, the evaluations at the configured boundaries, the checkpoints
with their manifests, ``evaluation_summary.json`` and ``fingerprints.jsonl``.
The last test proves a checkpoint re-scores and warm-starts through
``training.load_dir``.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from conftest import MockEnv, UnbatchedMockEnv
from entry_doubles import (
    ScriptedSimulator,
    scripted_environment,
    scripted_worker_spec,
)

from isaac_rl_posttraining.algorithms.replay_learner import ReplayLearner
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.algorithms.sac_edit import (
    ReferenceEditActor,
    ReferenceEditActorConfig,
    SACEditConfig,
)
from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler
from isaac_rl_posttraining.envs import LatentSpaceEnv
from isaac_rl_posttraining.envs.reference_action import ReferenceActionEnv
from isaac_rl_posttraining.modules.mlp import (
    MLPActor,
    MLPActorConfig,
    MLPCritic,
    MLPCriticConfig,
)
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy
from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    EditPolicySpec,
    FrozenBaseModelSpec,
    GymWorkerSpec,
    NativeRuntime,
    NativeRuntimeConfig,
    SparseOutcomeReward,
    SteeringPolicySpec,
)
from isaac_rl_posttraining.runtimes.learner import LocalLearnerGroup
from isaac_rl_posttraining.runtimes.policy_snapshot import (
    load_policy_snapshot,
    policy_state_tensors,
)
from isaac_rl_posttraining.training import (
    CheckpointStore,
    FingerprintSink,
    OffPolicyConfig,
    OffPolicyTrainer,
    RolloutEvaluator,
    RuntimeDataSource,
    ScheduleSpec,
)
from isaac_rl_posttraining.utils.checkpoint_retention import (
    EVALUATION_SUMMARY_FILE,
)
from isaac_rl_posttraining.utils.training_utils import (
    actor_fingerprint,
    load_checkpoint,
)

ACTION_DIM, STATE_DIM, DINO_DIM = 6, 4, 8
EXEC_LEN, HORIZON = 4, 12  # three decisions per episode
BUDGET, LEARNING_STARTS, EVAL_FREQ, SAVE_INTERVAL = 96, 24, 36, 36
EVAL_EPISODES, EVAL_MAX_DECISIONS = 2, 50
SEED = 3
STEERING_KEY_MAPPING = {"state": "state", "language": None, "image": ["ego_view_image"]}
CPU = torch.device("cpu")
SPARSE_REWARD = SparseOutcomeReward(
    success_reward=0.0, failure_reward=-1.0, step_reward=0.0
)
#: SAC-edit executes the mock base model's whole decoded chunk per decision; its
#: reference is made relative to the previous target so it stays inside the
#: actor's bounds despite the mock's position code.
EDIT_CHUNK, EDIT_STATE_DIM, EDIT_FEAT_DIM, RELATIVE_SCALE = 8, 6, 2, 100.0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_env() -> UnbatchedMockEnv:
    return UnbatchedMockEnv(
        MockEnv(
            episode_outcomes=("success", "fail", "fail"),
            horizon=HORIZON,
            dino_dim=DINO_DIM,
            action_dim=ACTION_DIM,
            hard_step_limit=10_000,
        )
    )


def _build_components(seed: int) -> SimpleNamespace:
    _seed_all(seed)
    base_model = MockBaseModelPolicy(action_horizon=8, action_dim=ACTION_DIM, seed=seed)
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "dino_embedding": gym.spaces.Box(-np.inf, np.inf, (DINO_DIM,), np.float32),
        }
    )
    latent_env = LatentSpaceEnv(
        observation_space,
        gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
        base_model,
        noise_dim=ACTION_DIM,
        chunk_size=None,
        action_bound=1.0,
        use_vlm_features=False,
    )
    # Gymnasium's Box sampler is otherwise unseeded, which would make the warm-up draws differ between runs.
    latent_env.action_space.seed(seed)
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
    buffer = ReplayBuffer(capacity=512)
    algorithm = SAC(
        SACConfig(
            env=latent_env,
            action_space=latent_env.action_space,
            actor=actor,
            critic=critic,
            buffer=buffer,
            sampler=UniformSampler(buffer),
            learning_starts=LEARNING_STARTS,
            batch_size=4,
            num_critics=2,
            n_critics_to_sample=2,
            num_critic_updates_per_actor_update=2,
            num_updates_per_train_step=2,
            pooled_critic_features=False,
            train_actor_with_kl_divergence=False,
            use_amp=False,
            ent_coef="auto",
        )
    )
    return SimpleNamespace(
        base_model=base_model,
        latent_env=latent_env,
        actor=actor,
        buffer=buffer,
        algorithm=algorithm,
    )


def _numeric(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if np.isscalar(value) and key != "train_step_total_time_s"
    }


def _fingerprints(save_dir: str) -> list[tuple[str, int, str]]:
    with open(os.path.join(save_dir, "fingerprints.jsonl")) as stream:
        records = [json.loads(line) for line in stream]
    return [
        (record["event"], record["step"], record["actor_sha256"]) for record in records
    ]


def _evaluation_history(save_dir: str) -> list[tuple[int, float]]:
    with open(os.path.join(save_dir, EVALUATION_SUMMARY_FILE)) as stream:
        summary = json.load(stream)
    return [(record["step"], record["success_rate"]) for record in summary["history"]]


def _checkpoint_tags(save_dir: str) -> list[str]:
    return sorted(
        entry
        for entry in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, entry))
    )


class _RecordingLearnerGroup:
    def __init__(
        self, group: LocalLearnerGroup, train_calls: list, step_at_call
    ) -> None:
        self.group = group
        self._train_calls = train_calls
        self._step_at_call = step_at_call

    def train_step(self, batches):
        results = self.group.train_step(batches)
        self._train_calls.append(
            (self._step_at_call(), _numeric(dict(results[0].metrics)))
        )
        return results

    def policy_state(self, *, version: int):
        return self.group.policy_state(version=version)

    def save_checkpoint(self, directory: str) -> None:
        self.group.save_checkpoint(directory)

    def hooks(self):
        return self.group.hooks()


class _RecordingEvaluator:
    def __init__(
        self, evaluator: RolloutEvaluator, eval_steps: list[int], step_at_call
    ) -> None:
        self.evaluator = evaluator
        self._eval_steps = eval_steps
        self._step_at_call = step_at_call

    def evaluate(self, policy_state, *, env_steps=0):
        self._eval_steps.append(self._step_at_call())
        return self.evaluator.evaluate(policy_state, env_steps=env_steps)


def _run_loop(
    components: SimpleNamespace,
    save_dir: str,
    *,
    train_spec: GymWorkerSpec | BatchedWorkerSpec,
    eval_spec: GymWorkerSpec | BatchedWorkerSpec,
    action_exec_len: int,
    num_envs: int = 1,
    num_rollout_workers: int = 1,
    schedule: str | ScheduleSpec = "sync",
    runtime_config: NativeRuntimeConfig | None = None,
) -> dict[str, Any]:
    runtime = NativeRuntime(runtime_config or NativeRuntimeConfig(device="cpu"))
    train_group = runtime.open_rollout_group(
        train_spec,
        num_rollout_workers=num_rollout_workers,
        num_envs=num_envs,
        name="train",
    )
    eval_group = runtime.open_rollout_group(
        eval_spec,
        num_rollout_workers=num_rollout_workers,
        num_envs=num_envs,
        name="eval",
    )
    train_calls: list[tuple[int, dict[str, float]]] = []
    eval_steps: list[int] = []
    system_records: list[dict[str, float]] = []
    source = RuntimeDataSource(
        train_group,
        InProcessBufferSink(components.buffer),
        request_count=1,
        count_by="episode" if isinstance(train_spec, BatchedWorkerSpec) else "decision",
    )
    learner_group = _RecordingLearnerGroup(
        LocalLearnerGroup(
            functools.partial(ReplayLearner, algorithm=components.algorithm),
            device="cpu",
        ),
        train_calls,
        lambda: trainer.total_env_steps,
    )
    evaluator = _RecordingEvaluator(
        RolloutEvaluator(
            group=eval_group,
            num_episodes=EVAL_EPISODES,
            max_decisions=EVAL_MAX_DECISIONS,
        ),
        eval_steps,
        lambda: trainer.total_env_steps,
    )
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner_group,
        buffer=components.buffer,
        draw_batch=components.algorithm.learner_batch,
        evaluator=evaluator,
        config=OffPolicyConfig(
            num_rollouts=BUDGET,
            learning_starts=LEARNING_STARTS,
            train_after_episode=True,
            eval_freq=EVAL_FREQ,
            eval_on_first_step=True,
            eval_at_episode_boundary=True,
            save_interval=SAVE_INTERVAL,
            schedule=schedule,
        ),
        metric_sink=lambda metrics, step: system_records.append(
            {key: value for key, value in metrics.items() if key.startswith("system/")}
        ),
        fingerprint_sink=FingerprintSink(save_dir),
        checkpoint_store=CheckpointStore(save_dir),
    )
    try:
        trainer.run()
    finally:
        runtime.close()
    return {
        "train_calls": train_calls,
        "eval_steps": eval_steps,
        "gradient_steps": components.algorithm.step_counter,
        "eval_history": _evaluation_history(save_dir),
        "checkpoint_tags": _checkpoint_tags(save_dir),
        "fingerprints": _fingerprints(save_dir),
        "algorithm": components.algorithm,
        "buffer": components.buffer,
        "system": [record for record in system_records if record],
    }


def _run_dsrl(
    save_dir: str,
    *,
    schedule: str | ScheduleSpec = "sync",
    runtime_config: NativeRuntimeConfig | None = None,
) -> dict[str, Any]:
    components = _build_components(SEED)

    def policy_spec() -> SteeringPolicySpec:
        return SteeringPolicySpec(
            actor=components.actor,
            base_model=FrozenBaseModelSpec(prebuilt=components.base_model),
            latent_sampler=components.latent_env,
            steering_key_mapping=STEERING_KEY_MAPPING,
            action_exec_len=EXEC_LEN,
        )

    return _run_loop(
        components,
        save_dir,
        train_spec=scripted_worker_spec(
            lambda context: _make_env(),
            policy_spec(),
            reward_transform=SPARSE_REWARD,
            supports_batched_policy=True,
        ),
        # Evaluation scores the raw environment reward.
        eval_spec=scripted_worker_spec(
            lambda context: _make_env(),
            policy_spec(),
            phase="eval",
            supports_batched_policy=True,
        ),
        action_exec_len=EXEC_LEN,
        schedule=schedule,
        runtime_config=runtime_config,
    )


def _assert_run_completed(run: dict[str, Any]) -> None:
    """The artifacts of one complete run: bursts, boundary evaluations, checkpoints, aligned digests."""
    assert run["train_calls"], "the run must train"
    train_steps = [step for step, _ in run["train_calls"]]
    assert train_steps == sorted(train_steps) and train_steps[0] >= LEARNING_STARTS
    for index, (_, metrics) in enumerate(run["train_calls"]):
        assert {"critic_loss", "actor_loss", "ent_coef", "q_values_mean"} <= set(
            metrics
        )
        for key, value in metrics.items():
            assert np.isfinite(value), f"call {index}: {key}={value}"
    assert run["eval_steps"][0] == 0 and len(run["eval_steps"]) >= 3
    assert run["gradient_steps"] > 0
    assert "final" in run["checkpoint_tags"] and len(run["checkpoint_tags"]) >= 3
    assert [step for step, _ in run["eval_history"]] == run["eval_steps"]
    # A checkpoint saved at an evaluation step holds exactly the evaluated weights.
    evals = {
        step: digest for event, step, digest in run["fingerprints"] if event == "eval"
    }
    saves = {
        step: digest for event, step, digest in run["fingerprints"] if event == "save"
    }
    shared = set(evals) & set(saves)
    assert shared, "expected checkpoints at evaluation steps"
    for step in shared:
        assert evals[step] == saves[step], f"digest mismatch at step {step}"


def test_dsrl_trains_evaluates_and_checkpoints_end_to_end(tmp_path) -> None:
    run = _run_dsrl(str(tmp_path / "dsrl"))
    _assert_run_completed(run)
    assert {record["system/policy_lag"] for record in run["system"]} == {0.0}


def test_dsrl_overlapped_schedule_overlaps_rounds_on_the_native_runtime(
    tmp_path,
) -> None:
    """The same run with the next round collecting during each burst: complete, with the rounds
    that followed a burst collected one publish behind, and none started into an evaluation."""
    run = _run_dsrl(str(tmp_path / "dsrl-overlapped"), schedule="overlapped")
    _assert_run_completed(run)
    lags = [record["system/policy_lag"] for record in run["system"]]
    assert set(lags) <= {0.0, 1.0} and 1.0 in lags
    # The rounds collected on the calling thread (the first, and the one after each of the
    # evaluations that fell inside the budget) have no lag; the run has at least those.
    assert (
        lags.count(0.0)
        >= len([step for step in run["eval_steps"] if 0 < step < BUDGET]) + 1
    )


def test_dsrl_async_schedule_runs_on_the_native_runtime(tmp_path) -> None:
    """Rounds re-submitted as soon as they return: weights published mid-round reach the workers
    when the round ends (no publish ever waits on a running collection), so data may lag
    several versions; the run still trains, evaluates and checkpoints on schedule."""
    run = _run_dsrl(
        str(tmp_path / "dsrl-async"), schedule=ScheduleSpec("async", publish_every=2)
    )
    _assert_run_completed(run)
    lags = [record["system/policy_lag"] for record in run["system"]]
    assert lags[0] == 0.0 and max(lags) >= 1.0


class _BatchedMockEnv:
    num_envs = 2

    def __init__(self, make_env=_make_env) -> None:
        self.envs = [make_env(), make_env()]

    @staticmethod
    def stack(observations):
        return {
            key: np.stack([item[key] for item in observations])
            for key in observations[0]
        }

    def reset(self, *, seed=None):
        observations = [env.reset(seed=seed)[0] for env in self.envs]
        return self.stack(observations), {}

    def step(self, actions):
        results = [
            env.step(action) for env, action in zip(self.envs, actions, strict=True)
        ]
        observations, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return (
            self.stack(observations),
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            list(infos),
        )

    def close(self):
        for env in self.envs:
            env.close()


def test_batched_dsrl_trains_evaluates_and_checkpoints_end_to_end(tmp_path) -> None:
    components = _build_components(SEED)
    policy = SteeringPolicySpec(
        actor=components.actor,
        base_model=FrozenBaseModelSpec(prebuilt=components.base_model),
        latent_sampler=components.latent_env,
        steering_key_mapping=STEERING_KEY_MAPPING,
        action_exec_len=EXEC_LEN,
    )
    entry = ScriptedSimulator(lambda context: _BatchedMockEnv())
    environment = scripted_environment(2, max_episode_steps=HORIZON)
    train_spec = BatchedWorkerSpec(
        entry,
        environment,
        "train",
        policy,
        HORIZON // EXEC_LEN,
        reward_transform=SPARSE_REWARD,
    )
    eval_spec = BatchedWorkerSpec(
        entry, environment, "eval", policy, HORIZON // EXEC_LEN
    )

    result = _run_loop(
        components,
        str(tmp_path / "batched-dsrl"),
        train_spec=train_spec,
        eval_spec=eval_spec,
        action_exec_len=EXEC_LEN,
        num_envs=2,
    )

    _assert_run_completed(result)


# ---------------------------------------------------------------------------
# SAC-edit: bounded edits of the frozen reference, executed as whole chunks
# ---------------------------------------------------------------------------


class _ScriptedTaskEnv(gym.Env):
    """``MockEnv``'s scripted outcomes behind the Gymnasium ``Env`` surface.

    Accepts one command or a whole ``[chunk, action_dim]`` chunk, which it
    executes row by row until the episode ends, reporting ``num_steps``.
    """

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, (EDIT_STATE_DIM,), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.steering_remove_obs_keys: list[str] = []
        self.episode_outcomes = ("success", "fail", "fail")
        self.episode_index = -1
        self._t = 0

    def _observation(self) -> dict[str, np.ndarray]:
        state = np.zeros(EDIT_STATE_DIM, dtype=np.float32)
        state[0] = float(self._t)
        return {"state": state}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.episode_index += 1
        self._t = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        outcome = self.episode_outcomes[self.episode_index % len(self.episode_outcomes)]
        executed = 0
        terminated = False
        for _ in np.asarray(action, dtype=np.float32).reshape(-1, ACTION_DIM):
            self._t += 1
            executed += 1
            terminated = self._t >= HORIZON
            if terminated:
                break
        success = terminated and outcome == "success"
        reward = -1.0 if terminated and not success else 0.0
        info = {"success": success, "is_success": success, "num_steps": executed}
        return self._observation(), reward, terminated, False, info

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _ReferenceBaseModel:
    """The mock frozen policy with the reference-action surface.

    Its native decode is the reference chunk; the pooled features are constant
    because the mock has no backbone.
    """

    def __init__(self, base_model: MockBaseModelPolicy) -> None:
        self.base_model = base_model
        self.policy = SimpleNamespace(model=SimpleNamespace(feat_dim=EDIT_FEAT_DIM))

    def infer_reference(
        self, observation: dict[str, Any], *, image_transform_mode: str = "eval"
    ):
        del image_transform_mode
        return torch.zeros(1, 2 * EDIT_FEAT_DIM), self.base_model.infer(
            observation, None
        )["actions"]


def _reference_env(
    base_model: _ReferenceBaseModel, *, image_transform_mode: str
) -> ReferenceActionEnv:
    return ReferenceActionEnv(
        gym.vector.SyncVectorEnv([_ScriptedTaskEnv]),
        base_model,
        action_horizon=EDIT_CHUNK,
        action_dim=ACTION_DIM,
        reference_action_transform="absolute_to_relative",
        relative_action_scale=RELATIVE_SCALE,
        reference_image_transform_mode=image_transform_mode,
    )


def _build_edit_components(seed: int) -> SimpleNamespace:
    _seed_all(seed)
    base_model = _ReferenceBaseModel(
        MockBaseModelPolicy(action_horizon=EDIT_CHUNK, action_dim=ACTION_DIM, seed=seed)
    )
    reference_env = _reference_env(base_model, image_transform_mode="train")
    remove_obs_keys = list(reference_env.learner_remove_obs_keys)
    critic_space = copy.deepcopy(reference_env.single_observation_space)
    for key in [*remove_obs_keys, ReferenceActionEnv.REFERENCE_KEY]:
        critic_space.spaces.pop(key)
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=critic_space,
            action_space=reference_env.single_action_space,
            remove_obs_keys=[*remove_obs_keys, ReferenceActionEnv.REFERENCE_KEY],
            hidden_dims=(16, 16),
            use_action=True,
        )
    )
    actor_space = copy.deepcopy(reference_env.single_observation_space)
    for key in [*remove_obs_keys, ReferenceActionEnv.ACTION_MASK_KEY]:
        actor_space.spaces.pop(key)
    actor = ReferenceEditActor(
        ReferenceEditActorConfig(
            observation_space=actor_space,
            action_space=reference_env.single_action_space,
            remove_obs_keys=remove_obs_keys,
            simnorm_dim=8,
            simnorm_groups=4,
            reference_action_key=ReferenceActionEnv.REFERENCE_KEY,
            edit_scale=0.02,
            hidden_dims=(16, 16),
            use_tanh_output=True,
            deterministic=False,
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
        )
    )
    buffer = ReplayBuffer(capacity=512, drop_components=remove_obs_keys)
    algorithm = SACEditConfig(
        env=reference_env,
        action_space=reference_env.single_action_space,
        actor=actor,
        critic=critic,
        buffer=buffer,
        sampler=UniformSampler(buffer),
        learning_starts=LEARNING_STARTS,
        batch_size=4,
        num_critics=2,
        n_critics_to_sample=2,
        num_critic_updates_per_actor_update=2,
        num_updates_per_train_step=2,
        pooled_critic_features=False,
        train_actor_with_kl_divergence=False,
        use_amp=False,
        ent_coef="auto",
        compute_chunked_gamma=True,
    ).create()
    return SimpleNamespace(
        base_model=base_model,
        reference_env=reference_env,
        actor=actor,
        buffer=buffer,
        algorithm=algorithm,
    )


def _run_edit(
    save_dir: str,
    *,
    runtime_config: NativeRuntimeConfig | None = None,
    layout: str = "one_environment",
) -> dict[str, Any]:
    components = _build_edit_components(SEED)
    reference_env = components.reference_env

    def policy_spec(image_transform_mode: str) -> EditPolicySpec:
        return EditPolicySpec(
            actor=components.actor,
            base_model=FrozenBaseModelSpec(prebuilt=components.base_model),
            action_indices=reference_env.action_indices,
            environment_action_defaults=reference_env.environment_action_defaults,
            action_horizon=EDIT_CHUNK,
            reference_action_transform="absolute_to_relative",
            relative_action_scale=RELATIVE_SCALE,
            reference_image_transform_mode=image_transform_mode,
            state_key="state",
            image_keys=(),
        )

    if layout == "batched_environments":
        entry = ScriptedSimulator(lambda context: _BatchedMockEnv(_ScriptedTaskEnv))
        environment = scripted_environment(2, max_episode_steps=HORIZON)
        chunks_per_episode = -(-HORIZON // EDIT_CHUNK)
        return _run_loop(
            components,
            save_dir,
            train_spec=BatchedWorkerSpec(
                entry,
                environment,
                "train",
                policy_spec("train"),
                chunks_per_episode,
                reward_transform=SPARSE_REWARD,
            ),
            eval_spec=BatchedWorkerSpec(
                entry,
                environment,
                "eval",
                policy_spec("eval"),
                chunks_per_episode,
                reward_transform=SPARSE_REWARD,
            ),
            action_exec_len=1,
            num_envs=2,
            runtime_config=runtime_config,
        )
    workers = 2 if layout == "two_workers" else 1
    # Both phases shape the reward, and the edit family earns one update per environment step.
    return _run_loop(
        components,
        save_dir,
        train_spec=scripted_worker_spec(
            lambda context: _ScriptedTaskEnv(),
            policy_spec("train"),
            reward_transform=SPARSE_REWARD,
            supports_batched_policy=True,
        ),
        eval_spec=scripted_worker_spec(
            lambda context: _ScriptedTaskEnv(),
            policy_spec("eval"),
            phase="eval",
            reward_transform=SPARSE_REWARD,
            supports_batched_policy=True,
        ),
        action_exec_len=1,
        num_envs=workers,
        num_rollout_workers=workers,
        runtime_config=runtime_config,
    )


def test_sac_edit_trains_evaluates_and_checkpoints_end_to_end(tmp_path) -> None:
    _assert_run_completed(_run_edit(str(tmp_path / "sac_edit")))


@pytest.mark.parametrize("layout", ["two_workers", "batched_environments"])
def test_sac_edit_trains_on_several_environments(tmp_path, layout) -> None:
    """Two environments (one per worker, or two in one batched worker) each keep their own reference
    decode, so every stored episode chains and the learner batches carry whole edit chunks."""
    run = _run_edit(str(tmp_path / layout), layout=layout)
    _assert_run_completed(run)
    buffer = run["buffer"]
    environments = {
        transition.metadata.environment_index for transition in buffer.transitions()
    }
    assert environments == {0, 1}
    episodes = buffer.episodes(complete=True)
    assert episodes and all(len(episode.transitions) == 2 for episode in episodes)
    for episode in episodes:
        # The decision after a replay next observation decides on that same decode.
        for previous, following in zip(
            episode.transitions, episode.transitions[1:], strict=False
        ):
            np.testing.assert_array_equal(
                previous.next_observation.to_mapping()[
                    ReferenceActionEnv.REFERENCE_KEY
                ],
                following.observation.to_mapping()[ReferenceActionEnv.REFERENCE_KEY],
            )
    batch = buffer.collate(range(len(buffer.transitions())))
    chunk = (EDIT_CHUNK, ACTION_DIM)
    assert batch["action"].shape[1:] == chunk
    assert batch["obs"][ReferenceActionEnv.REFERENCE_KEY].shape[1:] == chunk
    assert np.all(np.abs(batch["action"]) <= 1.0)


@pytest.mark.parametrize("family", ["dsrl", "sac_edit"])
def test_a_served_base_model_trains_like_todays_policy(tmp_path, family) -> None:
    """The base model served alone in one process (the actor on each worker, the training and
    evaluation groups sharing one base-model server) leaves the whole run unchanged."""
    run = _run_dsrl if family == "dsrl" else _run_edit
    today = run(str(tmp_path / "today"))
    served = run(
        str(tmp_path / "served"),
        runtime_config=NativeRuntimeConfig(device="cpu", base_mode="served"),
    )
    _assert_run_completed(served)
    for key in ("train_calls", "eval_steps", "gradient_steps", "eval_history"):
        assert served[key] == today[key], key
    assert [row[1:] for row in served["fingerprints"]] == [
        row[1:] for row in today["fingerprints"]
    ]


def test_checkpoint_loads_through_training_load_dir(tmp_path) -> None:
    """A checkpoint re-scores and warm-starts through ``training.load_dir``."""
    save_dir = str(tmp_path / "dsrl")
    trained = _run_dsrl(save_dir)
    checkpoint_dir = os.path.join(save_dir, "final")
    assert sorted(os.listdir(checkpoint_dir)) == [
        "learner.pt",
        "manifest.json",
        "policy_state.pt",
    ]

    # A differently initialized algorithm, so equality below comes from the load.
    fresh = _build_components(SEED + 1)
    rng_before = torch.get_rng_state()
    step = load_checkpoint(fresh.algorithm, checkpoint_dir)

    snapshot = policy_state_tensors(load_policy_snapshot(checkpoint_dir))
    actor_state = fresh.algorithm.actor.state_dict()
    assert set(actor_state) == set(snapshot)
    for key, value in snapshot.items():
        assert torch.equal(actor_state[key], value), key
    assert actor_fingerprint(fresh.algorithm.actor) == trained["fingerprints"][-1][2]
    with open(os.path.join(checkpoint_dir, "manifest.json")) as stream:
        manifest = json.load(stream)
    assert step == fresh.algorithm.step_counter == manifest["algo_step"] > 0
    # The warm start carries the critic, the temperature and the optimizer
    # moments too, and leaves the process's random stream to the run's seed.
    for key, value in trained["algorithm"].critic.state_dict().items():
        assert torch.equal(fresh.algorithm.critic.state_dict()[key], value), key
    assert torch.equal(
        fresh.algorithm.log_ent_coef.detach(),
        trained["algorithm"].log_ent_coef.detach(),
    )
    assert fresh.algorithm.actor_optimizer.state_dict()["state"]
    assert torch.equal(torch.get_rng_state(), rng_before)
