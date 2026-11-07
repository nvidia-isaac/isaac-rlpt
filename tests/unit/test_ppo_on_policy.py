# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for provider-neutral PPO collection and updates."""

from __future__ import annotations

import copy
import dataclasses
import time

import gymnasium as gym
import numpy as np
import pytest
import torch
from conftest import make_transition
from on_policy_doubles import episode_source

from isaac_rl_posttraining.algorithms.ppo import (
    PPO,
    DSRLPPOActorCritic,
    GaussianActorCritic,
    PPOConfig,
    build_rank_batches,
)
from isaac_rl_posttraining.buffers import load_buffer
from isaac_rl_posttraining.data import DynamicObservation
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
from isaac_rl_posttraining.rollouts import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    GymRolloutWorker,
    LocalLearnerGroup,
    PolicyState,
    RolloutGroupBase,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
)
from isaac_rl_posttraining.runtimes.concurrency import ThreadCall
from isaac_rl_posttraining.training import (
    OnPolicyConfig,
    OnPolicyTrainer,
    PersistSpec,
    trajectories_from_results,
)
from isaac_rl_posttraining.training.telemetry import open_run_telemetry, read_telemetry


class _ScriptedRolloutGroup(RolloutGroupBase):
    num_rollout_workers = 1
    num_envs = 2

    def __init__(self, model: GaussianActorCritic) -> None:
        self.model = copy.deepcopy(model)
        self.version = 0
        self.episode = 0
        self.requests = []
        self.closed = False

    def reset_policy_server_stats(self) -> None:
        # Test double: keeps no server stats.
        pass

    def policy_server_stats(self):
        return (
            {
                "operation_seconds": {"action": 0.25, "successor": 0.125},
                "operation_batch_sizes": {
                    "action": (2, 2),
                    "successor": (2,),
                },
                "request_latencies_seconds": (0.2, 0.4),
            },
        )

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        self.model.load_state_dict(
            {name: torch.from_numpy(value) for name, value in state.tensors.items()}
        )
        self.version = state.version
        return (self.version,)

    def _decision(self, request, env_index, episode_id, step):
        """One scripted two-step-episode decision acted by the shared model."""
        observation = torch.tensor(
            [[float(env_index), float(step), 1.0]], dtype=torch.float32
        )
        with torch.no_grad():
            action, log_prob, value = self.model.act(
                observation,
                deterministic=request.mode == "eval",
            )
            next_observation = observation + torch.tensor([[0.0, 1.0, 0.0]])
            _, _, next_value = self.model.act(
                next_observation,
                deterministic=True,
            )
        terminated = step == 1
        outputs = {}
        if request.mode == "train":
            outputs = {
                "log_prob": log_prob.numpy()[0],
                "value": value.numpy()[0],
                "next_value": next_value.numpy()[0],
            }
        transition = make_transition(
            episode_id=episode_id,
            step_in_episode=step,
            obs=observation.numpy()[0],
            action=action.numpy()[0],
            action_kind="chunk",
            reward=float(terminated),
            next_obs=next_observation.numpy()[0],
            done=terminated,
            info={"success": terminated},
            **outputs,
        )
        chunk = ActionChunk.from_arrays(
            learner_action=action.numpy(),
            environment_actions=action.numpy()[:, None, :],
            execution_length=1,
            learner_action_kind="chunk",
            policy_version=self.version,
            behavior_log_prob=(log_prob.numpy() if request.mode == "train" else None),
            behavior_value=(value.numpy() if request.mode == "train" else None),
            learner_observation={"features": observation.numpy()},
        )
        return CollectedDecision(env_index, transition, chunk)

    def collect(self, requests):
        request = requests[0]
        self.requests.append(request)
        assert isinstance(request, RolloutRequest)
        assert request.episodes_per_environment is not None
        decisions = []
        env_steps = 0
        for env_index in range(self.num_envs):
            for _ in range(request.episodes_per_environment):
                episode_id = self.episode
                self.episode += 1
                for step in range(2):
                    decisions.append(
                        self._decision(request, env_index, episode_id, step)
                    )
                    env_steps += 1
        return (
            RolloutResult(
                worker_index=0,
                decisions=tuple(decisions),
                env_steps=env_steps,
                episodes_completed=self.num_envs * request.episodes_per_environment,
                policy_version=self.version,
                timings={"action_inference_seconds": 0.5},
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_ppo_trainer_collects_complete_episodes_and_updates_local_learner():
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    algorithm = PPO(PPOConfig(epochs=1, minibatch_size=4, seed=8))
    learner = LocalLearnerGroup(
        algorithm.learner(lambda: copy.deepcopy(model)), device="cpu"
    )
    rollout = _ScriptedRolloutGroup(model)
    records = []
    evaluation_records = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(rollout),
        learner_group=learner,
        algorithm=algorithm,
        config=OnPolicyConfig(
            iterations=2,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=1,
            seed=9,
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
        evaluation_sink=lambda trajectories, step, state: evaluation_records.append(
            (
                step,
                tuple(trajectory.success for trajectory in trajectories),
                state.version,
            )
        ),
    )

    trainer.run()

    training = [metrics for step, metrics in records if "train/episodes" in metrics]
    evaluation = [metrics for step, metrics in records if "eval/episodes" in metrics]
    assert len(training) == 2
    assert len(evaluation) == 3
    assert all(
        metrics["train/success_rate"] == pytest.approx(1.0) for metrics in training
    )
    latest = training[-1]
    assert latest["system/env_steps"] == pytest.approx(8.0)
    assert latest["system/rollout_seconds"] >= 0.0
    assert latest["system/trajectory_build_seconds"] >= 0.0
    assert latest["system/batch_build_seconds"] >= 0.0
    assert latest["rollout_workers/action_inference_seconds_sum"] == pytest.approx(0.5)
    assert latest["rollout_workers/action_inference_seconds_max"] == pytest.approx(0.5)
    assert latest["policy_service/total_compute_seconds"] == pytest.approx(0.375)
    assert latest["policy_service/action_mean_batch_size"] == pytest.approx(2.0)
    assert np.isclose(
        latest["policy_service/action_mean_latency_seconds"],
        0.3,
    )
    assert rollout.version == 3
    eval_requests = [request for request in rollout.requests if request.mode == "eval"]
    assert [request.reset_seed for request in eval_requests] == [100_000] * 3
    train_requests = [
        request for request in rollout.requests if request.mode == "train"
    ]
    assert [request.reset_seed for request in train_requests] == [9, 10_000_028]
    assert evaluation_records == [
        (0, (True, True), 1),
        (1, (True, True), 2),
        (2, (True, True), 3),
    ]


def _trainer(model, *, iterations=2, eval_episodes=0, **source_options):
    algorithm = PPO(PPOConfig(epochs=1, minibatch_size=4, seed=8))
    learner = LocalLearnerGroup(
        algorithm.learner(lambda: copy.deepcopy(model)), device="cpu"
    )
    records = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(_ScriptedRolloutGroup(model), **source_options),
        learner_group=learner,
        algorithm=algorithm,
        config=OnPolicyConfig(
            iterations=iterations,
            episodes_per_environment=1,
            eval_episodes_per_environment=eval_episodes,
            seed=9,
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
    )
    return trainer, records


def test_training_rounds_reach_telemetry_and_persistence_through_the_data_source(
    tmp_path,
):
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    sink = open_run_telemetry(tmp_path / "run")
    trainer, _ = _trainer(
        model,
        eval_episodes=1,
        telemetry=sink,
        persist=PersistSpec(str(tmp_path / "episodes"), every_episodes=2),
    )
    trainer.run()
    trainer.train_source.close()
    sink.close()

    episodes = read_telemetry(tmp_path / "run", "episodes").to_pylist()
    assert sorted({row["source"] for row in episodes}) == ["eval", "train"]
    assert (
        sum(row["source"] == "train" for row in episodes) == 4
    )  # 2 iterations x 2 environments
    assert len(read_telemetry(tmp_path / "run", "learner")) == 2
    saved = load_buffer(tmp_path / "episodes")
    assert len(saved.episodes()) == 4 and {
        t.metadata.source for t in saved.transitions()
    } == {"train"}


def test_an_out_of_band_save_waits_for_the_update_in_flight(monkeypatch):
    trainer, _ = _trainer(GaussianActorCritic(3, 2, hidden_sizes=(16,)))
    order = []
    trainer.checkpoint_store, trainer._policy_state = object(), object()
    trainer._update_in_flight = ThreadCall(
        lambda: (time.sleep(0.2), order.append("update"))
    )
    monkeypatch.setattr(
        trainer, "_checkpoint", lambda tag, iteration: order.append(tag)
    )

    trainer.save_checkpoint("interrupted")

    assert order == ["update", "interrupted"]


def test_step_wise_collect_and_train_is_the_sync_loop():
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    looped, loop_records = _trainer(model, iterations=3)
    torch.manual_seed(5)  # the scripted workers sample from the global generator
    looped.run()
    stepped, step_records = _trainer(model, iterations=3)
    torch.manual_seed(5)
    for _ in range(3):
        assert stepped.collect()["train/episodes"] == 2.0
        stepped.train()
    with pytest.raises(RuntimeError, match="call collect\\(\\) first"):
        stepped.train()
    stepped.collect()
    with pytest.raises(RuntimeError, match="call train\\(\\) first"):
        stepped.collect()

    def learned(records):
        """The learner's metrics without its timings."""
        return [
            (
                step,
                {
                    name: value
                    for name, value in metrics.items()
                    if name.startswith("learner/") and "second" not in name
                },
            )
            for step, metrics in records
        ]

    assert learned(step_records) == learned(loop_records)
    stepped_weights = stepped.learner_group.policy_state(version=99).tensors
    for name, value in looped.learner_group.policy_state(version=99).tensors.items():
        np.testing.assert_array_equal(stepped_weights[name], value)


def test_ppo_trainer_isolates_fixed_evaluation_in_fresh_rollout_groups():
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    algorithm = PPO(PPOConfig(epochs=1, minibatch_size=4, seed=8))
    learner = LocalLearnerGroup(
        algorithm.learner(lambda: copy.deepcopy(model)), device="cpu"
    )
    created = [("eval", _ScriptedRolloutGroup(model))]

    def open_group(phase):
        group = _ScriptedRolloutGroup(model)
        created.append((phase, group))
        return group

    records = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(created[0][1]),
        open_rollout_group=open_group,
        rollout_phase="eval",
        learner_group=learner,
        algorithm=algorithm,
        config=OnPolicyConfig(
            iterations=2,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=1,
            seed=9,
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
    )

    trainer.run()

    assert [phase for phase, _ in created] == [
        "eval",
        "train",
        "eval",
        "train",
        "eval",
    ]
    assert all(group.closed for _, group in created[:-1])
    assert not created[-1][1].closed
    for phase, group in created:
        assert group.requests
        assert all(request.mode == phase for request in group.requests)
    evaluation = [metrics for _, metrics in records if "eval/episodes" in metrics]
    assert evaluation[0]["system/rollout_refresh_seconds"] == pytest.approx(0.0)
    assert all(
        metrics["system/rollout_refresh_seconds"] > 0.0 for metrics in evaluation[1:]
    )


def test_ppo_replays_fixed_evaluation_once_after_infrastructure_failure() -> None:
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    algorithm = PPO(PPOConfig(epochs=1, minibatch_size=4, seed=8))
    learner = LocalLearnerGroup(
        algorithm.learner(lambda: copy.deepcopy(model)), device="cpu"
    )

    class _FailedGroup(_ScriptedRolloutGroup):
        def collect(self, requests):
            raise RolloutInfrastructureError("worker lost")

    failed = _FailedGroup(model)
    replacement = _ScriptedRolloutGroup(model)
    records = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(failed),
        open_rollout_group=lambda phase: replacement,
        rollout_phase="eval",
        learner_group=learner,
        algorithm=algorithm,
        config=OnPolicyConfig(
            iterations=1,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=1,
            seed=9,
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
    )

    with pytest.warns(RuntimeWarning, match="replaying the same fixed seeds once"):
        trainer.evaluate()

    assert failed.closed
    assert trainer.rollout_group is replacement
    assert records[0][1]["eval/episodes"] == pytest.approx(2.0)
    assert records[0][1]["system/eval_retries"] == pytest.approx(1.0)


def test_trajectory_builder_uses_discounted_chunk_rewards_and_timeout_bootstrap():
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _ScriptedRolloutGroup(model)
    rollout.set_policy_state(
        PolicyState(
            version=1,
            tensors={
                name: value.detach().numpy()
                for name, value in model.state_dict().items()
            },
        )
    )
    result = rollout.collect((RolloutRequest(episodes_per_environment=1),))
    trajectories = trajectories_from_results(result, gamma=0.5)
    timed_out = trajectories[0]
    timed_out = type(timed_out)(
        **{
            **timed_out.__dict__,
            "terminated": np.asarray([False, False]),
            "truncated": np.asarray([False, True]),
            "rewards": np.asarray([0.0, 1.0], dtype=np.float32),
            "next_values": np.asarray([0.25, 2.0], dtype=np.float32),
        }
    )

    batches, metrics = build_rank_batches(
        (timed_out,),
        num_learner_ranks=1,
        minibatch_size=2,
        gamma=0.5,
        gae_lambda=0.0,
        rng=np.random.default_rng(3),
    )

    returns = np.sort(np.asarray(batches[0].tensors["returns"]))
    expected = np.sort(np.asarray([0.5 * 0.25, 1.0 + 0.5 * 2.0]))
    assert np.allclose(returns, expected)
    assert metrics["duplicated_samples"] == pytest.approx(0.0)


def test_evaluation_trajectory_does_not_require_training_metadata():
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _ScriptedRolloutGroup(model)
    rollout.set_policy_state(
        PolicyState(
            version=1,
            tensors={
                name: value.detach().numpy()
                for name, value in model.state_dict().items()
            },
        )
    )
    results = rollout.collect(
        (RolloutRequest(episodes_per_environment=1, mode="eval"),)
    )

    trajectories = trajectories_from_results(
        results,
        gamma=0.99,
        require_behavior_metadata=False,
    )

    assert len(trajectories) == rollout.num_envs
    assert all(
        np.count_nonzero(trajectory.log_probs) == 0 for trajectory in trajectories
    )
    assert all(np.count_nonzero(trajectory.values) == 0 for trajectory in trajectories)
    assert all(trajectory.initial_observation_sha256 for trajectory in trajectories)
    # Observations are name-keyed records, so every trajectory carries a digest per component.
    assert all(
        [name for name, _ in trajectory.initial_observation_sha256_by_key] == ["state"]
        for trajectory in trajectories
    )
    assert all(trajectory.first_execution_action_sha256 for trajectory in trajectories)
    assert all(trajectory.execution_action_sha256 for trajectory in trajectories)
    assert all(
        len(trajectory.observation_sha256_by_decision) == len(trajectory.actions)
        for trajectory in trajectories
    )
    assert all(
        len(trajectory.execution_action_sha256_by_decision) == len(trajectory.actions)
        for trajectory in trajectories
    )


def test_evaluation_trajectory_hashes_each_observation_key():
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _ScriptedRolloutGroup(model)
    results = rollout.collect(
        (RolloutRequest(episodes_per_environment=1, mode="eval"),)
    )
    mapped_decisions = tuple(
        CollectedDecision(
            environment_index=decision.environment_index,
            transition=dataclasses.replace(
                decision.transition,
                observation=DynamicObservation.from_mapping(
                    {
                        "ego_view_image": np.zeros((2, 2, 3), dtype=np.uint8),
                        "state": decision.transition.observation.arrays()["state"],
                    }
                ),
                metadata=dataclasses.replace(
                    decision.transition.metadata,
                    extra={
                        **decision.transition.metadata.extra,
                        "raw_observation_sha256": "raw-observation",
                        "raw_observation_sha256_by_key": {
                            "ego_view_image": "raw-image",
                            "state": "raw-state",
                        },
                    },
                ),
            ),
            action_chunk=decision.action_chunk,
        )
        for decision in results[0].decisions
    )
    mapped_results = (
        RolloutResult(
            worker_index=results[0].worker_index,
            decisions=mapped_decisions,
            env_steps=results[0].env_steps,
            episodes_completed=results[0].episodes_completed,
            policy_version=results[0].policy_version,
        ),
    )

    trajectories = trajectories_from_results(
        mapped_results,
        gamma=0.99,
        require_behavior_metadata=False,
    )

    for trajectory in trajectories:
        hashes = dict(trajectory.initial_observation_sha256_by_key)
        assert hashes.keys() == {"ego_view_image", "state"}
        assert all(len(value) == 64 for value in hashes.values())
        assert trajectory.initial_raw_observation_sha256 == "raw-observation"
        assert dict(trajectory.initial_raw_observation_sha256_by_key) == {
            "ego_view_image": "raw-image",
            "state": "raw-state",
        }


def test_dsrl_ppo_adapter_preserves_normalized_action_log_probability():
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
    actor = MLPActor(
        MLPActorConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(16,),
            use_tanh_output=True,
        )
    )
    value = MLPCritic(
        MLPCriticConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(16,),
            use_action=False,
        )
    )
    model = DSRLPPOActorCritic(actor, value)
    observation = {"state": torch.zeros(4, 3)}

    decoder_action, learner_action, old_log_prob, old_value = model.sample_action_chunk(
        observation
    )
    log_prob, entropy, value_again = model(observation, learner_action)

    assert torch.all(decoder_action.abs() <= 2.0)
    assert torch.all(learner_action.abs() <= 1.0)
    assert torch.allclose(log_prob, old_log_prob, atol=1e-5, rtol=1e-5)
    assert torch.equal(value_again, old_value)
    assert entropy.shape == (4,)


def test_dsrl_ppo_adapter_supports_default_chunked_transformer_geometry():
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
    shared = {
        "observation_space": observation_space,
        "action_space": action_space,
        "featurizer": {"state": [8]},
        "chunk_size": 3,
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "feature_hidden_dims": [8],
        "use_language_embeddings": False,
    }
    actor = TransformerActor(
        TransformerActorConfig(
            **shared,
            output_hidden_dims=[],
            use_tanh_output=True,
        )
    )
    value = TransformerCritic(
        TransformerCriticConfig(
            **shared,
            output_hidden_dims=[],
            use_action=False,
            pooling_strategy="first",
        )
    )
    model = DSRLPPOActorCritic(actor, value)
    observation = {"state": torch.zeros(4, 3)}

    decoder_action, learner_action, old_log_prob, old_value = model.sample_action_chunk(
        observation
    )
    log_prob, entropy, value_again = model(observation, learner_action)

    assert decoder_action.shape == (4, 3, 2)
    assert learner_action.shape == (4, 3, 2)
    assert old_log_prob.shape == (4,)
    assert torch.allclose(log_prob, old_log_prob, atol=1e-5, rtol=1e-5)
    assert torch.equal(value_again, old_value)
    assert entropy.shape == (4,)


def test_seeded_gym_collection_replays_the_same_environment_seeds():
    class _Env:
        def __init__(self) -> None:
            self.seeds = []

        def reset(self, *, seed=None):
            self.seeds.append(seed)
            return {"state": np.zeros(1, dtype=np.float32)}, {}

        def step(self, action):
            return (
                {"state": np.ones(1, dtype=np.float32)},
                1.0,
                True,
                False,
                {"success": True},
            )

        def close(self):
            # Test double: holds no resources to release.
            pass

    class _Policy:
        def __init__(self) -> None:
            self.seeds = []

        def reset_episode(self, *, seed=None):
            self.seeds.append(seed)

        def action_chunk(self, observation, *, mode):
            return ActionChunk.from_arrays(
                learner_action=None,
                environment_actions=np.zeros((1, 1, 1), dtype=np.float32),
                execution_length=1,
                policy_version=0,
            )

        def set_policy_state(self, state):
            return state.version

        def close(self):
            # Test double: holds no resources to release.
            pass

    env = _Env()
    policy = _Policy()
    worker = GymRolloutWorker(
        RolloutWorkerContext(0, 1, (7,), "cpu"),
        env,
        policy,
    )
    request = RolloutRequest(episodes_per_environment=2, reset_seed=50)

    first = worker.collect(request)
    second = worker.collect(request)

    assert first.episodes_completed == second.episodes_completed == 2
    assert first.timings.keys() == {
        "action_inference_seconds",
        "environment_reset_seconds",
        "environment_step_seconds",
        "environment_step_calls",
        "policy_calls",
    }
    assert all(seconds >= 0.0 for seconds in first.timings.values())
    expected = [50 + 7 * 1_000_003, 51 + 7 * 1_000_003]
    assert env.seeds == expected + expected
    assert policy.seeds == expected + expected
    assert [
        (
            item.transition.metadata.reset_seed,
            item.transition.metadata.reset_seed_scope,
        )
        for item in first.decisions
    ] == [
        (expected[0], "environment"),
        (expected[1], "environment"),
    ]
