# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO controller options: chunk-level discounting, overlapped collection,
learner-less evaluation, resume, snapshots, and frozen submodules."""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
import time

import numpy as np
import pytest
import torch
from conftest import make_transition
from on_policy_doubles import episode_source
from torch import nn

from isaac_rl_posttraining.algorithms.ppo import (
    PPO,
    GaussianActorCritic,
    PPOConfig,
    build_rank_batches,
)
from isaac_rl_posttraining.rollouts import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    LocalLearnerGroup,
    PolicyState,
    RolloutGroupBase,
    RolloutRequest,
    RolloutResult,
    ThreadCall,
    load_policy_snapshot,
    policy_state_from_tensors,
    policy_state_tensors,
    save_policy_snapshot,
)
from isaac_rl_posttraining.training import (
    ITERATION_CHECKPOINT_PREFIX,
    CheckpointStore,
    OnPolicyConfig,
    OnPolicyTrainer,
    ScheduleSpec,
    checkpoint_iteration,
    trajectories_from_results,
)


def _decision(
    env_index: int,
    step: int,
    *,
    value: float,
    next_value: float,
    done: bool,
    truncated: bool,
):
    transition = make_transition(
        episode_id=env_index,
        step_in_episode=step * 4,
        obs=np.asarray([float(env_index), float(step), 1.0], dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        action_kind="chunk",
        rewards=np.asarray([0.0, 0.0, 0.0, 1.0 if done else 0.0], dtype=np.float32),
        done=done,
        truncated=truncated,
        info={"success": done},
        log_prob=-0.5,
        value=value,
        next_value=next_value,
    )
    chunk = ActionChunk.from_arrays(
        learner_action=np.zeros((1, 2), dtype=np.float32),
        environment_actions=np.zeros((1, 4, 2), dtype=np.float32),
        execution_length=4,
        learner_action_kind="chunk",
        behavior_log_prob=np.asarray([-0.5]),
        behavior_value=np.asarray([value]),
    )
    return CollectedDecision(
        environment_index=env_index, transition=transition, action_chunk=chunk
    )


def _timed_out_result() -> RolloutResult:
    """One environment, two 4-step chunks, ending at the time limit."""
    decisions = (
        _decision(0, 0, value=0.5, next_value=0.25, done=False, truncated=False),
        _decision(0, 1, value=0.25, next_value=2.0, done=False, truncated=True),
    )
    return RolloutResult(
        worker_index=0,
        decisions=decisions,
        env_steps=8,
        episodes_completed=1,
        policy_version=0,
    )


def test_decision_level_discounting_ignores_chunk_length_and_time_limit_bootstrap():
    trajectories = trajectories_from_results(
        (_timed_out_result(),), gamma=0.5, discount_per="decision"
    )
    # Per decision the chunk reward is the plain sum of its step rewards.
    assert trajectories[0].rewards.tolist() == [0.0, 0.0]
    batches, _ = build_rank_batches(
        trajectories,
        num_learner_ranks=1,
        minibatch_size=2,
        gamma=0.5,
        gae_lambda=0.0,
        rng=np.random.default_rng(0),
        discount_per="decision",
        bootstrap_at_time_limit=False,
    )
    returns = np.sort(np.asarray(batches[0].tensors["returns"]))
    # lambda=0: return_t = r_t + gamma * next_value_t; the time limit zeroes the successor.
    assert np.allclose(returns, np.sort([0.0 + 0.5 * 0.25, 0.0]))

    per_step, _ = build_rank_batches(
        trajectories,
        num_learner_ranks=1,
        minibatch_size=2,
        gamma=0.5,
        gae_lambda=0.0,
        rng=np.random.default_rng(0),
    )
    # Per simulator step the same chunk spans gamma**4 and bootstraps at the time limit.
    expected = np.sort([0.5**4 * 0.25, 0.5**4 * 2.0])
    assert np.allclose(
        np.sort(np.asarray(per_step.__getitem__(0).tensors["returns"])), expected
    )


class _CountingRolloutGroup(RolloutGroupBase):
    """A rollout group whose collections take a little time and record concurrency."""

    num_rollout_workers = 1
    num_envs = 1

    def __init__(self, model: GaussianActorCritic, *, delay: float = 0.0) -> None:
        self.model = copy.deepcopy(model)
        self.version = 0
        self.delay = delay
        self.versions_seen: list[int] = []
        self.set_calls: list[int] = []
        self.collecting = False
        self.overlapped = 0
        self.lock = threading.Lock()
        # Held while a collection runs, as a rollout group holds its weights fixed.
        self.weights = threading.Lock()

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        with self.weights:
            self.model.load_state_dict(policy_state_tensors(state))
            self.version = state.version
            self.set_calls.append(state.version)
            return (self.version,)

    def collect_async(self, requests):
        """Pin the weights at submission; the collection itself runs on a thread."""
        self.weights.acquire()

        def run():
            try:
                return self.collect(requests)
            finally:
                self.weights.release()

        return ThreadCall(run)

    def collect(self, requests):
        request = requests[0]
        with self.lock:
            self.collecting = True
        time.sleep(self.delay)
        version = self.version
        self.versions_seen.append(version)
        decisions = []
        for step in range(2):
            observation = torch.tensor([[float(step), 1.0, 0.0]], dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, value = self.model.act(
                    observation, deterministic=request.mode == "eval"
                )
            terminated = step == 1
            transition = make_transition(
                episode_id=version,
                step_in_episode=step,
                obs=observation.numpy()[0],
                action=action.numpy()[0],
                action_kind="chunk",
                reward=float(terminated),
                done=terminated,
                info={"success": terminated},
                log_prob=float(log_prob[0]),
                value=float(value[0]),
                next_value=0.0,
            )
            chunk = ActionChunk.from_arrays(
                learner_action=action.numpy(),
                environment_actions=action.numpy()[:, None, :],
                execution_length=1,
                learner_action_kind="chunk",
                policy_version=version,
                behavior_log_prob=log_prob.numpy(),
                behavior_value=value.numpy(),
            )
            decisions.append(CollectedDecision(0, transition, chunk))
        with self.lock:
            self.collecting = False
        return (
            RolloutResult(
                worker_index=0,
                decisions=tuple(decisions),
                env_steps=2,
                episodes_completed=1,
                policy_version=version,
            ),
        )

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _ScoredRolloutGroup(_CountingRolloutGroup):
    """Evaluations succeed or fail on a fixed schedule; training episodes always succeed."""

    def __init__(self, model: GaussianActorCritic, eval_successes: list[bool]) -> None:
        super().__init__(model)
        self.eval_successes = list(eval_successes)

    def collect(self, requests):
        results = super().collect(requests)
        if requests[0].mode == "train":
            return results
        success = self.eval_successes.pop(0)
        result = results[0]
        *head, last = result.decisions
        transition = dataclasses.replace(
            last.transition,
            terminated=np.asarray([success]),
            truncated=np.asarray([not success]),
            success=np.asarray([success]),
            rewards={"env": np.asarray([float(success)], dtype=np.float32)},
            metadata=dataclasses.replace(
                last.transition.metadata,
                extra={**last.transition.metadata.extra, "success": success},
            ),
        )
        return (
            RolloutResult(
                worker_index=0,
                decisions=(*head, CollectedDecision(0, transition, last.action_chunk)),
                env_steps=result.env_steps,
                episodes_completed=1,
                policy_version=result.policy_version,
            ),
        )


#: The PPO every controller test trains with: one epoch, rank minibatch of two.
_ALGORITHM = PPO(PPOConfig(epochs=1, minibatch_size=2, seed=8))


def _local_learner(model: GaussianActorCritic) -> LocalLearnerGroup:
    return LocalLearnerGroup(
        _ALGORITHM.learner(lambda: copy.deepcopy(model)), device="cpu"
    )


def test_overlapped_collection_lags_one_policy_version_and_drains_for_evaluation():
    torch.manual_seed(1)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _CountingRolloutGroup(model)
    records: list[tuple[int, dict]] = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(rollout),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=4,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=2,
            schedule="overlapped",
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
    )
    trainer.run()
    training = {
        step: metrics for step, metrics in records if "train/episodes" in metrics
    }
    assert sorted(training) == [1, 2, 3, 4]
    # The first collection of a stretch runs with the newest policy; the ones
    # submitted during an update lag it by one version.
    assert [training[step]["system/policy_lag"] for step in (1, 2, 3, 4)] == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    assert all("system/collect_wait_seconds" in training[step] for step in training)
    evaluations = [step for step, metrics in records if "eval/episodes" in metrics]
    assert evaluations == [0, 2, 4]
    assert trainer.iteration == 4


class _BlockingRolloutGroup(_CountingRolloutGroup):
    """A group without a native non-blocking call; records the thread its collection ran on."""

    collect_async = RolloutGroupBase.collect_async

    def __init__(self, model: GaussianActorCritic) -> None:
        super().__init__(model)
        self.started = threading.Event()
        self.collecting_thread: threading.Thread | None = None

    def collect(self, requests):
        self.collecting_thread = threading.current_thread()
        self.started.set()
        return super().collect(requests)


def test_collect_async_falls_back_to_a_thread_for_blocking_groups():
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _BlockingRolloutGroup(model)
    call = rollout.collect_async((RolloutRequest(episodes_per_environment=1),))
    assert rollout.started.wait(timeout=10.0), "the collection never started"
    results = call.result()
    assert results[0].episodes_completed == 1
    assert rollout.collecting_thread is not threading.current_thread(), (
        "the blocking call must run on a helper thread"
    )


def test_learnerless_evaluation_publishes_the_given_snapshot_or_nothing(tmp_path):
    torch.manual_seed(2)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    records: list[tuple[int, dict]] = []
    rollout = _CountingRolloutGroup(model)
    snapshot = policy_state_from_tensors(model.state_dict(), version=7)
    trainer = OnPolicyTrainer(
        train_source=episode_source(rollout),
        learner_group=None,
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=1, episodes_per_environment=1, eval_episodes_per_environment=1
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
        initial_policy_state=snapshot,
        checkpoint_store=CheckpointStore(
            str(tmp_path), tag_prefix=ITERATION_CHECKPOINT_PREFIX
        ),
    )
    trainer.evaluate()
    assert rollout.set_calls == [1]
    assert records[0][0] == 0 and records[0][1]["eval/episodes"] == pytest.approx(1.0)
    # Evaluation-only runs report through the same summary file as training runs, and save nothing.
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert summary["latest"]["success_rate"] == records[0][1]["eval/success_rate"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "evaluation_summary.json"
    ]
    with pytest.raises(RuntimeError, match="training needs a learner group"):
        trainer.run()

    untouched = _CountingRolloutGroup(model)
    sink_states: list = []
    base_only = OnPolicyTrainer(
        train_source=episode_source(untouched),
        learner_group=None,
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=1, episodes_per_environment=1, eval_episodes_per_environment=1
        ),
        metric_sink=lambda metrics, step: None,
        evaluation_sink=lambda trajectories, step, state: sink_states.append(state),
    )
    base_only.evaluate()
    # Nothing was published: the sink learns that the workers scored their loaded weights.
    assert sink_states == [None]
    assert untouched.set_calls == []
    with pytest.raises(ValueError, match="initial_policy_state"):
        OnPolicyTrainer(
            train_source=episode_source(untouched),
            learner_group=_local_learner(model),
            algorithm=_ALGORITHM,
            config=OnPolicyConfig(iterations=1, episodes_per_environment=1),
            metric_sink=lambda metrics, step: None,
            initial_policy_state=snapshot,
        )


def test_checkpoints_carry_snapshots_and_resume_continues_the_iteration_count(tmp_path):
    torch.manual_seed(3)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    records: list[tuple[int, dict]] = []
    config = OnPolicyConfig(
        iterations=3, episodes_per_environment=1, checkpoint_every=1
    )
    trainer = OnPolicyTrainer(
        train_source=episode_source(_CountingRolloutGroup(model)),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=config,
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
        checkpoint_store=CheckpointStore(
            str(tmp_path),
            retention="best_last_final",
            tag_prefix=ITERATION_CHECKPOINT_PREFIX,
        ),
    )
    trainer.run()
    kept = sorted(path.name for path in tmp_path.iterdir())
    # No evaluation ran, so retention never guesses: every checkpoint survives,
    # and the run ends with a final checkpoint of the last published weights.
    assert kept == ["final", "iteration_1", "iteration_2", "iteration_3"]
    assert checkpoint_iteration(tmp_path / "iteration_3") == 3
    assert (tmp_path / "final" / "manifest.json").is_file()
    snapshot = load_policy_snapshot(tmp_path / "iteration_3")
    assert set(snapshot.tensors) == set(model.state_dict())

    resumed_records: list[tuple[int, dict]] = []
    resumed = OnPolicyTrainer(
        train_source=episode_source(_CountingRolloutGroup(model)),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=5,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=5,
        ),
        metric_sink=lambda metrics, step: resumed_records.append((step, dict(metrics))),
    )
    resumed.resume(tmp_path / "iteration_3")
    assert (resumed.iteration, resumed.env_steps) == (
        3,
        trainer.env_steps,
    ) and trainer.env_steps > 0
    resumed.run()
    training_steps = [
        step for step, metrics in resumed_records if "train/episodes" in metrics
    ]
    assert training_steps == [4, 5]
    # A resumed run skips the pre-training evaluation and evaluates on schedule.
    assert [
        step for step, metrics in resumed_records if "eval/episodes" in metrics
    ] == [5]
    # A named checkpoint (final, aborted) resumes from the iteration its manifest records.
    assert checkpoint_iteration(tmp_path / "final") == 3
    with pytest.raises(ValueError, match="iteration_<n>"):
        checkpoint_iteration(tmp_path)


@pytest.mark.parametrize(
    ("checkpoint_retention", "kept"),
    [
        ("best_last_final", ["iteration_2", "iteration_4"]),
        ("all", ["iteration_1", "iteration_2", "iteration_3", "iteration_4"]),
    ],
)
def test_retention_keeps_the_best_evaluated_and_the_newest_checkpoint(
    tmp_path, checkpoint_retention, kept
):
    torch.manual_seed(5)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    # Evaluations at iterations 0..4; only the actor of iteration 2 succeeds.
    rollout = _ScoredRolloutGroup(
        model, eval_successes=[False, False, True, False, False]
    )
    trainer = OnPolicyTrainer(
        train_source=episode_source(rollout),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=4,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=1,
            checkpoint_every=1,
        ),
        metric_sink=lambda metrics, step: None,
        checkpoint_store=CheckpointStore(
            str(tmp_path),
            retention=checkpoint_retention,
            tag_prefix=ITERATION_CHECKPOINT_PREFIX,
        ),
    )
    trainer.run()
    iteration_checkpoints = sorted(
        path.name
        for path in tmp_path.iterdir()
        if path.name.startswith(ITERATION_CHECKPOINT_PREFIX)
    )
    assert iteration_checkpoints == kept
    # The final checkpoint and the evaluation history sit next to the iteration checkpoints.
    assert (tmp_path / "final" / "policy_state.pt").is_file()
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert [record["iteration"] for record in summary["history"]] == [0, 1, 2, 3, 4]
    assert summary["latest"]["success_rate"] == pytest.approx(0.0)
    with pytest.raises(
        ValueError, match="checkpoint_every requires a checkpoint_store"
    ):
        OnPolicyTrainer(
            train_source=episode_source(rollout),
            learner_group=None,
            algorithm=_ALGORITHM,
            config=OnPolicyConfig(
                iterations=1, episodes_per_environment=1, checkpoint_every=1
            ),
            metric_sink=lambda metrics, step: None,
        )


def test_policy_state_round_trips_bfloat16_as_raw_bits(tmp_path):
    tensors = {
        "head.weight": torch.arange(6, dtype=torch.float32)
        .reshape(2, 3)
        .to(torch.bfloat16),
        "head.bias": torch.tensor([1.0, 2.0]),
    }
    state = policy_state_from_tensors(tensors, version=3, exclude_prefixes=("frozen.",))
    assert state.tensors["head.weight"].dtype == np.int16
    assert state.dtypes == {"head.weight": "bfloat16"}
    restored = policy_state_tensors(state)
    assert restored["head.weight"].dtype == torch.bfloat16
    assert torch.equal(restored["head.weight"], tensors["head.weight"])
    assert torch.equal(restored["head.bias"], tensors["head.bias"])
    file = save_policy_snapshot(state, tmp_path / "iteration_1")
    assert file.name == "policy_state.pt"
    loaded = load_policy_snapshot(tmp_path / "iteration_1", version=9)
    assert loaded.version == 9 and loaded.dtypes == {"head.weight": "bfloat16"}
    with pytest.raises(TypeError, match="stored as int16"):
        PolicyState(
            version=1,
            tensors={"x": np.zeros(2, dtype=np.float32)},
            dtypes={"x": "bfloat16"},
        )
    with pytest.raises(ValueError, match="unknown tensor"):
        PolicyState(version=1, tensors={}, dtypes={"x": "bfloat16"})


class _FrozenBackboneModel(nn.Module):
    """A tiny actor-critic with a frozen 'backbone' feeding a trainable head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.head = GaussianActorCritic(3, 2, hidden_sizes=(8,))

    def act(self, observations, *, deterministic=False):
        return self.head.act(self.backbone(observations), deterministic=deterministic)

    def forward(self, observations, actions):
        return self.head(self.backbone(observations), actions)


def test_frozen_submodules_stay_out_of_published_states_and_checkpoints(tmp_path):
    torch.manual_seed(5)
    model = _FrozenBackboneModel()
    learner = LocalLearnerGroup(
        PPO(PPOConfig(epochs=1, minibatch_size=2, seed=8)).learner(
            lambda: copy.deepcopy(model), frozen_submodules=("backbone",)
        ),
        device="cpu",
    )
    state = learner.policy_state(version=1)
    assert all(name.startswith("head.") for name in state.tensors)
    learner.save_checkpoint(str(tmp_path / "iteration_1"))
    path = tmp_path / "iteration_1" / "learner.pt"
    saved = torch.load(path, weights_only=False)  # NOSONAR numpy RNG state, own file
    assert all(name.startswith("head.") for name in saved["modules"]["actor_critic"])
    learner.load_checkpoint(str(tmp_path / "iteration_1"))
    with pytest.raises(ValueError, match="trainable parameters"):
        LocalLearnerGroup(
            PPO(PPOConfig(epochs=1, minibatch_size=2)).learner(
                lambda: copy.deepcopy(model), frozen_submodules=("head",)
            ),
            device="cpu",
        )


def test_learner_micro_batching_huber_and_value_rate_groups():
    torch.manual_seed(6)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    observations = torch.randn(8, 3)
    with torch.no_grad():
        actions, old_log_probs, old_values = model.act(observations)
    from isaac_rl_posttraining.runtimes import LearnerBatch

    batch = LearnerBatch(
        tensors={
            "observations": observations.numpy(),
            "actions": actions.numpy(),
            "old_log_probs": old_log_probs.numpy(),
            "advantages": np.linspace(-1, 1, 8, dtype=np.float32),
            "returns": (old_values + 3.0).numpy(),
            "old_values": old_values.numpy(),
        },
        samples=8,
    )
    algorithm = PPO(
        PPOConfig(
            epochs=1,
            minibatch_size=4,
            micro_batch_size=2,
            value_loss="huber",
            huber_delta=1.0,
            value_learning_rate=1e-3,
            optimizer="adamw",
            adam_betas=(0.9, 0.95),
            weight_decay=0.01,
            seed=8,
        )
    )
    learner = LocalLearnerGroup(
        algorithm.learner(lambda: copy.deepcopy(model)), device="cpu"
    )
    (result,) = learner.train_step([batch])
    metrics = result.metrics
    assert metrics["minibatches"] == pytest.approx(2.0)
    assert metrics["initial_clip_fraction"] == pytest.approx(0.0)
    assert abs(metrics["initial_approx_kl"]) < 1e-4
    assert 0.0 < metrics["value_loss"] < 3.0  # Huber grows linearly past delta
    assert "explained_variance" in metrics
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        PPOConfig(minibatch_size=4, micro_batch_size=3)


def test_evaluation_manifest_records_unpublished_weights_without_a_digest(tmp_path):
    from isaac_rl_posttraining.runners.ppo_launch import evaluation_sink

    evaluation_sink(tmp_path)((), 0, None)

    record = json.loads(
        (tmp_path / "ppo_evaluation_episodes.jsonl").read_text().splitlines()[0]
    )
    assert record["policy_version"] is None and record["policy_sha256"] is None
    assert record["policy_source"] is None
    assert record["episodes"] == []


def test_a_base_evaluation_fingerprints_the_checkpoint_it_scored(tmp_path):
    """play_ppo of the base checkpoint: the workers' start-up weights carry a digest of its weight files."""
    from isaac_rl_posttraining.runners.ppo_launch import evaluation_sink

    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    (checkpoint / "config.json").write_text("{}")
    out = tmp_path / "run"
    out.mkdir()
    sink = evaluation_sink(out, base_checkpoint=str(checkpoint))
    sink((), 0, None)
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"other weights")
    sink((), 0, None)  # computed once per run: the record names what the workers loaded
    first, second = (
        json.loads(line)
        for line in (out / "ppo_evaluation_episodes.jsonl").read_text().splitlines()
    )
    assert first["policy_source"] == "base_checkpoint"
    assert first["base_checkpoint"] == str(checkpoint)
    assert (
        len(first["policy_sha256"]) == 64
        and second["policy_sha256"] == first["policy_sha256"]
    )
    other = evaluation_sink(tmp_path, base_checkpoint=str(checkpoint))
    other((), 0, None)
    changed = json.loads(
        (tmp_path / "ppo_evaluation_episodes.jsonl").read_text().splitlines()[0]
    )
    assert changed["policy_sha256"] != first["policy_sha256"]
    # A published state is fingerprinted from its tensors, whatever the base.
    published = evaluation_sink(out, base_checkpoint=str(checkpoint))
    published((), 1, PolicyState(version=2, tensors={"w": np.ones(2, np.float32)}))
    record = json.loads(
        (out / "ppo_evaluation_episodes.jsonl").read_text().splitlines()[-1]
    )
    assert record["policy_source"] == "published" and record["base_checkpoint"] is None


def test_async_schedule_resubmits_at_once_and_publishes_on_its_cadence():
    """PPO under async: each collection is submitted right after the previous one returns.

    With ``publish_every=2`` the workers see a new version every second update;
    data lags at most one version, and evaluations drain the pipeline.
    """
    torch.manual_seed(1)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    rollout = _CountingRolloutGroup(model)
    records: list[tuple[int, dict]] = []
    trainer = OnPolicyTrainer(
        train_source=episode_source(rollout),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=4,
            episodes_per_environment=1,
            eval_episodes_per_environment=1,
            eval_every=4,
            schedule=ScheduleSpec("async", publish_every=2),
        ),
        metric_sink=lambda metrics, step: records.append((step, dict(metrics))),
    )
    assert trainer.config.schedule_spec.max_policy_lag == 1
    trainer.run()
    training = {
        step: metrics for step, metrics in records if "train/episodes" in metrics
    }
    assert sorted(training) == [1, 2, 3, 4]
    # Publishes after updates 2 and 4 only; collection 3 ran on the version
    # published before it, the others on the initial one.
    assert [training[step]["system/policy_version"] for step in (1, 2, 3, 4)] == [
        1,
        2,
        2,
        3,
    ]
    assert [training[step]["system/policy_lag"] for step in (1, 2, 3, 4)] == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert rollout.set_calls == [1, 2, 3]
    # The evaluation at 4 scored the newest weights.
    evaluations = [step for step, metrics in records if "eval/episodes" in metrics]
    assert evaluations == [0, 4]
    assert rollout.versions_seen[-1] == 3


@pytest.mark.parametrize(
    ("schedule", "match"),
    [
        (ScheduleSpec("async", max_policy_lag=2), "max_policy_lag=1"),
        (ScheduleSpec("async", max_policy_lag=0), "max_policy_lag=1"),
    ],
)
def test_on_policy_async_bounds_the_policy_lag_at_one(schedule, match):
    with pytest.raises(ValueError, match=match):
        OnPolicyConfig(iterations=1, episodes_per_environment=1, schedule=schedule)


def test_dsrl_ppo_resume_restores_the_iteration_counter(tmp_path):
    """DSRL-PPO resumes through the trainer, so a resumed run continues the iteration count."""
    from isaac_rl_posttraining.runners.dsrl_ppo_launch import resume_trainer

    torch.manual_seed(3)
    model = GaussianActorCritic(3, 2, hidden_sizes=(8,))
    OnPolicyTrainer(
        train_source=episode_source(_CountingRolloutGroup(model)),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(
            iterations=2, episodes_per_environment=1, checkpoint_every=1
        ),
        metric_sink=lambda metrics, step: None,
        checkpoint_store=CheckpointStore(
            str(tmp_path), tag_prefix=ITERATION_CHECKPOINT_PREFIX
        ),
    ).run()

    records: list[int] = []
    resumed = OnPolicyTrainer(
        train_source=episode_source(_CountingRolloutGroup(model)),
        learner_group=_local_learner(model),
        algorithm=_ALGORITHM,
        config=OnPolicyConfig(iterations=3, episodes_per_environment=1),
        metric_sink=lambda metrics, step: records.append(step),
    )
    resume_trainer(resumed, tmp_path / "iteration_2")
    assert resumed.iteration == 2
    resumed.run()
    assert records == [3]
