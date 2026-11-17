# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-SDE PPO with policy-server actors: runtime=native on several processes.

Two rollout-worker actors send their scenes to two policy-server actors that
hold the trainable policy, and two learner ranks on a gloo process group
train it; every publish reaches the servers by Ray object reference. The
tiny model's evaluation decodes are deterministic, so they must match one
process exactly; sampled training actions draw from each server's own random
streams (see ``FlowSDEActionChunkPolicy``), so training is checked for
consistency, not bit-equality with one process.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from flow_sde_doubles import TinyFlowSDE, tiny_worker_spec
from on_policy_doubles import episode_source

from isaac_rl_posttraining.algorithms.ppo import PPO, PPOConfig
from isaac_rl_posttraining.runtimes import (
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutRequest,
    WeightsRef,
    open_learner_group,
)
from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Placement,
    Resources,
    Served,
    plan_placement,
)
from isaac_rl_posttraining.training import OnPolicyConfig, OnPolicyTrainer

pytestmark = [pytest.mark.native_ray, pytest.mark.timeout(900)]

NUM_ENVS, WORKERS = 4, 2
EVALUATE = RolloutRequest(episodes_per_environment=1, mode="eval", reset_seed=7)


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.init(num_cpus=16, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _served_runtime(*, learner_ranks: int = 1) -> NativeRuntime:
    placement = Placement(
        envs=Envs(workers=WORKERS),
        policy=Served(replicas=2),
        learner=Learner(ranks=learner_ranks),
    )
    plan = plan_placement(
        placement,
        Resources.uniform(1, 0),
        served_supported=True,
        distributed_learner=True,
    )
    return NativeRuntime(
        NativeRuntimeConfig(device="cpu", seed=21, collect_policy_server_metrics=True),
        plan=plan,
    )


def _decisions(results) -> list[tuple]:
    return [
        (
            result.worker_index,
            decision.environment_index,
            decision.action_chunk.policy_version,
            decision.transition.metadata.policy_version,
            decision.action_chunk.environment_actions.tolist(),
            decision.action_chunk.learner_action.tolist(),
            float(decision.action_chunk.behavior_value[0]),
        )
        for result in results
        for decision in result.decisions
    ]


def _evaluate_in_one_process(weights: WeightsRef | None) -> list[tuple]:
    """The same evaluation with a copy of the policy in each of two in-process workers."""
    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu", policy_mode="local"))
    try:
        group = runtime.open_rollout_group(
            tiny_worker_spec(NUM_ENVS),
            num_rollout_workers=WORKERS,
            num_envs=NUM_ENVS,
            name="local",
        )
        if weights is not None:
            group.set_weights(WeightsRef.of(weights.resolve()))
        return _decisions(group.collect([EVALUATE] * WORKERS))
    finally:
        runtime.close()


def test_policy_server_actors_evaluate_like_local_copies_across_a_publish():
    runtime = _served_runtime()
    try:
        group = runtime.open_rollout_group(
            tiny_worker_spec(NUM_ENVS),
            num_rollout_workers=WORKERS,
            num_envs=NUM_ENVS,
            name="served",
        )
        before = _decisions(group.collect([EVALUATE] * WORKERS))
        model = TinyFlowSDE()
        state = {
            name: (0.5 * value).numpy()
            for name, value in model.state_dict().items()
            if not name.startswith("backbone.")
        }
        weights = WeightsRef.of(PolicyState(version=1, tensors=state))
        assert group.set_weights(weights) == (1,) * WORKERS
        after = _decisions(group.collect([EVALUATE] * WORKERS))
        replicas = group.policy_server_stats()
    finally:
        runtime.close()
    assert before == _evaluate_in_one_process(None)
    assert after == _evaluate_in_one_process(weights)
    assert {row[2] for row in before} == {0} and {row[2] for row in after} == {1}
    # Each worker's scenes went to its own replica, one batch per decision step.
    assert len(replicas) == 2 and all(stats["batch_sizes"] for stats in replicas)


def test_ppo_trains_through_policy_server_actors_with_two_learner_ranks():
    algorithm = PPO(
        PPOConfig(epochs=2, minibatch_size=4, normalize_advantages=False, seed=8)
    )
    runtime = _served_runtime(learner_ranks=2)
    records: list[dict[str, float]] = []
    try:
        learner = open_learner_group(
            runtime,
            algorithm.learner(
                TinyFlowSDE, frozen_submodules=TinyFlowSDE.frozen_submodules
            ),
            num_learner_ranks=2,
            name="ppo",
            local_device="cpu",
        )
        rollout = runtime.open_rollout_group(
            tiny_worker_spec(NUM_ENVS, cache_features=True),
            num_rollout_workers=WORKERS,
            num_envs=NUM_ENVS,
            name="ppo_rollout",
        )
        trainer = OnPolicyTrainer(
            train_source=episode_source(rollout, 2),
            learner_group=learner,
            algorithm=algorithm,
            config=OnPolicyConfig(
                iterations=2,
                episodes_per_environment=2,
                eval_episodes_per_environment=1,
                eval_every=1,
                seed=3,
            ),
            metric_sink=lambda metrics, step: records.append(dict(metrics)),
        )
        trainer.run()
        # The servers hold the learner's last published weights and stamp decisions with their version.
        served = _decisions(rollout.collect([EVALUATE] * WORKERS))
        newest = learner.publish(trainer.policy_version + 1)
        assert newest.handle is not None and newest.inline is None
        rollout.set_weights(newest)
        republished = _decisions(rollout.collect([EVALUATE] * WORKERS))
        # The handle's owner is a learner actor, so read it before close() kills that actor.
        newest = WeightsRef.of(newest.resolve())
    finally:
        runtime.close()
    training = [metrics for metrics in records if "train/episodes" in metrics]
    # Version 1 is the checkpoint published before the first evaluation; each update publishes the next.
    assert len(training) == 2 and trainer.policy_version == 3
    assert all(
        metrics["policy_service/action_mean_batch_size"] > 0 for metrics in training
    )
    assert {row[2] for row in served} == {row[3] for row in served} == {3}
    # The learner moved the policy away from the checkpoint, and the servers act with exactly its weights.
    assert [row[4] for row in served] != [
        row[4] for row in _evaluate_in_one_process(None)
    ]
    assert republished == _evaluate_in_one_process(newest)
