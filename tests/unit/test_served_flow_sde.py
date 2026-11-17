# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-SDE PPO with a served policy: the policy lives in a policy server the rollout workers call.

The tiny flow-SDE model of ``flow_sde_doubles`` runs the real sampling,
log-probability and value math. A served policy in this process decides like a
local copy (sampled training actions too, since one worker draws from the
same random streams in the same order), decisions carry the weight version the
server acted with, and a PPO run trains the same weights either way.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from flow_sde_doubles import (
    STEPS,
    TinyFlowSDE,
    TinyScenes,
    tiny_flow_sde_spec,
    tiny_worker_spec,
)
from on_policy_doubles import episode_source

from isaac_rl_posttraining.algorithms.ppo import PPO, PPOConfig
from isaac_rl_posttraining.runners import flow_sde_ppo_launch
from isaac_rl_posttraining.runners.build import plan_report
from isaac_rl_posttraining.runtimes import (
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutRequest,
    RolloutWorkerContext,
    WeightsRef,
    open_learner_group,
)
from isaac_rl_posttraining.runtimes.batched_worker import BatchedEpisodeRolloutWorker
from isaac_rl_posttraining.runtimes.serving import (
    Batcher,
    InProcessPolicyService,
    connect_policy,
)
from isaac_rl_posttraining.train import compose_train_config
from isaac_rl_posttraining.training import OnPolicyConfig, OnPolicyTrainer


def _runtime(policy_mode: str) -> NativeRuntime:
    return NativeRuntime(
        NativeRuntimeConfig(
            device="cpu",
            policy_mode=policy_mode,
            collect_policy_server_metrics=True,
        )
    )


def _fingerprint(results) -> list[tuple]:
    rows = []
    for result in results:
        for decision in result.decisions:
            chunk = decision.action_chunk
            rows.append(
                (
                    result.worker_index,
                    decision.environment_index,
                    chunk.policy_version,
                    decision.transition.metadata.policy_version,
                    chunk.environment_actions.tolist(),
                    chunk.learner_action.tolist(),
                    None
                    if chunk.behavior_log_prob is None
                    else chunk.behavior_log_prob.tolist(),
                    None
                    if chunk.behavior_value is None
                    else chunk.behavior_value.tolist(),
                    {
                        key: np.asarray(value).tolist()
                        for key, value in (chunk.learner_observation or {}).items()
                    },
                    {
                        key: np.asarray(value).tolist()
                        for key, value in (chunk.cached_features or {}).items()
                    },
                    decision.transition.rewards["env"].tolist(),
                )
            )
    return rows


def _weights(version: int, scale: float) -> WeightsRef:
    """The published (non-frozen) weights of the tiny model with its head scaled."""
    model = TinyFlowSDE()
    tensors = {
        name: (value * scale).numpy()
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
    }
    return WeightsRef.of(PolicyState(version=version, tensors=tensors))


def _collect(
    policy_mode: str, *, mode: str, workers: int, cache_features: bool = False
) -> list[tuple]:
    torch.manual_seed(11)
    runtime = _runtime(policy_mode)
    try:
        group = runtime.open_rollout_group(
            tiny_worker_spec(4, cache_features=cache_features),
            num_rollout_workers=workers,
            num_envs=4,
            name="flow_sde",
        )
        request = RolloutRequest(episodes_per_environment=1, mode=mode, reset_seed=5)
        results = list(group.collect([request] * workers))
        assert group.set_weights(_weights(1, 0.5)) == (1,) * workers
        results.extend(group.collect([request] * workers))
        return _fingerprint(results)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "cache_features", [False, True], ids=["inputs", "cached_features"]
)
def test_served_flow_sde_evaluates_like_a_local_copy(cache_features) -> None:
    """Mean decisions are deterministic, so two workers sharing one server match two local copies."""
    local = _collect("local", mode="eval", workers=2, cache_features=cache_features)
    served = _collect("served", mode="eval", workers=2, cache_features=cache_features)
    assert served == local
    assert {row[2] for row in served} == {row[3] for row in served} == {0, 1}


def test_one_worker_samples_training_actions_like_a_local_copy() -> None:
    """A server with one client draws its stochastic steps and noise in the local order."""
    local = _collect("local", mode="train", workers=1, cache_features=True)
    served = _collect("served", mode="train", workers=1, cache_features=True)
    assert served == local
    # Training decisions carry behaviour log-probabilities and the noised step.
    assert all(row[6] is not None and row[6] != [0.0] for row in served)
    assert {row[8]["stochastic_step"][0] for row in served} <= set(range(STEPS))


def test_a_publish_swaps_the_server_between_collections() -> None:
    """A publish issued during a collection waits for it; the next one acts with the new version."""
    runtime = _runtime("served")
    try:
        group = runtime.open_rollout_group(
            tiny_worker_spec(4), num_rollout_workers=2, num_envs=4, name="flow_sde"
        )
        request = RolloutRequest(episodes_per_environment=1, mode="eval", reset_seed=5)
        group.set_weights(_weights(1, 1.0))
        in_flight = group.collect_async([request] * 2)
        assert group.set_weights(_weights(2, 0.25)) == (2, 2)
        before = _fingerprint(in_flight.result())
        after = _fingerprint(group.collect([request] * 2))
    finally:
        runtime.close()
    assert {row[2] for row in before} == {row[3] for row in before} == {1}
    assert {row[2] for row in after} == {row[3] for row in after} == {2}
    assert [row[4] for row in before] != [row[4] for row in after]


def test_a_worker_refuses_decisions_of_weights_it_was_not_told() -> None:
    service = InProcessPolicyService(
        tiny_flow_sde_spec(), device="cpu", batcher=Batcher()
    )
    try:
        context = RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0, 1),
            device="cpu",
        )
        worker = BatchedEpisodeRolloutWorker(
            context,
            TinyScenes((0, 1)),
            connect_policy(service.handle(), route_index=0),
            chunks_per_episode=2,
            reward_transform=None,
            bootstrap_values="reuse_next_decision",
        )
        # The worker mirrors version 1, but nobody published it to the server.
        worker.set_policy_state(PolicyState(version=1, tensors={}))
        with pytest.raises(
            RuntimeError,
            match=r"policy returned versions \[0\], but worker serves version 1",
        ):
            worker.collect(RolloutRequest(episodes_per_environment=1, mode="eval"))
    finally:
        service.close()


def _train(policy_mode: str) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Two PPO iterations of the flow-SDE recipe's loop on the tiny model; the published weights and metrics."""
    torch.manual_seed(5)
    algorithm = PPO(
        PPOConfig(epochs=2, minibatch_size=8, normalize_advantages=False, seed=8)
    )
    runtime = _runtime(policy_mode)
    records: list[dict[str, float]] = []
    try:
        learner = open_learner_group(
            runtime,
            algorithm.learner(
                TinyFlowSDE, frozen_submodules=TinyFlowSDE.frozen_submodules
            ),
            num_learner_ranks=1,
            name="ppo",
            local_device="cpu",
        )
        rollout = runtime.open_rollout_group(
            tiny_worker_spec(4, cache_features=True),
            num_rollout_workers=1,
            num_envs=4,
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
        published = learner.publish(trainer.policy_version + 1).resolve()
        rollout.close()
        learner.close()
    finally:
        runtime.close()
    return {name: np.array(value) for name, value in published.tensors.items()}, records


def test_ppo_trains_the_same_weights_through_a_served_policy() -> None:
    local, local_records = _train("local")
    served, served_records = _train("served")
    assert served.keys() == local.keys() and not any(
        name.startswith("backbone.") for name in served
    )
    for name, value in local.items():
        np.testing.assert_array_equal(served[name], value, err_msg=name)
    # The policy moved, and the served run logged its server's batches.
    assert any(
        not np.array_equal(value, _weights(1, 1.0).resolve().tensors[name])
        for name, value in served.items()
    )
    training = [metrics for metrics in served_records if "train/episodes" in metrics]
    assert len(training) == 2 and all(
        metrics["system/policy_version"] >= 1 for metrics in training
    )
    assert any(
        key.startswith("policy_service/") for metrics in training for key in metrics
    )
    assert not any(
        key.startswith("policy_service/")
        for metrics in local_records
        for key in metrics
    )


# ----------------------------------------------------------------------
# The dry run
# ----------------------------------------------------------------------

#: The flow-SDE recipe served on one 8-GPU node: three simulators, two policy servers, three learner ranks.
SERVED_PPO = (
    "algorithm=ppo",
    "base_model.ckpt=/ckpt/RLinf-Gr00t-SFT-Stack-cube",
    "placement.gpus_per_node=8",
    "placement.policy.mode=served",
    "training.num_envs=48",
    "placement.envs.workers=3",
    "ppo.global_batch_size=576",
)


def _plan(*overrides: str):
    return plan_report(compose_train_config([*SERVED_PPO, *overrides]))


def test_the_dry_run_plans_a_served_ppo_policy_on_native() -> None:
    overrides = (
        "runtime=native",
        "placement.policy.gpus=2",
        "placement.learner.ranks=3",
    )
    report = _plan(*overrides)
    assert not report.checks.errors and "served policy" in report.checks.checked
    assert any(
        line.startswith("plan.placement.policy=served by 2 replica(s)")
        for line in report.lines
    )
    plan = flow_sde_ppo_launch.build_plan(
        compose_train_config([*SERVED_PPO, *overrides]), require_servers=False
    )
    assert flow_sde_ppo_launch.rollout_worker_spec(plan).supports_served_policy


def test_rlinf_serves_ppo_to_workers_that_simulate_on_their_own_gpus() -> None:
    """Each PPO worker simulates Isaac Lab on a GPU, so RLInf splits the cards after the learner with the servers."""
    report = _plan("runtime=rlinf", "placement.learner.ranks=3")
    assert not report.checks.errors
    assert "plan.placement.envs=3 worker(s) on node0 gpus [5, 6, 7]" in report.lines
    assert any(
        line.startswith(
            "plan.placement.policy=served by 2 replica(s) x 1 gpu on node0 gpus [3, 4]"
        )
        for line in report.lines
    )


def test_the_dry_run_refuses_a_served_policy_the_workers_cannot_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    building = flow_sde_ppo_launch.rollout_worker_spec
    monkeypatch.setattr(
        flow_sde_ppo_launch,
        "rollout_worker_spec",
        lambda plan: dataclasses.replace(building(plan), supports_served_policy=False),
    )
    report = _plan("runtime=native", "placement.learner.ranks=3")
    (error,) = report.checks.errors
    assert "the rollout workers of algorithm=ppo keep a copy of the policy each" in str(
        error
    )
    assert not _plan(
        "runtime=native", "placement.learner.ranks=3", "placement.policy.mode=local"
    ).checks.errors
