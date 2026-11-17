# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real RLInf scheduler smoke test for the provider-neutral contract.

Each test starts a Ray runtime, so the ``rlinf_cluster`` marker keeps them out
of the CPU lane unless ``ISAAC_RL_POSTTRAINING_RLINF_CLUSTER=1`` requests them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from isaac_rl_posttraining.data import (
    DynamicObservation,
    LearnerAction,
    PolicyOutputs,
    Transition,
    TransitionMetadata,
)
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    GymWorkerSpec,
    PolicyContext,
    PolicyState,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
)
from isaac_rl_posttraining.runtimes.rlinf import RLInfRuntime, RLInfRuntimeConfig
from isaac_rl_posttraining.runtimes.watchdog import RolloutTimeoutError


def _transition(environment_index: int) -> Transition:
    """One flat-state step, built here because a Ray worker cannot import the test helpers."""
    state = DynamicObservation.from_mapping(
        {"state": np.asarray([environment_index], dtype=np.float32)}
    )
    return Transition(
        episode_id=environment_index,
        observation=state,
        next_observation=state,
        learner_action=LearnerAction("latent", np.zeros(1, dtype=np.float32)),
        env_action=None,
        rewards={"env": np.zeros(1, dtype=np.float32)},
        terminated=np.asarray([False]),
        truncated=np.asarray([False]),
        success=np.asarray([False]),
        behavior_source="policy",
        metadata=TransitionMetadata(step_in_episode=0, extra={}),
        policy_outputs=PolicyOutputs(),
    )


class _RemoteWorker:
    def __init__(self, context: RolloutWorkerContext) -> None:
        self.context = context
        self.policy_version = 0

    def collect(self, request: RolloutRequest) -> RolloutResult:
        assert request.decisions_per_environment == 1
        decisions = []
        for environment_index in self.context.environment_indices:
            chunk = ActionChunk(
                learner_action=np.zeros((1, 1), dtype=np.float32),
                environment_actions=np.zeros((1, 1, 1), dtype=np.float32),
                execution_length=1,
                learner_action_kind="latent",
                policy_version=self.policy_version,
            )
            decisions.append(
                CollectedDecision(
                    environment_index=environment_index,
                    transition=_transition(environment_index),
                    action_chunk=chunk,
                )
            )
        return RolloutResult(
            worker_index=self.context.worker_index,
            decisions=tuple(decisions),
            env_steps=len(decisions),
            episodes_completed=0,
            policy_version=self.policy_version,
        )

    def set_policy_state(self, state: PolicyState) -> int:
        self.policy_version = state.version
        return state.version

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


def _worker_spec(context: RolloutWorkerContext) -> _RemoteWorker:
    return _RemoteWorker(context)


class _FrozenWorker(_RemoteWorker):
    """Worker 1's simulator never returns from its step."""

    def collect(self, request: RolloutRequest) -> RolloutResult:
        if self.context.worker_index == 1:
            import time

            time.sleep(3600)
        return super().collect(request)


def _frozen_worker_spec(context: RolloutWorkerContext) -> _FrozenWorker:
    return _FrozenWorker(context)


class _OneStepEnv:
    def reset(self):
        return {"state": np.zeros(1, dtype=np.float32)}, {}

    def step(self, action):
        assert np.asarray(action).shape == (1,)
        return (
            {"state": np.ones(1, dtype=np.float32)},
            1.0,
            True,
            False,
            {"success": True},
        )

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _BatchedPolicy:
    def __init__(self) -> None:
        self.version = 0

    def action_chunks(self, observations, *, mode, session_ids):
        batch_size = len(observations)
        return tuple(
            ActionChunk.from_arrays(
                learner_action=np.asarray(
                    [float(session_id), float(batch_size)],
                    dtype=np.float32,
                ),
                environment_actions=np.zeros((1, 1), dtype=np.float32),
                execution_length=1,
                learner_action_kind="latent",
                source="warmup" if decision_mode == "warmup" else "policy",
                policy_version=self.version,
            )
            for decision_mode, session_id in zip(mode, session_ids, strict=True)
        )

    def reset_episode(self, session_id: int) -> None:
        del session_id

    def set_policy_state(self, state: PolicyState) -> int:
        self.version = state.version
        return self.version


class _OneStepSimulator(SimulatorEntry):
    name = "one_step"

    def make_environment(self, spec, phase, context):
        del spec, phase
        assert len(context.environment_indices) == 1
        return _OneStepEnv()


def _policy_spec(context: PolicyContext) -> _BatchedPolicy:
    assert context.placement == "policy_server"
    return _BatchedPolicy()


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
def test_rlinf_distributes_workers_without_conflating_env_count() -> None:
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="cpu",
            serialize_specs_by_value=True,
        )
    )
    try:
        group = runtime.open_rollout_group(
            _worker_spec,
            num_rollout_workers=2,
            num_envs=5,
            name="contract_smoke",
        )
        results = group.collect(
            [
                RolloutRequest(decisions_per_environment=1),
                RolloutRequest(decisions_per_environment=1),
            ]
        )
        assert group.num_rollout_workers == 2
        assert group.num_envs == 5
        assert [len(result.decisions) for result in results] == [3, 2]
        assert {
            decision.environment_index
            for result in results
            for decision in result.decisions
        } == set(range(5))
    finally:
        runtime.close()


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
def test_rlinf_a_frozen_rollout_worker_fails_the_collection_naming_it() -> None:
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(device="cpu", rollout_timeout_seconds=3.0)
    )
    try:
        group = runtime.open_rollout_group(
            _frozen_worker_spec, num_rollout_workers=2, num_envs=2, name="frozen"
        )
        started = time.monotonic()
        with pytest.raises(
            RolloutTimeoutError,
            match=r"no result within 3 s from RLInf rollout worker 1 \(cpu\)",
        ):
            group.collect([RolloutRequest(decisions_per_environment=1)] * 2)
        assert time.monotonic() - started < 30.0
    finally:
        started = time.monotonic()
        runtime.close()
        # The frozen worker is killed, not asked to close behind its stuck collection.
        assert time.monotonic() - started < 30.0


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
def test_rlinf_shared_policy_server_microbatches_requests() -> None:
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="cpu",
            policy_mode="served",
            num_policy_servers=1,
            collect_policy_server_metrics=True,
        )
    )
    spec = GymWorkerSpec(
        entry=_OneStepSimulator(),
        environment=EnvironmentSpec(
            env_name="one_step/task", num_envs=4, max_episode_steps=1, seed=0
        ),
        phase="train",
        policy=_policy_spec,
        supports_batched_policy=True,
    )
    try:
        group = runtime.open_rollout_group(
            spec,
            num_rollout_workers=4,
            num_envs=4,
            name="policy_server_smoke",
        )
        assert group.set_policy_state(
            PolicyState(version=1, tensors={"weight": np.ones(1)})
        ) == (1, 1, 1, 1)
        results = group.collect([RolloutRequest(decisions_per_environment=1)] * 4)
        assert [result.worker_index for result in results] == [0, 1, 2, 3]
        stats = group.policy_server_stats()
        batch_sizes = [size for server in stats for size in server["batch_sizes"]]
        assert max(batch_sizes) > 1
    finally:
        runtime.close()


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
def test_rlinf_training_and_evaluation_groups_share_the_policy_servers() -> None:
    """An evaluation group adds its policy to the training group's servers instead of starting its own."""
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="cpu",
            policy_mode="served",
            num_policy_servers=1,
            collect_policy_server_metrics=True,
            policy_request_timeout_seconds=120.0,
        )
    )

    def spec(num_envs: int, phase: str) -> GymWorkerSpec:
        return GymWorkerSpec(
            entry=_OneStepSimulator(),
            environment=EnvironmentSpec(
                env_name="one_step/task", num_envs=num_envs, max_episode_steps=1, seed=0
            ),
            phase=phase,
            policy=_policy_spec,
            supports_batched_policy=True,
        )

    try:
        training = runtime.open_rollout_group(
            spec(4, "train"), num_rollout_workers=4, num_envs=4, name="training"
        )
        servers = runtime._policy_servers
        assert servers is not None
        evaluation = runtime.open_rollout_group(
            spec(2, "eval"), num_rollout_workers=2, num_envs=2, name="evaluation"
        )
        assert runtime._policy_servers is servers
        # Each group's policy keeps its own weights on the shared servers.
        assert training.set_policy_state(PolicyState(version=3, tensors={})) == (3,) * 4
        assert (
            evaluation.set_policy_state(PolicyState(version=5, tensors={})) == (5,) * 2
        )
        trained = training.collect([RolloutRequest(decisions_per_environment=1)] * 4)
        assert {
            decision.action_chunk.policy_version
            for result in trained
            for decision in result.decisions
        } == {3}
        evaluated = evaluation.collect(
            [RolloutRequest(decisions_per_environment=1)] * 2
        )
        assert {
            decision.action_chunk.policy_version
            for result in evaluated
            for decision in result.decisions
        } == {5}
        assert sum(
            len(server["batch_sizes"]) for server in evaluation.policy_server_stats()
        )
        evaluation.close()
        # The servers outlive the evaluation group.
        assert (
            len(training.collect([RolloutRequest(decisions_per_environment=1)] * 4))
            == 4
        )
    finally:
        runtime.close()


class _TwoStepBatchedEnv:
    """Two lockstep environments whose episodes end after two steps."""

    num_envs = 2

    def __init__(self) -> None:
        self.t = 0

    def _observation(self):
        return {"state": np.full((2, 1), float(self.t), dtype=np.float32)}

    def reset(self, *, seed=None):
        del seed
        self.t = 0
        return self._observation(), {}

    def step(self, actions):
        assert np.asarray(actions).shape == (2, 1)
        self.t += 1
        done = np.full(2, self.t >= 2)
        infos = [{"success": bool(flag)} for flag in done]
        return self._observation(), np.ones(2), done, np.zeros(2, bool), infos

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _BatchedSimulator(SimulatorEntry):
    name = "two_step"

    def make_environment(self, spec, phase, context):
        del spec, phase
        assert len(context.environment_indices) == 2
        return _TwoStepBatchedEnv()


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
def test_rlinf_policy_servers_serve_batched_rollout_workers() -> None:
    from isaac_rl_posttraining.runtimes import BatchedWorkerSpec

    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="cpu",
            policy_mode="served",
            num_policy_servers=1,
            collect_policy_server_metrics=True,
            policy_request_timeout_seconds=120.0,
        )
    )
    spec = BatchedWorkerSpec(
        entry=_BatchedSimulator(),
        environment=EnvironmentSpec(
            env_name="two_step/task", num_envs=4, max_episode_steps=2, seed=0
        ),
        phase="train",
        policy=_policy_spec,
        chunks_per_episode=2,
        bootstrap_values="reuse_next_decision",
        supports_served_policy=True,
    )
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=2, num_envs=4, name="batched_policy_server"
        )
        assert group.set_policy_state(
            PolicyState(version=1, tensors={"weight": np.ones(1)})
        ) == (1, 1)
        results = group.collect([RolloutRequest(episodes_per_environment=1)] * 2)
        assert [len(result.decisions) for result in results] == [4, 4]
        # Each batched worker's call reaches the server as one batch of its two sessions.
        for result in results:
            for decision in result.decisions:
                session_id, batch_size = decision.transition.learner_action.value
                assert session_id == decision.environment_index
                assert batch_size >= 2
        stats = group.policy_server_stats()
        assert max(size for server in stats for size in server["batch_sizes"]) >= 2
    finally:
        runtime.close()
