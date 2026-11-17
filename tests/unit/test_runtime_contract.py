# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral runtime contract and native-base-model tests."""

from __future__ import annotations

import dataclasses
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from conftest import make_transition
from entry_doubles import scripted_worker_spec

from isaac_rl_posttraining.data import Transition
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    GymRolloutWorker,
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
    SparseOutcomeReward,
    create_runtime,
    partition_environments,
    policy_state_from_tensors,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.concurrency import ThreadCall
from isaac_rl_posttraining.runtimes.rlinf.cluster import RLINF_VERSION, RLInfCluster
from isaac_rl_posttraining.runtimes.rlinf.config import RLInfRuntimeConfig
from isaac_rl_posttraining.runtimes.rlinf.layout import (
    policy_server_accelerators,
    resolve_policy_server_count,
    rollout_worker_accelerators,
)
from isaac_rl_posttraining.runtimes.rlinf.rollout import RLInfCollectFuture
from isaac_rl_posttraining.training import RuntimeDataSource
from isaac_rl_posttraining.training.data_source import collector_metrics


def _action_chunk(policy_version: int) -> ActionChunk:
    return ActionChunk(
        learner_action=np.zeros((1, 2), dtype=np.float32),
        environment_actions=np.zeros((1, 1, 3), dtype=np.float32),
        execution_length=1,
        learner_action_kind="latent",
        policy_version=policy_version,
    )


class _Worker:
    def __init__(self, context: RolloutWorkerContext) -> None:
        self.context = context
        self.policy_version = 0
        self.closed = False
        self._episode_ids = dict.fromkeys(context.environment_indices, 0)

    def collect(self, request: RolloutRequest) -> RolloutResult:
        if request.decisions_per_environment is not None:
            count = request.decisions_per_environment
        else:
            count = request.episodes_per_environment
        assert count is not None
        decisions: list[CollectedDecision] = []
        for environment_index in self.context.environment_indices:
            for _ in range(count):
                episode_id = self._episode_ids[environment_index]
                chunk = _action_chunk(self.policy_version)
                transition = make_transition(
                    episode_id=episode_id,
                    obs=np.asarray([environment_index], dtype=np.float32),
                    action=np.zeros(2, dtype=np.float32),
                    reward=1.0,
                    next_obs=np.asarray([environment_index + 1], dtype=np.float32),
                    done=True,
                    info={"success": True},
                )
                decisions.append(
                    CollectedDecision(
                        environment_index=environment_index,
                        transition=transition,
                        action_chunk=chunk,
                    )
                )
                self._episode_ids[environment_index] += 1
        return RolloutResult(
            worker_index=self.context.worker_index,
            decisions=tuple(decisions),
            env_steps=len(decisions),
            episodes_completed=len(decisions),
            policy_version=self.policy_version,
            timings={
                "action_inference_seconds": 0.5,
                "environment_step_seconds": 0.25,
                "policy_calls": 1.0,
                "environment_step_calls": float(len(decisions)),
            },
        )

    def set_policy_state(self, state: PolicyState) -> int:
        self.policy_version = state.version
        return self.policy_version

    def close(self) -> None:
        self.closed = True


class _SlowWorker(_Worker):
    """A worker whose collection takes long enough for a weight update to arrive meanwhile."""

    def collect(self, request: RolloutRequest) -> RolloutResult:
        time.sleep(0.2)
        return super().collect(request)


class _WorkerSpec:
    def __init__(self) -> None:
        self.contexts: list[RolloutWorkerContext] = []
        self.workers: list[_Worker] = []

    def __call__(self, context: RolloutWorkerContext) -> _Worker:
        self.contexts.append(context)
        worker = _Worker(context)
        self.workers.append(worker)
        return worker


class _Sink:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def push(self, transition: Transition) -> None:
        assert isinstance(transition, Transition)
        self.transitions.append(transition)


class _OneStepEnv:
    def __init__(self, *, succeeds: bool) -> None:
        self.succeeds = succeeds
        self.closed = False

    def reset(self):
        return {"state": np.zeros(1, dtype=np.float32)}, {}

    def step(self, action):
        del action
        return (
            {"state": np.ones(1, dtype=np.float32)},
            1.0 if self.succeeds else 0.0,
            self.succeeds,
            not self.succeeds,
            {"success": self.succeeds},
        )

    def close(self) -> None:
        self.closed = True


class _BatchedPolicy:
    def __init__(self) -> None:
        self.version = 0
        self.closed = False

    def action_chunks(self, observations, *, mode, session_ids):
        del mode, session_ids
        return tuple(_action_chunk(self.version) for _ in observations)

    def reset_episode(self, *, session_id):
        del session_id

    def set_policy_state(self, state):
        self.version = state.version
        return self.version

    def close(self) -> None:
        self.closed = True


def test_partition_environments_distinguishes_envs_from_workers() -> None:
    assert partition_environments(10, 3) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )
    assignments = partition_environments(512, 64)
    assert len(assignments) == 64
    assert all(len(assignment) == 8 for assignment in assignments)
    assert tuple(index for assignment in assignments for index in assignment) == tuple(
        range(512)
    )


def test_partition_environments_rejects_idle_workers() -> None:
    with pytest.raises(ValueError, match="cannot exceed num_envs"):
        partition_environments(2, 3)


def test_native_runtime_assigns_multiple_envs_per_worker() -> None:
    runtime = NativeRuntime()
    spec = _WorkerSpec()
    group = runtime.open_rollout_group(
        spec,
        num_rollout_workers=2,
        num_envs=5,
        name="test",
    )

    assert [context.environment_indices for context in spec.contexts] == [
        (0, 1, 2),
        (3, 4),
    ]
    results = group.collect(
        [
            RolloutRequest(decisions_per_environment=1),
            RolloutRequest(decisions_per_environment=1),
        ]
    )
    assert [len(result.decisions) for result in results] == [3, 2]
    assert [result.worker_index for result in results] == [0, 1]

    runtime.close()
    assert all(worker.closed for worker in spec.workers)


def test_native_collect_async_applies_new_weights_after_the_collection_in_flight() -> (
    None
):
    runtime = NativeRuntime()
    group = runtime.open_rollout_group(
        _SlowWorker, num_rollout_workers=1, num_envs=1, name="test"
    )
    call = group.collect_async([RolloutRequest(decisions_per_environment=1)])
    # Published while the collection runs: takes effect once it has finished.
    assert group.set_policy_state(
        PolicyState(version=1, tensors={"weight": np.ones(1)})
    ) == (1,)
    (result,) = call.result()
    assert result.policy_version == 0
    (result,) = group.collect([RolloutRequest(decisions_per_environment=1)])
    assert result.policy_version == 1
    runtime.close()


def test_native_gym_runtime_shares_one_policy_and_preserves_sparse_rewards() -> None:
    environments = []
    policies = []

    def make_environment(context):
        env = _OneStepEnv(succeeds=context.environment_indices[0] == 0)
        environments.append(env)
        return env

    def make_policy(context):
        del context
        policy = _BatchedPolicy()
        policies.append(policy)
        return policy

    spec = scripted_worker_spec(
        make_environment,
        make_policy,
        num_envs=2,
        reward_transform=SparseOutcomeReward(
            success_reward=0.0,
            failure_reward=-1.0,
            step_reward=0.0,
        ),
        supports_batched_policy=True,
    )
    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    group = runtime.open_rollout_group(
        spec,
        num_rollout_workers=2,
        num_envs=2,
        name="shared-native-policy",
    )

    results = group.collect(
        (
            RolloutRequest(episodes_per_environment=1),
            RolloutRequest(episodes_per_environment=1),
        )
    )

    assert len(policies) == 1
    # The workers record the simulator's reward and the shaped outcome side by side.
    assert [
        result.decisions[0].transition.decision_reward(
            "sparse_outcome", aggregation="last"
        )
        for result in results
    ] == [0.0, -1.0]
    assert [
        result.decisions[0].transition.reward("sparse_outcome").tolist()
        for result in results
    ] == [
        [0.0],
        [-1.0],
    ]
    assert [
        result.decisions[0].transition.reward("env").tolist() for result in results
    ] == [[1.0], [0.0]]
    assert group.set_policy_state(PolicyState(version=1, tensors={})) == (1, 1)
    runtime.close()
    assert policies[0].closed
    assert all(environment.closed for environment in environments)


def test_runtime_data_source_ingests_each_environment_once() -> None:
    runtime = NativeRuntime()
    spec = _WorkerSpec()
    group = runtime.open_rollout_group(
        spec,
        num_rollout_workers=2,
        num_envs=5,
        name="data-source",
    )
    sink = _Sink()
    source = RuntimeDataSource(group, sink)
    source.publish(
        policy_state_from_tensors(torch.nn.Linear(1, 1).state_dict(), version=1)
    )

    metrics = source.run(can_train=True)

    assert source.num_envs == 5
    assert source.decisions_collected_last_run == 5
    assert source.total_episodes == 5
    assert len(sink.transitions) == 5
    assert len({transition.episode_id for transition in sink.transitions}) == 5
    assert metrics["num_episodes"] == 5
    # two workers, one policy call each, 0.5 s per call; 5 step calls of 0.25 s per worker
    assert source.last_timings["policy_calls"] == pytest.approx(2.0)
    assert metrics["collector/decode_s_per_decision"] == pytest.approx(0.5)
    assert metrics["collector/cycle_s_per_decision"] == pytest.approx(0.75)
    assert metrics["collector/env_step_s_per_step"] == pytest.approx(0.1)
    source.close()


def test_collector_metrics_report_only_the_phases_the_workers_timed() -> None:
    assert collector_metrics({}) == {}
    assert collector_metrics({"environment_reset_seconds": 1.5}) == {
        "collector/env_reset_s": 1.5
    }
    full = collector_metrics(
        {
            "action_inference_seconds": 2.0,
            "environment_step_seconds": 8.0,
            "successor_inference_seconds": 1.0,
            "environment_reset_seconds": 3.0,
            "policy_calls": 4.0,
            "environment_step_calls": 32.0,
        }
    )
    assert full == {
        "collector/decode_s_per_decision": 0.5,
        "collector/cycle_s_per_decision": 2.75,
        "collector/env_step_s_per_step": 0.25,
        "collector/env_reset_s": 3.0,
    }


def test_runtime_factory_defaults_to_native_without_optional_imports() -> None:
    runtime = create_runtime("native")
    assert isinstance(runtime, NativeRuntime)
    runtime.close()


def test_action_chunk_keeps_learner_and_environment_actions_separate() -> None:
    chunk = ActionChunk.from_arrays(
        learner_action=np.asarray([1.0, 2.0], dtype=np.float32),
        environment_actions=np.arange(12, dtype=np.float32).reshape(4, 3),
        execution_length=2,
        learner_action_kind="latent",
    )
    with pytest.raises(ValueError, match="needs its kind"):
        ActionChunk.from_arrays(
            learner_action=np.asarray([1.0, 2.0], dtype=np.float32),
            environment_actions=np.arange(12, dtype=np.float32).reshape(4, 3),
            execution_length=2,
        )

    assert chunk.learner_action is not None
    assert chunk.learner_action.shape == (1, 2)
    assert chunk.environment_actions.shape == (1, 4, 3)
    assert chunk.execution_actions.shape == (1, 2, 3)


def test_action_chunk_carries_cached_features_per_batch_element() -> None:
    def chunk(features: float) -> ActionChunk:
        return ActionChunk.from_arrays(
            learner_action=np.zeros((1, 2), dtype=np.float32),
            environment_actions=np.zeros((1, 4, 3), dtype=np.float32),
            execution_length=2,
            learner_action_kind="latent",
            cached_features={"backbone": np.full((1, 5), features, dtype=np.float32)},
        )

    combined = ActionChunk.concatenate([chunk(1.0), chunk(2.0)])

    assert combined.cached_features is not None
    assert combined.cached_features["backbone"].shape == (2, 5)
    np.testing.assert_array_equal(
        combined.for_environment(1).cached_features["backbone"], [[2.0] * 5]
    )
    with pytest.raises(ValueError, match="cached_features"):
        ActionChunk.from_arrays(
            learner_action=None,
            environment_actions=np.zeros((1, 4, 3), dtype=np.float32),
            execution_length=2,
            cached_features={"backbone": np.zeros(5, dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="cached features"):
        ActionChunk.concatenate(
            [
                chunk(1.0),
                ActionChunk.from_arrays(
                    learner_action=np.zeros((1, 2), dtype=np.float32),
                    environment_actions=np.zeros((1, 4, 3), dtype=np.float32),
                    execution_length=2,
                    learner_action_kind="latent",
                ),
            ]
        )


def test_rlinf_auto_policy_serving_is_capability_driven() -> None:
    gpu = RLInfRuntimeConfig(device="gpu")
    cpu = RLInfRuntimeConfig(device="cpu")

    assert gpu.serves_policy(supports_batched_policy=True)
    assert not gpu.serves_policy(supports_batched_policy=False)
    assert not cpu.serves_policy(supports_batched_policy=True)


def test_rlinf_served_policy_requires_batched_policy_support() -> None:
    config = RLInfRuntimeConfig(policy_mode="served")
    with pytest.raises(ValueError, match="batched inference"):
        config.serves_policy(supports_batched_policy=False)


def test_rlinf_auto_policy_server_count_is_capacity_bounded() -> None:
    assert (
        resolve_policy_server_count(
            num_envs=8,
            available_accelerators=4,
            requested=None,
        )
        == 2
    )
    assert (
        resolve_policy_server_count(
            num_envs=128,
            available_accelerators=8,
            requested=None,
        )
        == 8
    )
    # No more servers than rollout workers: a server no worker routes to would sit idle.
    assert (
        resolve_policy_server_count(
            num_envs=16, available_accelerators=3, requested=None, num_rollout_workers=2
        )
        == 2
    )


@pytest.mark.parametrize(
    ("accelerators", "workers", "served", "simulates", "servers", "placed"),
    [
        # LIBERO, local policy: one card each while they last, then sharing them evenly.
        (5, 4, False, False, (), (1, 2, 3, 4)),
        (2, 4, False, False, (), (1, 1, 1, 1)),
        (3, 4, False, False, (), (1, 1, 2, 2)),
        # Isaac Lab, served base model: the servers take their share, the simulators the rest.
        (4, 2, True, True, (1,), (2, 3)),
        # One rollout card: servers and workers share it.
        (2, 4, True, True, (1,), (1, 1, 1, 1)),
    ],
)
def test_rlinf_layout_places_servers_and_workers_by_one_rule(
    accelerators, workers, served, simulates, servers, placed
) -> None:
    config = RLInfRuntimeConfig(device="gpu", first_accelerator=1)
    found: tuple[int, ...] = ()
    if served:
        found = policy_server_accelerators(
            config,
            accelerators=accelerators,
            num_envs=16,
            num_rollout_workers=workers,
            workers_need_accelerators=simulates,
        )
    assert found == servers
    assert (
        rollout_worker_accelerators(
            config,
            accelerators=accelerators,
            num_rollout_workers=workers,
            servers=found,
        )
        == placed
    )


def test_rlinf_an_evaluation_group_starts_on_the_cards_training_leaves_free() -> None:
    """Evaluation workers holding a policy copy go next to the open training workers, not onto them."""
    from isaac_rl_posttraining.runtimes.rlinf import RLInfRuntime

    config = RLInfRuntimeConfig(device="gpu", first_accelerator=1)
    runtime = RLInfRuntime(config)
    scheduler = SimpleNamespace(FlexiblePlacementStrategy=lambda cards: cards)
    cluster = SimpleNamespace(num_accelerators=4)

    def place(workers: int) -> tuple[int, ...]:
        return runtime._rollout_placement(
            scheduler, cluster, num_rollout_workers=workers, needs_accelerators=True
        )[1]

    training = place(2)
    assert training == (1, 2)
    runtime._groups.append(SimpleNamespace(closed=False, accelerators=training))
    assert place(1) == (3,)
    runtime._groups[0].closed = True
    assert place(1) == (1,)


def test_rlinf_served_workers_without_a_gpu_simulator_leave_the_cards_to_servers() -> (
    None
):
    config = RLInfRuntimeConfig(device="gpu", first_accelerator=1)
    assert policy_server_accelerators(
        config,
        accelerators=8,
        num_envs=16,
        num_rollout_workers=16,
        workers_need_accelerators=False,
    ) == (1, 2, 3, 4)
    shared = RLInfRuntimeConfig(
        device="gpu", first_accelerator=0, rollout_workers_share_accelerator=True
    )
    assert rollout_worker_accelerators(
        shared, accelerators=2, num_rollout_workers=3
    ) == (0, 0, 0)


def test_rlinf_rollout_groups_past_the_last_gpu_say_how_to_share_it() -> None:
    config = RLInfRuntimeConfig(device="gpu", first_accelerator=1)
    with pytest.raises(ValueError, match="set runtime.rlinf.first_accelerator=0"):
        rollout_worker_accelerators(config, accelerators=1, num_rollout_workers=1)


def test_rlinf_exclusive_accelerators_refuse_a_second_group_on_one_card() -> None:
    cluster = RLInfCluster(RLInfRuntimeConfig(device="gpu"))
    assert cluster.claim_accelerators(
        1, 1, "rollout workers 'training'", exclusive=True
    ) == (1,)
    with pytest.raises(
        ValueError, match="placement overlap.*evaluation.*accelerator 1.*training"
    ):
        cluster.claim_accelerators(1, 1, "rollout workers 'evaluation'", exclusive=True)
    cluster.release_accelerators((1,))
    assert cluster.claim_accelerators(
        1, 1, "rollout workers 'evaluation'", exclusive=True
    ) == (1,)


def test_rlinf_shared_accelerators_let_two_groups_share_a_card() -> None:
    """With sharing allowed the registry records nothing: there is no overlap to catch."""
    config = RLInfRuntimeConfig(device="gpu", exclusive_accelerators=False)
    cluster = RLInfCluster(config)
    assert cluster.claim_accelerators(
        0, 1, "rollout workers 'training'", exclusive=False
    ) == (0,)
    assert cluster.claim_accelerators(
        0, 1, "rollout workers 'evaluation'", exclusive=False
    ) == (0,)
    cluster.release_accelerators((0,))


def test_rlinf_cluster_start_keeps_the_drivers_interpreter(monkeypatch) -> None:
    from isaac_rl_posttraining.runtimes.rlinf import cluster as cluster_module

    seen: list[str] = []

    class FakeCluster:
        def __init__(self, num_nodes: int = 1) -> None:
            seen.append("cluster")

        @staticmethod
        def has_initialized() -> bool:
            return False

    monkeypatch.setattr(
        cluster_module,
        "load_scheduler",
        lambda: types.SimpleNamespace(Cluster=FakeCluster),
    )
    monkeypatch.setattr(
        cluster_module, "keep_driver_interpreter", lambda: seen.append("keep")
    )
    RLInfCluster(RLInfRuntimeConfig(device="gpu")).ensure()
    assert seen == ["keep", "cluster"]


def test_rlinf_evaluation_reuses_the_training_policy_servers_accelerators() -> None:
    """A second rollout group with a shared policy adds it to the run's servers and claims no GPU.

    Before, the evaluation group started servers of its own on ``first_accelerator``,
    which the training group's servers held (``exclusive_accelerators=true``).
    """
    pytest.importorskip("ray")  # the RLInf runtime's policy handles are Ray handles
    from isaac_rl_posttraining.runtimes.rlinf import RLInfRuntime

    runtime = RLInfRuntime(RLInfRuntimeConfig(device="gpu", first_accelerator=1))
    added: list[str] = []

    class _Call:
        def __init__(self, value=None) -> None:
            self.value = value

        def wait(self):
            return self.value

    group = SimpleNamespace(
        worker_info_list=[SimpleNamespace(worker=object())],
        add_policy=lambda name, spec, *_: added.append(name) or _Call([0]),
    )

    def open_policy_group(scheduler, cluster, spec, *, num_envs, name, policy_name):
        claimed = runtime._cluster_owner.claim_accelerators(
            1, 1, f"policy servers {name!r}", exclusive=True
        )
        added.append(policy_name)
        return group, claimed, 1

    runtime._open_policy_group = open_policy_group  # type: ignore[method-assign]
    training = runtime._serve_policy(None, None, object(), num_envs=4, name="training")
    evaluation = runtime._serve_policy(
        None, None, object(), num_envs=4, name="evaluation"
    )
    assert len(added) == 2 and added[0] != added[1]
    assert (training.handle().policy, evaluation.handle().policy) == tuple(added)
    assert training.handle().actors == evaluation.handle().actors
    assert runtime._cluster_owner._accelerator_claims == {1: "policy servers 'policy'"}


def test_rlinf_exclusive_accelerators_must_be_a_boolean() -> None:
    with pytest.raises(TypeError, match="exclusive_accelerators"):
        RLInfRuntimeConfig(exclusive_accelerators="no")  # type: ignore[arg-type]


def test_rlinf_rollout_workers_share_accelerator_must_be_a_boolean() -> None:
    with pytest.raises(TypeError, match="rollout_workers_share_accelerator"):
        RLInfRuntimeConfig(rollout_workers_share_accelerator="yes")  # type: ignore[arg-type]


def test_thread_call_stamps_when_its_work_ended_not_when_it_was_read() -> None:
    """A controller that reads a round late still learns the round's own duration."""
    call = ThreadCall(lambda: time.sleep(0.05) or "done")
    assert call.result() == "done"
    read_at = time.time()
    time.sleep(0.05)
    finished_at = call.finished_at()
    assert finished_at is not None and finished_at <= read_at + 1e-3
    assert time.time() - finished_at >= 0.05


def test_rlinf_collect_future_ends_with_its_last_worker() -> None:
    """The actors stamp their own clocks; the round's end is the last stamp, whatever order the results arrive in."""
    results = {
        "k0": RolloutResult(
            worker_index=0,
            decisions=(),
            env_steps=0,
            episodes_completed=0,
            policy_version=0,
            timings={},
        ),
        "k1": RolloutResult(
            worker_index=1,
            decisions=(),
            env_steps=0,
            episodes_completed=0,
            policy_version=0,
            timings={},
        ),
    }

    class _Channel:
        def get(self, *, key):
            return results[key]

    fake_ray = types.SimpleNamespace(get=lambda handles: [(1, 200.5), (0, 190.0)])
    future = RLInfCollectFuture(("h1", "h0"), _Channel(), ("k0", "k1"), ((0,), (1,)))
    assert future.finished_at() is None
    with mock.patch.dict(sys.modules, {"ray": fake_ray}):
        assert [result.worker_index for result in future.result()] == [0, 1]
    assert future.finished_at() == pytest.approx(200.5)


def test_rlinf_runtime_pin_matches_optional_extra() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    extra = re.search(r"^rlinf = \[(.*)\]$", pyproject, re.MULTILINE)
    assert extra is not None, "pyproject must declare the [rlinf] extra"
    assert f'"rlinf=={RLINF_VERSION}"' in extra.group(1)


def test_unbatched_gym_spec_defaults_to_one_worker_per_env() -> None:
    spec = scripted_worker_spec(lambda context: None, lambda: None, num_envs=8)
    assert resolve_num_rollout_workers(spec, num_envs=8, configured=None) == 8


class _ImageEnv:
    """Three-step episodes with a small camera image in every observation."""

    def __init__(self) -> None:
        self.t = 0

    def _observation(self):
        return {
            "state": np.full(2, float(self.t), dtype=np.float32),
            "ego_view_image": np.full((4, 6, 3), self.t, dtype=np.uint8),
        }

    def reset(self, *, seed=None):
        del seed
        self.t = 0
        return self._observation(), {
            "initial_state_fingerprint": "scalar-case",
            "case_seed": 42,
            "round_index": 3,
        }

    def step(self, action):
        del action
        self.t += 1
        return self._observation(), 0.0, False, False, {"success": False}

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _ConstantPolicy:
    def __init__(self) -> None:
        self.modes = []

    def action_chunk(self, observation, *, mode):
        self.modes.append(mode)
        return ActionChunk(
            learner_action=np.zeros((1, 2), dtype=np.float32),
            environment_actions=np.zeros((1, 1, 3), dtype=np.float32),
            execution_length=1,
            learner_action_kind="latent",
        )

    def set_policy_state(self, state):
        return state.version


def test_rollout_request_mode_derives_can_train() -> None:
    assert RolloutRequest(decisions_per_environment=1).mode == "train"
    assert RolloutRequest(decisions_per_environment=1, mode="warmup").can_train is False
    assert RolloutRequest(decisions_per_environment=1, mode="train").can_train is True
    assert RolloutRequest(episodes_per_environment=1, mode="eval").can_train is True
    with pytest.raises(ValueError, match="mode"):
        RolloutRequest(decisions_per_environment=1, mode="explore")
    with pytest.raises(ValueError, match="max_decisions_per_episode"):
        RolloutRequest(episodes_per_environment=1, max_decisions_per_episode=0)


def test_gym_worker_forwards_mode_and_captures_frames() -> None:
    policy = _ConstantPolicy()
    worker = GymRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        ),
        _ImageEnv(),
        policy,
    )

    result = worker.collect(
        RolloutRequest(
            episodes_per_environment=1,
            mode="eval",
            capture_frames=True,
            max_decisions_per_episode=3,
        )
    )

    assert policy.modes == ["eval", "eval", "eval"]
    assert result.episodes_completed == 1
    assert [
        bool(decision.transition.truncated[-1]) for decision in result.decisions
    ] == [
        False,
        False,
        True,
    ]
    # The cap ended the episode, so the last decision carries its cause and no success.
    assert (
        result.decisions[-1].transition.metadata.extra["decision_cap_reached"] is True
    )
    assert (
        result.decisions[-1].transition.metadata.extra["terminal_reason"]
        == "decision_cap"
    )
    assert not result.decisions[-1].transition.succeeded
    first_extra = result.decisions[0].transition.metadata.extra
    assert first_extra["initial_state_fingerprint"] == "scalar-case"
    assert first_extra["case_seed"] == 42
    assert first_extra["round_index"] == 3
    assert (
        "initial_state_fingerprint" not in result.decisions[1].transition.metadata.extra
    )
    # One frame per decision plus the terminal observation, images only.
    assert len(result.frames) == 4
    assert all(set(frame) == {"ego_view_image"} for frame in result.frames)
    assert [int(frame["ego_view_image"][0, 0, 0]) for frame in result.frames] == [
        0,
        1,
        2,
        3,
    ]
    plain = worker.collect(
        RolloutRequest(episodes_per_environment=1, max_decisions_per_episode=2)
    )
    assert plain.frames is None


def test_runtime_data_source_publish_requires_advancing_versions() -> None:
    runtime = NativeRuntime()
    spec = _WorkerSpec()
    group = runtime.open_rollout_group(
        spec, num_rollout_workers=1, num_envs=1, name="publish"
    )
    source = RuntimeDataSource(group, _Sink())

    source.publish(PolicyState(version=3, tensors={"weight": np.ones(1)}))
    assert spec.workers[0].policy_version == 3
    with pytest.raises(ValueError, match="advance beyond 3"):
        source.publish(PolicyState(version=3, tensors={"weight": np.ones(1)}))
    source.run(can_train=False)
    source.close()


def test_runtime_data_source_finish_checks_the_version_the_round_was_submitted_with() -> (
    None
):
    """A publish issued while a round is in flight applies afterwards, so the round's results carry
    the version at submission; ``finish`` expects that one, the next ``run`` the new one."""
    runtime = NativeRuntime()
    group = runtime.open_rollout_group(
        _SlowWorker, num_rollout_workers=1, num_envs=1, name="ahead"
    )
    sink = _Sink()
    source = RuntimeDataSource(group, sink)

    pending = source.start(can_train=False)
    assert pending.mode == "warmup" and pending.policy_version == 0
    source.publish(PolicyState(version=1, tensors={"weight": np.ones(1)}))
    source.finish(pending)
    assert source.total_episodes == 1 and len(sink.transitions) == 1

    # The new weights applied once the round had ended: the next round carries version 1.
    source.run(can_train=True)
    assert source.total_episodes == 2 and len(sink.transitions) == 2
    source.close()


def test_runtime_data_source_allows_one_round_in_flight() -> None:
    runtime = NativeRuntime()
    group = runtime.open_rollout_group(
        _WorkerSpec(), num_rollout_workers=1, num_envs=1, name="one"
    )
    source = RuntimeDataSource(group, _Sink())

    pending = source.start(can_train=False)
    with pytest.raises(RuntimeError, match="already in flight"):
        source.start(can_train=False)
    source.finish(pending)
    with pytest.raises(RuntimeError, match="last start"):
        source.finish(pending)
    source.close()


def test_runtime_data_source_close_waits_for_the_round_in_flight() -> None:
    runtime = NativeRuntime()
    spec = _WorkerSpec()
    group = runtime.open_rollout_group(
        spec, num_rollout_workers=1, num_envs=1, name="closing"
    )
    spec.workers[0].__class__ = _SlowWorker
    source = RuntimeDataSource(group, _Sink())

    source.start(can_train=False)
    source.close()

    assert spec.workers[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        source.run(can_train=False)


class _ChunkEnv:
    """Executes whole chunks; ends the episode after five simulator steps."""

    def __init__(self) -> None:
        self.t = 0
        self.chunk_calls = []

    def reset(self, *, seed=None):
        del seed
        self.t = 0
        return {"state": np.zeros(1, np.float32)}, {}

    def step(self, action):
        raise AssertionError("a chunked worker must not step one action at a time")

    def step_chunk(self, actions):
        self.chunk_calls.append(np.asarray(actions).shape)
        executed = min(len(actions), 5 - self.t)
        self.t += executed
        done = self.t >= 5
        rewards = (0.0,) * (executed - 1) + (1.0 if done else 0.0,)
        return (
            {"state": np.full(1, float(self.t), np.float32)},
            rewards,
            done,
            False,
            {"success": done},
        )

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


class _ThreeStepPolicy:
    def action_chunk(self, observation, *, mode):
        return ActionChunk(
            learner_action=np.zeros((1, 1), np.float32),
            environment_actions=np.zeros((1, 3, 2), np.float32),
            execution_length=3,
            learner_action_kind="latent",
        )

    def set_policy_state(self, state):
        return state.version


def test_gym_worker_executes_whole_chunks_when_chunked_step_is_on() -> None:
    env = _ChunkEnv()
    worker = GymRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        ),
        env,
        _ThreeStepPolicy(),
        chunked_step=True,
    )

    result = worker.collect(RolloutRequest(episodes_per_environment=1))

    assert env.chunk_calls == [(3, 2), (3, 2)]
    assert result.env_steps == 5
    assert [d.transition.steps for d in result.decisions] == [3, 2]
    assert [
        d.transition.decision_reward("env", aggregation="last")
        for d in result.decisions
    ] == [0.0, 1.0]
    assert [d.transition.env_action.commands.shape for d in result.decisions] == [
        (3, 2),
        (2, 2),
    ]
    assert (
        result.decisions[-1].transition.terminated[-1]
        and result.decisions[-1].transition.succeeded
    )
    with pytest.raises(TypeError, match="step_chunk"):
        GymRolloutWorker(
            worker.context,
            _OneStepEnv(succeeds=True),
            _ThreeStepPolicy(),
            chunked_step=True,
        )


def test_rollout_result_without_frames_survives_a_tensor_field_skeleton() -> None:
    """RLInf sends dataclasses as a skeleton whose tensor-classified fields are None; empty containers count as such."""
    result = RolloutResult(
        worker_index=0,
        decisions=(),
        env_steps=0,
        episodes_completed=0,
        policy_version=1,
        timings={"collect": 0.0},
    )
    assert result.frames is None
    skeleton = dataclasses.replace(result, frames=None)
    assert skeleton.frames is None and skeleton.policy_version == 1
    # A worker that recorded no timings sends an empty mapping, which travels as None too.
    skeleton = dataclasses.replace(result, timings=None)
    assert skeleton.timings is None and skeleton.policy_version == 1


def test_rlinf_server_count_refusal_names_the_placement_key() -> None:
    with pytest.raises(ValueError, match=r"placement\.policy\.replicas=3 .* 2 GPU"):
        resolve_policy_server_count(num_envs=8, available_accelerators=2, requested=3)
