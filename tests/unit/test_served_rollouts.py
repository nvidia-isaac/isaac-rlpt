# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout workers driving a served policy decide exactly as with a policy of their own."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from entry_doubles import ScriptedSimulator, scripted_environment

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    BatchedEpisodeRolloutWorker,
    BatchedWorkerSpec,
    GymRolloutWorker,
    GymWorkerSpec,
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutRequest,
    RolloutWorkerContext,
)
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy


class _NoisyPolicy(RolloutPolicy):
    """Acts with the observation plus process-RNG noise, so call order shows in the decisions."""

    def __init__(
        self, *, valued: bool = False, encodes: bool = True, execution_length: int = 2
    ) -> None:
        self.valued = valued
        self.encodes = encodes
        self.execution_length = execution_length
        self.version = 0
        self.features = PolicyFeatures(
            encodes_observations=True, values_observations=True, behavior_values=valued
        )

    def action_chunks(self, observations, *, mode, session_ids):
        chunks = []
        for observation in observations:
            state = np.asarray(observation["state"], dtype=np.float32)
            action = float(state[0]) + float(torch.randn(1).item())
            extra = {}
            if self.valued:
                extra = {
                    "behavior_log_prob": np.asarray([-0.5], np.float32),
                    "behavior_value": np.asarray([float(state.sum())], np.float32),
                }
            chunks.append(
                ActionChunk.from_arrays(
                    learner_action=np.asarray([[action]], np.float32),
                    environment_actions=np.full(
                        (1, self.execution_length, 1), action, np.float32
                    ),
                    execution_length=self.execution_length,
                    learner_action_kind="latent",
                    policy_version=self.version,
                    learner_observation={"state": state[None]}
                    if self.encodes
                    else None,
                    **extra,
                )
            )
        return tuple(chunks)

    def encode_observations(self, observations, *, session_ids):
        return tuple(
            {"state": np.asarray(o["state"], np.float32) + 100.0} for o in observations
        )

    def encode_and_value_next_observations(self, observations, *, session_ids):
        return tuple(
            (
                {"state": np.asarray(o["state"], np.float32) + 100.0},
                np.asarray([float(np.sum(o["state"])) * 0.5], np.float32),
            )
            for o in observations
        )

    def set_policy_state(self, state: PolicyState) -> int:
        self.version = state.version
        return state.version


class _CountingEnv:
    """One environment whose state counts its steps; it ends after ``horizon`` steps."""

    def __init__(self, offset: float, horizon: int = 5) -> None:
        self.offset = offset
        self.horizon = horizon
        self.t = 0

    def _observation(self):
        return {"state": np.asarray([self.offset + self.t, 1.0], np.float32)}

    def reset(self, *, seed=None):
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        return (
            self._observation(),
            float(np.asarray(action).sum()),
            done,
            False,
            {"success": done},
        )

    def close(self):
        pass


class _CountingBatchedEnv:
    """Two lockstep environments; the first ends after 3 steps, the second never does."""

    num_envs = 2

    def __init__(self) -> None:
        self.t = 0
        self.received: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

    def _observation(self):
        return {"state": np.asarray([[self.t, 0.0], [10.0 + self.t, 1.0]], np.float32)}

    def reset(self, *, seed=None):
        self.t = 0
        return self._observation(), {}

    def step(self, actions):
        self.received.append(np.array(actions, copy=True))
        self.t += 1
        terminated = np.asarray([self.t == 3, False])
        return (
            self._observation(),
            np.asarray(actions, np.float64).sum(axis=1),
            terminated,
            np.zeros(2, dtype=bool),
            [{"success": bool(terminated[0])}, {"success": False}],
        )

    def close(self):
        pass


class _MaskedBatchedEnv(_CountingBatchedEnv):
    accepts_active_mask = True

    def step(self, actions, active=None):
        self.masks.append(np.array(active, copy=True))
        return super().step(actions)


def _gym_spec(**options) -> GymWorkerSpec:
    return GymWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: _CountingEnv(offset=10.0 * context.environment_indices[0])
        ),
        environment=scripted_environment(3),
        phase="train",
        policy=lambda context: _NoisyPolicy(**options),
        supports_batched_policy=True,
    )


def _batched_spec(env_factory=_CountingBatchedEnv, **options) -> BatchedWorkerSpec:
    return BatchedWorkerSpec(
        entry=ScriptedSimulator(lambda context: env_factory()),
        environment=scripted_environment(2),
        phase="train",
        policy=lambda context: _NoisyPolicy(),
        chunks_per_episode=3,
        supports_served_policy=True,
        **options,
    )


def _fingerprint(results) -> list[tuple]:
    rows = []
    for result in results:
        for decision in result.decisions:
            transition = decision.transition
            rows.append(
                (
                    decision.environment_index,
                    transition.learner_action.value.tolist(),
                    transition.env_action.commands.tolist(),
                    {
                        key: value.tolist()
                        for key, value in transition.next_observation.arrays().items()
                    },
                    None
                    if transition.policy_outputs.next_value is None
                    else float(transition.policy_outputs.next_value),
                    transition.rewards["env"].tolist(),
                )
            )
    return rows


def _collect(spec, *, policy_mode: str, num_envs: int, workers: int, requests):
    torch.manual_seed(0)
    # One worker after another: the policy draws from one global generator, so the order is fixed.
    runtime = NativeRuntime(
        NativeRuntimeConfig(
            device="cpu", policy_mode=policy_mode, concurrent_collection=False
        )
    )
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=workers, num_envs=num_envs, name="parity"
        )
        results = []
        for request in requests:
            results.extend(group.collect([request] * workers))
        group.set_policy_state(PolicyState(version=1, tensors={}))
        results.extend(group.collect([requests[-1]] * workers))
        return _fingerprint(results)
    finally:
        runtime.close()


def test_served_and_local_gym_workers_decide_identically() -> None:
    requests = [
        RolloutRequest(decisions_per_environment=2, mode="warmup"),
        RolloutRequest(episodes_per_environment=1, mode="train"),
    ]
    local = _collect(
        _gym_spec(),
        policy_mode="local",
        num_envs=3,
        workers=3,
        requests=requests,
    )
    served = _collect(
        _gym_spec(), policy_mode="served", num_envs=3, workers=3, requests=requests
    )
    assert local == served
    assert len(local) > 6


def test_served_and_local_batched_workers_decide_identically() -> None:
    requests = [RolloutRequest(episodes_per_environment=1, mode="train")]
    local = _collect(
        _batched_spec(),
        policy_mode="local",
        num_envs=2,
        workers=1,
        requests=requests,
    )
    served = _collect(
        _batched_spec(),
        policy_mode="served",
        num_envs=2,
        workers=1,
        requests=requests,
    )
    assert local == served
    # The terminal next observation is still encoded (the +100 layout).
    assert all(row[3]["state"][0] >= 100.0 for row in local)


def test_auto_serving_keeps_batched_workers_local() -> None:
    config = NativeRuntimeConfig(device="cpu")
    assert not config.serves_policy(_batched_spec())
    assert config.serves_policy(_gym_spec())
    with pytest.raises(ValueError, match="supports_served_policy"):
        NativeRuntimeConfig(device="cpu", policy_mode="served").serves_policy(
            BatchedWorkerSpec(
                entry=ScriptedSimulator(lambda context: _CountingBatchedEnv()),
                environment=scripted_environment(2),
                phase="train",
                policy=lambda context: _NoisyPolicy(),
                chunks_per_episode=3,
            )
        )


def test_concurrent_collection_batches_workers_through_the_shared_server() -> None:
    runtime = NativeRuntime(
        NativeRuntimeConfig(
            device="cpu",
            policy_mode="served",
            concurrent_collection=True,
            policy_max_wait_ms=50.0,
            collect_policy_server_metrics=True,
        )
    )
    try:
        # Decisions without learner observations send only acting requests,
        # so the three workers act in lockstep.
        group = runtime.open_rollout_group(
            _gym_spec(encodes=False),
            num_rollout_workers=3,
            num_envs=3,
            name="concurrent",
        )
        results = group.collect(
            [RolloutRequest(decisions_per_environment=3, mode="train")] * 3
        )
        assert [result.worker_index for result in results] == [0, 1, 2]
        assert all(len(result.decisions) == 3 for result in results)
        (stats,) = group.policy_server_stats()
        assert max(stats["batch_sizes"]) > 1
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("policy_mode", "workers", "setting", "concurrent"),
    [
        (
            "served",
            3,
            "auto",
            True,
        ),  # a served policy with several workers batches them
        ("served", 1, "auto", False),
        (
            "auto",
            3,
            "auto",
            False,
        ),  # the default in-process sharing keeps today's order
        ("local", 3, "auto", False),
        ("served", 3, False, False),  # an explicit setting wins
        ("auto", 3, True, True),
    ],
)
def test_concurrent_collection_auto_is_on_for_served_policies_with_several_workers(
    policy_mode, workers, setting, concurrent
) -> None:
    config = NativeRuntimeConfig(
        device="cpu", policy_mode=policy_mode, concurrent_collection=setting
    )
    runtime = NativeRuntime(config)
    try:
        group = runtime.open_rollout_group(
            _gym_spec(encodes=False),
            num_rollout_workers=workers,
            num_envs=workers,
            name="auto",
        )
        assert group._concurrent is concurrent
        group.collect(
            [RolloutRequest(decisions_per_environment=2, mode="train")] * workers
        )
        # Served runs record the server's batch sizes by default (policy_service/* metrics).
        stats = group.policy_server_stats()
        assert bool(stats) is (policy_mode == "served")
    finally:
        runtime.close()


def test_concurrent_collection_takes_auto_or_a_boolean() -> None:
    with pytest.raises(TypeError, match="concurrent_collection"):
        NativeRuntimeConfig(concurrent_collection="yes")  # type: ignore[arg-type]
    assert not NativeRuntimeConfig(
        collect_policy_server_metrics="auto"
    ).collects_server_metrics
    assert NativeRuntimeConfig(
        base_mode="served", collect_policy_server_metrics="auto"
    ).collects_server_metrics


def _gym_worker(*, fused: bool) -> GymRolloutWorker:
    return GymRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        ),
        _CountingEnv(offset=0.0, horizon=5),
        _NoisyPolicy(valued=True),
        fused_act_and_value=fused,
    )


def test_fused_act_and_value_matches_two_requests() -> None:
    fingerprints = {}
    for fused in (False, True):
        torch.manual_seed(0)
        worker = _gym_worker(fused=fused)
        results = [
            worker.collect(RolloutRequest(episodes_per_environment=2, mode="train")),
            worker.collect(RolloutRequest(decisions_per_environment=2, mode="train")),
            worker.collect(RolloutRequest(decisions_per_environment=1, mode="train")),
        ]
        fingerprints[fused] = _fingerprint(results)
        if fused:
            # The fused request replaced the separate bootstrap-value requests
            # of decisions whose episode went on within the same collection.
            assert (
                results[0].timings.get("next_observation_inference_seconds", 0.0) > 0.0
            )
    assert fingerprints[True] == fingerprints[False]
    assert all(row[4] is not None for row in fingerprints[True])


def test_fused_act_and_value_needs_a_valuing_policy() -> None:
    class _Valueless(_NoisyPolicy):
        def __init__(self):
            super().__init__()
            self.features = PolicyFeatures(encodes_observations=True)

    with pytest.raises(TypeError, match="fused_act_and_value"):
        GymRolloutWorker(
            RolloutWorkerContext(
                worker_index=0,
                num_rollout_workers=1,
                environment_indices=(0,),
                device="cpu",
            ),
            _CountingEnv(offset=0.0),
            _Valueless(),
            fused_act_and_value=True,
        )


def _batched_worker(env, **options) -> BatchedEpisodeRolloutWorker:
    return BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0, 1),
            device="cpu",
        ),
        env,
        _NoisyPolicy(),
        chunks_per_episode=3,
        reward_transform=None,
        **options,
    )


def test_finished_environments_execute_zeros_by_default_or_hold_their_last_action() -> (
    None
):
    for inactive_action in ("zeros", "hold"):
        torch.manual_seed(0)
        env = _CountingBatchedEnv()
        result = _batched_worker(env, inactive_action=inactive_action).collect(
            RolloutRequest(episodes_per_environment=1, mode="train")
        )
        # Environment 0 ends at step 3, the first step of its second chunk.
        last_command = result.decisions[2].transition.env_action.commands[-1]
        after_end = [received[0] for received in env.received[3:]]
        if inactive_action == "zeros":
            assert all(np.all(command == 0.0) for command in after_end)
        else:
            assert all(np.allclose(command, last_command) for command in after_end)
            assert not np.allclose(last_command, 0.0)


def test_environments_that_accept_it_receive_the_active_mask() -> None:
    env = _MaskedBatchedEnv()
    _batched_worker(env).collect(
        RolloutRequest(episodes_per_environment=1, mode="train")
    )
    assert env.masks[0].tolist() == [True, True]
    assert env.masks[-1].tolist() == [False, True]


def test_bootstrap_values_accept_their_old_names() -> None:
    spec = _batched_spec(successor_values="next_decision")
    assert spec.bootstrap_values == "reuse_next_decision"
    assert spec.successor_values is None
    assert _batched_spec(bootstrap_values="policy").bootstrap_values == (
        "evaluate_next_observation"
    )
    worker = _batched_worker(_CountingBatchedEnv(), successor_values="next_decision")
    assert worker.bootstrap_values == "reuse_next_decision"
    with pytest.raises(ValueError, match="contradicts"):
        _batched_spec(successor_values="policy", bootstrap_values="reuse_next_decision")
    with pytest.raises(ValueError, match="bootstrap_values must be one of"):
        _batched_spec(bootstrap_values="previous_decision")


def test_time_limit_bootstrapping_needs_a_value_pass() -> None:
    with pytest.raises(ValueError, match="bootstrap_at_time_limit"):
        _batched_spec(
            bootstrap_values="reuse_next_decision", bootstrap_at_time_limit=True
        )
    assert _batched_spec(bootstrap_at_time_limit=True).bootstrap_at_time_limit
