# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""runtime=native on several processes: rollout workers, policy servers and learner ranks as Ray actors.

Each test compares a multi-process run with the same run in one process:
deterministic workers and policies give bit-identical decisions, and learner
ranks on a gloo process group give the one-rank weights up to float rounding.
The ``native_ray`` marker skips the module without the ``ray`` extra.
"""

from __future__ import annotations

import copy
import functools
import time

import numpy as np
import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from entry_doubles import ScriptedSimulator, scripted_environment

from isaac_rl_posttraining.algorithms.ppo import (
    PPO,
    GaussianActorCritic,
    PPOConfig,
)
from isaac_rl_posttraining.algorithms.replay_learner import (
    ReplayLearner,
    replica_learner_spec,
)
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig
from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler
from isaac_rl_posttraining.modules.mlp import (
    MLPActor,
    MLPActorConfig,
    MLPCritic,
    MLPCriticConfig,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    GymWorkerSpec,
    LearnerBatch,
    LocalLearnerGroup,
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.base import ActorFailedError
from isaac_rl_posttraining.runtimes.native_ray.cluster import seed_process
from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Local,
    Placement,
    Resources,
    Served,
    plan_placement,
)
from isaac_rl_posttraining.runtimes.policy import (
    PolicyFeatures,
    RolloutPolicy,
)
from isaac_rl_posttraining.runtimes.watchdog import RolloutTimeoutError

pytestmark = [pytest.mark.native_ray, pytest.mark.timeout(900)]

CPU_ONLY = Resources.uniform(1, 0)


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    """One local Ray instance for the module; each runtime joins it instead of starting its own."""
    ray.init(num_cpus=16, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _plan(placement: Placement):
    return plan_placement(
        placement, CPU_ONLY, served_supported=True, distributed_learner=True
    )


def _runtime(placement: Placement | None, **settings) -> NativeRuntime:
    config = NativeRuntimeConfig(device="cpu", **settings)
    return NativeRuntime(config, plan=None if placement is None else _plan(placement))


class _CountingEnv:
    """An environment whose state counts its steps from an offset; it ends after ``horizon`` steps."""

    def __init__(self, offset: float, horizon: int = 4) -> None:
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
        reward = float(np.asarray(action, np.float64).sum())
        return self._observation(), reward, done, False, {"success": done}

    def close(self):
        pass


class _CountingBatchedEnv:
    """The environments of one worker in lockstep; environment ``i`` ends after ``3 + i % 2`` steps."""

    def __init__(self, indices: tuple[int, ...]) -> None:
        self.indices = np.asarray(indices, np.float32)
        self.num_envs = len(indices)
        self.t = 0

    def _observation(self):
        return {
            "state": np.stack(
                [10.0 * self.indices + self.t, np.ones(self.num_envs, np.float32)],
                axis=1,
            ).astype(np.float32)
        }

    def reset(self, *, seed=None):
        self.t = 0
        return self._observation(), {}

    def step(self, actions):
        self.t += 1
        terminated = self.t >= 3 + (self.indices % 2)
        return (
            self._observation(),
            np.asarray(actions, np.float64).sum(axis=1),
            terminated,
            np.zeros(self.num_envs, dtype=bool),
            [{"success": bool(done)} for done in terminated],
        )

    def close(self):
        pass


class _ScaledPolicy(RolloutPolicy):
    """Acts ``scale * state[0] + session``: deterministic, so any routing or batching shows."""

    features = PolicyFeatures(encodes_observations=False)

    def __init__(self) -> None:
        self.scale = 1.0
        self.version = 0

    def action_chunks(self, observations, *, mode, session_ids):
        chunks = []
        for observation in observations:
            state = np.asarray(observation["state"], np.float32)
            action = np.float32(self.scale * state[0] + 0.01 * state[1])
            chunks.append(
                ActionChunk.from_arrays(
                    learner_action=np.asarray([[action]], np.float32),
                    environment_actions=np.full((1, 2, 1), action, np.float32),
                    execution_length=2,
                    learner_action_kind="latent",
                    policy_version=self.version,
                )
            )
        return tuple(chunks)

    def set_policy_state(self, state: PolicyState) -> int:
        if "scale" in state.tensors:
            self.scale = float(state.tensors["scale"][0])
        self.version = state.version
        return state.version


def _policy_spec(context) -> _ScaledPolicy:
    return _ScaledPolicy()


def _gym_spec(num_envs: int) -> GymWorkerSpec:
    return GymWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: _CountingEnv(offset=10.0 * context.environment_indices[0])
        ),
        environment=scripted_environment(num_envs),
        phase="train",
        policy=_policy_spec,
        supports_batched_policy=True,
    )


def _batched_spec(num_envs: int) -> BatchedWorkerSpec:
    return BatchedWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: _CountingBatchedEnv(context.environment_indices)
        ),
        environment=scripted_environment(num_envs),
        phase="train",
        policy=_policy_spec,
        chunks_per_episode=3,
        supports_served_policy=True,
    )


def _fingerprint(results) -> list[tuple]:
    return [
        (
            result.worker_index,
            decision.environment_index,
            decision.action_chunk.policy_version,
            decision.transition.learner_action.value.tolist(),
            {
                key: value.tolist()
                for key, value in decision.transition.next_observation.arrays().items()
            },
            decision.transition.rewards["env"].tolist(),
            bool(decision.transition.terminated[-1]),
        )
        for result in results
        for decision in result.decisions
    ]


def _weights(version: int, scale: float) -> WeightsRef:
    return WeightsRef.of(
        PolicyState(version=version, tensors={"scale": np.asarray([scale], np.float32)})
    )


def _collect(runtime: NativeRuntime, spec, *, num_envs: int, workers: int, request):
    """Two rounds with a weight update between them, the second started asynchronously."""
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=workers, num_envs=num_envs, name="parity"
        )
        assert group.set_weights(_weights(1, 1.0)) == (1,) * workers
        results = list(group.collect([request] * workers))
        assert group.set_weights(_weights(2, 0.5)) == (2,) * workers
        pending = group.collect_async([request] * workers)
        results.extend(pending.result())
        assert pending.finished_at() is not None
        return _fingerprint(results)
    finally:
        runtime.close()


@pytest.mark.parametrize("batched", [False, True], ids=["gym", "batched"])
def test_rollout_workers_with_their_own_policy_match_one_process(batched):
    num_envs, workers = (4, 2) if batched else (3, 3)
    spec = _batched_spec(num_envs) if batched else _gym_spec(num_envs)
    request = (
        RolloutRequest(episodes_per_environment=1)
        if batched
        else RolloutRequest(decisions_per_environment=3)
    )
    single = _collect(
        _runtime(None, policy_mode="local"),
        spec,
        num_envs=num_envs,
        workers=workers,
        request=request,
    )
    several = _collect(
        _runtime(Placement(envs=Envs(workers=workers), policy=Local())),
        spec,
        num_envs=num_envs,
        workers=workers,
        request=request,
    )
    assert several == single
    assert {row[2] for row in several} == {1, 2}


@pytest.mark.parametrize("routing", ["modulo", "least_loaded"])
def test_a_policy_served_by_two_replicas_matches_one_in_process_server(routing):
    spec = _gym_spec(4)
    request = RolloutRequest(decisions_per_environment=3)
    single = _collect(
        _runtime(None, policy_mode="served"),
        spec,
        num_envs=4,
        workers=4,
        request=request,
    )
    runtime = _runtime(
        Placement(envs=Envs(workers=4), policy=Served(replicas=2, routing=routing)),
        collect_policy_server_metrics=True,
    )
    group = runtime.open_rollout_group(
        spec, num_rollout_workers=4, num_envs=4, name="served"
    )
    try:
        group.set_weights(_weights(1, 1.0))
        results = list(group.collect([request] * 4))
        group.set_weights(_weights(2, 0.5))
        results.extend(group.collect([request] * 4))
        served = [
            len(replica["batch_sizes"]) for replica in group.policy_server_stats()
        ]
        router = group._owned_service.handle().router
        clients = None if router is None else ray.get(router.clients.remote())
    finally:
        runtime.close()
    assert _fingerprint(results) == single
    # Both replicas answered requests; least-loaded routing spread the four clients evenly.
    assert len(served) == 2 and all(count > 0 for count in served)
    assert clients == (None if routing == "modulo" else [2, 2])


class _EndlessBatchedEnv(_CountingBatchedEnv):
    """Lockstep environments that never end, so every decision of a round has all of them."""

    def step(self, actions):
        observation, reward, _, truncated, infos = super().step(actions)
        return (
            observation,
            reward,
            np.zeros(self.num_envs, dtype=bool),
            truncated,
            infos,
        )


class _SlowPolicy(_ScaledPolicy):
    """A policy whose every call takes as long as a small decode."""

    def action_chunks(self, observations, *, mode, session_ids):
        time.sleep(0.02)
        return super().action_chunks(observations, mode=mode, session_ids=session_ids)


def _slow_policy_spec(context) -> _SlowPolicy:
    return _SlowPolicy()


def test_a_policy_server_batches_the_requests_of_concurrent_rollout_workers():
    """Three batched workers of four scenes share one replica: training calls coalesce, evaluation calls do not."""
    spec = BatchedWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: _EndlessBatchedEnv(context.environment_indices)
        ),
        environment=scripted_environment(12),
        phase="train",
        policy=_slow_policy_spec,
        chunks_per_episode=8,
        supports_served_policy=True,
    )
    # The default batching (no cap, a 5 ms window); before, the window was 0 and every call held one worker's scenes.
    runtime = _runtime(
        Placement(envs=Envs(workers=3), policy=Served(replicas=1)),
        collect_policy_server_metrics=True,
    )
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=3, num_envs=12, name="coalesce"
        )
        group.set_weights(_weights(1, 1.0))
        group.collect([RolloutRequest(decisions_per_environment=8, mode="train")] * 3)
        (training,) = group.policy_server_stats()
        group.reset_policy_server_stats()
        group.collect([RolloutRequest(decisions_per_environment=2, mode="eval")] * 3)
        (evaluation,) = group.policy_server_stats()
    finally:
        runtime.close()
    sizes = training["batch_sizes"]
    assert sum(sizes) == 3 * 4 * 8
    assert max(sizes) == 12 and np.mean(sizes) > 6, sizes
    assert set(evaluation["batch_sizes"]) == {4}


def _ppo_learner(model, ranks: int):
    """One minibatch per update over every rank's samples (8 in all), so one rank and two take the same step."""
    algorithm = PPO(
        PPOConfig(
            epochs=2, minibatch_size=8 // ranks, normalize_advantages=False, seed=8
        )
    )
    return algorithm.learner(functools.partial(copy.deepcopy, model))


def _ppo_batch(rank_count: int, seed: int) -> list[LearnerBatch]:
    """One PPO update's data: every rank gets its slice of the same 8 samples."""
    generator = np.random.default_rng(seed)
    samples = 8
    tensors = {
        "observations": generator.normal(size=(samples, 3)).astype(np.float32),
        "actions": generator.uniform(-0.9, 0.9, size=(samples, 2)).astype(np.float32),
        "old_log_probs": generator.normal(-2.0, 0.1, size=samples).astype(np.float32),
        "advantages": generator.normal(size=samples).astype(np.float32),
        "returns": generator.normal(size=samples).astype(np.float32),
        "old_values": generator.normal(size=samples).astype(np.float32),
    }
    return list(LearnerBatch(tensors=tensors, samples=samples).split(rank_count))


def _sac_algorithm(seed: int) -> SAC:
    import gymnasium as gym

    torch.manual_seed(seed)
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    actor = MLPActor(
        MLPActorConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(16,),
            use_tanh_output=True,
        )
    )
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=(16,),
            use_action=True,
        )
    )
    buffer = ReplayBuffer(capacity=64)
    env = gym.Env()
    env.observation_space, env.action_space = observation_space, action_space
    return SAC(
        SACConfig(
            env=env,
            action_space=action_space,
            actor=actor,
            critic=critic,
            buffer=buffer,
            sampler=UniformSampler(buffer),
            learning_starts=0,
            batch_size=8,
            num_critics=2,
            n_critics_to_sample=2,
            use_amp=False,
            ent_coef=0.1,
        )
    )


def _sac_batch(rank_count: int, seed: int) -> list[LearnerBatch]:
    """Every rank trains on the whole draw: SAC samples actions, and ranks seeded alike draw alike."""
    generator = np.random.default_rng(seed)
    samples = 8

    def state():
        return {"state": generator.normal(size=(samples, 4)).astype(np.float32)}

    tensors = {
        "0/replay/obs": state(),
        "0/replay/next_obs": state(),
        "0/replay/action": generator.uniform(-1, 1, size=(samples, 2)).astype(
            np.float32
        ),
        "0/replay/reward": generator.normal(size=samples).astype(np.float32),
        "0/replay/done": (generator.uniform(size=samples) < 0.2).astype(np.float32),
        "0/replay/truncated": np.zeros(samples, np.float32),
        "0/replay/bootstrap_steps": np.full(samples, 4, np.int64),
    }
    return [LearnerBatch(tensors=tensors, samples=samples)] * rank_count


def _train(group, make_batches, steps: int) -> dict[str, np.ndarray]:
    ranks = group.num_learner_ranks
    for step in range(steps):
        group.train_step(make_batches(ranks, step))
    weights = group.publish(steps)
    return {name: np.array(value) for name, value in weights.resolve().tensors.items()}


@pytest.mark.parametrize("family", ["ppo", "sac"])
def test_two_learner_ranks_on_gloo_match_one_rank(family, tmp_path):
    """PPO shards its model with FSDP, SAC keeps whole replicas; both average gradients over the ranks.

    PPO's ranks train on halves of one draw, so the averaged step is the
    one-rank step on the whole draw. SAC's ranks each train on the whole draw
    with the same seed, so averaging must leave the one-rank step unchanged.
    """
    if family == "ppo":
        torch.manual_seed(4)
        model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
        spec, batches = _ppo_learner(model, 2), _ppo_batch
        one_rank = LocalLearnerGroup(_ppo_learner(model, 1), device="cpu")
    else:
        spec, batches = replica_learner_spec(_sac_algorithm(5)), _sac_batch
        one_rank = LocalLearnerGroup(
            functools.partial(ReplayLearner, algorithm=_sac_algorithm(5)), device="cpu"
        )
    seed_process(
        NativeRuntimeConfig().learner_seed
    )  # the seed every learner rank starts from
    expected = _train(one_rank, batches, steps=3)
    one_rank.close()
    runtime = _runtime(Placement(learner=Learner(ranks=2)))
    try:
        group = runtime.open_learner_group(spec, num_learner_ranks=2, name=family)
        published = _train(group, batches, steps=3)
        if family == "sac":
            assert group.hooks().gradient_steps() == 3 * 1
        checkpoint = str(tmp_path / "learner")
        group.save_checkpoint(checkpoint)
        group.load_checkpoint(checkpoint)
        assert group.publish(4).handle is not None
    finally:
        runtime.close()
    assert published.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(
            published[name], value, rtol=1e-4, atol=1e-5, err_msg=name
        )


def test_the_async_schedule_streams_rounds_from_worker_actors():
    """The trainer's async schedule keeps a collection in flight on the actors while it trains."""
    runtime = _runtime(Placement(envs=Envs(workers=2)))
    group = runtime.open_rollout_group(
        _gym_spec(2), num_rollout_workers=2, num_envs=2, name="async"
    )
    try:
        group.set_weights(_weights(1, 1.0))
        first = group.collect_async([RolloutRequest(decisions_per_environment=2)] * 2)
        # Weights published while a round is in flight reach the workers after it.
        assert group.set_weights(_weights(2, 0.5)) == (2, 2)
        assert {
            d.action_chunk.policy_version for r in first.result() for d in r.decisions
        } == {1}
        second = group.collect([RolloutRequest(decisions_per_environment=2)] * 2)
        assert {d.action_chunk.policy_version for r in second for d in r.decisions} == {
            2
        }
    finally:
        runtime.close()


def _fails_quickly(call, error: type[Exception], match: str) -> None:
    started = time.monotonic()
    with pytest.raises(error, match=match):
        call()
    assert time.monotonic() - started < 15.0


def test_a_dead_policy_server_fails_the_collection_within_seconds():
    runtime = _runtime(Placement(envs=Envs(workers=2), policy=Served(replicas=2)))
    try:
        group = runtime.open_rollout_group(
            _gym_spec(2), num_rollout_workers=2, num_envs=2, name="served"
        )
        group.set_weights(_weights(1, 1.0))
        ray.kill(group._owned_service.handle().actors[1])
        _fails_quickly(
            lambda: group.collect([RolloutRequest(decisions_per_environment=2)] * 2),
            RolloutInfrastructureError,
            "rollout worker 1 .* failed: .*dead actor policy_server1_",
        )
    finally:
        runtime.close()


class _FrozenEnv(_CountingEnv):
    """Its second step never returns."""

    def step(self, action):
        if self.t >= 1:
            time.sleep(3600)
        return super().step(action)


def test_a_frozen_rollout_worker_fails_the_collection_naming_it():
    spec = GymWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: (_FrozenEnv if context.worker_index == 1 else _CountingEnv)(
                offset=0.0
            )
        ),
        environment=scripted_environment(2),
        phase="train",
        policy=_policy_spec,
    )
    runtime = _runtime(Placement(envs=Envs(workers=2)), rollout_timeout_seconds=2.0)
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=2, num_envs=2, name="frozen"
        )
        group.set_weights(_weights(1, 1.0))
        _fails_quickly(
            lambda: group.collect([RolloutRequest(decisions_per_environment=3)] * 2),
            RolloutTimeoutError,
            r"rollout group 'frozen' collecting: no result within 2 s from rollout worker 1 \(node0, cpu\)",
        )
    finally:
        started = time.monotonic()
        runtime.close()
        # The frozen worker is killed, not asked to close behind its stuck collection.
        assert time.monotonic() - started < 15.0


def test_a_frozen_rollout_worker_fails_the_publish_behind_its_collection():
    spec = GymWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: (_FrozenEnv if context.worker_index == 1 else _CountingEnv)(
                offset=0.0
            )
        ),
        environment=scripted_environment(2),
        phase="train",
        policy=_policy_spec,
    )
    runtime = _runtime(Placement(envs=Envs(workers=2)), rollout_timeout_seconds=2.0)
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=2, num_envs=2, name="frozen"
        )
        group.set_weights(_weights(1, 1.0))
        # The overlapped schedule publishes while a collection is in flight.
        group.collect_async([RolloutRequest(decisions_per_environment=3)] * 2)
        _fails_quickly(
            lambda: group.set_weights(_weights(2, 2.0)),
            RolloutTimeoutError,
            r"rollout group 'frozen' publishing weights: no result within 2 s from rollout worker 1 \(node0, cpu\)",
        )
    finally:
        started = time.monotonic()
        runtime.close()
        assert time.monotonic() - started < 15.0


def test_a_dead_learner_rank_fails_the_update_within_seconds():
    runtime = _runtime(Placement(learner=Learner(ranks=2)))
    try:
        group = runtime.open_learner_group(
            replica_learner_spec(_sac_algorithm(5)), num_learner_ranks=2, name="sac"
        )
        group.train_step(_sac_batch(2, 0))
        ray.kill(group._actors[1])
        _fails_quickly(
            lambda: group.train_step(_sac_batch(2, 1)),
            ActorFailedError,
            "learner rank 1 .* failed",
        )
        # The surviving rank may be stuck in the collective; the group refuses further calls.
        with pytest.raises(ActorFailedError, match="unusable"):
            group.publish(1)
    finally:
        started = time.monotonic()
        runtime.close()
        assert time.monotonic() - started < 15.0


def test_a_node_that_never_joins_fails_after_the_node_join_timeout():
    from isaac_rl_posttraining.runtimes.native_ray.cluster import RayCluster

    plan = plan_placement(
        Placement(learner=Learner(ranks=2)),
        Resources.uniform(2, 0),
        served_supported=True,
        distributed_learner=True,
    )
    start = time.monotonic()
    with pytest.raises(
        ActorFailedError, match="only 1 joined the Ray cluster within 3 s"
    ):
        RayCluster(
            plan.resources,
            plan.roles(),
            startup_timeout_seconds=900.0,
            node_join_timeout_seconds=3.0,
        )
    assert time.monotonic() - start < 30.0
