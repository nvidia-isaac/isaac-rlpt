# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for natively batched simulator rollouts."""

from __future__ import annotations

import numpy as np
import pytest
from entry_doubles import ScriptedSimulator, scripted_environment

from isaac_rl_posttraining.rollouts import ActionChunk
from isaac_rl_posttraining.runtimes import (
    BatchedEpisodeRolloutWorker,
    BatchedWorkerSpec,
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutWorkerContext,
)


class _BatchedEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.ticks = 0
        self.seeds = []
        self.need_obs = True
        self.need_obs_history = []

    def _observation(self):
        return {
            "state": np.asarray(
                [[0.0, self.ticks], [1.0, self.ticks]], dtype=np.float32
            )
        }

    def reset(self, *, seed=None):
        self.ticks = 0
        self.seeds.append(seed)
        return self._observation(), {
            "initial_state_fingerprint": ["case-0", "case-1"],
            "case_seed": [100, 101],
            "round_index": 7,
        }

    def step(self, actions):
        assert actions.shape == (2, 1)
        self.need_obs_history.append(self.need_obs)
        self.ticks += 1
        terminated = np.asarray([self.ticks == 1, False])
        return (
            self._observation(),
            np.asarray([1.0, 0.5]),
            terminated,
            np.zeros(2, dtype=bool),
            [
                {"success": bool(terminated[0])},
                {"success": False},
            ],
        )

    def close(self):
        # Test double: holds no resources to release.
        pass


class _Policy:
    def __init__(self) -> None:
        self.version = 0

    def action_chunks(self, observations, *, mode, session_ids):
        return tuple(
            ActionChunk.from_arrays(
                learner_action=np.asarray([[float(session_id)]], dtype=np.float32),
                environment_actions=np.ones((1, 2, 1), dtype=np.float32),
                execution_length=2,
                learner_action_kind="latent",
                policy_version=self.version,
                behavior_log_prob=np.asarray([-0.5], dtype=np.float32),
                behavior_value=np.asarray([0.25], dtype=np.float32),
                learner_observation={"state": np.asarray(observation["state"])[None]},
            )
            for observation, session_id in zip(observations, session_ids, strict=True)
        )

    def prepare_successors(self, observations, *, session_ids):
        return tuple(
            (
                {
                    "state": np.asarray(observation["state"], dtype=np.float32),
                    "prepared": np.ones(1, dtype=np.float32),
                },
                np.asarray([0.75], dtype=np.float32),
            )
            for observation in observations
        )

    def reset_episode(self, *, session_id):
        # Test double: keeps no per-episode state.
        pass

    def set_policy_state(self, state):
        self.version = state.version
        return self.version

    def close(self):
        # Test double: holds no resources to release.
        pass


class _ValuelessPolicy:
    """A batched policy without a value head."""

    def action_chunks(self, observations, *, mode, session_ids):
        return _Policy().action_chunks(observations, mode=mode, session_ids=session_ids)

    def reset_episode(self, *, session_id):
        # Test double: keeps no per-episode state.
        pass

    def set_policy_state(self, state):
        return state.version


class _OffPolicy(_Policy):
    def action_chunks(self, observations, *, mode, session_ids):
        return tuple(
            ActionChunk.from_arrays(
                learner_action=np.ones((1, 1), dtype=np.float32),
                environment_actions=np.ones((1, 2, 1), dtype=np.float32),
                execution_length=2,
                learner_action_kind="latent",
                policy_version=self.version,
                learner_observation={
                    "features": np.asarray(observation["state"])[None] + 10
                },
            )
            for observation in observations
        )

    def prepare_successors(self, observations, *, session_ids):
        raise AssertionError("SAC actors do not provide behavior values")

    def prepare_observations(self, observations, *, session_ids):
        return tuple(
            {"features": np.asarray(observation["state"]) + 10}
            for observation in observations
        )


def test_off_policy_successors_keep_encoded_observations_without_actor_values():
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        _BatchedEnv(),
        _OffPolicy(),
        chunks_per_episode=2,
        reward_transform=None,
    )

    result = worker.collect(RolloutRequest(episodes_per_environment=1))

    assert result.episodes_completed == 2
    expected_successors = [[10, 11], [11, 12], [11, 14]]
    for decision, expected in zip(result.decisions, expected_successors, strict=True):
        transition = decision.transition
        assert transition.policy_outputs.value is None
        assert transition.policy_outputs.next_value is None
        assert set(transition.observation.arrays()) == {"features"}
        assert set(transition.next_observation.arrays()) == {"features"}
        np.testing.assert_array_equal(
            transition.next_observation.arrays()["features"], expected
        )


def test_batched_worker_needs_a_valuing_policy_for_policy_successor_values():
    context = RolloutWorkerContext(0, 1, (0, 1), "cpu")
    with pytest.raises(TypeError, match="prepare_successors"):
        BatchedEpisodeRolloutWorker(
            context,
            _BatchedEnv(),
            _ValuelessPolicy(),
            chunks_per_episode=2,
            reward_transform=None,
        )
    worker = BatchedEpisodeRolloutWorker(
        context,
        _BatchedEnv(),
        _ValuelessPolicy(),
        chunks_per_episode=2,
        reward_transform=None,
        successor_values="next_decision",
    )
    result = worker.collect(RolloutRequest(episodes_per_environment=1))
    assert result.episodes_completed == 2


def test_batched_worker_stops_each_trajectory_at_its_first_episode_end():
    context = RolloutWorkerContext(
        worker_index=0,
        num_rollout_workers=1,
        environment_indices=(4, 5),
        device="cpu",
    )
    worker = BatchedEpisodeRolloutWorker(
        context,
        _BatchedEnv(),
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )
    worker.set_policy_state(PolicyState(version=1, tensors={"weight": np.ones(1)}))

    result = worker.collect(RolloutRequest(episodes_per_environment=1))

    env4 = [item for item in result.decisions if item.environment_index == 4]
    env5 = [item for item in result.decisions if item.environment_index == 5]
    assert len(env4) == 1
    assert env4[0].transition.terminated[-1]
    assert env4[0].transition.steps == 1
    assert len(env5) == 2
    assert env5[-1].transition.truncated[-1]
    assert result.env_steps == 5
    assert result.episodes_completed == 2
    assert float(env5[-1].transition.policy_outputs.next_value) == pytest.approx(0.75)
    # A terminal decision bootstraps nothing; its successor value is zero.
    assert float(env4[0].transition.policy_outputs.next_value) == pytest.approx(0.0)
    # Its replay successor is still the policy's prepared observation, like every
    # other successor, so a replay batch has one observation layout throughout.
    assert "prepared" in env4[0].transition.next_observation.arrays()
    assert all("prepared" in item.transition.next_observation.arrays() for item in env5)
    assert result.timings.keys() == {
        "action_inference_seconds",
        "environment_reset_seconds",
        "environment_step_seconds",
        "environment_step_calls",
        "policy_calls",
        "next_observation_inference_seconds",
        "successor_inference_seconds",
    }
    assert (
        result.timings["successor_inference_seconds"]
        == result.timings["next_observation_inference_seconds"]
    )
    assert all(seconds >= 0.0 for seconds in result.timings.values())
    # Two chunks, one policy call each; every chunk ticks the vector env execution_length times.
    assert result.timings["policy_calls"] == pytest.approx(2.0)
    assert result.timings["environment_step_calls"] == pytest.approx(4.0)


def test_batched_decisions_preserve_episode_state_and_apply_new_policy():
    env = _BatchedEnv()
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        env,
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )
    request = RolloutRequest(decisions_per_environment=1)
    first = worker.collect(request)
    assert first.env_steps == 3 and first.episodes_completed == 1
    assert not first.decisions[1].transition.ended
    worker.set_policy_state(PolicyState(version=2, tensors={"weight": np.ones(1)}))
    second = worker.collect(request)
    assert second.env_steps == 2 and second.episodes_completed == 1
    assert env.seeds == [None]
    assert (
        second.decisions[0].transition.episode_id
        == first.decisions[1].transition.episode_id
    )
    assert second.decisions[0].transition.metadata.decision_index == 1
    assert second.decisions[0].transition.metadata.policy_version == 2
    assert second.timings["environment_step_seconds"] > 0
    third = worker.collect(request)
    assert env.seeds == [None, None]
    assert (
        third.decisions[0].transition.episode_id
        != first.decisions[0].transition.episode_id
    )


def test_batched_worker_applies_a_shorter_request_cap_only_to_unfinished_episodes():
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        _BatchedEnv(),
        _Policy(),
        chunks_per_episode=3,
        reward_transform=None,
    )

    result = worker.collect(
        RolloutRequest(episodes_per_environment=1, max_decisions_per_episode=1)
    )

    env0 = [item for item in result.decisions if item.environment_index == 0]
    env1 = [item for item in result.decisions if item.environment_index == 1]
    assert env0[0].transition.succeeded
    assert "decision_cap_reached" not in env0[0].transition.metadata.extra
    assert env1[0].transition.truncated[-1]
    assert not env1[0].transition.succeeded
    assert env1[0].transition.metadata.extra["decision_cap_reached"] is True
    assert env1[0].transition.metadata.extra["terminal_reason"] == "decision_cap"


def test_batched_worker_fixed_horizon_is_not_an_artificial_request_cap():
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        _BatchedEnv(),
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )

    result = worker.collect(
        RolloutRequest(episodes_per_environment=1, max_decisions_per_episode=2)
    )

    env1_terminal = [item for item in result.decisions if item.environment_index == 1][
        -1
    ].transition
    assert env1_terminal.truncated[-1]
    assert "decision_cap_reached" not in env1_terminal.metadata.extra


def test_batched_worker_replays_seeded_evaluation_epochs():
    env = _BatchedEnv()
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        env,
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )
    request = RolloutRequest(episodes_per_environment=2, reset_seed=90)

    first = worker.collect(request)
    second = worker.collect(request)

    assert env.seeds == [90, 91, 90, 91]
    assert {
        (item.transition.metadata.reset_seed, item.transition.metadata.reset_seed_scope)
        for item in (*first.decisions, *second.decisions)
    } == {(90, "worker_batch"), (91, "worker_batch")}
    for decision in (*first.decisions, *second.decisions):
        extra = decision.transition.metadata.extra
        if decision.transition.metadata.step_in_episode == 0:
            assert len(extra["raw_observation_sha256"]) == 64
            assert extra["raw_observation_sha256_by_key"].keys() == {"state"}
            local_index = decision.environment_index
            assert extra["initial_state_fingerprint"] == f"case-{local_index}"
            assert extra["case_seed"] == 100 + local_index
            assert extra["round_index"] == 7
        else:
            assert "raw_observation_sha256" not in extra


def test_batched_worker_skips_intermediate_observations():
    env = _BatchedEnv()
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        env,
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )

    worker.collect(RolloutRequest(episodes_per_environment=1))

    assert env.need_obs_history == [False, True, False, True]
    assert env.need_obs is True


class _ChainedPolicy(_Policy):
    """A policy whose decisions carry cached features and that owns no successor pass."""

    def action_chunks(self, observations, *, mode, session_ids):
        return tuple(
            ActionChunk.from_arrays(
                learner_action=np.asarray([[float(session_id)]], dtype=np.float32),
                environment_actions=np.ones((1, 2, 1), dtype=np.float32),
                execution_length=2,
                learner_action_kind="latent",
                policy_version=self.version,
                behavior_log_prob=np.asarray([-0.5], dtype=np.float32),
                behavior_value=np.asarray(
                    [0.25 + observation["state"][1]], dtype=np.float32
                ),
                learner_observation={"state": np.asarray(observation["state"])[None]},
                cached_features={
                    "features": np.asarray([[session_id, 7]], dtype=np.int16)
                },
            )
            for observation, session_id in zip(observations, session_ids, strict=True)
        )

    def prepare_successors(self, observations, *, session_ids):
        raise AssertionError("next_decision successor values need no policy pass")


def test_batched_worker_takes_the_next_decisions_value_as_the_successor_value():
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (4, 5), "cpu"),
        _BatchedEnv(),
        _ChainedPolicy(),
        chunks_per_episode=2,
        reward_transform=None,
        successor_values="next_decision",
    )

    result = worker.collect(RolloutRequest(episodes_per_environment=1))

    env4 = [item for item in result.decisions if item.environment_index == 4]
    env5 = [item for item in result.decisions if item.environment_index == 5]
    # Environment 4 terminates in its first chunk; environment 5 runs to the time limit.
    assert float(env4[0].transition.policy_outputs.next_value) == pytest.approx(0.0)
    assert float(env5[0].transition.policy_outputs.next_value) == float(
        env5[1].transition.policy_outputs.value
    )
    assert float(env5[1].transition.policy_outputs.next_value) == pytest.approx(0.0)
    # Without a successor pass, the stored successor is the raw observation after the chunk.
    assert all(
        "state" in item.transition.next_observation.arrays()
        for item in result.decisions
    )
    assert "next_observation_inference_seconds" not in result.timings
    assert "successor_inference_seconds" not in result.timings
    np.testing.assert_array_equal(
        env5[0].transition.policy_outputs.features["features"], [5, 7]
    )
    assert env5[0].transition.policy_outputs.features["features"].dtype == np.int16


class _FlakyEnv(_BatchedEnv):
    """Loses its simulation on the first step of an episode until restarted."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_step = True
        self.restarts = 0

    def step(self, actions):
        if self.fail_next_step:
            self.fail_next_step = False
            raise RolloutInfrastructureError("simulation died")
        return super().step(actions)

    def restart(self) -> None:
        self.restarts += 1
        self.ticks = 0


def test_batched_worker_redoes_an_episode_after_an_infrastructure_failure():
    env = _FlakyEnv()
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        env,
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
        episode_attempts=2,
    )

    result = worker.collect(RolloutRequest(episodes_per_environment=1, reset_seed=7))

    # The redone episode replays the same seed and yields the complete episode.
    assert env.restarts == 1 and env.seeds == [7, 7]
    assert result.episodes_completed == 2 and len(result.decisions) == 3

    failing = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        _FlakyEnv(),
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )
    with pytest.raises(RolloutInfrastructureError, match="1 attempt"):
        failing.collect(RolloutRequest(episodes_per_environment=1))
    with pytest.raises(TypeError, match="restart"):
        BatchedEpisodeRolloutWorker(
            RolloutWorkerContext(0, 1, (0, 1), "cpu"),
            _BatchedEnv(),
            _Policy(),
            chunks_per_episode=2,
            reward_transform=None,
            episode_attempts=2,
        )


def test_batched_worker_spec_validates_its_additions():
    entry = ScriptedSimulator(lambda context: _BatchedEnv())
    with pytest.raises(ValueError, match="successor_values"):
        BatchedWorkerSpec(
            entry=entry,
            environment=scripted_environment(2),
            phase="train",
            policy=lambda context: _Policy(),
            chunks_per_episode=1,
            successor_values="previous_decision",
        )
    with pytest.raises(ValueError, match="episode_attempts"):
        BatchedWorkerSpec(
            entry=entry,
            environment=scripted_environment(2),
            phase="train",
            policy=lambda context: _Policy(),
            chunks_per_episode=1,
            episode_attempts=0,
        )


def test_batched_worker_spec_builds_the_entrys_environment_on_the_worker():
    built = []

    def make_env(context):
        built.append(context.environment_indices)
        return _BatchedEnv()

    spec = BatchedWorkerSpec(
        entry=ScriptedSimulator(make_env),
        environment=scripted_environment(2),
        phase="train",
        policy=lambda context: _Policy(),
        chunks_per_episode=2,
    )
    worker = spec(RolloutWorkerContext(0, 1, (0, 1), "cpu"))

    assert isinstance(worker, BatchedEpisodeRolloutWorker)
    assert built == [(0, 1)]
    worker.close()


def test_batched_worker_keeps_derived_seeds_within_32_bits():
    env = _BatchedEnv()
    worker = BatchedEpisodeRolloutWorker(
        RolloutWorkerContext(0, 1, (0, 1), "cpu"),
        env,
        _Policy(),
        chunks_per_episode=2,
        reward_transform=None,
    )
    top = 2**32 - 1
    result = worker.collect(RolloutRequest(episodes_per_environment=2, reset_seed=top))

    assert env.seeds == [top, 0]
    assert all(0 <= seed < 2**32 for seed in env.seeds)
    # The first decision of every episode records the policy seed the session was reset with.
    recorded = {
        (
            decision.environment_index,
            decision.transition.metadata.reset_seed,
        ): decision.transition.metadata.extra["policy_seed"]
        for decision in result.decisions
        if decision.transition.metadata.step_in_episode == 0
    }
    expected = {
        (environment_index, epoch_seed): (epoch_seed + environment_index * 1_000_003)
        % 2**32
        for environment_index in (0, 1)
        for epoch_seed in (top, 0)
    }
    assert recorded == expected
    assert all(0 <= seed < 2**32 for seed in recorded.values())
