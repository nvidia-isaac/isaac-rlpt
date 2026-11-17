# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the runtime evaluator scores policy states, pinned on scripted rollout groups.

The properties the serial evaluator guaranteed carry over: success is read at
the terminal decision only, a capped episode is a failure, the video pass is
separate from the scored pass, and every published state reaches the workers
before they act.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from conftest import make_transition
from entry_doubles import scripted_worker_spec

from isaac_rl_posttraining.buffers import ReplayBuffer
from isaac_rl_posttraining.envs import LatentSpaceEnv
from isaac_rl_posttraining.modules.mlp import MLPActor, MLPActorConfig
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy
from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    CollectedDecision,
    FrozenBaseModelSpec,
    GymRolloutWorker,
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutGroupBase,
    RolloutResult,
    SteeringPolicySpec,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import (
    policy_state_from_tensors,
)
from isaac_rl_posttraining.training import (
    RolloutEvaluator,
    RuntimeDataSource,
    episode_outcomes,
)
from isaac_rl_posttraining.training.evaluation import _worker_time_summary


def _decision(environment_index, *, reward, done, truncated, success, version=1):
    return CollectedDecision(
        environment_index=environment_index,
        transition=make_transition(
            reward=reward,
            done=done,
            truncated=truncated,
            info={"is_success": success, "success": success},
        ),
        action_chunk=ActionChunk(
            learner_action=None,
            environment_actions=np.zeros((1, 1, 1), np.float32),
            execution_length=1,
            source="native",
            policy_version=version,
        ),
    )


def _result(decisions, *, version=1, frames=()):
    return RolloutResult(
        worker_index=0,
        decisions=tuple(decisions),
        env_steps=len(decisions),
        episodes_completed=sum(1 for d in decisions if d.transition.ended),
        policy_version=version,
        frames=frames,
    )


class _ScriptedGroup(RolloutGroupBase):
    """A rollout group returning scripted results stamped with the served policy version."""

    num_rollout_workers = 1

    def __init__(self, episodes, *, num_envs=1):
        self.episodes = list(
            episodes
        )  # per collect call: list of decisions, or a ready result
        self.num_envs = num_envs
        self.requests = []
        self.versions = []
        self.version = 0
        self.closed = 0

    def collect(self, requests):
        self.requests.append(requests[0])
        scripted = self.episodes.pop(0)
        if isinstance(scripted, RolloutResult):
            return (scripted,)
        decisions = [
            _decision(
                decision.environment_index,
                reward=float(decision.transition.reward("env")[-1]),
                done=bool(decision.transition.terminated[-1]),
                truncated=bool(decision.transition.truncated[-1]),
                success=bool(decision.transition.metadata.extra["success"]),
                version=self.version,
            )
            for decision in scripted
        ]
        return (_result(decisions, version=self.version),)

    def set_policy_state(self, state):
        self.versions.append(state.version)
        self.version = state.version
        return (state.version,)

    def close(self):
        self.closed += 1


def _state(version=1):
    return PolicyState(
        version=version, tensors={"weight": np.ones(1, dtype=np.float32)}
    )


def test_transient_midepisode_success_scores_as_failure() -> None:
    group = _ScriptedGroup(
        [
            [
                _decision(0, reward=0.0, done=False, truncated=False, success=True),
                _decision(0, reward=-1.0, done=True, truncated=False, success=False),
            ]
        ]
    )
    evaluator = RolloutEvaluator(group=group, num_episodes=1, max_decisions=10)

    metrics = evaluator.evaluate(_state())

    assert metrics["num_eval_episodes"] == 1
    assert metrics["success_rate"] == pytest.approx(0.0)
    assert metrics["avg_reward"] == -1.0
    assert metrics["avg_steps"] == pytest.approx(2.0)
    (request,) = group.requests
    assert request.mode == "eval"
    assert request.episodes_per_environment == 1
    assert request.max_decisions_per_episode == 10
    assert request.reset_seed is None
    assert not request.capture_frames


def test_persistent_group_receives_each_new_state_once() -> None:
    def one_success():
        return [_decision(0, reward=1.0, done=True, truncated=False, success=True)]

    group = _ScriptedGroup([one_success() for _ in range(3)])
    evaluator = RolloutEvaluator(group=group, num_episodes=1, max_decisions=10)

    evaluator.evaluate(_state(1))
    evaluator.evaluate(_state(1))
    evaluator.evaluate(_state(2))

    assert group.versions == [1, 2]
    # A worker that acted with stale weights is caught.
    group.episodes.append(_result(one_success(), version=1))
    with pytest.raises(RuntimeError, match="policy version"):
        evaluator.evaluate(_state(3))


def test_a_dedicated_group_clears_its_server_counters_after_each_evaluation() -> None:
    group = _ScriptedGroup(
        [[_decision(0, reward=1.0, done=True, truncated=False, success=True)]] * 2
    )
    resets = []
    group.policy_server_stats = lambda: ()
    group.reset_policy_server_stats = lambda: resets.append(True)
    evaluator = RolloutEvaluator(group=group, num_episodes=1, max_decisions=10)
    evaluator.evaluate(_state(1))
    evaluator.evaluate(_state(2))
    assert len(resets) == 2


def test_fresh_worker_groups_are_opened_and_closed_per_evaluation() -> None:
    opened = []

    def open_group():
        group = _ScriptedGroup(
            [[_decision(0, reward=0.0, done=False, truncated=True, success=False)]]
        )
        opened.append(group)
        return group

    evaluator = RolloutEvaluator(
        open_group=open_group, num_episodes=1, max_decisions=10
    )

    evaluator.evaluate(_state(4))
    evaluator.evaluate(_state(5))

    assert [group.versions for group in opened] == [[4], [5]]
    assert all(group.closed == 1 for group in opened)


def test_episode_count_requires_and_preserves_exact_whole_rounds(tmp_path) -> None:
    decisions = []
    for environment_index in (0, 1, 2):
        decisions.append(
            _decision(
                environment_index,
                reward=1.0,
                done=True,
                truncated=False,
                success=environment_index != 1,
            )
        )
        decisions.append(
            _decision(
                environment_index, reward=0.0, done=False, truncated=True, success=False
            )
        )
    group = _ScriptedGroup([decisions], num_envs=3)
    evaluator = RolloutEvaluator(
        group=group,
        num_episodes=6,
        max_decisions=10,
        record_video=True,
        video_dir=str(tmp_path),
    )

    metrics = evaluator.evaluate(_state())

    assert group.requests[0].episodes_per_environment == 2
    assert metrics["num_eval_episodes"] == 6
    assert metrics["success_rate"] == pytest.approx(2 / 6)
    # Video is skipped for batched evaluation, so only the scored request ran.
    assert len(group.requests) == 1
    assert "video_saved" not in metrics


def test_episode_count_rejects_nonmultiples_and_less_than_one_round() -> None:
    group = _ScriptedGroup([], num_envs=3)
    with pytest.raises(ValueError, match="exact multiple"):
        RolloutEvaluator(group=group, num_episodes=4, max_decisions=10).evaluate(
            _state()
        )
    with pytest.raises(ValueError, match="at least"):
        RolloutEvaluator(group=group, num_episodes=2, max_decisions=10).evaluate(
            _state()
        )


def test_task_metrics_are_aggregated_and_episode_records_are_written(tmp_path) -> None:
    first = _decision(0, reward=1.0, done=True, truncated=False, success=True)
    second = _decision(1, reward=0.0, done=True, truncated=False, success=False)
    from dataclasses import replace

    enriched = []
    for index, decision in enumerate((first, second)):
        transition = decision.transition
        extra = {
            **transition.metadata.extra,
            "policy_seed": 900 + index,
            "case_seed": 1200 + index,
            "round_index": 4,
            "initial_state_fingerprint": f"case-{index}",
            "terminal_reason": "success" if index == 0 else "lift",
            "task_metrics": {
                "lift_hold_success": index == 0,
                "final_height_ok": True,
            },
        }
        metadata = replace(
            transition.metadata,
            environment_index=index,
            policy_version=1,
            reset_seed=100,
            reset_seed_scope="worker_batch",
            extra=extra,
        )
        enriched.append(
            replace(decision, transition=replace(transition, metadata=metadata))
        )
    group = _ScriptedGroup([_result(enriched)], num_envs=2)
    path = tmp_path / "evaluation_episodes.jsonl"
    evaluator = RolloutEvaluator(
        group=group,
        num_episodes=2,
        max_decisions=10,
        episode_record_path=str(path),
    )

    metrics = evaluator.evaluate(_state())

    assert metrics["task/lift_hold_success_rate"] == pytest.approx(0.5)
    assert metrics["task/final_height_ok_rate"] == pytest.approx(1.0)
    import json

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["initial_state_fingerprint"] == "case-0"
    assert records[0]["case_seed"] == 1200
    assert records[0]["round_index"] == 4
    assert records[1]["terminal_reason"] == "lift"
    assert records[1]["task_metrics"]["final_height_ok"] is True


def test_unfinished_episodes_are_a_contract_violation() -> None:
    with pytest.raises(RuntimeError, match="unfinished"):
        episode_outcomes(
            [
                _result(
                    [
                        _decision(
                            0, reward=0.0, done=False, truncated=False, success=False
                        )
                    ]
                )
            ]
        )


def test_video_pass_records_one_extra_episode_and_does_not_shadow_the_score(
    tmp_path,
) -> None:
    pytest.importorskip("cv2")
    scored = [_decision(0, reward=1.0, done=True, truncated=False, success=True)]
    frames = tuple(
        {
            "ego_view_image": np.full((8, 8, 3), value, dtype=np.uint8),
            "critic_image": np.zeros((2, 2, 3), np.uint8),
        }
        for value in (0, 128, 255)
    )
    video_episode = _result(
        [
            _decision(0, reward=0.0, done=False, truncated=False, success=False),
            _decision(0, reward=-1.0, done=True, truncated=False, success=False),
        ],
        frames=frames,
    )
    group = _ScriptedGroup([scored, video_episode])
    evaluator = RolloutEvaluator(
        group=group,
        num_episodes=1,
        max_decisions=10,
        record_video=True,
        video_dir=str(tmp_path),
    )

    metrics = evaluator.evaluate(_state())

    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["video_success"] == pytest.approx(0.0)
    assert metrics["video_steps"] == pytest.approx(2.0)
    assert metrics["video_saved"] == pytest.approx(1.0)
    assert group.requests[1].capture_frames
    assert group.requests[1].episodes_per_environment == 1
    videos = list(tmp_path.glob("eval_*_all_cameras.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 0


class _CappedEnv:
    """Never terminates on its own."""

    def __init__(self) -> None:
        self.t = 0

    def reset(self, *, seed=None):
        self.t = 0
        return {"state": np.zeros(1, np.float32)}, {}

    def step(self, action):
        self.t += 1
        return (
            {"state": np.full(1, float(self.t), np.float32)},
            0.0,
            False,
            False,
            {"success": True},
        )

    def close(self):
        # Test double: holds no resources to release.
        pass


class _NativePolicy:
    def action_chunk(self, observation, *, mode):
        assert mode == "eval"
        return ActionChunk(
            learner_action=None,
            environment_actions=np.zeros((1, 2, 1), np.float32),
            execution_length=2,
            source="native",
            policy_version=self.version,
        )

    version = 0

    def set_policy_state(self, state):
        self.version = state.version
        return self.version


def test_decision_cap_ends_a_non_terminating_episode_as_a_failure() -> None:
    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    policy = _NativePolicy()
    group = runtime.open_rollout_group(
        lambda context: GymRolloutWorker(context, _CappedEnv(), policy),
        num_rollout_workers=1,
        num_envs=1,
        name="capped",
    )
    evaluator = RolloutEvaluator(group=group, num_episodes=2, max_decisions=3)

    metrics = evaluator.evaluate(_state())

    # Three decisions of two simulator steps each, then a time-limit cut;
    # the transient success flag of a capped episode does not count.
    assert metrics["num_eval_episodes"] == 2
    assert metrics["avg_steps"] == pytest.approx(3.0)
    assert metrics["success_rate"] == pytest.approx(0.0)
    runtime.close()


class _SuccessfulEpisodeEnv:
    """Succeeds after ``horizon`` simulator steps; observations carry a ready DINO embedding."""

    def __init__(self, horizon: int = 4) -> None:
        self.horizon = horizon
        self.t = 0

    def _observation(self):
        return {
            "state": np.full(1, float(self.t), np.float32),
            "dino_embedding": np.zeros(2, np.float32),
        }

    def reset(self, *, seed=None):
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        return self._observation(), float(done), done, False, {"success": done}

    def close(self):
        # Test double: holds no resources to release.
        pass


def test_shared_group_evaluation_does_not_republish_the_data_sources_version() -> None:
    """On a shared simulator the data source owns the policy version; the evaluator only scores."""
    base_model = MockBaseModelPolicy(action_horizon=4, action_dim=2, seed=0)
    latent_env = LatentSpaceEnv(
        gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                "dino_embedding": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            }
        ),
        gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
        base_model,
        noise_dim=2,
        chunk_size=None,
        action_bound=1.0,
        use_vlm_features=False,
    )
    actor = MLPActor(
        MLPActorConfig(
            observation_space=latent_env.observation_space,
            action_space=latent_env.action_space,
            hidden_dims=(4,),
            use_tanh_output=True,
            log_std_init=-1.0,
        )
    )
    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    group = runtime.open_rollout_group(
        scripted_worker_spec(
            lambda context: _SuccessfulEpisodeEnv(),
            SteeringPolicySpec(
                actor=actor,
                base_model=FrozenBaseModelSpec(prebuilt=base_model),
                latent_sampler=latent_env,
                steering_key_mapping={"state": "state", "language": None, "image": []},
                action_exec_len=2,
            ),
            supports_batched_policy=True,
        ),
        num_rollout_workers=1,
        num_envs=1,
        name="shared",
    )
    source = RuntimeDataSource(
        group,
        InProcessBufferSink(ReplayBuffer(capacity=16)),
        count_by="episode",
    )
    state = policy_state_from_tensors(actor.state_dict(), version=1)
    source.publish(state)
    source.run(can_train=True)

    # An evaluator that owned the group would publish the version the
    # training workers already act with, which the policy rejects as stale.
    with pytest.raises(ValueError, match="version must increase beyond 1, got 1"):
        RolloutEvaluator(group=group, num_episodes=1, max_decisions=10).evaluate(state)

    evaluator = RolloutEvaluator(
        group=group, shared_group=True, num_episodes=1, max_decisions=10
    )
    metrics = evaluator.evaluate(state)

    assert metrics["num_eval_episodes"] == 1
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["avg_steps"] == pytest.approx(2.0)
    # Training continues on the same workers with the same version, and a
    # state the data source never published is still caught after the fact.
    assert source.run(can_train=True)["num_episodes"] == 2
    with pytest.raises(RuntimeError, match="policy version"):
        evaluator.evaluate(policy_state_from_tensors(actor.state_dict(), version=2))
    runtime.close()


def test_evaluator_validates_its_arguments() -> None:
    group = _ScriptedGroup([])
    with pytest.raises(ValueError, match="exactly one"):
        RolloutEvaluator(
            group=group, open_group=lambda: group, num_episodes=1, max_decisions=1
        )
    with pytest.raises(ValueError, match="shared_group requires"):
        RolloutEvaluator(
            open_group=lambda: group, shared_group=True, num_episodes=1, max_decisions=1
        )
    with pytest.raises(ValueError, match="num_episodes"):
        RolloutEvaluator(group=group, num_episodes=0, max_decisions=1)
    with pytest.raises(ValueError, match="video_dir"):
        RolloutEvaluator(
            group=group, num_episodes=1, max_decisions=1, record_video=True
        )


def _two_step_decision(environment_index, *, rewards, done):
    """One executed chunk of two simulator steps whose per-step rewards differ."""
    return CollectedDecision(
        environment_index=environment_index,
        transition=make_transition(
            rewards=rewards,
            steps=2,
            done=done,
            truncated=False,
            info={"is_success": done, "success": done},
        ),
        action_chunk=ActionChunk(
            learner_action=None,
            environment_actions=np.zeros((1, 2, 1), np.float32),
            execution_length=2,
            source="native",
            policy_version=1,
        ),
    )


def test_episode_return_counts_each_decision_the_way_the_learner_does():
    """A dense two-step chunk: ``last`` reads the outcome step, ``sum`` adds both steps."""
    decisions = (
        _two_step_decision(0, rewards=[1.0, 5.0], done=False),
        _two_step_decision(0, rewards=[2.0, 3.0], done=True),
    )
    result = RolloutResult(
        worker_index=0,
        decisions=decisions,
        env_steps=4,
        episodes_completed=1,
        policy_version=1,
        frames=(),
    )
    (last,) = episode_outcomes([result])
    (summed,) = episode_outcomes([result], reward_aggregation="sum")
    assert (last.total_reward, last.decisions) == (8.0, 2)
    assert summed.total_reward == pytest.approx(11.0)
    with pytest.raises(ValueError, match="reward_aggregation"):
        RolloutEvaluator(
            group=_ScriptedGroup([]),
            num_episodes=1,
            max_decisions=1,
            reward_aggregation="mean",
        )


def test_worker_time_summary_reports_counters_as_counts_and_timings_as_seconds():
    def result(worker_index, timings):
        return RolloutResult(
            worker_index=worker_index,
            decisions=(),
            env_steps=0,
            episodes_completed=0,
            policy_version=1,
            timings=timings,
        )

    results = [
        result(
            0,
            {
                "environment_step_seconds": 100.4,
                "policy_calls": 3.0,
                "environment_step_calls": 24.0,
            },
        ),
        result(1, {"environment_step_seconds": 50.0, "policy_calls": 2.0}),
    ]
    assert _worker_time_summary(results) == (
        "; worker time: environment_step_calls=24, environment_step_seconds=150s, policy_calls=5"
    )
    assert _worker_time_summary([]) == ""


def test_the_video_row_carries_the_evaluation_env_step(tmp_path) -> None:
    pytest.importorskip("cv2")
    scored = [_decision(0, reward=1.0, done=True, truncated=False, success=True)]
    frames = tuple(
        {"ego_view_image": np.full((8, 8, 3), value, dtype=np.uint8)}
        for value in (0, 255)
    )
    video_episode = _result(
        [_decision(0, reward=0.0, done=True, truncated=False, success=False)],
        frames=frames,
    )
    logged: list[tuple[str, int | None, str | None]] = []

    class _Logger:
        def log_video(self, name, video, step=None, prefix=None, **kwargs):
            logged.append((name, step, prefix))

    evaluator = RolloutEvaluator(
        group=_ScriptedGroup([scored, video_episode]),
        num_episodes=1,
        max_decisions=10,
        record_video=True,
        video_dir=str(tmp_path),
        logger=_Logger(),
    )

    evaluator.evaluate(_state(), env_steps=4808)

    assert logged == [("eval_video", 4808, "eval")]
