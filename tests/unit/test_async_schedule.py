# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The async off-policy schedule: immediate re-submission, publish cadence, policy-lag accounting and drops."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
    make_recording_runtime_evaluator,
    make_transition,
    single_learner_batch,
)

from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.training import (
    OffPolicyConfig,
    OffPolicyTrainer,
    ScheduleSpec,
)
from isaac_rl_posttraining.training.schedule import (
    on_policy_schedule,
    schedule_from_config,
)


class _VersionedSource(ScriptedRuntimeSource):
    """Scripted rounds whose decisions carry their policy version; drops them as the data source does."""

    def __init__(self, script, sink, max_policy_lag):
        super().__init__(script, action_exec_len=8)
        self.sink = sink
        self.max_policy_lag = max_policy_lag
        self._round_version = 0
        self._learner_version = 0

    def run(self, can_train, *, learner_version=None):
        self._round_version = self._policy_version
        self._learner_version = (
            self._policy_version if learner_version is None else learner_version
        )
        return super().run(can_train)

    def finish(self, pending, *, learner_version=None):
        self._round_version = pending.policy_version
        self._learner_version = (
            self._policy_version if learner_version is None else learner_version
        )
        return super().finish(pending)

    def _consume(self, can_train):
        metrics = super()._consume(can_train)
        decisions = self.decisions_collected_last_run
        lag = self._learner_version - self._round_version
        self.dropped_stale_last_run = 0
        if self.max_policy_lag is not None and lag > self.max_policy_lag:
            self.dropped_stale_last_run = decisions
            self.dropped_stale_total += decisions
            self.decisions_collected_last_run = 0
            self.last_transitions = ()
            return metrics
        transitions = []
        for step in range(decisions):
            transition = make_transition(step_in_episode=step, reward=1.0)
            transition = dataclasses.replace(
                transition,
                metadata=dataclasses.replace(
                    transition.metadata, policy_version=self._round_version
                ),
            )
            self.sink.push(transition)
            transitions.append(transition)
        self.last_transitions = tuple(transitions)
        return metrics


def _async_run(*, rounds, max_policy_lag, publish_every, eval_freq=0):
    buffer = RecordingBuffer()
    source = _VersionedSource(
        [{"steps": 8, "episodes": 1, "decisions": 8}] * rounds,
        InProcessBufferSink(buffer),
        max_policy_lag,
    )
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    records: list[dict] = []
    eval_record: list[int] = []
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=buffer,
        draw_batch=single_learner_batch,
        evaluator=make_recording_runtime_evaluator(eval_record, source),
        config=OffPolicyConfig(
            num_rollouts=8 * rounds,
            learning_starts=0,
            eval_freq=eval_freq,
            schedule=ScheduleSpec(
                "async", max_policy_lag=max_policy_lag, publish_every=publish_every
            ),
        ),
        metric_sink=lambda metrics, step: records.append(dict(metrics)),
    )
    trainer.run()
    system = [metrics for metrics in records if "system/policy_lag" in metrics]
    return source, learner, buffer, system, eval_record


def test_async_drops_rounds_beyond_the_lag_bound_and_counts_them():
    """Eight decisions per round, a publish every four updates, at most one version of lag.

    Round 1 runs on the calling thread. Round 2 is submitted at once and
    collects on version 1 while the learner publishes versions 2 and 3, so it
    lags two versions and is dropped; nothing trains on it. Round 3 collects on
    version 3 (delivered when round 2 ended) and is kept; round 4 is dropped again.
    """
    source, learner, buffer, system, _ = _async_run(
        rounds=4, max_policy_lag=1, publish_every=4
    )

    assert [metrics["system/policy_lag"] for metrics in system] == [0, 2, 0, 2]
    assert [metrics["system/dropped_stale_decisions"] for metrics in system] == [
        0,
        8,
        0,
        8,
    ]
    assert system[-1]["system/dropped_stale_decisions_total"] == 16
    assert len(buffer.adds) == 16
    assert {transition.metadata.policy_version for transition in buffer.adds} == {1, 3}
    assert len(learner.train_step_calls) == 16
    # Rounds are re-submitted as soon as the previous one returns; versions 2 and 4
    # were published while a round was in flight and never reached the workers.
    assert source.published_versions == [1, 3, 5]
    assert [event[0] for event in source.events].count("start") == 3
    # The learner sees only the rewards of what reached the buffer.
    assert sum(len(stream[0]) for stream in learner.observed_reward_streams) == 16


def test_async_without_a_bound_keeps_every_round_and_reports_its_lag():
    _, learner, buffer, system, _ = _async_run(
        rounds=3, max_policy_lag=None, publish_every=2
    )

    assert len(buffer.adds) == 24
    assert len(learner.train_step_calls) == 24
    assert [metrics["system/policy_lag"] for metrics in system] == [0, 4, 4]
    assert all("system/dropped_stale_decisions" not in metrics for metrics in system)


def test_async_evaluation_scores_the_newest_weights_on_idle_workers():
    """An update not yet published is published before the evaluation; no round runs into it."""
    source, learner, _, _, eval_record = _async_run(
        rounds=2, max_policy_lag=None, publish_every=3, eval_freq=16
    )

    assert eval_record == [16]
    # 16 updates at a publish every 3: versions 2..6 by cadence, 7 for the last update before the evaluation.
    assert learner.policy_versions[-1] == 7
    assert source.published_versions[-1] == 7
    assert source.events[-1] == ("publish", 7)


def test_async_with_a_bound_needs_a_data_source_with_the_same_bound():
    with pytest.raises(ValueError, match="max_policy_lag=1"):
        OffPolicyTrainer(
            train_source=ScriptedRuntimeSource([], action_exec_len=8),
            learner_group=ScriptedLearnerGroup(),
            buffer=RecordingBuffer(),
            draw_batch=single_learner_batch,
            evaluator=None,
            config=OffPolicyConfig(
                num_rollouts=8,
                learning_starts=0,
                schedule=ScheduleSpec("async", max_policy_lag=1),
            ),
        )


def test_every_update_is_a_learner_telemetry_row():
    source, learner, _, _, _ = _async_run(
        rounds=2, max_policy_lag=None, publish_every=4
    )
    assert len(source.learner_rows) == len(learner.train_step_calls) == 16
    assert [row[1] for row in source.learner_rows][:8] == [8] * 8


def test_schedule_section():
    assert schedule_from_config(None) == ScheduleSpec("sync")
    assert schedule_from_config("async") == ScheduleSpec("async")
    assert schedule_from_config(
        {"kind": "async", "max_policy_lag": 2, "publish_every": 4}
    ) == ScheduleSpec("async", max_policy_lag=2, publish_every=4)
    assert schedule_from_config(
        {"kind": "overlapped", "max_policy_lag": None, "publish_every": None}
    ) == (ScheduleSpec("overlapped"))
    with pytest.raises(ValueError, match="unknown schedule keys"):
        schedule_from_config({"kind": "async", "lag": 1})
    assert on_policy_schedule(ScheduleSpec("async")).max_policy_lag == 1
    assert on_policy_schedule(ScheduleSpec("overlapped")) == ScheduleSpec("overlapped")
    assert ScheduleSpec("async", max_policy_lag=2).staleness_bound == 2
    assert ScheduleSpec("overlapped", max_policy_lag=2).staleness_bound is None
    assert OffPolicyConfig(
        num_rollouts=1, learning_starts=0, schedule="overlapped"
    ).schedule == (ScheduleSpec("overlapped"))


def test_async_default_publishes_once_per_burst_so_a_long_burst_costs_one_publish_of_lag():
    """A round collected during a 1,100-update burst is one publish old, not 1,100 (the default)."""
    rounds, decisions = 4, 1100
    buffer = RecordingBuffer()
    source = _VersionedSource(
        [{"steps": decisions, "episodes": 1, "decisions": decisions}] * rounds,
        InProcessBufferSink(buffer),
        2,
    )
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    records: list[dict] = []
    OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=buffer,
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=decisions * rounds,
            learning_starts=0,
            schedule=ScheduleSpec("async", max_policy_lag=2),
        ),
        metric_sink=lambda metrics, step: records.append(dict(metrics)),
    ).run()
    system = [metrics for metrics in records if "system/policy_lag" in metrics]

    assert [metrics["system/policy_lag"] for metrics in system] == [0, 1, 1, 1]
    assert system[-1]["system/dropped_stale_decisions_total"] == 0
    assert len(buffer.adds) == len(learner.train_step_calls) == rounds * decisions
    # One publish before training plus one per burst; the last burst's weights reach the workers at the end.
    assert source.published_versions == [1, 2, 3, 4, 5]


def test_async_default_keeps_a_lag_bound_of_one():
    _, _, buffer, system, _ = _async_run(rounds=4, max_policy_lag=1, publish_every=None)
    assert [metrics["system/policy_lag"] for metrics in system] == [0, 1, 1, 1]
    assert system[-1]["system/dropped_stale_decisions_total"] == 0
    assert len(buffer.adds) == 32


def test_publish_every_is_per_burst_unless_set():
    assert ScheduleSpec("async").updates_per_publish is None
    assert ScheduleSpec("async", publish_every=4).updates_per_publish == 4
    # The old default ``1`` is still accepted outside async and means the same as unset.
    assert ScheduleSpec("sync", publish_every=1) == ScheduleSpec("sync")
    assert ScheduleSpec("overlapped", publish_every=1).updates_per_publish is None
    assert schedule_from_config(
        {"kind": "async", "publish_every": None}
    ) == ScheduleSpec("async")


def test_async_checkpoint_holds_the_published_weights(tmp_path):
    """A checkpoint saved while a round is in flight publishes pending updates first."""
    from isaac_rl_posttraining.training import CheckpointStore

    class _Learner(ScriptedLearnerGroup):
        def policy_state(self, *, version):
            self.published_at = len(self.train_step_calls)
            return super().policy_state(version=version)

        def save_checkpoint(self, directory):
            self.saves.append((self.published_at, len(self.train_step_calls)))
            super().save_checkpoint(directory)

    buffer = RecordingBuffer()
    source = _VersionedSource(
        [{"steps": 8, "episodes": 1, "decisions": 8}] * 3,
        InProcessBufferSink(buffer),
        None,
    )
    learner = _Learner(step_at_call=lambda: source.cum_steps)
    learner.saves = []
    OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=buffer,
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=24,
            learning_starts=0,
            save_interval=8,
            schedule=ScheduleSpec("async", max_policy_lag=None, publish_every=3),
        ),
        checkpoint_store=CheckpointStore(str(tmp_path)),
    ).run()
    assert len(learner.saves) == 4  # 8, 16, 24 and final
    assert all(published == trained for published, trained in learner.saves), (
        learner.saves
    )


def test_an_interrupted_save_holds_the_learner_weights(tmp_path):
    """A signal mid-burst saves after publishing the updates the burst already applied."""
    from isaac_rl_posttraining.training import CheckpointStore

    class _Learner(ScriptedLearnerGroup):
        def train_step(self, batches):
            result = super().train_step(batches)
            if len(self.train_step_calls) == 5:
                raise KeyboardInterrupt
            return result

        def policy_state(self, *, version):
            self.published_at = len(self.train_step_calls)
            return super().policy_state(version=version)

        def save_checkpoint(self, directory):
            self.saves.append((self.published_at, len(self.train_step_calls)))
            super().save_checkpoint(directory)

    buffer = RecordingBuffer()
    source = _VersionedSource(
        [{"steps": 8, "episodes": 1, "decisions": 8}] * 3,
        InProcessBufferSink(buffer),
        None,
    )
    learner = _Learner(step_at_call=lambda: source.cum_steps)
    learner.saves = []
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=buffer,
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=24,
            learning_starts=0,
            schedule=ScheduleSpec("async", max_policy_lag=None),
        ),
        checkpoint_store=CheckpointStore(str(tmp_path)),
    )
    with pytest.raises(KeyboardInterrupt):
        trainer.run()
    trainer.save_checkpoint("interrupted")
    assert learner.saves == [(5, 5)]
