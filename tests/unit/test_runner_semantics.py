# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Off-policy loop semantics on scripted collection and learner doubles.

Pins the training-loop contract:

* burst sizing: with ``train_after_episode``, each completed episode triggers
  one learner update per stored decision, including short terminal chunks,
* evaluation is scheduled at ``eval_freq`` env-step boundaries (5k cadence in
  the shipped configs) and executes at episode boundaries,
* ``eval_on_first_step`` runs one evaluation at step 0, before any training,
* checkpoints are written every ``save_interval`` env steps plus a final one,
* SAVE-FAILURE RESILIENCE: a checkpoint save that raises ``OSError`` (e.g. a
  network-filesystem blip) is retried once, then skipped -- and the run
  continues (regression for a real shared-filesystem outage that used to kill
  multi-hour runs),
* no learner updates happen before ``learning_starts`` env steps,
* the ``overlapped`` schedule starts the next round before every burst except
  those an evaluation or the budget's end follows, with a policy lag of one.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
    make_recording_runtime_evaluator,
    single_learner_batch,
)

from isaac_rl_posttraining.training import (
    CheckpointStore,
    OffPolicyConfig,
    OffPolicyTrainer,
)


def _make_setup(
    script,
    action_exec_len,
    num_rollouts,
    learning_starts=0,
    eval_freq=0,
    eval_on_first_step=False,
    save_dir=None,
    save_interval=None,
    with_evaluator=True,
    schedule="sync",
    metric_sink=None,
):
    """Build (trainer, source, learner, eval_record) around scripted doubles.

    ``learner`` is the ScriptedLearnerGroup; it exposes ``train_step_calls``
    and ``save_dirs``.
    """
    eval_record = []
    source = ScriptedRuntimeSource(script, action_exec_len)
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(len_fn=lambda: source.cum_steps),
        draw_batch=single_learner_batch,
        evaluator=make_recording_runtime_evaluator(eval_record, source)
        if with_evaluator
        else None,
        config=OffPolicyConfig(
            num_rollouts=num_rollouts,
            learning_starts=learning_starts,
            train_after_episode=True,
            eval_freq=eval_freq,
            eval_on_first_step=eval_on_first_step,
            eval_at_episode_boundary=True,
            save_interval=save_interval,
            schedule=schedule,
        ),
        metric_sink=metric_sink,
        checkpoint_store=None if save_dir is None else CheckpointStore(save_dir),
    )
    return trainer, source, learner, eval_record


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Neutralize retry backoffs so the resilience tests stay fast; record durations."""
    calls = []
    monkeypatch.setattr("time.sleep", lambda seconds: calls.append(seconds))
    return calls


@pytest.mark.parametrize("schedule", ["sync", "overlapped"])
def test_burst_size_equals_stored_decisions(schedule):
    """Terminal partial chunks receive the same update credit as full chunks, on either schedule."""
    trainer, source, learner, _ = _make_setup(
        script=[
            {"steps": 39, "episodes": 1, "decisions": 5},
            {"steps": 25, "episodes": 1, "decisions": 4},
        ],
        action_exec_len=8,
        num_rollouts=64,
        schedule=schedule,
    )
    trainer.run()

    assert source.cum_steps == 64
    assert len(learner.train_step_calls) == 9, (
        f"expected 5 + 4 = 9 train steps, got {len(learner.train_step_calls)}"
    )
    assert source.actor_updates >= 2, (
        "the workers must receive the refreshed weights after training"
    )


@pytest.mark.parametrize("schedule", ["sync", "overlapped"])
def test_eval_scheduled_at_freq_boundaries(schedule):
    """With eval_freq=5000, evaluations land at the 5k / 10k / 15k env-step boundaries."""
    trainer, _, _, eval_record = _make_setup(
        script=[{"steps": 1000, "episodes": 1}] * 15,
        action_exec_len=8,
        num_rollouts=15000,
        eval_freq=5000,
        schedule=schedule,
    )
    trainer.run()

    assert eval_record == [5000, 10000, 15000], (
        f"evaluations must fire at every 5k env-step boundary; got {eval_record}"
    )


def test_sync_schedule_never_starts_a_round_ahead():
    """The reference schedule collects on the calling thread only: no start / finish calls."""
    trainer, source, _, _ = _make_setup(
        script=[{"steps": 1000, "episodes": 1}] * 4,
        action_exec_len=8,
        num_rollouts=4000,
        eval_freq=2000,
    )
    trainer.run()

    kinds = [event[0] for event in source.events]
    assert "start" not in kinds and "finish" not in kinds
    assert kinds.count("run") == 4


class _EventLearnerGroup(ScriptedLearnerGroup):
    """Writes every burst's first update into the source's event log, to order it against the rounds."""

    def __init__(self, source):
        super().__init__(step_at_call=lambda: source.cum_steps)
        self._events = source.events

    def train_step(self, batches):
        if not self._events or self._events[-1][0] != "train":
            self._events.append(("train",))
        return super().train_step(batches)


def _run_overlapped(script, *, num_rollouts, eval_freq, learning_starts=0):
    """An overlapped run on the scripted doubles; returns (source, learner, eval_record, system records)."""
    records = []
    trainer, source, _, eval_record = _make_setup(
        script=script,
        action_exec_len=8,
        num_rollouts=num_rollouts,
        learning_starts=learning_starts,
        eval_freq=eval_freq,
        schedule="overlapped",
        metric_sink=lambda metrics, step: records.append(dict(metrics)),
    )
    learner = _EventLearnerGroup(source)
    trainer.learner_group = learner
    trainer.run()
    system = [metrics for metrics in records if "system/policy_lag" in metrics]
    return source, learner, eval_record, system


def test_overlapped_starts_the_next_round_before_each_burst_but_not_into_an_evaluation():
    """Six rounds of 1000 steps with evaluations at 3000 and at the budget's end (6000).

    Rounds 2, 3, 5 and 6 are started before the burst of the iteration that
    precedes them; iterations 3 and 6 start nothing because an evaluation
    follows their burst, so round 4 is collected on the calling thread again.
    A round finished after a burst was collected on the weights before it
    (policy lag 1); rounds collected on the calling thread have no lag.
    """
    source, _, eval_record, system = _run_overlapped(
        [{"steps": 1000, "episodes": 1}] * 6, num_rollouts=6000, eval_freq=3000
    )

    assert eval_record == [3000, 6000]
    assert source.events == [
        ("publish", 1),  # the initial weights, before any round
        ("run", True),
        ("start", True),
        ("train",),
        ("publish", 2),
        ("finish",),
        ("start", True),
        ("train",),
        ("publish", 3),
        ("finish",),
        ("train",),
        ("publish", 4),
        ("run", True),
        ("start", True),
        ("train",),
        ("publish", 5),
        ("finish",),
        ("start", True),
        ("train",),
        ("publish", 6),
        ("finish",),
        ("train",),
        ("publish", 7),
    ]
    assert [metrics["system/policy_lag"] for metrics in system] == [0, 1, 1, 0, 1, 1]
    assert all(metrics["system/collect_wall_s"] >= 0.0 for metrics in system)


def test_overlapped_submits_the_next_round_with_the_recomputed_training_gate():
    """The round started ahead uses the gate as the next iteration would compute it: the
    round that crosses learning_starts is followed by a training round, not a warm-up one."""
    source, learner, _, _ = _run_overlapped(
        [{"steps": 1000, "episodes": 1}] * 4,
        num_rollouts=4000,
        eval_freq=0,
        learning_starts=1500,
    )

    starts = [flag for kind, *rest in source.events if kind == "start" for flag in rest]
    assert starts == [False, True, True]
    assert source.can_train_flags == [False, False, True, True]
    assert all(at >= 1500 for at in learner.train_step_calls)


def test_eval_waits_for_episode_boundary():
    """A scheduled eval stays pending until an episode actually completes."""
    trainer, _, _, eval_record = _make_setup(
        script=[{"steps": 5000, "episodes": 0}, {"steps": 1000, "episodes": 1}],
        action_exec_len=8,
        num_rollouts=6000,
        eval_freq=5000,
    )
    trainer.run()

    assert eval_record == [6000], (
        "eval was scheduled at 5000 but no episode had completed; it must execute at the "
        f"first episode boundary (6000). Got {eval_record}"
    )


def test_pending_eval_runs_when_the_budget_ends_without_an_episode_boundary():
    """An evaluation scheduled at the last freq boundary must not be dropped because
    the budget ran out before another episode completed (several envs overshoot)."""
    trainer, _, _, eval_record = _make_setup(
        script=[{"steps": 5000, "episodes": 1}, {"steps": 5100, "episodes": 0}],
        action_exec_len=8,
        num_rollouts=10000,
        eval_freq=5000,
    )
    trainer.run()

    assert eval_record == [5000, 10100], (
        f"the 10k evaluation was pending at the end of the budget and must run then; got {eval_record}"
    )


def test_final_evaluation_runs_when_the_last_periodic_one_fired_late():
    """Evaluations fire at the first check past a multiple of eval_freq, so with a
    5100-step burst the 5k evaluation lands at 5100 and the 10k one would only be
    due at 10100, past the budget; the end of the budget must still be scored."""
    trainer, _, _, eval_record = _make_setup(
        script=[{"steps": 5100, "episodes": 1}, {"steps": 4900, "episodes": 1}],
        action_exec_len=8,
        num_rollouts=10000,
        eval_freq=5000,
    )
    trainer.run()

    assert eval_record == [5100, 10000], (
        f"the run must end with an evaluation at the budget even though 10000 - 5100 < eval_freq; got {eval_record}"
    )


@pytest.mark.parametrize(
    ("eval_on_first_step", "expected"),
    [(True, [0, 1000]), (False, [1000])],
    ids=["step0_eval", "no_step0_eval"],
)
def test_step0_evaluation_follows_eval_on_first_step(eval_on_first_step, expected):
    """eval_on_first_step=True scores step 0; the budget end (1000 < eval_freq) is always scored."""
    trainer, _, _, eval_record = _make_setup(
        script=[{"steps": 1000, "episodes": 1}],
        action_exec_len=8,
        num_rollouts=1000,
        eval_freq=5000,
        eval_on_first_step=eval_on_first_step,
    )
    trainer.run()

    assert eval_record == expected, (
        f"expected evaluations at {expected}, got {eval_record}"
    )


def test_no_evaluator_means_no_evaluation():
    """A collect-only run (no evaluator) trains the full budget and never evaluates."""
    trainer, source, _, eval_record = _make_setup(
        script=[{"steps": 1000, "episodes": 1}] * 15,
        action_exec_len=8,
        num_rollouts=15000,
        eval_freq=5000,
        eval_on_first_step=True,
        with_evaluator=False,
    )
    trainer.run()

    assert trainer.evaluator is None
    assert eval_record == []
    assert source.cum_steps == 15000, (
        "collection and training must run the full budget without an evaluator"
    )
    with pytest.raises(RuntimeError):
        trainer.evaluate()


def test_save_interval_cadence(tmp_path):
    """Checkpoints land every save_interval env steps, plus a final checkpoint."""
    save_dir = tmp_path / "ckpts"
    trainer, _, learner, _ = _make_setup(
        script=[{"steps": 10, "episodes": 1}] * 6,
        action_exec_len=10,
        num_rollouts=60,
        save_dir=str(save_dir),
        save_interval=20,
    )
    trainer.run()

    assert len(learner.save_dirs) == 4, (
        f"expected saves at 20/40/60 plus a final checkpoint, got {learner.save_dirs}"
    )
    step_tags = []
    for d in learner.save_dirs:
        tag = os.path.basename(os.path.normpath(d))
        try:
            step_tags.append(int(tag))
        except ValueError:
            step_tags.append(tag)
    assert step_tags[:3] == [20, 40, 60], f"periodic checkpoint tags off: {step_tags}"
    for d in learner.save_dirs:
        assert os.path.isfile(os.path.join(d, "algo_state.pt")), (
            f"checkpoint payload missing in {d}"
        )


class _FailNTimes:
    """Callable torch.save replacement that raises OSError for its first N calls."""

    def __init__(self, n_failures, real_save):
        self.remaining = n_failures
        self.real_save = real_save
        self.attempts = 0

    def __call__(self, *args, **kwargs):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise OSError(
                122, "Disk quota exceeded (simulated shared-filesystem outage)"
            )
        return self.real_save(*args, **kwargs)


def test_save_failure_retries_once_then_skips_and_the_run_goes_on(
    tmp_path, monkeypatch, _fast_sleep
):
    """An OSError-ing save is retried exactly once, that checkpoint is skipped, and the run completes
    with later saves and the evaluation cadence intact."""
    save_dir = tmp_path / "ckpts"
    trainer, source, learner, eval_record = _make_setup(
        script=[{"steps": 10, "episodes": 1}] * 4,
        action_exec_len=10,
        num_rollouts=40,
        eval_freq=20,
        save_dir=str(save_dir),
        save_interval=10,
    )
    failer = _FailNTimes(2, torch.save)
    monkeypatch.setattr(torch, "save", failer)

    trainer.run()  # must NOT raise: the outage costs one checkpoint, not the run

    # The first checkpoint (step 10) was attempted twice (initial + one retry), then skipped.
    tag10_attempts = [
        d for d in learner.save_dirs if os.path.basename(os.path.normpath(d)) == "10"
    ]
    assert len(tag10_attempts) == 2, (
        f"expected exactly one retry (2 attempts) for the failing checkpoint, got "
        f"{len(tag10_attempts)} attempts: {learner.save_dirs}"
    )
    assert not os.path.isfile(os.path.join(str(save_dir), "10", "algo_state.pt")), (
        "the failed checkpoint must be skipped, not silently half-written"
    )
    assert len(_fast_sleep) == 1, (
        f"expected a single retry backoff sleep, got sleeps: {_fast_sleep}"
    )
    assert source.cum_steps == 40, (
        "the run must reach its env-step budget despite the outage"
    )
    assert len(learner.train_step_calls) == 4, (
        "training must continue after the skipped save"
    )
    assert eval_record == [20, 40], "evaluation cadence must survive the skipped save"
    for tag in ("20", "30", "40"):
        assert os.path.isfile(os.path.join(str(save_dir), tag, "algo_state.pt")), (
            f"checkpoint {tag} must succeed once the filesystem recovers"
        )


def test_runner_respects_learning_starts():
    """No learner updates before learning_starts env steps have been collected."""
    trainer, source, learner, _ = _make_setup(
        script=[{"steps": 10, "episodes": 1}] * 4,
        action_exec_len=10,
        num_rollouts=40,
        learning_starts=30,
    )
    trainer.run()

    assert learner.train_step_calls, (
        "training must start once learning_starts is reached"
    )
    assert all(at >= 30 for at in learner.train_step_calls), (
        f"train steps happened before learning_starts=30 env steps: {learner.train_step_calls}"
    )
    # Only post-gate steps earn update credit: the 30-step warmup backlog is
    # exploration data, not a burst (a backlog burst would front-load
    # thousands of updates on the fresh buffer in real recipes). One
    # iteration collects 10 steps past the gate -> 10 // 10 = 1 burst.
    assert len(learner.train_step_calls) == 1, (
        f"expected 10//10 = 1 post-gate update, got {len(learner.train_step_calls)}"
    )
    assert source.can_train_flags[0] is False, (
        "the very first rollout must be exploratory"
    )
    assert source.can_train_flags[-1] is True
