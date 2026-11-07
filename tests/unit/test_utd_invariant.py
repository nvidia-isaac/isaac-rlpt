# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UTD (update-to-data) invariant under parallel collection.

With ``N`` envs a single vector step collects ``N`` decisions, so the number of
``train_step()`` calls must scale with ``N`` for the update-to-data ratio
(gradient updates per collected decision) to stay INVARIANT to ``num_envs``.
Each ``train_step()`` runs ``num_updates_per_train_step`` gradient updates, so the
invariant reduces to: cumulative ``train_step()`` calls == cumulative decisions
collected, for any ``N`` and on both training cadences.

This drives the real :class:`~isaac_rl_posttraining.training.OffPolicyTrainer`
with lightweight fakes (no sim, no models) so the accounting under test is
exactly the one the training loop uses.
"""

import time

import pytest

torch = pytest.importorskip("torch")

from conftest import (
    ScriptedLearnerGroup,
    single_learner_batch,
)

from isaac_rl_posttraining.training import (
    OffPolicyConfig,
    OffPolicyTrainer,
    PendingCollection,
)

SCHEDULES = ("sync", "overlapped")
SYSTEM_KEYS = {
    "system/collect_wall_s",
    "system/collect_wait_s",
    "system/train_wall_s",
    "system/publish_wall_s",
    "system/iteration_wall_s",
    "system/train_updates",
    "system/policy_lag",
    "system/env_steps_per_s",
    "system/iteration_env_steps_per_s",
}


class _Submitted:
    """The fake's asynchronous call: the round is produced when ``finish`` ingests it, and ends at once."""

    def __init__(self) -> None:
        self._finished_at = time.time()

    def result(self):
        return ()

    def finished_at(self):
        return self._finished_at


class _FakeBuffer:
    reward_channel = "env"
    reward_aggregation = "last"

    def __len__(self) -> int:
        return 0

    def get_statistics(self):
        return {}

    def reward_statistics(self):
        return {}

    def transitions(self):  # pragma: no cover - only read when logger set
        return []


class _FakeRuntimeSource:
    """A data source that 'collects' ``num_envs`` decisions per ``run()`` (one vector step).

    Every env closes an episode every three vector steps; no sim or model runs.
    """

    _EP_EVERY_VEC_STEPS = 3
    max_policy_lag = None
    dropped_stale_last_run = 0
    dropped_stale_total = 0

    def __init__(self, num_envs: int, action_exec_len: int) -> None:
        self.num_envs = num_envs
        self.action_exec_len = action_exec_len
        self.total_episodes = 0
        self.decisions_collected_last_run = 0
        self.last_transitions = ()
        self._env_steps = 0
        self._vec_steps = 0
        self._policy_version = 0
        self._in_flight = None

    def run(self, can_train: bool, *, learner_version: int | None = None) -> dict:
        # One vector step == num_envs decisions == num_envs*action_exec_len env
        # steps (every env's steps are summed into total_steps).
        self.decisions_collected_last_run = self.num_envs
        self._env_steps += self.num_envs * self.action_exec_len
        self._vec_steps += 1
        if self._vec_steps % self._EP_EVERY_VEC_STEPS == 0:
            self.total_episodes += self.num_envs
        return {"total_steps": self._env_steps, "success_rate": 0.0}

    def start(self, can_train: bool) -> PendingCollection:
        assert self._in_flight is None, (
            "the trainer started a round while one was in flight"
        )
        started = time.time()
        self._in_flight = PendingCollection(
            call=_Submitted(),
            policy_version=self._policy_version,
            mode="train" if can_train else "warmup",
            started=started,
        )
        return self._in_flight

    def finish(
        self, pending: PendingCollection, *, learner_version: int | None = None
    ) -> dict:
        assert pending is self._in_flight, (
            "the trainer finished a round it did not start"
        )
        self._in_flight = None
        return self.run(pending.mode == "train")

    def set_weights(self, weights) -> None:
        self._policy_version = weights.version

    def log_learner(self, step, env_steps, metrics, *, publish_version=None) -> None:
        pass


def _run_trainer(
    num_envs: int,
    action_exec_len: int,
    n_vec_steps: int,
    train_after_episode: bool,
    max_updates_per_burst: int | None = None,
    learning_starts: int = 0,
    schedule: str = "sync",
):
    max_steps = num_envs * action_exec_len * n_vec_steps
    total_decisions = n_vec_steps * num_envs
    learner = ScriptedLearnerGroup()
    trainer = OffPolicyTrainer(
        train_source=_FakeRuntimeSource(num_envs, action_exec_len),
        learner_group=learner,
        buffer=_FakeBuffer(),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=max_steps,
            learning_starts=learning_starts,
            train_after_episode=train_after_episode,
            max_updates_per_burst=max_updates_per_burst,
            schedule=schedule,
        ),
    )
    trainer.run()
    return len(learner.train_step_calls), total_decisions


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("num_envs", [1, 4, 8])
@pytest.mark.parametrize("train_after_episode", [False, True])
def test_updates_per_decision_is_invariant_to_num_envs(
    num_envs, train_after_episode, schedule
):
    # 9 vector steps, an episode boundary every 3 -> the run ends ON a boundary
    # so the train_after_episode cadence has trained every collected decision.
    # The schedule moves the rounds in time, never the update credit.
    action_exec_len = 20
    train_step_calls, total_decisions = _run_trainer(
        num_envs,
        action_exec_len,
        n_vec_steps=9,
        train_after_episode=train_after_episode,
        schedule=schedule,
    )
    # The invariant: exactly one train_step() (== num_updates_per_train_step
    # gradient updates) per collected decision, regardless of num_envs.
    assert total_decisions == 9 * num_envs
    assert train_step_calls == total_decisions


@pytest.mark.parametrize("train_after_episode", [False, True])
def test_max_updates_per_burst_caps_each_training_phase(train_after_episode):
    """With many envs the per-decision rule owes num_envs x more updates per
    phase; the cap bounds every phase, in both cadences."""
    # 8 envs, 9 vector steps, episode boundary every 3 -> 3 bursts of 24
    # decisions (burst mode) or 9 phases of 8 decisions (per-iteration mode).
    calls, total_decisions = _run_trainer(
        8,
        20,
        n_vec_steps=9,
        train_after_episode=train_after_episode,
        max_updates_per_burst=5,
    )
    phases = 3 if train_after_episode else 9
    assert total_decisions == 72
    assert calls == 5 * phases  # every phase hit the cap
    # A cap above the per-phase demand changes nothing.
    calls_loose, _ = _run_trainer(
        8,
        20,
        n_vec_steps=9,
        train_after_episode=train_after_episode,
        max_updates_per_burst=1000,
    )
    assert calls_loose == total_decisions


def test_max_updates_per_burst_rejects_non_positive():
    with pytest.raises(ValueError, match="max_updates_per_burst"):
        _run_trainer(
            1, 20, n_vec_steps=3, train_after_episode=False, max_updates_per_burst=0
        )


def test_capped_first_burst_credits_the_warm_up_backlog():
    """With a cap, the first burst fires at learning_starts on the warm-up data
    (bounded by the cap); uncapped, the backlog is still not update credit."""
    # 8 envs x 20 steps = 160 env steps per vector step; episodes close every 3
    # vector steps (steps 480, 960, ...). learning_starts = 480 lands exactly on
    # an episode boundary, like a round-sized warm-up does at N envs.
    capped, _ = _run_trainer(
        8,
        20,
        n_vec_steps=3,
        train_after_episode=True,
        max_updates_per_burst=5,
        learning_starts=480,
    )
    assert capped == 5  # 24 decisions of warm-up data, capped to 5 updates, at step 480
    uncapped, _ = _run_trainer(
        8, 20, n_vec_steps=3, train_after_episode=True, learning_starts=480
    )
    assert (
        uncapped == 0
    )  # no cap: nothing collected after learning_starts yet -> no burst
    # The credit is consumed: the next burst counts only the steps since.
    capped_two_rounds, _ = _run_trainer(
        8,
        20,
        n_vec_steps=6,
        train_after_episode=True,
        max_updates_per_burst=1000,
        learning_starts=480,
    )
    assert (
        capped_two_rounds == 48
    )  # 24 warm-up decisions + 24 post-warm-up decisions, no double counting


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_trainer_logs_the_iteration_wall_split_every_iteration(schedule):
    """Each iteration sinks the collect / wait / train / publish / whole-iteration walls, the update
    count, the policy lag and the two rates, whatever the schedule."""
    records: list[tuple[dict, int]] = []
    num_envs, action_exec_len, n_vec_steps = 4, 8, 6
    learner = ScriptedLearnerGroup()
    trainer = OffPolicyTrainer(
        train_source=_FakeRuntimeSource(num_envs, action_exec_len),
        learner_group=learner,
        buffer=_FakeBuffer(),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=num_envs * action_exec_len * n_vec_steps,
            learning_starts=0,
            train_after_episode=True,
            schedule=schedule,
        ),
        metric_sink=lambda metrics, step: records.append((dict(metrics), step)),
    )
    trainer.run()
    system = [metrics for metrics, _ in records if "system/collect_wall_s" in metrics]
    assert len(system) == n_vec_steps
    for metrics in system:
        assert metrics.keys() == SYSTEM_KEYS
        for key in SYSTEM_KEYS:
            assert metrics[key] >= 0.0
        assert metrics["system/iteration_wall_s"] >= metrics["system/train_wall_s"]
    assert sum(metrics["system/train_updates"] for metrics in system) == len(
        learner.train_step_calls
    )
    # Bursts run on the episode ends (every third round); under the overlapped schedule the round
    # that follows a burst was collected on the weights before it.
    lags = [metrics["system/policy_lag"] for metrics in system]
    assert lags == (
        [0, 0, 0, 1, 0, 0] if schedule == "overlapped" else [0] * n_vec_steps
    )
    # The learner's own metrics reach the sink under learner/ after every burst that trained.
    learner_records = [
        metrics for metrics, _ in records if "learner/critic_loss" in metrics
    ]
    assert len(learner_records) == sum(
        1 for metrics in system if metrics["system/train_updates"] > 0
    )
    assert learner_records and all(
        metrics.keys()
        == {
            "learner/critic_loss",
            "learner/actor_loss",
            "learner/ent_coef",
            "learner/q_values_mean",
            "learner/updates",
        }
        for metrics in learner_records
    )


@pytest.mark.parametrize("episode_cadence", [False, True])
def test_128_environment_550_step_boundary_update_counts(episode_cadence):
    class Source:
        total_episodes = 0
        decisions_collected_last_run = 0
        last_transitions = ()
        max_policy_lag = None
        dropped_stale_last_run = 0
        dropped_stale_total = 0

        def __init__(self):
            self.decisions = 0
            self.boundaries = []

        def run(self, can_train, *, learner_version=None):
            self.decisions += 138 if episode_cadence else 1
            rounds, remainder = divmod(self.decisions, 138)
            self.total_episodes = rounds * 128
            self.decisions_collected_last_run = 17664 if episode_cadence else 128
            return {
                "total_steps": rounds * 70400 + remainder * 512,
                "success_rate": 0.5,
            }

        def set_weights(self, weights):
            if self.decisions and self.decisions % 138 == 0:
                self.boundaries.append(len(learner.train_step_calls))

        def log_learner(self, step, env_steps, metrics, *, publish_version=None):
            pass

    source = Source()
    learner = ScriptedLearnerGroup()
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=_FakeBuffer(),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=140800,
            learning_starts=70400,
            train_after_episode=episode_cadence,
            max_updates_per_burst=17664 if episode_cadence else 128,
        ),
    )
    trainer.run()
    assert source.total_episodes == 256
    assert source.boundaries == ([17664, 35328] if episode_cadence else [128, 17792])
