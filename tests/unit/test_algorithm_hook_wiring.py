# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training loop must actually invoke the algorithm's observation hooks.

``observe_rewards`` and ``target_update_count`` are BaseAlgorithm contract
methods that default to no-op/None, so a missing call-site fails silently:
SAC keeps training and no test notices. These tests pin the call-sites so the
hooks stay live for algorithms (SAC-edit today, future variants) that
override them. The loop reaches the hooks through the learner group's
``call`` seam, which the scripted learner group records.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
    make_transition,
    single_learner_batch,
)
from loguru import logger

from isaac_rl_posttraining.training import (
    OffPolicyConfig,
    OffPolicyTrainer,
)


class _TransitionSource(ScriptedRuntimeSource):
    """A scripted source whose every ``run`` also exposes one collected transition."""

    def run(self, can_train, *, learner_version=None):
        metrics = super().run(can_train)
        self.last_transitions = (
            make_transition(
                obs={"state": np.zeros(1, dtype=np.float32)},
                action=np.zeros(1, dtype=np.float32),
                reward=1.0,
                next_obs={"state": np.zeros(1, dtype=np.float32)},
            ),
        )
        return metrics


def _make_trainer(source, num_rollouts):
    """Wire the loop around scripted doubles, evaluation disabled."""
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(len_fn=lambda: source.cum_steps),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=num_rollouts,
            learning_starts=0,
            train_after_episode=True,
            eval_at_episode_boundary=True,
        ),
    )
    return trainer, learner


def test_observe_rewards_forwarded_from_the_collected_transitions():
    """Each collection burst hands its transitions' rewards, dones and truncations to the learner."""
    source = _TransitionSource(
        script=[{"steps": 10, "episodes": 1}, {"steps": 10, "episodes": 1}],
        action_exec_len=10,
    )
    trainer, learner = _make_trainer(source, num_rollouts=20)

    trainer.run()

    stream = ([1.0], [False], [False], [0])
    assert learner.observed_reward_streams == [stream, stream], (
        "the loop must forward the live reward stream to observe_rewards every burst; "
        f"got {learner.observed_reward_streams}"
    )


def test_observe_rewards_skipped_without_transitions():
    """A burst that ingested nothing must not call observe_rewards."""
    source = ScriptedRuntimeSource(
        script=[{"steps": 10, "episodes": 1}], action_exec_len=10
    )
    trainer, learner = _make_trainer(source, num_rollouts=10)

    trainer.run()

    assert learner.observed_reward_streams == []


def test_target_update_count_queried_with_running_env_steps():
    """The loop queries the cumulative-update target with the current step count."""
    source = ScriptedRuntimeSource(
        script=[{"steps": 10, "episodes": 1}, {"steps": 10, "episodes": 1}],
        action_exec_len=10,
    )
    trainer, learner = _make_trainer(source, num_rollouts=20)

    trainer.run()

    # One query per trained burst, each carrying the cumulative env-step count.
    assert learner.target_update_count_calls == [10, 20], (
        f"target_update_count must be queried with total_env_steps each burst; "
        f"got {learner.target_update_count_calls}"
    )


def test_a_budget_spent_before_any_update_warns():
    """A budget that ends inside the first post-warm-up round trains nothing and says so."""
    source = ScriptedRuntimeSource(
        script=[{"steps": 3200, "episodes": 8}], action_exec_len=10
    )
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(len_fn=lambda: source.cum_steps),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=3000, learning_starts=1000, train_after_episode=True
        ),
    )

    warnings: list[str] = []
    handle = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        trainer.run()
    finally:
        logger.remove(handle)

    assert any("no learner update" in line for line in warnings)


def test_a_budget_with_no_episode_end_after_warm_up_names_the_episode_cadence():
    """Decisions collected after warm-up wait for an episode end; the warning says so."""
    source = ScriptedRuntimeSource(
        script=[{"steps": 500, "episodes": 0}, {"steps": 500, "episodes": 0}],
        action_exec_len=10,
    )
    learner = ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps)
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(len_fn=lambda: source.cum_steps),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(
            num_rollouts=1000, learning_starts=400, train_after_episode=True
        ),
    )

    warnings: list[str] = []
    handle = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        trainer.run()
    finally:
        logger.remove(handle)

    assert any("waited for an episode to end" in line for line in warnings), warnings
