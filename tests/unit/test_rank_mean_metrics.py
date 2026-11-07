# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run.train returns what the trainer logs: an update's metrics averaged over learner ranks."""

from types import SimpleNamespace

from isaac_rl_posttraining.runners.off_policy_session import OffPolicySession
from isaac_rl_posttraining.runtimes.learner import LearnerStepResult
from isaac_rl_posttraining.training.on_policy_metrics import rank_mean_metrics


def _results(*losses: float) -> tuple[LearnerStepResult, ...]:
    return tuple(
        LearnerStepResult(
            learner_rank=rank,
            step=1,
            samples=4,
            metrics={"loss": loss, f"rank{rank}_only": 1.0},
        )
        for rank, loss in enumerate(losses)
    )


class _Batch:
    def at_env_steps(self, env_steps):
        return self

    def split(self, num_ranks):
        return [self] * num_ranks


class _TwoRanks:
    num_learner_ranks = 2

    def train_step(self, batches):
        return _results(1.0, 3.0)

    def hooks(self):
        return SimpleNamespace(gradient_steps=lambda: 1)

    def policy_state(self, *, version):
        return SimpleNamespace(version=version)


def test_one_rank_keeps_its_metrics_and_several_ranks_average_the_shared_ones():
    (single,) = _results(2.0)
    assert rank_mean_metrics((single,)) is single.metrics
    assert rank_mean_metrics(_results(1.0, 3.0)) == {"loss": 2.0}


def test_session_train_returns_the_rank_mean_the_trainer_logs():
    logged = []
    session = OffPolicySession(
        trainer=SimpleNamespace(draw_batch=_Batch),
        train_source=SimpleNamespace(
            log_learner=lambda step, env_steps, metrics, **kwargs: logged.append(
                dict(metrics)
            ),
            publish=lambda state: None,
        ),
        learner_group=_TwoRanks(),
        evaluator=None,
        learning_starts=0,
        run_dir="",
    )
    assert session.train(steps=2) == {"loss": 2.0}
    assert logged == [{"loss": 2.0}, {"loss": 2.0}]
