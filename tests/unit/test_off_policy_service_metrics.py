# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The off-policy loop logs the policy servers' batch sizes every iteration and resets them."""

from __future__ import annotations

from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
    single_learner_batch,
)

from isaac_rl_posttraining.training import OffPolicyConfig, OffPolicyTrainer


class _ServedGroup:
    """A rollout group whose policy server ran two action batches since the last reset."""

    def __init__(self) -> None:
        self.resets = 0

    def policy_server_stats(self):
        return (
            {
                "operation_seconds": {"action": 0.5},
                "operation_batch_sizes": {"action": (16, 32)},
                "request_latencies_seconds": (0.1, 0.3),
            },
        )

    def reset_policy_server_stats(self) -> None:
        self.resets += 1


def test_off_policy_iterations_log_policy_server_batches():
    source = ScriptedRuntimeSource(
        [{"steps": 8, "episodes": 1, "decisions": 8}] * 2, action_exec_len=8
    )
    source.group = _ServedGroup()
    records: list[dict] = []
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=ScriptedLearnerGroup(step_at_call=lambda: source.cum_steps),
        buffer=RecordingBuffer(),
        draw_batch=single_learner_batch,
        evaluator=None,
        config=OffPolicyConfig(num_rollouts=16, learning_starts=0, eval_freq=0),
        metric_sink=lambda metrics, step: records.append(dict(metrics)),
    )
    trainer.run()

    served = [m for m in records if "policy_service/action_mean_batch_size" in m]
    assert len(served) == 2
    assert served[0]["policy_service/action_mean_batch_size"] == 24.0
    assert served[0]["policy_service/action_max_batch_size"] == 32.0
    assert source.group.resets == 2
