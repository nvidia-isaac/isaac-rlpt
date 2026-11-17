# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline evaluation executes the chunk ``infer`` decodes."""

from __future__ import annotations

import numpy as np

from isaac_rl_posttraining.runners.offline_launch import _InferActionChunkPolicy
from isaac_rl_posttraining.runtimes.gym_worker import GymRolloutWorker


class _StubInferPolicy:
    action_window = 40
    frozen_submodules = ()

    def eval(self):
        return self

    def infer(self, observations):
        assert observations["instruction"] == "pick up the drill"
        return np.zeros((1, 16, 35), dtype=np.float32)

    def state_dict(self):
        return {}


def test_offline_eval_executes_the_decoded_chunk_not_the_configured_horizon():
    policy = _InferActionChunkPolicy(
        _StubInferPolicy(), instruction="pick up the drill"
    )
    (chunk,) = policy.action_chunks(
        (
            {
                "ego_view_image": np.zeros((4, 4, 3), dtype=np.uint8),
                "state": np.zeros(31, dtype=np.float32),
            },
        ),
        mode=("eval",),
        session_ids=(0,),
    )
    assert chunk.execution_length == 16
    assert chunk.environment_actions.shape == (1, 16, 35)


def test_a_gym_worker_accepts_the_offline_eval_policy():
    """Its decisions carry no behaviour values, so the worker needs no next-observation value."""
    policy = _InferActionChunkPolicy(
        _StubInferPolicy(), instruction="pick up the drill"
    )
    context = type("Context", (), {"environment_indices": (0,)})()
    GymRolloutWorker(context, env=object(), policy=policy)
