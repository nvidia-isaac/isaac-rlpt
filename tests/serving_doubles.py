# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small served policy and the statistics every policy-service replica reports."""

from __future__ import annotations

import numpy as np

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import PolicyContext, PolicyState
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy

REPLICA_STATS = {
    "policy_server_index",
    "initialization_seconds",
    "batch_sizes",
    "inference_seconds",
    "request_latencies_seconds",
    "operation_seconds",
    "operation_batch_sizes",
    "policy_metadata",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
}


class ScaledPolicy(RolloutPolicy):
    """Acts with ``scale * state``; the weights set ``scale``."""

    features = PolicyFeatures(multi_session=True)

    def __init__(self, context: PolicyContext) -> None:
        self.index = context.policy_index
        self.scale, self.version = 1.0, 0

    def action_chunks(self, observations, *, mode, session_ids):
        return tuple(
            ActionChunk.from_arrays(
                learner_action=self.scale * np.asarray(o["state"], np.float32)[None],
                environment_actions=self.scale
                * np.asarray(o["state"], np.float32)[None, None],
                execution_length=1,
                learner_action_kind="chunk",
                policy_version=self.version,
            )
            for o in observations
        )

    def set_policy_state(self, state: PolicyState) -> int:
        self.scale = float(state.tensors["scale"][0])
        self.version = state.version
        return state.version

    def benchmark_metadata(self):
        return {"replica": self.index}
