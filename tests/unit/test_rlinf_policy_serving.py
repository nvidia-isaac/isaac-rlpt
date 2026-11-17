# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf policy-server worker behaviour that does not require RLInf or Ray."""

from __future__ import annotations

import asyncio

import numpy as np

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.rlinf.serving import make_policy_worker
from isaac_rl_posttraining.runtimes.serving import Batcher, PolicyRequest


class _WorkerBase:
    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1


def _chunk(session_id) -> ActionChunk:
    return ActionChunk.from_arrays(
        learner_action=np.full((1, 1), float(session_id), dtype=np.float32),
        environment_actions=np.zeros((1, 1, 1), dtype=np.float32),
        execution_length=1,
        learner_action_kind="latent",
    )


class _RecordingPolicy(RolloutPolicy):
    features = PolicyFeatures(encodes_observations=True, values_observations=True)

    def __init__(self) -> None:
        self.action_batches: list[tuple[int, ...]] = []
        self.value_batches: list[tuple[int, ...]] = []
        self.resets: list[tuple[int, int | None]] = []

    def action_chunks(self, observations, *, mode, session_ids):
        del observations, mode
        self.action_batches.append(tuple(session_ids))
        return tuple(_chunk(session_id) for session_id in session_ids)

    def encode_observations(self, observations, *, session_ids):
        return tuple({"state": np.asarray([s], np.float32)} for s in session_ids)

    def encode_and_value_next_observations(self, observations, *, session_ids):
        self.value_batches.append(tuple(session_ids))
        return tuple(
            ({"state": np.asarray([s], np.float32)}, np.asarray([0.5], np.float32))
            for s in session_ids
        )

    def reset_episode(self, session_id=0, seed=None):
        self.resets.append((session_id, seed))

    def set_policy_state(self, state) -> int:
        return state.version


def _requests(kind, sessions, mode):
    return [
        PolicyRequest(kind, session_id, {"state": session_id}, mode, index)
        for index, session_id in enumerate(sessions)
    ]


def _worker(policy, *, deterministic_evaluation: bool):
    batcher = Batcher(
        max_batch=8, max_wait_ms=10.0, serve_eval_unbatched=deterministic_evaluation
    )
    return make_policy_worker(_WorkerBase)(lambda context: policy, "cpu", batcher, True)


def test_deterministic_evaluation_serves_each_evaluation_call_alone() -> None:
    async def exercise() -> None:  # NOSONAR - the queue needs a running event loop
        policy = _RecordingPolicy()
        worker = _worker(policy, deterministic_evaluation=True)

        # Two Gym workers' evaluation calls arrive together; each is answered alone.
        await asyncio.gather(
            worker.serve_requests(_requests("act", (4,), "eval")),
            worker.serve_requests(_requests("act", (7,), "eval")),
        )
        await asyncio.gather(
            worker.serve_requests(_requests("value", (4,), None)),
            worker.serve_requests(_requests("value", (7,), None)),
        )
        # Training calls are batched together.
        await asyncio.gather(
            worker.serve_requests(_requests("act", (4,), "train")),
            worker.serve_requests(_requests("act", (7,), "train")),
        )
        await asyncio.gather(
            worker.serve_requests(_requests("value", (4,), None)),
            worker.serve_requests(_requests("value", (7,), None)),
        )

        assert policy.action_batches == [(4,), (7,), (4, 7)]
        assert policy.value_batches == [(4,), (7,), (4, 7)]
        stats = worker.stats()
        assert stats["batch_sizes"] == (1, 1, 2)
        assert stats["operation_batch_sizes"]["action"] == (1, 1, 2)
        assert stats["operation_batch_sizes"]["successor"] == (1, 1, 2)
        assert stats["operation_seconds"]["action"] > 0.0
        assert len(stats["request_latencies_seconds"]) == 4
        await worker.close()

    asyncio.run(exercise())


def test_dynamic_evaluation_batching_remains_available() -> None:
    async def exercise() -> None:  # NOSONAR - the queue needs a running event loop
        policy = _RecordingPolicy()
        worker = _worker(policy, deterministic_evaluation=False)
        responses = await asyncio.gather(
            worker.serve_requests(_requests("act", (2,), "eval")),
            worker.serve_requests(_requests("act", (3,), "eval")),
        )
        assert policy.action_batches == [(2, 3)]
        assert [
            response.action_chunk.learner_action[0, 0] for (response,) in responses
        ] == [
            2.0,
            3.0,
        ]
        await worker.close()

    asyncio.run(exercise())


def test_worker_resets_sessions_and_publishes_weights_once_per_version() -> None:
    from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef

    async def exercise() -> None:  # NOSONAR - close() is a coroutine
        policy = _RecordingPolicy()
        worker = _worker(policy, deterministic_evaluation=True)
        worker.reset_sessions([3, 5], [11, None])
        assert policy.resets == [(3, 11), (5, None)]
        state = PolicyState(version=2, tensors={})
        assert await worker.set_weights(WeightsRef.of(state)) == 2
        # A second group publishing the same version is acknowledged without reloading.
        assert await worker.set_weights(WeightsRef.of(state)) == 2
        assert worker.features().values_observations
        await worker.close()

    asyncio.run(exercise())
