# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime-agnostic policy server: batching, sessions, evaluation isolation, clients."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import PolicyContext, PolicyState, WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.policy_adapters import as_rollout_policy
from isaac_rl_posttraining.runtimes.serving import (
    Batcher,
    BuiltinEngine,
    InProcessPolicyService,
    PolicyRequest,
    PolicyServer,
    ServedPolicy,
    ServiceHandle,
    connect_policy,
)
from isaac_rl_posttraining.runtimes.serving.batcher import QueuedRequest
from isaac_rl_posttraining.runtimes.serving.clients import InProcessTransport
from isaac_rl_posttraining.runtimes.serving.queues import (
    AsyncRequestQueue,
    ThreadRequestQueue,
)


def _client(queue: ThreadRequestQueue) -> ServedPolicy:
    return ServedPolicy(InProcessTransport(queue))


def _chunk(value: float, version: int = 0) -> ActionChunk:
    return ActionChunk.from_arrays(
        learner_action=np.full((1, 1), value, dtype=np.float32),
        environment_actions=np.full((1, 1, 1), value, dtype=np.float32),
        execution_length=1,
        learner_action_kind="latent",
        policy_version=version,
    )


class _Policy(RolloutPolicy):
    """Records every batch it answers; acts with ``session_id + 0.5``."""

    features = PolicyFeatures(encodes_observations=True, values_observations=True)

    def __init__(self, *, mixed_modes: bool = True, fail: bool = False) -> None:
        self.calls: list[tuple[str, tuple[int, ...], tuple[str | None, ...]]] = []
        self.resets: list[tuple[int, int | None]] = []
        self.version = 0
        self.fail = fail
        self.closed = False
        if not mixed_modes:
            self.features = PolicyFeatures(
                encodes_observations=True,
                values_observations=True,
                mixed_mode_batches=False,
            )

    def action_chunks(self, observations, *, mode, session_ids):
        if self.fail:
            raise RuntimeError("policy failed")
        self.calls.append(("act", tuple(session_ids), tuple(mode)))
        return tuple(_chunk(s + 0.5, self.version) for s in session_ids)

    def encode_observations(self, observations, *, session_ids):
        self.calls.append(("encode", tuple(session_ids), (None,) * len(session_ids)))
        return tuple(
            {"state": np.asarray(o["state"], np.float32)} for o in observations
        )

    def encode_and_value_next_observations(self, observations, *, session_ids):
        self.calls.append(("value", tuple(session_ids), (None,) * len(session_ids)))
        return tuple(
            (
                {"state": np.asarray(o["state"], np.float32)},
                np.asarray([2.0 * s], np.float32),
            )
            for o, s in zip(observations, session_ids, strict=True)
        )

    def reset_episode(self, session_id=0, seed=None):
        self.resets.append((session_id, seed))

    def set_policy_state(self, state: PolicyState) -> int:
        if state.version <= self.version:
            raise ValueError("version must increase")
        self.version = state.version
        return state.version

    def close(self):
        self.closed = True


def _server(policy: _Policy, batcher: Batcher) -> PolicyServer:
    engine = BuiltinEngine()
    engine.load(
        lambda context: policy,
        PolicyContext(
            policy_index=0, num_policies=1, device="cpu", placement="policy_server"
        ),
    )
    return PolicyServer(engine, batcher, collect_metrics=True)


def _queued(kind, session_id, mode, group, request_id=0) -> QueuedRequest:
    return QueuedRequest(
        PolicyRequest(
            kind, session_id, {"state": [float(session_id)]}, mode, request_id
        ),
        group,
        0.0,
    )


def test_server_answers_kinds_in_order_and_keeps_request_order() -> None:
    policy = _Policy()
    server = _server(policy, Batcher())
    batch = [
        _queued("value", 3, None, 0),
        _queued("act", 1, "train", 1),
        _queued("encode", 2, None, 2),
        _queued("act", 4, "warmup", 3),
    ]
    responses = server.serve(batch)
    # One mixed-mode act batch first, then encodings, then values.
    assert [call[:2] for call in policy.calls] == [
        ("act", (1, 4)),
        ("encode", (2,)),
        ("value", (3,)),
    ]
    assert [response.kind for response in responses] == [
        "value",
        "act",
        "encode",
        "act",
    ]
    assert responses[0].value == pytest.approx(6.0)
    assert responses[1].action_chunk.learner_action[0, 0] == pytest.approx(1.5)


def test_evaluation_requests_batch_only_within_one_client_call() -> None:
    policy = _Policy()
    server = _server(policy, Batcher(serve_eval_unbatched=True))
    server.serve(
        [
            _queued("act", 0, "eval", group=0),
            _queued("act", 1, "eval", group=0),
            _queued("act", 2, "eval", group=1),
            _queued("act", 3, "train", group=2),
            _queued("act", 4, "train", group=3),
        ]
    )
    # A batched worker's call stays one batch; another worker's evaluation is
    # answered alone; training requests of different calls batch together.
    assert [call[1] for call in policy.calls] == [(0, 1), (2,), (3, 4)]
    policy.calls.clear()
    # The sessions stay evaluation sessions for their value requests.
    server.serve(
        [
            _queued("value", 0, None, group=4),
            _queued("value", 2, None, group=5),
            _queued("value", 3, None, group=6),
            _queued("value", 4, None, group=7),
        ]
    )
    assert [call[1] for call in policy.calls] == [(0,), (2,), (3, 4)]


def test_a_policy_without_mixed_mode_batches_gets_one_mode_per_call() -> None:
    policy = _Policy(mixed_modes=False)
    server = _server(policy, Batcher())
    server.serve(
        [
            _queued("act", 0, "train", 0),
            _queued("act", 1, "eval", 1),
            _queued("act", 2, "train", 2),
        ]
    )
    assert [call[1:] for call in policy.calls] == [
        ((0, 2), ("train", "train")),
        ((1,), ("eval",)),
    ]


def test_a_failed_batch_answers_every_request_with_the_error() -> None:
    server = _server(_Policy(fail=True), Batcher())
    answers = server.serve(
        [_queued("act", 0, "train", 0), _queued("act", 1, "train", 1)]
    )
    assert all(isinstance(answer, RuntimeError) for answer in answers)
    queue = ThreadRequestQueue(server)
    with pytest.raises(RuntimeError, match="policy failed"):
        _client(queue).action_chunks(
            ({"state": [0.0]},), mode=("train",), session_ids=(0,)
        )


def test_batcher_validates_and_takes_at_most_max_batch() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        Batcher(max_batch=0)
    with pytest.raises(ValueError, match="max_wait_ms"):
        Batcher(max_wait_ms=-1.0)
    pending = [_queued("act", index, "train", index) for index in range(5)]
    assert [item.request.session_id for item in Batcher(max_batch=2).take(pending)] == [
        0,
        1,
    ]
    assert len(pending) == 3


def test_thread_queue_batches_concurrent_callers() -> None:
    policy = _Policy()
    queue = ThreadRequestQueue(_server(policy, Batcher(max_wait_ms=200.0, max_batch=4)))
    clients = [_client(queue) for _ in range(4)]
    barrier = threading.Barrier(4)
    results: dict[int, float] = {}

    def act(session_id: int) -> None:
        barrier.wait()
        (chunk,) = clients[session_id].action_chunks(
            ({"state": [float(session_id)]},),
            mode=("train",),
            session_ids=(session_id,),
        )
        results[session_id] = float(chunk.learner_action[0, 0])

    threads = [threading.Thread(target=act, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert results == {0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5}
    # The four callers were answered in one batch (they arrived within the wait).
    assert sorted(len(call[1]) for call in policy.calls) == [4]


def _act(session_id: int) -> list[PolicyRequest]:
    return [PolicyRequest("act", session_id, {"state": [0.0]}, "train", session_id)]


def test_a_lone_client_is_answered_without_waiting_out_max_wait() -> None:
    queue = ThreadRequestQueue(_server(_Policy(), Batcher(max_wait_ms=5_000.0)))
    client = _client(queue)
    started = time.monotonic()
    for _ in range(3):
        client.action_chunks(({"state": [0.0]},), mode=("train",), session_ids=(0,))
    assert time.monotonic() - started < 2.0
    # A caller that never connected may have company coming, so it still waits.
    other = ThreadRequestQueue(_server(_Policy(), Batcher(max_wait_ms=300.0)))
    started = time.monotonic()
    other.submit(_act(0))
    assert time.monotonic() - started >= 0.3


def test_async_queue_waits_only_for_connected_clients_without_a_request() -> None:
    async def scenario() -> tuple[float, list[tuple[int, ...]]]:
        policy = _Policy()
        queue = AsyncRequestQueue(_server(policy, Batcher(max_wait_ms=5_000.0)))
        first, second = queue.connect(), queue.connect()
        started = time.monotonic()
        # Both clients' requests arrive: the batch is ready long before 5 s.
        await asyncio.gather(
            queue.submit(_act(0), first), queue.submit(_act(1), second)
        )
        queue.disconnect(second)
        await queue.submit(_act(0), first)  # the only client left
        elapsed = time.monotonic() - started
        await queue.close()
        return elapsed, [call[1] for call in policy.calls]

    elapsed, batches = asyncio.run(scenario())
    assert elapsed < 2.0
    assert batches == [(0, 1), (0,)]


def test_client_surface_matches_the_local_policy() -> None:
    local = _Policy()
    served_policy = _Policy()
    client = _client(ThreadRequestQueue(_server(served_policy, Batcher())))
    observations = ({"state": [1.0]}, {"state": [2.0]})
    assert client.features == local.features
    for name in ("encode_observations", "encode_and_value_next_observations"):
        expected = getattr(local, name)(observations, session_ids=(1, 2))
        actual = getattr(client, name)(observations, session_ids=(1, 2))
        assert repr(actual) == repr(tuple(expected))
    fused = client.act_and_value(
        observations, mode=("train", "train"), session_ids=(1, 2)
    )
    assert [float(value[0]) for _, _, value in fused] == [2.0, 4.0]
    # The fused request values first and acts second, as two requests would.
    assert [call[0] for call in served_policy.calls[-2:]] == ["value", "act"]
    client.reset_episodes((1, 2), (7, None))
    assert served_policy.resets == [(1, 7), (2, None)]
    assert client.set_policy_state(PolicyState(version=1, tensors={})) == 1
    with pytest.raises(ValueError, match="must increase"):
        client.set_policy_state(PolicyState(version=1, tensors={}))


def test_in_process_service_publishes_each_version_once_and_closes_its_policy() -> None:
    policy = _Policy()
    service = InProcessPolicyService(
        lambda context: policy, device="cpu", batcher=Batcher()
    )
    handle = service.handle()
    assert isinstance(handle, ServiceHandle)
    client = connect_policy(handle, route_index=5)
    ref = WeightsRef.of(PolicyState(version=3, tensors={}))
    assert service.set_weights(ref) == (3,)
    # A second group publishing the same version does not reload it.
    assert service.set_weights(ref) == (3,)
    (chunk,) = client.action_chunks(
        ({"state": [5.0]},), mode=("eval",), session_ids=(5,)
    )
    assert chunk.policy_version == 3
    service.close()
    assert policy.closed
    with pytest.raises(LookupError, match="no in-process policy service"):
        connect_policy(handle, route_index=0)


def test_legacy_policies_are_adapted_once_with_their_optional_methods() -> None:
    class _Legacy:
        def __init__(self):
            self.resets = []

        def action_chunk(self, observation, *, mode):
            return _chunk(1.0)

        def prepare_observation(self, observation):
            return {"state": np.zeros(1, np.float32)}

        def reset_episode(self, *, seed=None):
            self.resets.append(seed)

        def set_policy_state(self, state):
            return state.version

    legacy = _Legacy()
    adapted = as_rollout_policy(legacy)
    assert adapted.features.encodes_observations
    assert not adapted.features.values_observations
    assert not adapted.features.multi_session
    assert adapted.features.behavior_values is None
    adapted.reset_episode(0, seed=4)
    assert legacy.resets == [4]
    with pytest.raises(TypeError, match="does not value observations"):
        adapted.encode_and_value_next_observations(({},), session_ids=(0,))
    with pytest.raises(TypeError, match="several sessions"):
        BuiltinEngine().load(
            lambda context: _Legacy(),
            PolicyContext(
                policy_index=0, num_policies=1, device="cpu", placement="policy_server"
            ),
        )


def test_a_policy_declaring_a_capability_must_implement_it() -> None:
    class _Undeclared(RolloutPolicy):
        features = PolicyFeatures(values_observations=True)

        def action_chunks(self, observations, *, mode, session_ids):
            return ()

        def set_policy_state(self, state):
            return state.version

    with pytest.raises(TypeError, match="encode_and_value_next_observations"):
        as_rollout_policy(_Undeclared())
