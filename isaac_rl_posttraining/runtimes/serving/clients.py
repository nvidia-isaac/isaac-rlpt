# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy clients: a rollout worker's :class:`RolloutPolicy` backed by a policy server.

A client turns each policy call into typed :class:`PolicyRequest`s and sends
them over a transport; the worker cannot tell it from a local policy. Weights
are published to the server by whoever owns it, so a client only mirrors the
version it is told.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from typing import Protocol

import numpy as np

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import DecisionMode, PolicyState
from isaac_rl_posttraining.runtimes.policy import (
    EncodedObservation,
    Observation,
    PolicyFeatures,
    RolloutPolicy,
    ValuedObservation,
)
from isaac_rl_posttraining.runtimes.serving.messages import (
    PolicyRequest,
    PolicyResponse,
    RequestKind,
)
from isaac_rl_posttraining.runtimes.serving.queues import ThreadRequestQueue

__all__ = ["InProcessTransport", "PolicyTransport", "ServedPolicy"]


class PolicyTransport(Protocol):
    """Moves requests to one policy server and its answers back."""

    def submit(self, requests: Sequence[PolicyRequest]) -> list[PolicyResponse]: ...

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None: ...

    def features(self) -> PolicyFeatures: ...

    def close(self) -> None:
        """Disconnect from the server."""
        ...


class ServedPolicy(RolloutPolicy):
    """The :class:`RolloutPolicy` surface over a transport to one policy server."""

    def __init__(self, transport: PolicyTransport) -> None:
        self._transport = transport
        self.features = transport.features()
        self._request_ids = itertools.count()
        self._policy_version = 0

    def _request(
        self,
        kind: RequestKind,
        observations: Sequence[Observation],
        session_ids: Sequence[Hashable],
        modes: Sequence[DecisionMode | None],
    ) -> list[PolicyResponse]:
        if not (len(observations) == len(session_ids) == len(modes)):
            raise ValueError(
                "observations, session_ids and modes must have equal lengths"
            )
        requests = [
            PolicyRequest(kind, session_id, observation, mode, next(self._request_ids))
            for observation, session_id, mode in zip(
                observations, session_ids, modes, strict=True
            )
        ]
        if not requests:
            return []
        responses = self._transport.submit(requests)
        if [response.request_id for response in responses] != [
            request.request_id for request in requests
        ]:
            raise RuntimeError("policy server answered requests out of order")
        return responses

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> tuple[ActionChunk, ...]:
        responses = self._request("act", observations, session_ids, mode)
        return tuple(_chunk(response) for response in responses)

    def encode_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> tuple[EncodedObservation, ...] | None:
        if not self.features.encodes_observations:
            return None
        responses = self._request(
            "encode", observations, session_ids, (None,) * len(observations)
        )
        return tuple(response.observation for response in responses)

    def encode_and_value_next_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> tuple[ValuedObservation, ...]:
        responses = self._request(
            "value", observations, session_ids, (None,) * len(observations)
        )
        return tuple((response.observation, _value(response)) for response in responses)

    def act_and_value(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> tuple[tuple[ActionChunk, EncodedObservation, np.ndarray], ...]:
        responses = self._request("act_and_value", observations, session_ids, mode)
        return tuple(
            (_chunk(response), response.observation, _value(response))
            for response in responses
        )

    def reset_episode(self, session_id: Hashable = 0, seed: int | None = None) -> None:
        self._transport.reset_sessions((session_id,), (seed,))

    def reset_episodes(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        self._transport.reset_sessions(tuple(session_ids), tuple(seeds))

    def set_policy_state(self, state: PolicyState) -> int:
        # The service's owner publishes the weights once; workers mirror the version.
        if state.version <= self._policy_version:
            raise ValueError(
                f"policy state version must increase beyond {self._policy_version}, "
                f"got {state.version}"
            )
        self._policy_version = state.version
        return self._policy_version

    def close(self) -> None:
        self._transport.close()


def _chunk(response: PolicyResponse) -> ActionChunk:
    if response.action_chunk is None:
        raise RuntimeError(f"a {response.kind!r} response carries no action chunk")
    return response.action_chunk


def _value(response: PolicyResponse) -> np.ndarray:
    if response.value is None:
        raise RuntimeError(f"a {response.kind!r} response carries no value")
    return np.asarray([response.value], dtype=np.float32)


class InProcessTransport:
    """Requests to a policy server in this process."""

    def __init__(self, queue: ThreadRequestQueue) -> None:
        self._queue = queue
        self._client = queue.connect()

    def submit(self, requests: Sequence[PolicyRequest]) -> list[PolicyResponse]:
        return self._queue.submit(requests, self._client)

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        self._queue.reset_sessions(session_ids, seeds)

    def features(self) -> PolicyFeatures:
        return self._queue.server.features

    def close(self) -> None:
        self._queue.disconnect(self._client)
