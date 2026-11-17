# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The acting-policy contract every rollout worker and policy server drives."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import (
    DECISION_MODES,
    DecisionMode,
    PolicyState,
)

__all__ = [
    "DECISION_MODES",
    "ActionChunkPolicy",
    "BatchedActionChunkPolicy",
    "BootstrapValuePolicy",
    "DecisionMode",
    "EncodedObservation",
    "InfoValue",
    "Observation",
    "ObservationValue",
    "PolicyFeatures",
    "RolloutPolicy",
    "SessionPolicy",
    "ValuedObservation",
]

ObservationValue: TypeAlias = np.ndarray | str | int | float | bool
Observation: TypeAlias = np.ndarray | Mapping[str, ObservationValue]
InfoValue: TypeAlias = np.ndarray | dict[str, str] | str | int | float | bool | None
#: An observation in the learner's layout, one array per key without a batch axis.
EncodedObservation: TypeAlias = Mapping[str, np.ndarray]
#: An encoded observation and its value under the behaviour policy, shape ``(1,)``.
ValuedObservation: TypeAlias = tuple[EncodedObservation, np.ndarray]


@dataclass(frozen=True)
class PolicyFeatures:
    """What an acting policy can do, declared once and checked when a worker is built.

    ``encodes_observations``: :meth:`RolloutPolicy.encode_observations` turns an
    observation into the learner's layout (decisions then carry learner
    observations, and replay needs the next observation encoded the same way).
    ``values_observations``: :meth:`RolloutPolicy.encode_and_value_next_observations`
    also returns the bootstrap value of the next observation.
    ``behavior_values``: decisions carry behaviour log-probabilities and values
    (on-policy actors); ``None`` when unknown (an adapted legacy policy).
    ``mixed_mode_batches``: one ``action_chunks`` call may mix decision modes.
    ``multi_session``: one ``action_chunks`` call may serve several sessions.
    """

    encodes_observations: bool = False
    values_observations: bool = False
    behavior_values: bool | None = False
    mixed_mode_batches: bool = True
    multi_session: bool = True

    def __post_init__(self) -> None:
        for name in (
            "encodes_observations",
            "values_observations",
            "mixed_mode_batches",
            "multi_session",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"policy capability {name} must be a boolean")
        if self.behavior_values is not None and not isinstance(
            self.behavior_values, bool
        ):
            raise TypeError(
                "policy capability behavior_values must be a boolean or None"
            )


class RolloutPolicy:
    """Base class of every acting policy: one or many environment sessions, decided in batches.

    A subclass implements :meth:`action_chunks` and :meth:`set_policy_state`
    and sets ``features``; the remaining methods have no-op defaults and
    are overridden when a capability says so. Sessions are environment
    indices; a policy keeps its per-session state (recurrent actor state, the
    encoded observation cached for the next decision, evaluation RNG) under
    them.
    """

    features: PolicyFeatures = PolicyFeatures()

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> Sequence[ActionChunk]:
        """One batch-one chunk per session."""
        raise NotImplementedError

    def set_policy_state(self, state: PolicyState) -> int:
        """Apply published weights; returns the version now acted with."""
        raise NotImplementedError

    def encode_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[EncodedObservation] | None:
        """Encode next observations for replay and cache them for each session's next decision.

        ``None`` when the policy has no learner layout (``encodes_observations`` false).
        """
        del observations, session_ids
        return None

    def encode_and_value_next_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[ValuedObservation]:
        """Encode next observations (as :meth:`encode_observations`) and return their bootstrap values."""
        del observations, session_ids
        raise TypeError(
            f"{type(self).__name__} does not value observations "
            "(features.values_observations is false)"
        )

    def act_and_value(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> Sequence[tuple[ActionChunk, EncodedObservation, np.ndarray]]:
        """Value each observation as a bootstrap value, then act on it: one call for both.

        The same two calls in the same order as a worker that asks for the
        value first and the next chunk second, so the results are identical.
        """
        valued = self.encode_and_value_next_observations(
            observations, session_ids=session_ids
        )
        chunks = self.action_chunks(observations, mode=mode, session_ids=session_ids)
        return tuple(
            (chunk, encoded, value)
            for chunk, (encoded, value) in zip(chunks, valued, strict=True)
        )

    def reset_episode(self, session_id: Hashable = 0, seed: int | None = None) -> None:
        """Start a new episode for one session; ``seed`` seeds its evaluation randomness."""
        del session_id, seed

    def reset_episodes(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        """Start a new episode for several sessions."""
        for session_id, seed in zip(session_ids, seeds, strict=True):
            self.reset_episode(session_id, seed=seed)

    def close(self) -> None:
        """Release resources the policy holds."""

    def benchmark_metadata(self) -> Mapping[str, object]:
        """Facts a benchmark reports next to this policy's timings (dtypes, sizes)."""
        return {}

    # Single-session helpers for a worker that owns session 0.
    def action_chunk(
        self, observation: Observation, *, mode: DecisionMode
    ) -> ActionChunk:
        return self.action_chunks((observation,), mode=(mode,), session_ids=(0,))[0]

    def encode_observation(self, observation: Observation) -> EncodedObservation:
        encoded = self.encode_observations((observation,), session_ids=(0,))
        if encoded is None:
            raise TypeError(f"{type(self).__name__} does not encode observations")
        return encoded[0]

    def encode_and_value_next_observation(
        self, observation: Observation
    ) -> ValuedObservation:
        return self.encode_and_value_next_observations(
            (observation,), session_ids=(0,)
        )[0]

    # Names before the rename, kept for one release.
    def prepare_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[EncodedObservation] | None:
        return self.encode_observations(observations, session_ids=session_ids)

    def prepare_successors(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[ValuedObservation]:
        return self.encode_and_value_next_observations(
            observations, session_ids=session_ids
        )

    def prepare_observation(self, observation: Observation) -> EncodedObservation:
        return self.encode_observation(observation)

    def prepare_successor(self, observation: Observation) -> ValuedObservation:
        return self.encode_and_value_next_observation(observation)


@runtime_checkable
class ActionChunkPolicy(Protocol):
    """The legacy single-session surface; adapted with ``as_rollout_policy``."""

    def action_chunk(
        self, observation: Observation, *, mode: DecisionMode
    ) -> ActionChunk: ...

    def set_policy_state(self, state: PolicyState) -> int: ...


@runtime_checkable
class BatchedActionChunkPolicy(Protocol):
    """The legacy batched surface; adapted with ``as_rollout_policy``."""

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> Sequence[ActionChunk]: ...

    def set_policy_state(self, state: PolicyState) -> int: ...


@runtime_checkable
class BootstrapValuePolicy(Protocol):
    """A batched policy that also values next observations (the legacy ``prepare_successors`` surface)."""

    def prepare_successors(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[ValuedObservation]: ...


#: The name before the rename, kept for one release.


@runtime_checkable
class SessionPolicy(ActionChunkPolicy, BatchedActionChunkPolicy, Protocol):
    """A policy serving one environment session directly and several as one batch."""
