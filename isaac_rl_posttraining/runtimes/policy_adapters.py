# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn any acting policy into a checked :class:`RolloutPolicy`.

Policies written against the earlier duck-typed surface (``action_chunk`` or
``action_chunks``, optional ``prepare_observation(s)``,
``prepare_successor(s)``, ``reset_episode`` with or without ``seed``) keep
working: their optional methods are discovered once, here, and never probed
again on the hot path.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

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

__all__ = ["as_rollout_policy", "validate_rollout_policy"]

#: Each declared capability and the method a policy must override to honour it.
_CAPABILITY_METHODS = {
    "encodes_observations": "encode_observations",
    "values_observations": "encode_and_value_next_observations",
}


def _accepts_keyword(method: Callable[..., Any], name: str) -> bool:
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(method).parameters.values()
    )


def validate_rollout_policy(policy: RolloutPolicy) -> RolloutPolicy:
    """Check that a policy implements what its features declare; returns it."""
    if not isinstance(policy, RolloutPolicy):
        raise TypeError(f"expected a RolloutPolicy, got {type(policy).__name__}")
    features = policy.features
    if not isinstance(features, PolicyFeatures):
        raise TypeError(f"{type(policy).__name__}.features must be a PolicyFeatures")
    policy_type = type(policy)
    for method in ("action_chunks", "set_policy_state"):
        if getattr(policy_type, method) is getattr(RolloutPolicy, method):
            raise TypeError(f"{policy_type.__name__} must implement {method}(...)")
    for capability, method in _CAPABILITY_METHODS.items():
        if getattr(features, capability) and getattr(policy_type, method) is getattr(
            RolloutPolicy, method
        ):
            raise TypeError(
                f"{policy_type.__name__} declares {capability} but does not implement {method}(...)"
            )
    return policy


def as_rollout_policy(policy: object) -> RolloutPolicy:
    """``policy`` itself when it is a :class:`RolloutPolicy`, else an adapter of the legacy surface."""
    if isinstance(policy, RolloutPolicy):
        return validate_rollout_policy(policy)
    return validate_rollout_policy(_LegacyPolicy(policy))


class _LegacyPolicy(RolloutPolicy):
    """A duck-typed policy behind the :class:`RolloutPolicy` surface."""

    def __init__(self, policy: Any) -> None:
        batched = getattr(policy, "action_chunks", None)
        single = getattr(policy, "action_chunk", None)
        if not callable(batched) and not callable(single):
            raise TypeError(
                f"{type(policy).__name__} is not an acting policy: it has neither "
                "action_chunks(...) nor action_chunk(...)"
            )
        if not callable(getattr(policy, "set_policy_state", None)):
            raise TypeError(
                f"{type(policy).__name__} must implement set_policy_state(...)"
            )
        self.policy = policy
        self._batched = batched if callable(batched) else None
        self._single = single if callable(single) else None
        self._encode_many = _method(policy, "prepare_observations")
        self._encode_one = _method(policy, "prepare_observation")
        self._value_many = _method(policy, "prepare_successors")
        self._value_one = _method(policy, "prepare_successor")
        self._close = _method(policy, "close")
        self._metadata = _method(policy, "benchmark_metadata")
        reset = _method(policy, "reset_episode")
        self._reset = reset
        self._reset_takes_session = reset is not None and _accepts_keyword(
            reset, "session_id"
        )
        self._reset_takes_seed = reset is not None and _accepts_keyword(reset, "seed")
        self.features = PolicyFeatures(
            encodes_observations=self._encode_many is not None
            or self._encode_one is not None,
            values_observations=self._value_many is not None
            or self._value_one is not None,
            behavior_values=None,
            multi_session=self._batched is not None,
        )

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> Sequence[ActionChunk]:
        if self._batched is not None:
            return self._batched(observations, mode=mode, session_ids=session_ids)
        assert self._single is not None
        return tuple(
            self._single(observation, mode=decision_mode)
            for observation, decision_mode in zip(observations, mode, strict=True)
        )

    def encode_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[EncodedObservation] | None:
        if self._encode_many is not None:
            return self._encode_many(observations, session_ids=session_ids)
        if self._encode_one is not None:
            return tuple(self._encode_one(observation) for observation in observations)
        return None

    def encode_and_value_next_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> Sequence[ValuedObservation]:
        if self._value_many is not None:
            return self._value_many(observations, session_ids=session_ids)
        if self._value_one is not None:
            return tuple(self._value_one(observation) for observation in observations)
        return super().encode_and_value_next_observations(
            observations, session_ids=session_ids
        )

    def reset_episode(self, session_id: Hashable = 0, seed: int | None = None) -> None:
        if self._reset is None:
            return
        arguments: dict[str, Any] = {}
        if self._reset_takes_session:
            arguments["session_id"] = session_id
        if self._reset_takes_seed:
            arguments["seed"] = seed
        self._reset(**arguments)

    def set_policy_state(self, state: PolicyState) -> int:
        return self.policy.set_policy_state(state)

    def close(self) -> None:
        if self._close is not None:
            self._close()

    def benchmark_metadata(self) -> Mapping[str, object]:
        if self._metadata is None:
            return {}
        return dict(self._metadata())


def _method(policy: object, name: str) -> Callable[..., Any] | None:
    method = getattr(policy, name, None)
    return method if callable(method) else None


def first_value(value: np.ndarray | float) -> float:
    """The scalar bootstrap value of one session, whatever its array shape."""
    return float(np.asarray(value).reshape(-1)[0])
