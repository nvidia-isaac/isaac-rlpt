# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving engines: what runs the model inside a policy server.

The ``builtin`` engine runs today's acting policies (steering, edit,
flow-SDE) in the server's process, so a served decision is computed by the
same code, in the same order, as a local one.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaac_rl_posttraining.runtimes.base import DecisionMode, PolicyContext, WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.policy_adapters import as_rollout_policy
from isaac_rl_posttraining.runtimes.serving.messages import (
    BASE_MODEL_KINDS,
    BasePartInputs,
    RequestKind,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.base_part import LocalBasePart

__all__ = ["SERVING_ENGINES", "BuiltinEngine", "make_engine"]


class BuiltinEngine:
    """Run an acting policy built from a policy spec, one same-kind batch at a time.

    Loaded with a :class:`~isaac_rl_posttraining.runtimes.frozen_base_model.FrozenBaseModelSpec`
    it serves the base model alone (``decode``, ``reference`` and ``features``
    requests) to rollout workers that run the actor themselves; a frozen base
    model takes no weights.
    """

    name = "builtin"
    #: GPUs one replica can use: the model is loaded on a single device.
    max_gpus_per_replica = 1

    def __init__(self) -> None:
        self._policy: RolloutPolicy | None = None
        self._base: LocalBasePart | None = None
        self._version = 0

    @property
    def policy(self) -> RolloutPolicy:
        if self._policy is None:
            raise RuntimeError("the serving engine has no policy loaded")
        return self._policy

    @property
    def serves_base_part(self) -> bool:
        """Whether this engine serves a base model alone (its actor runs on the rollout workers)."""
        return self._base is not None

    @property
    def features(self) -> PolicyFeatures:
        if self._base is not None:
            return PolicyFeatures()
        return self.policy.features

    @property
    def version(self) -> int:
        return self._version

    def load(
        self, spec: Callable[[PolicyContext], Any], context: PolicyContext
    ) -> None:
        if self._policy is not None or self._base is not None:
            raise RuntimeError("the serving engine already holds a policy")
        from isaac_rl_posttraining.runtimes.base_part import LocalBasePart
        from isaac_rl_posttraining.runtimes.frozen_base_model import FrozenBaseModelSpec

        if isinstance(spec, FrozenBaseModelSpec):
            base = LocalBasePart(spec(context), torch.device(context.device))
            base.check_servable()
            self._base = base
            return
        policy = as_rollout_policy(spec(context))
        if not policy.features.multi_session:
            policy.close()
            raise TypeError(
                "a policy server needs a policy that decides for several sessions "
                "at once (action_chunks(...))"
            )
        self._policy = policy

    def run(
        self,
        kind: RequestKind,
        observations: Sequence[Any],
        sessions: Sequence[Hashable],
        modes: Sequence[DecisionMode | None],
        inputs: Sequence[BasePartInputs | None] | None = None,
    ) -> list[Any]:
        if kind in BASE_MODEL_KINDS:
            if self._base is None:
                raise TypeError(
                    f"{kind!r} requests need a server of the base model alone "
                    "(placement.parts.base.mode=served)"
                )
            return self._base.answer(
                kind,
                observations,
                sessions,
                modes,
                [None] * len(observations) if inputs is None else inputs,
            )
        if self._base is not None:
            raise TypeError(
                f"a base-model server answers decode, reference and features requests, not {kind!r}"
            )
        policy = self.policy
        if kind == "act":
            return list(
                policy.action_chunks(
                    observations, mode=_decision_modes(modes), session_ids=sessions
                )
            )
        if kind == "encode":
            encoded = policy.encode_observations(observations, session_ids=sessions)
            if encoded is None:
                raise TypeError(f"{type(policy).__name__} does not encode observations")
            return list(encoded)
        if kind == "value":
            return list(
                policy.encode_and_value_next_observations(
                    observations, session_ids=sessions
                )
            )
        if kind == "act_and_value":
            return list(
                policy.act_and_value(
                    observations, mode=_decision_modes(modes), session_ids=sessions
                )
            )
        raise ValueError(f"unknown request kind {kind!r}")

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        if self._base is not None:
            # The workers keep the random streams; only a stateful base model starts episodes.
            for session_id in session_ids:
                self._base.start_episode(session_id)
            return
        self.policy.reset_episodes(session_ids, seeds)

    def load_weights(self, ref: WeightsRef) -> int:
        """Apply published weights; a version already served is acknowledged without reloading.

        Several rollout groups (training and evaluation) may share one service
        and each publishes the same version; the server loads it once.
        """
        if ref.version == self._version:
            return self._version
        if self._base is not None:
            # A frozen base model takes no weights; it acknowledges the version.
            self._version = ref.version
            return self._version
        applied = int(self.policy.set_policy_state(ref.resolve()))
        if applied != ref.version:
            raise RuntimeError(
                f"served policy acknowledged version {applied}, expected {ref.version}"
            )
        self._version = applied
        return applied

    def metadata(self) -> Mapping[str, object]:
        if self._base is not None:
            return {"base_model": type(self._base.base_model).__name__}
        return self.policy.benchmark_metadata()

    def close(self) -> None:
        self._base = None
        if self._policy is not None:
            policy, self._policy = self._policy, None
            policy.close()


def _decision_modes(modes: Sequence[DecisionMode | None]) -> tuple[DecisionMode, ...]:
    decision_modes: list[DecisionMode] = []
    for mode in modes:
        if mode is None:
            raise ValueError("acting requests need a decision mode")
        decision_modes.append(mode)
    return tuple(decision_modes)


#: Serving engines by the name a placement selects them with.
SERVING_ENGINES: dict[str, type[BuiltinEngine]] = {"builtin": BuiltinEngine}


def make_engine(name: str) -> BuiltinEngine:
    try:
        factory = SERVING_ENGINES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown serving engine {name!r}; available: {sorted(SERVING_ENGINES)}"
        ) from exc
    return factory()
