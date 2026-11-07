# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The base part of a steering or edit policy (``placement.parts.base``): its frozen base model, here or served.

A steering policy (DSRL, RFS) or an edit policy (SAC-edit) is a small actor
over a large frozen base model. The actor's step (encoding the observation,
selecting a latent or an edit, warm-up draws, per-session randomness) runs
where the policy runs; the base model's work (decoding the selected latents
into action chunks, its own reference action and features) runs either in
the same process (:class:`LocalBasePart`) or in a policy server shared by
many rollout workers (:class:`ServedBasePart`, built from
:class:`ServedBasePartSpec`).

A served decode draws the same numbers as a local one. A seeded evaluation
session's random streams travel with its ``decode`` request; the server
draws the session's flow noise from them and returns the continued streams,
so the actor's next draw continues where the base model's stopped. Unseeded
draws come from the server process's generators: the same generators as
local in a single-process run, others when the server is another process.
"""

from __future__ import annotations

import inspect
import itertools
import uuid
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import torch

from isaac_rl_posttraining.rollouts.steering_step import SteeringStep
from isaac_rl_posttraining.runtimes.base import DecisionMode, PolicyContext
from isaac_rl_posttraining.runtimes.frozen_base_model import FrozenBaseModelSpec
from isaac_rl_posttraining.runtimes.policy import Observation
from isaac_rl_posttraining.runtimes.policy_batching import (
    batch_observation,
    combine_observations,
)
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.messages import (
    BasePartInputs,
    DecodeOptions,
    PolicyRequest,
    PolicyResponse,
    RequestKind,
)
from isaac_rl_posttraining.runtimes.session_randomness import SessionRandomness

__all__ = [
    "BasePart",
    "BasePartPolicySpec",
    "DecodeOptions",
    "LocalBasePart",
    "ServedBasePart",
    "ServedBasePartSpec",
    "base_part",
    "with_served_base_part",
]


@runtime_checkable
class BasePart(Protocol):
    """What a steering policy asks of its base part, wherever the base model runs."""

    def decode(
        self,
        observations: Sequence[Observation],
        actions: Sequence[np.ndarray] | None,
        *,
        session_ids: Sequence[Hashable],
        modes: Sequence[DecisionMode],
        randomness: SessionRandomness,
        options: DecodeOptions,
    ) -> np.ndarray:
        """Decode one action chunk per session (``actions=None``: the base model's own noise)."""
        ...

    def start_episode(self, session_id: Hashable) -> None:
        """Start a session's episode in a stateful base model."""
        ...


@runtime_checkable
class BasePartPolicySpec(Protocol):
    """A policy spec whose base part may be served on its own (``placement.parts.base``)."""

    base_model: FrozenBaseModelSpec | ServedBasePartSpec

    def with_base_part(self, base_model: ServedBasePartSpec) -> BasePartPolicySpec:
        """The same policy, with its base model served."""
        ...


def _episode_starter(base_model: Any) -> Literal["none", "session", "single"]:
    """How a frozen base model starts an episode: per session, once for its one session, or not at all."""
    start_episode = getattr(base_model, "start_episode", None)
    if not callable(start_episode):
        return "none"
    session_aware = any(
        parameter.name == "session_id"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(start_episode).parameters.values()
    )
    return "session" if session_aware else "single"


class LocalBasePart:
    """The base part in this process: batched decodes, seeded per-session noise, references."""

    def __init__(self, base_model: Any, device: torch.device) -> None:
        self.base_model = base_model
        self.device = torch.device(device)
        self._step = SteeringStep(
            base_model=base_model, device=self.device, steering_key_mapping={}
        )
        self._start_episode = _episode_starter(base_model)

    def decode(
        self,
        observations: Sequence[Observation],
        actions: Sequence[np.ndarray] | None,
        *,
        session_ids: Sequence[Hashable],
        modes: Sequence[DecisionMode],
        randomness: SessionRandomness,
        options: DecodeOptions,
    ) -> np.ndarray:
        """Decode every session's environment actions, batched where that cannot change a result."""
        evaluation_modes = [mode == "eval" for mode in modes]
        selected = [] if actions is None else list(actions)
        prepared = self._seeded_batch_noise(
            selected, session_ids, evaluation_modes, randomness, native=actions is None
        )
        if actions is None:
            seeded = any(randomness.seeded(session_id) for session_id in session_ids)
            batch_allowed = not seeded or prepared is not None
        else:
            batch_allowed = not any(evaluation_modes) or prepared is not None
        can_batch_decode = (
            len(observations) > 1
            and batch_allowed
            and bool(getattr(self.base_model, "supports_batched_infer", False))
            and options.action_modulator is None
            and options.batched
        )
        if can_batch_decode:
            combined = None if actions is None else np.concatenate(selected, axis=0)
            decoded = self._decode_batches(
                observations, prepared, combined, options.max_batch
            )
        else:
            # Without latents every session decodes with the base model's own noise.
            per_session: list[np.ndarray | None] = (
                list(selected) if selected else [None] * len(observations)
            )
            decoded = np.concatenate(
                [
                    self._decode_for_session(
                        observation,
                        action,
                        session_id=session_id,
                        evaluation=evaluation,
                        randomness=randomness,
                        modulator=options.action_modulator,
                    )
                    for observation, action, evaluation, session_id in zip(
                        observations,
                        per_session,
                        evaluation_modes,
                        session_ids,
                        strict=True,
                    )
                ],
                axis=0,
            )
        if decoded.shape[0] != len(observations):
            raise RuntimeError(
                "frozen decoder returned the wrong batch size: "
                f"{decoded.shape[0]} != {len(observations)}"
            )
        return decoded

    def _seeded_batch_noise(
        self,
        selected: Sequence[np.ndarray],
        session_ids: Sequence[Hashable],
        evaluation_modes: Sequence[bool],
        randomness: SessionRandomness,
        *,
        native: bool,
    ) -> np.ndarray | None:
        """Per-session seeded decoder noise for one batched evaluation decode, if the base model can."""
        base_model = self.base_model
        # GR00T N1.5 cannot decode prepared noise in one batch; it evaluates per session.
        # Seeded batch noise only serves a batched decode; a single observation
        # decodes through the per-session path and would discard it.
        if not (
            len(session_ids) > 1
            and all(evaluation_modes)
            and getattr(base_model, "supports_seeded_batch_decode", True)
            and callable(getattr(base_model, "infer_prepared_batch", None))
        ):
            return None
        prepared = []
        if native:
            prepare_native_noise = getattr(
                base_model, "prepare_seeded_native_noise", None
            )
            if not callable(prepare_native_noise):
                return None
            for session_id in session_ids:
                with randomness.stream(session_id, enabled=True):
                    prepared.append(prepare_native_noise())
        else:
            prepare_noise = getattr(base_model, "prepare_seeded_batch_noise", None)
            if not callable(prepare_noise):
                return None
            for action, session_id in zip(selected, session_ids, strict=True):
                with randomness.stream(session_id, enabled=True):
                    prepared.append(prepare_noise(action))
        return np.concatenate(prepared, axis=0)

    def _decode_batches(
        self,
        observations: Sequence[Observation],
        prepared: np.ndarray | None,
        combined: np.ndarray | None,
        max_batch: int | None,
    ) -> np.ndarray:
        """Decode the sessions in batches of at most ``max_batch``."""
        decoded_batches = []
        batch_size = max_batch or len(observations)
        for start in range(0, len(observations), batch_size):
            stop = min(start + batch_size, len(observations))
            combined_observation = combine_observations(
                [
                    batch_observation(observation)
                    for observation in observations[start:stop]
                ]
            )
            if prepared is not None:
                decoded_batch = self._step.decode_prepared_batch(
                    combined_observation, prepared[start:stop]
                )
            elif combined is None:
                decoded_batch = self._step.decode_batch(combined_observation, None)
            else:
                decoded_batch = self._step.decode_batch(
                    combined_observation, combined[start:stop]
                )
            decoded_batches.append(decoded_batch)
        return np.concatenate(decoded_batches, axis=0)

    def _decode_for_session(
        self,
        observation: Observation,
        action: np.ndarray | None,
        *,
        session_id: Hashable,
        evaluation: bool,
        randomness: SessionRandomness,
        modulator: Any,
    ) -> np.ndarray:
        latents, residuals = action, None
        if action is not None and modulator is not None:
            # RFS: the joint action is flow noise plus a residual added after the decode.
            latents, residuals = modulator.split(action)
        with randomness.stream(session_id, enabled=evaluation):
            decoded = self._step.decode(batch_observation(observation), latents)
        if residuals is None:
            return decoded
        return modulator.apply(decoded, residuals)

    def infer_reference(
        self, observations: dict[str, Any], *, image_transform_mode: str = "eval"
    ) -> tuple[torch.Tensor, np.ndarray]:
        infer_reference = getattr(self.base_model, "infer_reference", None)
        if not callable(infer_reference):
            raise TypeError(
                "reference-conditioned policies require a base model with infer_reference()"
            )
        with torch.inference_mode():
            return infer_reference(
                observations, image_transform_mode=image_transform_mode
            )

    def get_features(self, observations: dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode():
            return self.base_model.get_features(observations)

    def start_episode(self, session_id: Hashable) -> None:
        if self._start_episode == "session":
            self.base_model.start_episode(session_id=session_id)
        elif self._start_episode == "single":
            if session_id != 0:
                raise RuntimeError(
                    "a stateful frozen base model must accept start_episode(session_id=...) "
                    "before it can be shared across rollout sessions"
                )
            self.base_model.start_episode()

    def check_servable(self) -> None:
        """Fail unless several rollout workers may share this base model."""
        if not getattr(self.base_model, "servable", True):
            raise TypeError(
                f"{type(self.base_model).__name__} declares it cannot be served (servable=False); keep "
                "the base model local (placement.policy.mode=local, placement.parts.base.mode=local)"
            )
        if self._start_episode == "single":
            raise TypeError(
                f"{type(self.base_model).__name__}.start_episode() keeps one episode's state without a "
                "session id, so several rollout workers cannot share it; keep the base model "
                "local (placement.parts.base.mode=local)"
            )

    def answer(
        self,
        kind: RequestKind,
        observations: Sequence[Observation],
        sessions: Sequence[Hashable],
        modes: Sequence[DecisionMode | None],
        inputs: Sequence[BasePartInputs | None],
    ) -> list[tuple[Any, Mapping[str, Any] | None]]:
        """A policy server's answer to same-kind base-model requests: ``(result, random_state)`` each."""
        if kind == "reference":
            answers: list[tuple[Any, Mapping[str, Any] | None]] = []
            for observation, request in zip(observations, inputs, strict=True):
                mode = "eval" if request is None else request.image_transform_mode
                features, reference = self.infer_reference(
                    cast(dict[str, Any], observation), image_transform_mode=mode
                )
                answers.append(((_to_cpu(features), np.asarray(reference)), None))
            return answers
        if kind == "features":
            return [
                (_to_cpu(self.get_features(cast(dict[str, Any], observation))), None)
                for observation in observations
            ]
        if kind == "decode":
            return self._answer_decode(observations, sessions, modes, inputs)
        raise ValueError(f"a served base model does not answer {kind!r} requests")

    def _answer_decode(
        self,
        observations: Sequence[Observation],
        sessions: Sequence[Hashable],
        modes: Sequence[DecisionMode | None],
        inputs: Sequence[BasePartInputs | None],
    ) -> list[tuple[Any, Mapping[str, Any] | None]]:
        """Decode same-option requests together, in the order they arrived."""
        groups: list[tuple[tuple[DecodeOptions, bool], list[int]]] = []
        for index, request in enumerate(inputs):
            if request is None:
                raise ValueError("a decode request needs its base-model inputs")
            key = (request.options, request.action is None)
            for existing, members in groups:
                if existing == key:
                    members.append(index)
                    break
            else:
                groups.append((key, [index]))
        answers: list[tuple[Any, Mapping[str, Any] | None]] = [(None, None)] * len(
            observations
        )
        for (options, native), members in groups:
            randomness = SessionRandomness(self.device)
            for index in members:
                request = inputs[index]
                assert request is not None
                randomness.load(sessions[index], _state(request.random_state))
            decoded = self.decode(
                [observations[index] for index in members],
                None
                if native
                else [np.asarray(inputs[index].action) for index in members],  # type: ignore[union-attr]
                session_ids=[sessions[index] for index in members],
                modes=[modes[index] or "train" for index in members],
                randomness=randomness,
                options=options,
            )
            for position, index in enumerate(members):
                answers[index] = (
                    decoded[position : position + 1],
                    randomness.export(sessions[index]),
                )
        return answers


def _state(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if state is None else dict(state)


def _to_cpu(value: Any) -> Any:
    """A tensor on the CPU, so the answer can leave a GPU process."""
    return value.detach().cpu() if torch.is_tensor(value) else value


class _Transport(Protocol):
    def submit(self, requests: Sequence[PolicyRequest]) -> list[PolicyResponse]: ...

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None: ...


class ServedBasePart:
    """The base part in a policy server, as the actor's side of a steering or edit policy sees it.

    Sessions are namespaced by this client, so many rollout workers share
    one server without their session ids meeting.
    """

    def __init__(self, transport: _Transport) -> None:
        self._transport = transport
        self._client = uuid.uuid4().hex[:12]
        self._request_ids = itertools.count()

    def _session(self, session_id: Hashable) -> tuple[str, Hashable]:
        return (self._client, session_id)

    def _submit(self, requests: list[PolicyRequest]) -> list[PolicyResponse]:
        responses = self._transport.submit(requests)
        if [response.request_id for response in responses] != [
            request.request_id for request in requests
        ]:
            raise RuntimeError("the base-model server answered requests out of order")
        return responses

    def decode(
        self,
        observations: Sequence[Observation],
        actions: Sequence[np.ndarray] | None,
        *,
        session_ids: Sequence[Hashable],
        modes: Sequence[DecisionMode],
        randomness: SessionRandomness,
        options: DecodeOptions,
    ) -> np.ndarray:
        requests = [
            PolicyRequest(
                "decode",
                self._session(session_id),
                observation,
                mode,
                next(self._request_ids),
                base=BasePartInputs(
                    action=None if actions is None else actions[index],
                    random_state=randomness.export(session_id),
                    options=options,
                ),
            )
            for index, (observation, session_id, mode) in enumerate(
                zip(observations, session_ids, modes, strict=True)
            )
        ]
        if not requests:
            raise ValueError("decode needs at least one observation")
        responses = self._submit(requests)
        for session_id, response in zip(session_ids, responses, strict=True):
            if randomness.seeded(session_id):
                randomness.load(session_id, _state(response.random_state))
        return np.concatenate([np.asarray(r.result) for r in responses], axis=0)

    def _one(
        self, kind: RequestKind, observation: Any, base: BasePartInputs | None
    ) -> Any:
        request = PolicyRequest(
            kind,
            self._session(None),
            observation,
            None,
            next(self._request_ids),
            base=base,
        )
        return self._submit([request])[0].result

    def infer_reference(
        self, observations: dict[str, Any], *, image_transform_mode: str = "eval"
    ) -> tuple[torch.Tensor, np.ndarray]:
        """The base model's reference chunk and pooled features (the base model's ``infer_reference``)."""
        features, reference = self._one(
            "reference",
            observations,
            BasePartInputs(image_transform_mode=image_transform_mode),
        )
        return features, reference

    def get_features(self, observations: dict[str, Any]) -> torch.Tensor:
        return self._one("features", observations, None)

    def start_episode(self, session_id: Hashable) -> None:
        self._transport.reset_sessions((self._session(session_id),), (None,))


@dataclass(frozen=True)
class ServedBasePartSpec:
    """A policy spec's base part, served by a policy service; picklable.

    A rollout worker that builds its policy connects to the replica its
    worker index routes to.
    """

    handle: ServiceHandle

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ServiceHandle):
            raise TypeError("handle must be a ServiceHandle")

    def __call__(self, context: PolicyContext) -> ServedBasePart:
        from isaac_rl_posttraining.runtimes.serving.local import connect_transport

        return ServedBasePart(
            connect_transport(self.handle, route_index=context.policy_index)
        )


def base_part(base_model: Any, device: torch.device) -> LocalBasePart | ServedBasePart:
    """The base part of a policy: a served one as is, a base model wrapped to run here."""
    if isinstance(base_model, ServedBasePart):
        return base_model
    return LocalBasePart(base_model, device)


def with_served_base_part(
    spec: Any, open_base_service: Callable[[FrozenBaseModelSpec], Any]
) -> tuple[Any, Any]:
    """A rollout worker spec whose policy calls a served base model, and that service.

    ``open_base_service`` returns the service for a base model (a runtime keeps
    one per base model, so a run's training and evaluation groups share it).
    A spec whose base model is already served is returned as is, with ``None``.
    """
    policy = getattr(spec, "policy", None)
    if isinstance(policy, ServiceHandle):
        raise TypeError(
            "the rollout workers' policy is already served whole; serve either the whole "
            "policy (placement.policy.mode=served) or its base model alone"
        )
    if not isinstance(policy, BasePartPolicySpec):
        raise TypeError(
            "placement.parts.base serves the frozen base model of a steering or edit policy; "
            f"{type(policy).__name__} has none (serve the whole policy instead)"
        )
    if isinstance(policy.base_model, ServedBasePartSpec):
        return spec, None
    service = open_base_service(policy.base_model)
    served = policy.with_base_part(ServedBasePartSpec(service.handle()))
    return replace(spec, policy=served), service
