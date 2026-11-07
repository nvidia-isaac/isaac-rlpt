# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requests a policy server answers and the responses it returns."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from isaac_rl_posttraining.runtimes.base import DECISION_MODES, DecisionMode

if TYPE_CHECKING:
    from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk

__all__ = [
    "BASE_MODEL_KINDS",
    "DECISION_KINDS",
    "REQUEST_KINDS",
    "BasePartInputs",
    "DecodeOptions",
    "PolicyRequest",
    "PolicyResponse",
    "RequestKind",
]

#: ``act``: choose an action chunk. ``encode``: turn an observation into the
#: learner's input (and cache it for the next ``act``). ``value``: encode and
#: return the value of the observation (the bootstrap value after a chunk).
#: ``act_and_value``: both in one round trip.
#: A served base model (the actor stays on the rollout worker) answers
#: ``decode``: decode the actor's latents for these observations;
#: ``reference``: the base model's own action chunk and pooled features (SAC-edit,
#: reference-conditioned RFS); ``features``: the base model's features.
RequestKind = Literal[
    "act", "encode", "value", "act_and_value", "decode", "reference", "features"
]
REQUEST_KINDS: tuple[RequestKind, ...] = (
    "act",
    "encode",
    "value",
    "act_and_value",
    "decode",
    "reference",
    "features",
)
BASE_MODEL_KINDS: tuple[RequestKind, ...] = ("decode", "reference", "features")
#: Kinds that carry the session's decision mode.
DECISION_KINDS: tuple[RequestKind, ...] = ("act", "act_and_value", "decode")
_ACTING_KINDS = ("act", "act_and_value")
_VALUE_KINDS = ("value", "act_and_value")


def _check_kind(kind: object) -> None:
    if kind not in REQUEST_KINDS:
        raise ValueError(f"request kind must be one of {REQUEST_KINDS}, got {kind!r}")


def _check_request_id(request_id: object) -> None:
    if (
        isinstance(request_id, bool)
        or not isinstance(request_id, int)
        or request_id < 0
    ):
        raise ValueError("request_id must be a non-negative integer")


@dataclass(frozen=True)
class DecodeOptions:
    """Settings of one rollout worker's decode call, sent with each of its ``decode`` requests.

    ``action_modulator`` splits RFS's joint actions into latents and residuals
    and adds the residual after the decode; ``batched=False`` decodes session
    by session (reference-conditioned actors); ``max_batch`` caps one decode.
    """

    action_modulator: Any = None
    batched: bool = True
    max_batch: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.batched, bool):
            raise TypeError("batched must be a boolean")
        if self.max_batch is not None and (
            isinstance(self.max_batch, bool)
            or not isinstance(self.max_batch, int)
            or self.max_batch < 1
        ):
            raise ValueError("max_batch must be a positive integer or None")


@dataclass(frozen=True)
class BasePartInputs:
    """What a base-part request carries besides its observation.

    ``decode``: ``action`` is the actor's selected action for the session
    (``None``: the base model draws its own noise), ``random_state`` the
    session's seeded evaluation streams (continued by the server and returned
    in the response), ``options`` the caller's :class:`DecodeOptions`.
    ``reference``: ``image_transform_mode`` (``train`` or ``eval``).
    """

    action: Any = None
    random_state: Mapping[str, Any] | None = None
    options: DecodeOptions = DecodeOptions()
    image_transform_mode: str = "eval"

    def __post_init__(self) -> None:
        if not isinstance(self.options, DecodeOptions):
            raise TypeError("options must be DecodeOptions")
        if self.image_transform_mode not in ("train", "eval"):
            raise ValueError("image_transform_mode must be train or eval")


@dataclass(frozen=True)
class PolicyRequest:
    """One rollout worker's request for one session (environment).

    ``mode`` is required for the kinds that decide (``act``,
    ``act_and_value``, ``decode``) and ignored by the others, which may
    leave it ``None``. ``base`` carries a base-model request's inputs.
    """

    kind: RequestKind
    session_id: Hashable
    observation: Any
    mode: DecisionMode | None = None
    request_id: int = 0
    base: BasePartInputs | None = None

    def __post_init__(self) -> None:
        _check_kind(self.kind)
        _check_request_id(self.request_id)
        if not isinstance(self.session_id, Hashable):
            raise TypeError("session_id must be hashable")
        if self.kind in DECISION_KINDS and self.mode is None:
            raise ValueError(f"a {self.kind!r} request needs a decision mode")
        if self.mode is not None and self.mode not in DECISION_MODES:
            raise ValueError(f"mode must be one of {DECISION_MODES}, got {self.mode!r}")
        if self.base is not None and not isinstance(self.base, BasePartInputs):
            raise TypeError("base must be BasePartInputs")
        if self.kind in ("decode", "reference") and self.base is None:
            raise ValueError(f"a {self.kind!r} request needs its base-model inputs")


@dataclass(frozen=True)
class PolicyResponse:
    """The answer to one :class:`PolicyRequest`.

    ``action_chunk`` is set for the acting kinds, ``value`` for the value
    kinds, ``observation`` (the encoded observation) for ``encode``,
    ``value`` and ``act_and_value``. A base-model kind returns ``result``
    (decoded actions, ``(features, reference)`` or features) and, for a
    seeded ``decode``, the session's continued ``random_state``.
    ``policy_version`` is the weight version that answered.
    """

    request_id: int
    kind: RequestKind
    policy_version: int
    action_chunk: ActionChunk | None = None
    observation: Any = None
    value: float | None = None
    result: Any = None
    random_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_kind(self.kind)
        _check_request_id(self.request_id)
        version = self.policy_version
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        if (self.action_chunk is not None) != (self.kind in _ACTING_KINDS):
            raise ValueError(
                f"a {self.kind!r} response carries an action chunk only when it acts"
            )
        if self.kind in _VALUE_KINDS and self.value is None:
            raise ValueError(f"a {self.kind!r} response needs a value")
        if self.kind in BASE_MODEL_KINDS and self.result is None:
            raise ValueError(f"a {self.kind!r} response needs its result")
