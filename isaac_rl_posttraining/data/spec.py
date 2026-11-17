# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a record looks like, and how two sides of a hand-off compare.

Every array a record carries has a shape and a dtype. A shape entry is a fixed
size, a *named dimension* whose size another component supplies (``"state_dim"``
is fixed by the simulator's proprioception, ``"act_dim"`` by the policy's
checkpoint), or ``None`` for a size nobody fixes (a camera's native height).
The specs here are the types of the records in :mod:`.records`; a producer and
a consumer each state the spec they emit or read, and
:func:`compare_observation_specs` / :func:`compare_action_specs` report every
disagreement, plus what stays unverified until a named dimension is bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ActionSpec",
    "ArraySpec",
    "DimSize",
    "ObservationSpec",
    "SpecComparison",
    "SpecMismatch",
    "compare_action_specs",
    "compare_observation_specs",
]

#: One axis of an array spec: a fixed size, a named dimension, or a free size.
DimSize = int | str | None


class SpecMismatch(ValueError):
    """Two sides of a hand-off disagree on the data that crosses it."""


def _format_shape(shape: tuple[DimSize, ...]) -> str:
    return "[" + ", ".join("?" if size is None else str(size) for size in shape) + "]"


def _check_dim(size: DimSize) -> None:
    if size is None:
        return
    if isinstance(size, bool):
        raise TypeError("a dimension must be an int, a name or None, not a bool")
    if isinstance(size, int):
        if size < 1:
            raise ValueError(f"a fixed dimension must be positive, got {size}")
        return
    if isinstance(size, str):
        if not size:
            raise ValueError("a named dimension must not be empty")
        return
    raise TypeError(f"a dimension must be an int, a name or None, got {type(size).__name__}")


def _resolve(size: DimSize, bindings: Mapping[str, int]) -> DimSize:
    if isinstance(size, str) and size in bindings:
        return int(bindings[size])
    return size


def _dtype_name(dtype: Any) -> str:
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"unknown dtype {dtype!r}") from exc
    if resolved.hasobject:
        raise TypeError("records carry numeric or boolean arrays, not object dtype")
    return resolved.name


@dataclass(frozen=True)
class ArraySpec:
    """The shape and dtype of one array component."""

    shape: tuple[DimSize, ...]
    dtype: str

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        for size in shape:
            _check_dim(size)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _dtype_name(self.dtype))

    @classmethod
    def of(cls, value: Any) -> ArraySpec:
        """The spec a real array satisfies, every axis fixed."""
        array = np.asarray(value)
        return cls(tuple(int(size) for size in array.shape), array.dtype.name)

    @classmethod
    def from_space(cls, space: Any) -> ArraySpec:
        """The spec a Gymnasium ``Box`` (anything with ``shape`` and ``dtype``) declares."""
        return cls(tuple(int(size) for size in space.shape), np.dtype(space.dtype).name)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def names(self) -> tuple[str, ...]:
        """The named dimensions this spec leaves to be bound."""
        return tuple(size for size in self.shape if isinstance(size, str))

    @property
    def is_image(self) -> bool:
        """Height, width and channels of unsigned bytes: a camera frame."""
        return self.dtype == "uint8" and self.rank == 3

    @property
    def width(self) -> int | None:
        """The fixed size of the last axis, or ``None`` when it is named or free."""
        if not self.shape:
            return None
        last = self.shape[-1]
        return last if isinstance(last, int) else None

    def bind(self, bindings: Mapping[str, int]) -> ArraySpec:
        """This spec with every bound named dimension replaced by its size."""
        return ArraySpec(tuple(_resolve(size, bindings) for size in self.shape), self.dtype)

    def to_json(self) -> dict[str, Any]:
        """The JSON form: ``{"shape": [...], "dtype": ...}`` with names and free sizes kept as such."""
        return {"shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ArraySpec:
        return cls(tuple(data["shape"]), data["dtype"])

    def mismatch(self, value: Any, *, bindings: Mapping[str, int] | None = None) -> str | None:
        """Why ``value`` does not satisfy this spec, or ``None`` when it does.

        A named dimension that is not bound accepts any size, but the same
        name must take the same size everywhere it appears in the array.
        """
        return self.measure(value, dict(bindings or {}))

    def measure(self, value: Any, bindings: dict[str, int]) -> str | None:
        """Check ``value`` and record the sizes its named dimensions take in ``bindings``.

        Sharing one ``bindings`` dict across the arrays of a record makes a name
        mean one size everywhere it appears.
        """
        array = np.asarray(value)
        if array.dtype.name != self.dtype:
            return f"dtype {array.dtype.name}, expected {self.dtype}"
        if array.ndim != self.rank:
            return f"shape {list(array.shape)}, expected {self}"
        for actual, expected in zip(array.shape, self.shape):
            expected = _resolve(expected, bindings)
            if expected is None:
                continue
            if isinstance(expected, str):
                bindings[expected] = int(actual)
                continue
            if int(actual) != expected:
                return f"shape {list(array.shape)}, expected {self}"
        return None

    def compare(
        self, other: ArraySpec, *, bindings: Mapping[str, int] | None = None
    ) -> tuple[list[str], list[str]]:
        """Disagreements between this spec and ``other``, and what stays unverified.

        Axis by axis: two fixed sizes must be equal, a free size accepts
        anything, and a named dimension that neither side nor ``bindings`` fixes
        is reported as unverified rather than accepted. An axis one side leaves
        free is accepted but listed as unverified too, naming the axis, so a
        summary never reads "checked" for a size nobody compared.
        """
        if self.dtype != other.dtype:
            return [f"dtype {self.dtype} vs {other.dtype}"], []
        if self.rank != other.rank:
            return [f"shape {self} vs {other}"], []
        resolved = dict(bindings or {})
        unverified: list[str] = []
        for axis, (mine, theirs) in enumerate(zip(self.shape, other.shape)):
            mine, theirs = _resolve(mine, resolved), _resolve(theirs, resolved)
            if isinstance(mine, int) and isinstance(theirs, int):
                if mine != theirs:
                    return [f"shape {self} vs {other}"], []
                continue
            note = self._unverified_axis(other, axis, mine, theirs)
            if note is not None:
                unverified.append(note)
        return [], unverified

    def _unverified_axis(self, other: ArraySpec, axis: int, mine: DimSize, theirs: DimSize) -> str | None:
        """Why an axis whose resolved sizes are not both fixed stays unverified, or None if it matches."""
        if mine is None or theirs is None:
            if mine is not None or theirs is not None:
                return f"axis {axis} free on one side ({self} vs {other})"
            return None
        if isinstance(mine, str) and isinstance(theirs, str) and mine == theirs:
            return None
        names = sorted({size for size in (mine, theirs) if isinstance(size, str)})
        return f"{' and '.join(names)} unbound ({self} vs {other})"

    def __str__(self) -> str:
        return f"{self.dtype}{_format_shape(self.shape)}"


@dataclass(frozen=True)
class ObservationSpec:
    """What an :class:`~isaac_rl_posttraining.data.records.Observation` carries.

    ``arrays`` maps component names to their specs; ``text`` names the string
    entries (a language instruction). A producer's spec lists everything it
    emits; a consumer's spec lists only what it reads.
    """

    arrays: Mapping[str, ArraySpec]
    text: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        arrays = dict(self.arrays)
        text = frozenset(self.text)
        for name in list(arrays) + list(text):
            if not isinstance(name, str) or not name:
                raise ValueError("observation component names must be non-empty strings")
        for name, spec in arrays.items():
            if not isinstance(spec, ArraySpec):
                raise TypeError(f"observation component {name!r} needs an ArraySpec")
        shared = sorted(set(arrays) & text)
        if shared:
            raise ValueError(f"components cannot be both arrays and text: {shared}")
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "text", text)

    @classmethod
    def from_space(cls, space: Any, *, text: tuple[str, ...] = ()) -> ObservationSpec:
        """The spec a Gymnasium ``Dict`` space of boxes declares."""
        return cls({name: ArraySpec.from_space(box) for name, box in space.spaces.items()}, frozenset(text))

    @property
    def images(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.arrays.items() if spec.is_image)

    def bind(self, bindings: Mapping[str, int]) -> ObservationSpec:
        return ObservationSpec({name: spec.bind(bindings) for name, spec in self.arrays.items()}, self.text)

    def to_json(self) -> dict[str, Any]:
        """The JSON form: ``{"arrays": {name: array spec}, "text": [names]}``."""
        return {"arrays": {name: spec.to_json() for name, spec in self.arrays.items()}, "text": sorted(self.text)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ObservationSpec:
        return cls(
            {name: ArraySpec.from_json(spec) for name, spec in data["arrays"].items()},
            frozenset(data.get("text", ())),
        )

    def mismatches(
        self,
        components: Mapping[str, Any],
        text: Mapping[str, str],
        *,
        bindings: Mapping[str, int] | None = None,
    ) -> list[str]:
        """Every way real components fail this spec: missing entries, wrong dtype or shape."""
        problems: list[str] = []
        measured: dict[str, int] = dict(bindings or {})
        for name, spec in self.arrays.items():
            if name not in components:
                problems.append(f"{name}: missing")
                continue
            reason = spec.measure(components[name], measured)
            if reason is not None:
                problems.append(f"{name}: {reason}")
        for name in sorted(self.text):
            if name not in text:
                problems.append(f"{name}: missing text")
        return problems

    def describe(self) -> str:
        parts = [f"{name}: {spec}" for name, spec in self.arrays.items()]
        parts.extend(f"{name}: text" for name in sorted(self.text))
        return "{" + ", ".join(parts) + "}"


@dataclass(frozen=True)
class ActionSpec:
    """The per-step action a simulator accepts, or the chunk a policy emits."""

    width: DimSize
    dtype: str = "float32"
    horizon: DimSize = None

    def __post_init__(self) -> None:
        _check_dim(self.width)
        _check_dim(self.horizon)
        if self.width is None:
            raise ValueError("an action spec needs a fixed or named width")
        object.__setattr__(self, "dtype", _dtype_name(self.dtype))

    def to_json(self) -> dict[str, Any]:
        """The JSON form: ``{"width": ..., "dtype": ..., "horizon": ...}``."""
        return {"width": self.width, "dtype": self.dtype, "horizon": self.horizon}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ActionSpec:
        return cls(width=data["width"], dtype=data["dtype"], horizon=data.get("horizon"))

    def as_array_spec(self) -> ArraySpec:
        """The ``[horizon, width]`` chunk this spec describes (free horizon when unset)."""
        return ArraySpec((self.horizon, self.width), self.dtype)

    def mismatch(self, actions: Any, *, bindings: Mapping[str, int] | None = None) -> str | None:
        """Why a ``[steps, width]`` chunk does not fit, or ``None`` when it does."""
        return ArraySpec((None, self.width), self.dtype).mismatch(actions, bindings=bindings)

    def __str__(self) -> str:
        horizon = "" if self.horizon is None else f" x {self.horizon} steps"
        return f"{self.dtype}[{self.width}]{horizon}"


@dataclass(frozen=True)
class SpecComparison:
    """The outcome of comparing two sides of a hand-off."""

    checks: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def merged(self, other: SpecComparison) -> SpecComparison:
        return SpecComparison(
            self.checks + other.checks,
            self.mismatches + other.mismatches,
            self.unverified + other.unverified,
        )

    def raise_on_mismatch(self, context: str) -> None:
        if self.mismatches:
            raise SpecMismatch(f"{context}: " + "; ".join(self.mismatches))

    def summary(self) -> str:
        parts = [f"checked: {', '.join(self.checks) if self.checks else 'nothing'}"]
        if self.mismatches:
            parts.append(f"mismatches: {'; '.join(self.mismatches)}")
        if self.unverified:
            parts.append(f"unverified: {'; '.join(self.unverified)}")
        return " | ".join(parts)


def compare_observation_specs(
    emitted: ObservationSpec,
    consumed: ObservationSpec,
    *,
    bindings: Mapping[str, int] | None = None,
) -> SpecComparison:
    """Check that a producer's observation covers everything a consumer reads.

    Components the producer emits but the consumer ignores are fine; every
    component the consumer reads must exist with a compatible spec.
    """
    checks: list[str] = []
    mismatches: list[str] = []
    unverified: list[str] = []
    for name, wanted in consumed.arrays.items():
        if name not in emitted.arrays:
            mismatches.append(f"observation {name!r} is read but not emitted (emitted: {emitted.describe()})")
            continue
        found_mismatches, found_unverified = emitted.arrays[name].compare(wanted, bindings=bindings)
        checks.append(f"observation {name!r}")
        mismatches.extend(f"observation {name!r}: {reason}" for reason in found_mismatches)
        unverified.extend(f"observation {name!r}: {reason}" for reason in found_unverified)
    for name in sorted(consumed.text):
        if name in emitted.text:
            checks.append(f"text {name!r}")
        else:
            mismatches.append(f"text {name!r} is read but not emitted")
    return SpecComparison(tuple(checks), tuple(mismatches), tuple(unverified))


def compare_action_specs(
    emitted: ActionSpec,
    accepted: ActionSpec,
    *,
    bindings: Mapping[str, int] | None = None,
) -> SpecComparison:
    """Check that the action a policy emits fits what the simulator accepts, step by step."""
    found_mismatches, found_unverified = ArraySpec((emitted.width,), emitted.dtype).compare(
        ArraySpec((accepted.width,), accepted.dtype), bindings=bindings
    )
    return SpecComparison(
        ("action width",),
        tuple(f"action: emitted {emitted} vs accepted {accepted}" for _ in found_mismatches),
        tuple(f"action: {reason}" for reason in found_unverified),
    )
