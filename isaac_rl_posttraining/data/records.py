# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-validating records whose field annotations are their schema, and the observation built on them.

Declaring a new observation type takes a dataclass with annotated fields::

    @dataclass(frozen=True)
    class ArenaObservation(Observation):
        ego_view_image: Array[np.uint8, dim.H, dim.W, 3]
        state: Array[np.float32, dim.state_dim]

The annotations are the schema: ``ArenaObservation.declared_spec()`` is what the
type promises, ``instance.spec()`` what one instance carries, and every
instance validates itself on construction, including that a named dimension
such as ``dim.state_dim`` takes one size everywhere it appears. Observation
types may hold other observation types; their components are flattened as
``"field/component"``. The records of one decision and one episode build on
the same base in :mod:`isaac_rl_posttraining.data.transitions`.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np

from isaac_rl_posttraining.data.spec import ArraySpec, ObservationSpec, SpecMismatch

__all__ = [
    "Array",
    "ArrayMap",
    "DynamicObservation",
    "Observation",
    "Record",
    "Text",
    "coerce_array",
    "dim",
    "require_finite",
]


class _TextMarker:
    """Annotation marker of a string component."""


class _ArrayMapMarker:
    """Annotation marker of a mapping whose entries are arrays named only at runtime."""


_TEXT = _TextMarker()
_ARRAY_MAP = _ArrayMapMarker()

#: A string component of a record, such as a language instruction.
Text = Annotated[str, _TEXT]
#: Arrays whose names are only known at runtime (cached activations keyed by sub-module).
ArrayMap = Annotated[Mapping[str, np.ndarray], _ARRAY_MAP]


class Array:
    """``Array[dtype, *dims]`` annotates an array field.

    ``dims`` are fixed sizes, named dimensions (``dim.<name>``) or ``None``
    for a free size: ``Array[np.float32, dim.state_dim]``,
    ``Array[np.uint8, dim.H, dim.W, 3]``, ``Array[np.float32]`` for a scalar.
    """

    def __class_getitem__(cls, params: Any) -> Any:
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError("Array[...] needs a dtype")
        return Annotated[np.ndarray, ArraySpec(tuple(params[1:]), params[0])]


class _DimensionNames:
    """``dim.state_dim`` names a dimension whose size binds from config or the first instance.

    A named dimension is just its name string at runtime, but a string written
    inside an annotation reads as a forward reference to linters, which flag
    it as an undefined name and strip its quotes; the attribute spelling is
    what they leave alone.
    """

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        return name


dim = _DimensionNames()


@dataclass(frozen=True)
class _FieldKind:
    """How one annotated field of a record is stored and checked."""

    name: str
    kind: Literal["array", "text", "arrays", "record"]
    optional: bool
    array: ArraySpec | None = None
    record: type[Record] | None = None


def _unwrap_optional(owner: type, name: str, hint: Any) -> tuple[Any, bool]:
    """``(inner hint, optional)`` for an ``X | None`` annotation, else ``(hint, False)``."""
    origin = get_origin(hint)
    if origin is not typing.Union and origin is not types.UnionType:
        return hint, False
    args = get_args(hint)
    inner = [arg for arg in args if arg is not type(None)]
    if len(args) != 2 or len(inner) != 1:
        raise TypeError(
            f"{owner.__name__}.{name}: only 'X | None' unions are supported"
        )
    return inner[0], True


def _annotated_kind(name: str, hint: Any, optional: bool) -> _FieldKind | None:
    """The field kind an ``Annotated`` marker declares, or None."""
    for marker in get_args(hint)[1:]:
        if isinstance(marker, ArraySpec):
            return _FieldKind(name, "array", optional, array=marker)
        if marker is _TEXT:
            return _FieldKind(name, "text", optional)
        if marker is _ARRAY_MAP:
            return _FieldKind(name, "arrays", optional)
    return None


def _classify(owner: type, name: str, hint: Any) -> _FieldKind:
    hint, optional = _unwrap_optional(owner, name, hint)
    if get_origin(hint) is Annotated:
        kind = _annotated_kind(name, hint, optional)
        if kind is not None:
            return kind
    if isinstance(hint, type) and issubclass(hint, Record):
        return _FieldKind(name, "record", optional, record=hint)
    raise TypeError(
        f"{owner.__name__}.{name}: unsupported annotation {hint!r}; use Array[...], Text, "
        "ArrayMap or another Record type"
    )


_DECLARED: dict[type, tuple[_FieldKind, ...]] = {}


def coerce_array(owner: str, name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{owner}.{name} must be a numeric or boolean array, got object dtype")
    return array


def require_finite(owner: str, name: str, array: np.ndarray) -> None:
    """Reject NaN or inf in a floating-point array; other dtypes cannot hold them."""
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{owner}.{name} must be finite, found NaN or inf")


def coerce_array_map(owner: str, name: str, value: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner}.{name} must map names to arrays")
    arrays: dict[str, np.ndarray] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{owner}.{name} keys must be non-empty strings")
        arrays[key] = coerce_array(owner, f"{name}[{key!r}]", entry)
    return arrays


def _field_problems(
    spec: _FieldKind, value: Any, measured: dict[str, int]
) -> list[str]:
    """Annotation violations of one present field, measuring named dimensions into ``measured``."""
    problems: list[str] = []
    if spec.kind == "array":
        reason = spec.array.measure(value, measured)
        if reason is not None:
            problems.append(f"{spec.name}: {reason}")
    elif spec.kind == "record":
        nested = spec.record.declared_spec(include_optional=True).arrays
        for key, array in value.arrays().items():
            if key not in nested:
                continue
            reason = nested[key].measure(array, measured)
            if reason is not None:
                problems.append(f"{spec.name}/{key}: {reason}")
    return problems


@dataclass(frozen=True)
class Record:
    """Base of every typed record: annotated fields, coercion and validation on construction.

    Subclasses declare fields with :class:`Array`, :class:`Text`,
    :class:`ArrayMap` or another :class:`Record` type, optionally ``| None``.
    """

    def __post_init__(self) -> None:
        owner = type(self).__name__
        for spec in type(self).declared_fields():
            value = getattr(self, spec.name)
            if value is None:
                if not spec.optional:
                    raise ValueError(f"{owner}.{spec.name} is required")
                continue
            if spec.kind == "array":
                object.__setattr__(self, spec.name, coerce_array(owner, spec.name, value))
            elif spec.kind == "text" and not isinstance(value, str):
                raise TypeError(f"{owner}.{spec.name} must be a string")
            elif spec.kind == "arrays":
                object.__setattr__(self, spec.name, coerce_array_map(owner, spec.name, value))
            elif spec.kind == "record" and not isinstance(value, spec.record):
                raise TypeError(f"{owner}.{spec.name} must be a {spec.record.__name__}")
        self.validate()

    @classmethod
    def declared_fields(cls) -> tuple[_FieldKind, ...]:
        """The annotated fields of this record type, classified once per class."""
        cached = _DECLARED.get(cls)
        if cached is None:
            hints = get_type_hints(cls, include_extras=True)
            fields = {attribute.name for attribute in dataclasses.fields(cls)}
            # A subclass that skipped @dataclass keeps its annotations but declares no fields.
            undeclared = [
                name
                for name in inspect.get_annotations(cls)
                if name not in fields and get_origin(hints[name]) is not ClassVar
            ]
            if undeclared:
                raise TypeError(
                    f"{cls.__qualname__} annotates {undeclared} but is not a dataclass; "
                    "decorate it with @dataclass(frozen=True)"
                )
            cached = tuple(
                _classify(cls, attribute.name, hints[attribute.name])
                for attribute in dataclasses.fields(cls)
            )
            _DECLARED[cls] = cached
        return cached

    @classmethod
    def declared_spec(
        cls, bindings: Mapping[str, int] | None = None, *, include_optional: bool = False
    ) -> ObservationSpec:
        """The schema this type promises, from its annotations alone.

        Optional fields are left out unless ``include_optional`` is set, because a
        producer only promises what it always emits.
        """
        arrays: dict[str, ArraySpec] = {}
        text: set[str] = set()
        for spec in cls.declared_fields():
            if spec.optional and not include_optional:
                continue
            if spec.kind == "array":
                arrays[spec.name] = spec.array
            elif spec.kind == "text":
                text.add(spec.name)
            elif spec.kind == "record":
                nested = spec.record.declared_spec(include_optional=include_optional)
                arrays.update({f"{spec.name}/{key}": entry for key, entry in nested.arrays.items()})
                text.update(f"{spec.name}/{key}" for key in nested.text)
        declared = ObservationSpec(arrays, frozenset(text))
        return declared if bindings is None else declared.bind(bindings)

    def arrays(self) -> dict[str, np.ndarray]:
        """Every array this instance carries, nested records and runtime maps flattened as ``name/key``."""
        flattened: dict[str, np.ndarray] = {}
        for spec in type(self).declared_fields():
            value = getattr(self, spec.name)
            if value is None:
                continue
            if spec.kind == "array":
                flattened[spec.name] = value
            elif spec.kind == "arrays":
                flattened.update({f"{spec.name}/{key}": entry for key, entry in value.items()})
            elif spec.kind == "record":
                flattened.update({f"{spec.name}/{key}": entry for key, entry in value.arrays().items()})
        return flattened

    def text(self) -> dict[str, str]:
        """Every string this instance carries, nested records flattened as ``name/key``."""
        strings: dict[str, str] = {}
        for spec in type(self).declared_fields():
            value = getattr(self, spec.name)
            if value is None:
                continue
            if spec.kind == "text":
                strings[spec.name] = value
            elif spec.kind == "record":
                strings.update({f"{spec.name}/{key}": entry for key, entry in value.text().items()})
        return strings

    def bindings(self) -> dict[str, int]:
        """The sizes this instance gives the named dimensions of its type."""
        measured: dict[str, int] = {}
        declared = type(self).declared_spec(include_optional=True).arrays
        for name, array in self.arrays().items():
            if name in declared:
                declared[name].measure(array, measured)
        return measured

    def spec(self) -> ObservationSpec:
        """The schema of this instance, every axis fixed to what it carries."""
        return ObservationSpec(
            {name: ArraySpec.of(array) for name, array in self.arrays().items()},
            frozenset(self.text()),
        )

    def validate(self, bindings: Mapping[str, int] | None = None) -> None:
        """Raise :class:`SpecMismatch` listing every field that violates the annotations.

        A named dimension must take one size across all fields; ``bindings``
        fixes sizes up front (for example from the simulator's config).
        """
        measured: dict[str, int] = dict(bindings or {})
        problems: list[str] = []
        for spec in type(self).declared_fields():
            value = getattr(self, spec.name)
            if value is not None:
                problems.extend(_field_problems(spec, value, measured))
        if problems:
            raise SpecMismatch(f"{type(self).__name__}: " + "; ".join(problems))


@dataclass(frozen=True)
class Observation(Record):
    """What a policy sees at one decision; subclass it with annotated fields.

    ``Observation.from_mapping(...)`` on this base class builds a
    :class:`DynamicObservation`, the shape observations take while they still
    arrive as plain dictionaries from the environment wrappers.
    """

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Observation:
        """Build this observation type from a name-keyed mapping."""
        if cls is Observation:
            return DynamicObservation.from_mapping(mapping)
        declared = cls.declared_fields()
        names = {spec.name for spec in declared}
        unknown = sorted(set(mapping) - names)
        if unknown:
            raise ValueError(f"{cls.__name__} has no components {unknown}")
        missing = sorted(spec.name for spec in declared if not spec.optional and spec.name not in mapping)
        if missing:
            raise ValueError(f"{cls.__name__} is missing components {missing}")
        return cls(**{spec.name: mapping.get(spec.name) for spec in declared})

    def to_mapping(self) -> dict[str, np.ndarray | str]:
        """The flat name-keyed view the environment wrappers and frozen policies exchange."""
        return {**self.arrays(), **self.text()}


@dataclass(frozen=True)
class DynamicObservation(Observation):
    """An observation whose components are named at runtime.

    The environment wrappers assemble observations as dictionaries whose keys
    depend on the run (a DINO embedding, a language embedding); this record
    types that dictionary without a declared field per component.
    """

    components: Mapping[str, np.ndarray] = field(default_factory=dict)
    strings: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def declared_fields(cls) -> tuple[_FieldKind, ...]:
        return ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", coerce_array_map("DynamicObservation", "components", self.components))
        for name, value in self.strings.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise TypeError("DynamicObservation.strings maps non-empty names to strings")
        shared = sorted(set(self.components) & set(self.strings))
        if shared:
            raise ValueError(f"components cannot be both arrays and strings: {shared}")
        object.__setattr__(self, "strings", dict(self.strings))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DynamicObservation:
        components: dict[str, Any] = {}
        strings: dict[str, str] = {}
        for name, value in mapping.items():
            if isinstance(value, str):
                strings[name] = value
            else:
                components[name] = value
        return cls(components=components, strings=strings)

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(self.components)

    def text(self) -> dict[str, str]:
        return dict(self.strings)

    def bindings(self) -> dict[str, int]:
        return {}

    def validate(self, bindings: Mapping[str, int] | None = None) -> None:
        return None


