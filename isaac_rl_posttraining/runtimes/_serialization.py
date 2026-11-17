# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side helpers for serializing user-defined specs into remote workers."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import sys
import sysconfig
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

_STDLIB_NAMES = frozenset(getattr(sys, "stdlib_module_names", ()))
#: The checkout this package lives in; its packages are importable by name on every worker, its tests are not.
_REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def _library_roots() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    roots = {paths.get(key) for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(os.path.realpath(root) for root in roots if root)


def _imports_by_name(module: ModuleType) -> bool:
    """Whether every worker can import ``module`` by name, so it must not be pickled by value.

    True for the standard library, built-ins, installed third-party packages
    and this repository's own packages. Registering those by value makes
    cloudpickle serialize their classes themselves, which fails on C-level
    descriptors and on library singletons (``functools.partial`` and torch's
    config objects were the first casualties). Only modules outside all of
    them, a user's script or a test's throwaway module, take the by-value path.
    """
    top = module.__name__.partition(".")[0]
    if top in _STDLIB_NAMES or top in sys.builtin_module_names:
        return True
    file = getattr(module, "__file__", None)
    if not file:
        return True
    path = os.path.realpath(file)
    if path.startswith(_REPO_ROOT + os.sep):
        # The checkout's packages are installed on every worker; its test
        # directories are on the driver's path only (pytest puts them there).
        return "tests" not in Path(os.path.relpath(path, _REPO_ROOT)).parts
    if any(part in ("site-packages", "dist-packages") for part in Path(path).parts):
        return True
    return any(path.startswith(root + os.sep) for root in _library_roots())


def _defining_module(value: Any) -> ModuleType | None:
    while isinstance(value, functools.partial):
        value = value.func
    module_name = getattr(value, "__module__", None)
    if module_name is None and callable(value):
        module_name = getattr(type(value), "__module__", None)
    if not isinstance(module_name, str):
        return None
    module = sys.modules.get(module_name)
    if module is None or _imports_by_name(module):
        return None
    return module


#: How deep the scan follows nested specs (a served worker spec wraps a
#: worker spec that carries the entry and the policy spec).
_MAX_DEPTH = 4


def _spec_candidates(value: Any, depth: int, seen: set[int]) -> Iterator[Any]:
    """The spec itself and, through nested dataclasses and containers, the callables and entries it carries."""
    if id(value) in seen or depth > _MAX_DEPTH:
        return
    seen.add(id(value))
    if (
        callable(value)
        or isinstance(value, SimulatorEntry)
        or dataclasses.is_dataclass(value)
    ):
        yield value
    if isinstance(value, functools.partial):
        children: Iterable[Any] = (value.func, *value.args, *value.keywords.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif isinstance(value, Mapping):
        children = value.values()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        children = (
            getattr(value, field.name, None) for field in dataclasses.fields(value)
        )
    else:
        try:
            children = vars(value).values()
        except TypeError:
            return
    for child in children:
        if isinstance(child, (str, bytes, int, float, bool, type(None))):
            continue
        yield from _spec_candidates(child, depth + 1, seen)


def _spec_modules(spec: Any) -> tuple[ModuleType, ...]:
    """The top-level packages defining a spec and the callables and simulator entries it carries, however nested."""
    modules: dict[str, ModuleType] = {}
    for candidate in _spec_candidates(spec, 0, set()):
        module = _defining_module(candidate)
        if module is not None:
            # Registering the top-level package also ships the sibling modules it imports.
            top = sys.modules.get(module.__name__.partition(".")[0])
            if top is not None and not _imports_by_name(top):
                module = top
            modules[module.__name__] = module
    return tuple(modules.values())


@contextmanager
def serialize_specs_by_value(spec: Any, *, enabled: bool) -> Iterator[None]:
    """Temporarily make the modules defining script-level specs cloudpickle by value.

    Workers of a Ray-based runtime import a spec's module by name unless the
    module is registered for by-value pickling; a spec (or the simulator entry
    and policy spec it carries, at any nesting such as a served worker spec
    wrapping a worker spec) defined in a script or test needs that.
    """
    if not enabled:
        yield
        return
    cloudpickle = importlib.import_module("ray.cloudpickle")
    modules = _spec_modules(spec)
    for module in modules:
        cloudpickle.register_pickle_by_value(module)
    try:
        yield
    finally:
        for module in reversed(modules):
            cloudpickle.unregister_pickle_by_value(module)
