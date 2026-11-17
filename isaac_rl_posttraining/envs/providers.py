# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named environment providers: one entry per simulator integration.

An *environment provider* is the environment class a simulator integration
contributes (see ``simulators/README.md``): a Gymnasium environment whose
observations, actions, and ``info`` fields follow the contract the steering
wrappers and rollout workers consume. Providers are resolved by name so that
configs (the ``simulator`` Hydra group) and entry points never hard-code an
import path, and so that providers with heavy or optional dependencies are
imported only when actually requested.

Two kinds of provider exist behind the same seam:

* **remote** providers (``"g1_arena"``) drive a simulator -- or a real robot
  bridge -- that runs in another process, through an
  :class:`~isaac_rl_posttraining.envs.transport.EnvTransport`;
* **in-process** providers (``"libero"``, ``"isaaclab"``) step the simulator
  inside the learner process.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

import gymnasium as gym

from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "available_providers",
    "available_simulator_entries",
    "find_simulator_entry",
    "get_env_builder",
    "get_provider",
    "get_simulator_entry",
    "register_env_builder",
    "register_provider",
    "register_simulator_entry",
    "simulator_segment",
]

#: name -> (module, class) of every known provider. Modules are imported
#: lazily by :func:`get_provider`.
_PROVIDERS: dict[str, tuple[str, str]] = {
    "g1_arena": ("simulators.g1_arena.env", "ArenaEnv"),
    "v2d": ("simulators.v2d.env", "V2DEnv"),
    "libero": ("simulators.libero.env", "LiberoEnv"),
    "isaaclab": ("simulators.isaaclab.env", "IsaacLabGr00tEnv"),
}

#: Optional third-party modules a provider needs at construction time. A
#: provider is listed as available only when all of them are installed.
_OPTIONAL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "libero": ("libero", "robosuite"),
    "isaaclab": ("isaaclab",),
}


def _missing_requirements(name: str) -> list[str]:
    """Return the provider's optional requirements that are not installed."""
    missing = []
    for module in _OPTIONAL_REQUIREMENTS.get(name, ()):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
    return missing


def register_provider(
    name: str,
    module: str,
    attr: str,
    requires: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register an environment provider under a name.

    The provider module is *not* imported here; :func:`get_provider` imports
    it on first use, so registration stays free of the provider's
    dependencies.

    Args:
        name: Provider name, matching the ``provider`` field of the
            simulator's config (``configs/simulator/<name>.yaml``).
        module: Dotted module path holding the environment class.
        attr: Attribute name of the environment class inside ``module``.
        requires: Optional third-party module names the provider needs at
            construction time; when any is missing the provider is reported
            as unavailable and :func:`get_provider` raises ``ImportError``.
        override: Allow replacing an existing registration. Without it a
            duplicate name raises, so two integrations cannot silently
            shadow each other.

    Raises:
        ValueError: If ``name`` is already registered and ``override`` is
            not set.
    """
    if name in _PROVIDERS and not override:
        raise ValueError(
            f"Environment provider {name!r} is already registered "
            f"(pass override=True to replace it)"
        )
    _PROVIDERS[name] = (module, attr)
    if requires:
        _OPTIONAL_REQUIREMENTS[name] = tuple(requires)
    else:
        _OPTIONAL_REQUIREMENTS.pop(name, None)


def available_providers() -> list[str]:
    """Names accepted by :func:`get_provider` whose dependencies are installed.

    ``"g1_arena"`` is always listed (the Arena client only needs the base
    package; the simulator itself lives behind a transport). ``"libero"`` and
    ``"isaaclab"`` are listed only when their simulator stacks are installed.
    """
    return [name for name in _PROVIDERS if not _missing_requirements(name)]


def get_provider(name: str) -> type[gym.Env]:
    """Resolve an environment provider class by name.

    Args:
        name: A registered provider name, e.g. ``"g1_arena"`` or
            ``"libero"``.

    Returns:
        The environment class (not an instance); construction kwargs come
        from the simulator's config entry.

    Raises:
        KeyError: If ``name`` is not a registered provider.
        ImportError: If the provider's optional dependencies are missing.
    """
    if name not in _PROVIDERS:
        raise KeyError(
            f"Unknown environment provider {name!r}; available: {available_providers()}"
        )
    missing = _missing_requirements(name)
    if missing:
        raise ImportError(
            f"Environment provider {name!r} requires optional dependencies that "
            f"are not installed: {missing}; install them in the environment that runs this simulator"
        )
    module_name, class_name = _PROVIDERS[name]
    return getattr(importlib.import_module(module_name), class_name)


#: Simulator segment of ``env_name`` (``"<simulator>/<task>"``) -> "module:function"
#: of the simulator's training-env builder (constructs the fully-wrapped,
#: vectorized env for a run). Like the provider map, the modules are imported
#: lazily (in :func:`get_env_builder`), so core never statically imports a
#: simulator. The segment is matched exactly, so ``isaac`` and ``isaaclab`` or
#: ``libero`` and ``libero_pi0`` can never shadow each other.
_ENV_BUILDERS: dict[str, str] = {
    "arena": "simulators.g1_arena.env:build_dsrl_env",
    "isaac": "simulators.g1_arena.env:build_dsrl_env",
    "libero": "simulators.libero.policies.gr00t:build_dsrl_env",
    "libero_pi0": "simulators.libero.policies.pi0:build_dsrl_env",
    "libero_dreamzero": "simulators.libero.policies.dreamzero:build_dsrl_env",
}

#: Simulator segment -> "module:Class" of the simulator's
#: :class:`~isaac_rl_posttraining.envs.simulator_entry.SimulatorEntry` subclass,
#: the attachment the runtime layer builds rollout workers from for every
#: algorithm family. Imported lazily and matched on the exact segment; an entry
#: registered from Python (:func:`register_simulator_entry`) is stored as is.
_SIMULATOR_ENTRIES: dict[str, str | type[SimulatorEntry] | SimulatorEntry] = {
    "arena": "simulators.g1_arena.runtime:G1Arena",
    "isaac": "simulators.g1_arena.runtime:G1Arena",
    "isaaclab": "simulators.isaaclab.runtime:IsaacLab",
    "libero": "simulators.libero.runtime:Libero",
    "v2d": "simulators.v2d.runtime:V2D",
    "libero_pi0": "simulators.libero.runtime:LiberoPi0",
    "libero_dreamzero": "simulators.libero.runtime:LiberoDreamZero",
}

#: One instance per entry class, so every phase and every dry run sees the same object.
_ENTRY_INSTANCES: dict[type[SimulatorEntry], SimulatorEntry] = {}


def simulator_segment(env_name: str) -> str:
    """The simulator part of ``"<simulator>/<task>"`` (the whole name without a task)."""
    return env_name.split("/", 1)[0]


def _import_target(target: str):
    module_name, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def register_env_builder(name: str, target: str, *, override: bool = False) -> None:
    """Register a simulator's training-env builder for one env_name simulator segment.

    Args:
        name: Simulator segment of the env names this builder handles (the
            ``arena`` of ``arena/drill_lift``).
        target: ``"module:function"`` of the builder; imported lazily.
        override: Allow replacing an existing registration. Without it a
            duplicate name raises, so two integrations cannot silently
            shadow each other.

    Raises:
        ValueError: If ``name`` is already registered and ``override`` is
            not set.
    """
    if name in _ENV_BUILDERS and not override:
        raise ValueError(
            f"Env builder {name!r} is already registered (pass override=True to replace it)"
        )
    _ENV_BUILDERS[name] = target


def get_env_builder(env_name: str):
    """Return the simulator training-env builder for ``env_name``, or ``None``.

    The builder builds the fully-wrapped (vectorized) environment for its
    simulator and is imported lazily, so core stays free of simulator imports.
    ``None`` means no simulator matches ``env_name`` and the caller should use
    the generic Gymnasium path.
    """
    target = _ENV_BUILDERS.get(simulator_segment(env_name))
    if target is None:
        return None
    return _import_target(target)


def register_simulator_entry(
    name: str,
    entry: str | type[SimulatorEntry] | SimulatorEntry,
    *,
    override: bool = False,
) -> None:
    """Register a simulator entry for one env_name simulator segment.

    Args:
        name: Simulator segment of the env names the entry serves.
        entry: The entry class, an instance of it, or its ``"module:Class"``
            path imported lazily on first use.
        override: Allow replacing an existing registration. Without it a
            duplicate name raises, so two integrations cannot silently
            shadow each other.

    Raises:
        ValueError: If ``name`` is already registered and ``override`` is
            not set.
        TypeError: If ``entry`` is neither a :class:`SimulatorEntry` (class or
            instance) nor a ``"module:Class"`` string.
    """
    if name in _SIMULATOR_ENTRIES and not override:
        raise ValueError(
            f"Simulator entry {name!r} is already registered (pass override=True to replace it)"
        )
    if isinstance(entry, str):
        if ":" not in entry:
            raise ValueError(
                f"simulator entry for {name!r} must be written as 'module:Class', got {entry!r}"
            )
    elif not isinstance(entry, SimulatorEntry) and not (
        isinstance(entry, type) and issubclass(entry, SimulatorEntry)
    ):
        raise TypeError(
            f"simulator entry for {name!r} must subclass SimulatorEntry, got {entry!r}"
        )
    _SIMULATOR_ENTRIES[name] = entry


def _instantiate_entry(
    target: str | type[SimulatorEntry] | SimulatorEntry,
) -> SimulatorEntry:
    """The one instance of a registered entry, importing and constructing it on first use."""
    if isinstance(target, str):
        target = _import_target(target)
    if isinstance(target, SimulatorEntry):
        return target
    if not (isinstance(target, type) and issubclass(target, SimulatorEntry)):
        raise TypeError(f"registered simulator entry {target!r} does not subclass SimulatorEntry")
    instance = _ENTRY_INSTANCES.get(target)
    if instance is None:
        try:
            instance = target()
        except TypeError as exc:
            # An abstract subclass: name the missing method instead of Python's generic message.
            raise TypeError(
                f"{target.__module__}.{target.__qualname__} is not a complete simulator entry: {exc}"
            ) from exc
        _ENTRY_INSTANCES[target] = instance
    return instance


def available_simulator_entries() -> list[str]:
    """The simulator segments with a registered entry (the ``arena`` of ``arena/drill_lift``), sorted."""
    return sorted(_SIMULATOR_ENTRIES)


def find_simulator_entry(env_name: str) -> SimulatorEntry | None:
    """The simulator entry registered for ``env_name``'s simulator segment, or ``None``."""
    target = _SIMULATOR_ENTRIES.get(simulator_segment(env_name))
    if target is None:
        return None
    return _instantiate_entry(target)


def get_simulator_entry(env_name: str) -> SimulatorEntry:
    """The simulator entry for ``env_name``.

    Raises:
        KeyError: If no registered simulator matches the environment name.
    """
    entry = find_simulator_entry(env_name)
    if entry is None:
        raise KeyError(
            f"no simulator entry matches environment {env_name!r}; registered simulators: "
            f"{sorted(_SIMULATOR_ENTRIES)}"
        )
    return entry
