# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base models wrapping pretrained robot foundation policies.

Every base model implements the :class:`~isaac_rl_posttraining.policies.base.
BaseModelPolicy` contract; :func:`get_base_model` resolves a base model class by
name and :func:`register_base_model` adds one. Base models with heavy or optional
dependencies are imported lazily, so importing this package never requires the
GR00T, pi0, or DreamZero client stacks.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect

from isaac_rl_posttraining.policies.base import (
    DEFAULT_LATENT_CLIP,
    BaseModelPolicy,
    MockBaseModelPolicy,
)

__all__ = [
    "DEFAULT_LATENT_CLIP",
    "BaseModelPolicy",
    "MockBaseModelPolicy",
    "available_base_models",
    "base_model_servable",
    "base_model_steerable",
    "get_base_model",
    "is_gr00t_base_model",
    "register_base_model",
]

_GR00T_BASE_MODEL = ("isaac_rl_posttraining.policies.gr00t.base_model", "Gr00tBaseModelPolicy")

#: name -> (module, class) of every known base model. Modules are imported
#: lazily by get_base_model.
_BASE_MODELS: dict[str, tuple[str, str]] = {
    "mock": ("isaac_rl_posttraining.policies.base", "MockBaseModelPolicy"),
    "gr00t": _GR00T_BASE_MODEL,
    # One frozen GR00T base model serves both generations (the checkpoint says which);
    # the generation-specific names exist so fine-tuning can pick its adapter by
    # ``base_model.name`` alone (see ``policies/trainable.py``).
    "gr00t_n15": _GR00T_BASE_MODEL,
    "gr00t_n17": _GR00T_BASE_MODEL,
    "pi0": ("isaac_rl_posttraining.policies.pi0.base_model", "Pi0BaseModelPolicy"),
    "dreamzero_local": (
        "isaac_rl_posttraining.policies.dreamzero.local_base_model",
        "DreamZeroBaseModelPolicy",
    ),
    "dreamzero_server": (
        "isaac_rl_posttraining.policies.dreamzero.server_base_model",
        "DreamZeroServerBaseModelPolicy",
    ),
}

#: Optional modules a base model needs at construction time. A base model is
#: listed as available only when all of them are installed.
_OPTIONAL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "pi0": ("openpi", "openpi_client"),
    "dreamzero_local": ("groot", "einops", "safetensors"),
    "dreamzero_server": ("websockets", "msgpack"),
}


def is_gr00t_base_model(name: str) -> bool:
    """Whether base model ``name`` is the GR00T base model under any of its registered names.

    GR00T decodes head outputs through an embodiment adapter that only the entry
    point may resolve (the policy layer never imports the env layer), so the
    launchers hand it in for every name that maps to the GR00T base model class.
    """
    if name not in _BASE_MODELS:
        raise KeyError(f"unknown base model {name!r}; registered: {sorted(_BASE_MODELS)}")
    return _BASE_MODELS[name] == _BASE_MODELS["gr00t"]


def register_base_model(
    name: str,
    target: str,
    requires: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register a base model under a name.

    Mirrors :func:`~isaac_rl_posttraining.envs.providers.register_provider` /
    :func:`~isaac_rl_posttraining.envs.providers.register_env_builder`: the
    base model module is *not* imported here; :func:`get_base_model` imports it on
    first use, so registration stays free of the base model's dependencies. This
    lets an out-of-core integration (e.g. a simulator package) contribute a
    base model without editing this module.

    Args:
        name: Base model name, matching the ``base_model.name`` config field.
        target: ``"module:attr"`` of the base model class; imported lazily.
        requires: Optional third-party module names the base model needs at
            construction time; when any is missing the base model is reported as
            unavailable and :func:`get_base_model` raises ``ImportError``.
        override: Allow replacing an existing registration. Without it a
            duplicate name raises, so two integrations cannot silently shadow
            each other.

    Raises:
        ValueError: If ``name`` is already registered and ``override`` is not
            set, or if ``target`` is not a ``"module:attr"`` string.
    """
    if name in _BASE_MODELS and not override:
        raise ValueError(
            f"Base model {name!r} is already registered "
            f"(pass override=True to replace it)"
        )
    if ":" not in target:
        raise ValueError(f"target must be 'module:attr', got {target!r}")
    module, attr = target.split(":", 1)
    _BASE_MODELS[name] = (module, attr)
    if requires:
        _OPTIONAL_REQUIREMENTS[name] = tuple(requires)
    else:
        _OPTIONAL_REQUIREMENTS.pop(name, None)


def _missing_requirements(name: str) -> list[str]:
    """Return the base model's optional requirements that are not installed."""
    missing = []
    for module in _OPTIONAL_REQUIREMENTS.get(name, ()):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
    return missing


def available_base_models() -> list[str]:
    """Names accepted by :func:`get_base_model` whose dependencies are installed.

    ``"mock"`` and ``"gr00t"`` are always listed (the GR00T base model class
    imports without the ``gr00t`` package; only constructing a policy needs
    it). The remote base models are listed only when their optional dependencies
    are installed.
    """
    return [name for name in _BASE_MODELS if not _missing_requirements(name)]


def get_base_model(name: str) -> type[BaseModelPolicy]:
    """Resolve a base model class by name.

    Args:
        name: A registered base model name, e.g. ``"mock"``, ``"gr00t"``,
            ``"pi0"``, or ``"dreamzero_local"``.

    Returns:
        The base model class (not an instance).

    Raises:
        KeyError: If ``name`` is not a known base model.
        ImportError: If the base model's optional dependencies are missing.
    """
    if name not in _BASE_MODELS:
        raise KeyError(
            f"Unknown base model {name!r}; available: {available_base_models()}"
        )
    missing = _missing_requirements(name)
    if missing:
        raise ImportError(
            f"Base model {name!r} requires optional dependencies that are not "
            f"installed: {missing}"
        )
    module_name, class_name = _BASE_MODELS[name]
    return getattr(importlib.import_module(module_name), class_name)


def _base_model_class(name: str) -> type[BaseModelPolicy] | None:
    """The class of base model ``name`` without checking its optional dependencies; ``None`` if it does not import."""
    if name not in _BASE_MODELS:
        raise KeyError(f"unknown base model {name!r}; registered: {sorted(_BASE_MODELS)}")
    module_name, class_name = _BASE_MODELS[name]
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except ImportError:
        return None


def base_model_servable(name: str) -> bool | None:
    """Whether base model ``name`` declares it may be served (:attr:`BaseModelPolicy.servable`).

    Reads the class attribute without constructing the base model or checking its
    optional dependencies; ``None`` when the class cannot be imported here.
    """
    base_model = _base_model_class(name)
    return None if base_model is None else bool(getattr(base_model, "servable", True))


def base_model_steerable(name: str) -> bool | None:
    """Whether base model ``name`` accepts steering latents (:attr:`BaseModelPolicy.supports_steering`).

    Reads the class without constructing it; ``None`` when the class does not
    import here or decides per instance (overrides the property).
    """
    base_model = _base_model_class(name)
    if base_model is None:
        return None
    steering = inspect.getattr_static(base_model, "supports_steering")
    if isinstance(steering, property):
        return True if steering is BaseModelPolicy.supports_steering else None
    return bool(steering)
