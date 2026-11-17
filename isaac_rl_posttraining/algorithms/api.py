# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithms as Python objects: ``DSRL(base="gr00t_n17", base_ckpt=...)``.

Each constructor names a registered algorithm, the base model it trains or
steers, and keyword hyperparameters. :meth:`Algorithm.overrides` turns that
into exactly the Hydra overrides ``scripts/train.py`` would get, so a run
launched from Python and one launched from the command line compose the same
config::

    DSRL(base="gr00t_n17", base_ckpt="/ckpt", batch_size=128).overrides()
    # ['algorithm=dsrl_sac', 'base_model=gr00t_n17', "base_model.ckpt='/ckpt'", 'algorithm.batch_size=128']

A keyword goes to the first config section that declares it: the
``algorithm`` group (off-policy learners), then the algorithm's own sections
(``dsrl``, ``rfs``, ``direct_action``, ``ppo``) in its registered order.
Anything else (``training.*``, ``eval.*``) is an override passed to
:func:`~isaac_rl_posttraining.launch`.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from isaac_rl_posttraining.algorithms.registry import (
    AlgorithmSpec,
    algorithm_spec,
    load_target,
)

__all__ = [
    "DSRL",
    "DSRLPPO",
    "PPO",
    "RFS",
    "Algorithm",
    "BaseModel",
    "SACEdit",
    "override_value",
]


def override_value(value: Any) -> str:
    """``value`` written in Hydra's override grammar."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(f"{key}:{override_value(item)}" for key, item in value.items())
            + "}"
        )
    if isinstance(value, Sequence):
        return "[" + ",".join(override_value(item) for item in value) + "]"
    raise TypeError(
        f"cannot write {type(value).__name__} value {value!r} as a config override"
    )


class BaseModel:
    """The pretrained base model a run starts from: which one, its checkpoint and its settings.

    A description, not the model: ``name`` is a ``base_model`` config option
    (``"gr00t_n17"``, ``"pi0"``, ``"mock"``) or a
    :class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` subclass,
    which is registered under its dotted path; ``ckpt`` sets ``base_model.ckpt``
    and ``options`` set ``base_model.<key>``. Where the base model runs is
    ``placement.parts.base``.
    """

    def __init__(
        self, name: str | type, ckpt: str | None = None, **options: Any
    ) -> None:
        self.name = name
        self.ckpt = ckpt
        self.options = dict(options)

    def __repr__(self) -> str:
        options = "".join(f", {key}={value!r}" for key, value in self.options.items())
        return f"BaseModel({self.name!r}, ckpt={self.ckpt!r}{options})"

    def overrides(self) -> list[str]:
        """The ``base_model`` overrides selecting this base model."""
        if isinstance(self.name, str):
            selected = [f"base_model={self.name}"]
        else:
            from isaac_rl_posttraining.policies import (
                available_base_models,
                register_base_model,
            )

            dotted = f"{self.name.__module__}.{self.name.__qualname__}"
            if dotted not in available_base_models():
                register_base_model(
                    dotted, f"{self.name.__module__}:{self.name.__qualname__}"
                )
            selected = [f"base_model.name={dotted}"]
        if self.ckpt is not None:
            selected.append(f"base_model.ckpt={override_value(self.ckpt)}")
        return selected + [
            f"base_model.{key}={override_value(value)}"
            for key, value in self.options.items()
        ]


def _base_model(
    base: str | type | BaseModel | None, base_ckpt: str | None
) -> BaseModel | None:
    if isinstance(base, BaseModel):
        if base_ckpt is not None:
            raise TypeError(
                "give the checkpoint either as BaseModel(ckpt=...) or as base_ckpt=, not both"
            )
        return base
    if base is None:
        # The config's own base model (an algorithm recipe may pin one), with this checkpoint.
        return (
            None
            if base_ckpt is None
            else _CheckpointOnly("<config default>", base_ckpt)
        )
    return BaseModel(base, base_ckpt)


class _CheckpointOnly(BaseModel):
    """The config's own base model with a different checkpoint."""

    def overrides(self) -> list[str]:
        return [f"base_model.ckpt={override_value(self.ckpt)}"]


def _section_fields(section: str, spec: AlgorithmSpec) -> set[str]:
    """The settings a config section declares."""
    from isaac_rl_posttraining.configs.normalize import CONFIG_ALIASES
    from isaac_rl_posttraining.configs.schema import TrainConfig

    if section == "algorithm":
        if spec.config is None:
            return set()
        schema = load_target(spec.config).yaml_schema()
        names = {item.name for item in dataclasses.fields(schema)} - {"name", "family"}
    else:
        hint = typing.get_type_hints(TrainConfig)[section]
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        names = {item.name for item in dataclasses.fields(args[0] if args else hint)}
    # Deprecated aliases are reachable only by their canonical names.
    deprecated = {
        alias.old.split(".", 1)[1]
        for alias in CONFIG_ALIASES
        if alias.old.startswith(f"{section}.")
    }
    return names - deprecated


class Algorithm:
    """A registered algorithm with its base model and hyperparameters.

    Subclasses set :attr:`name`; :meth:`named` builds one for any registered
    name (an out-of-core algorithm).
    """

    name: ClassVar[str]

    def __init__(
        self,
        base: str | type | BaseModel | None = None,
        base_ckpt: str | None = None,
        **settings: Any,
    ) -> None:
        self.base = _base_model(base, base_ckpt)
        self.settings = dict(settings)
        self._routed = self._route(self.settings)

    @staticmethod
    def named(
        name: str,
        base: str | type | BaseModel | None = None,
        base_ckpt: str | None = None,
        **settings: Any,
    ) -> Algorithm:
        """An :class:`Algorithm` for the registered algorithm ``name``."""
        algorithm_spec(name)
        named = type(f"Algorithm[{name}]", (Algorithm,), {"name": name})
        return named(base, base_ckpt, **settings)

    @property
    def spec(self) -> AlgorithmSpec:
        """The algorithm's registry entry."""
        return algorithm_spec(self.name)

    def _sections(self) -> tuple[str, ...]:
        spec = self.spec
        if spec.loop == "off_policy":
            return ("algorithm", *spec.sections)
        # An on-policy algorithm trains from its own sections; its algorithm group only sizes networks.
        return (*spec.sections, "algorithm")

    def _route(self, settings: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
        sections = [
            (section, _section_fields(section, self.spec))
            for section in self._sections()
        ]
        routed, unknown = [], []
        for key, value in settings.items():
            section = next((name for name, names in sections if key in names), None)
            if section is None:
                unknown.append(key)
            else:
                routed.append((section, key, value))
        if unknown:
            raise TypeError(
                f"{type(self).__name__}({self.name}) got unknown settings {sorted(unknown)}; it accepts the "
                f"fields of {', '.join(name for name, names in sections if names)} (other keys go in overrides=)"
            )
        return routed

    def overrides(self) -> list[str]:
        """The Hydra overrides ``scripts/train.py`` would get for this algorithm."""
        overrides = [f"algorithm={self.name}"]
        if self.base is not None:
            overrides += self.base.overrides()
        return overrides + [
            f"{section}.{key}={override_value(value)}"
            for section, key, value in self._routed
        ]

    def __repr__(self) -> str:
        settings = "".join(f", {key}={value!r}" for key, value in self.settings.items())
        return f"{type(self).__name__}(base={self.base!r}{settings})"


class DSRL(Algorithm):
    """DSRL: SAC over the frozen base model's flow noise (steering latents)."""

    name = "dsrl_sac"


class RFS(Algorithm):
    """Residual flow steering: steering latents plus a bounded residual after the decode."""

    name = "rfs"


class SACEdit(Algorithm):
    """SAC-edit: a bounded edit of the base model's decoded action chunk."""

    name = "sac_edit"


class PPO(Algorithm):
    """PPO fine-tuning of the base model's flow head (the base model is trained)."""

    name = "ppo"


class DSRLPPO(Algorithm):
    """PPO over the steering actor of a frozen base model."""

    name = "dsrl_ppo"
