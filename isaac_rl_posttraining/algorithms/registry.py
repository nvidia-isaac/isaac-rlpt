# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of RL algorithms: the single source of truth for what exists and how it runs.

Each algorithm is one :class:`AlgorithmSpec`: the training loop it uses
(``off_policy`` replay, ``on_policy`` rollouts or ``offline``
demonstrations), the action space its actor acts in, its config class and
the launcher that builds its run. Everything the entry point routes on
(the trainer, the acting policy, the learner networks, the data-spec check)
is derived from these fields, so registering a new algorithm needs no edit
elsewhere in the core.

Modules are named as ``"module:attr"`` strings and imported lazily, so
importing this module never pulls an algorithm's dependencies.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from isaac_rl_posttraining.algorithms.base import BaseAlgorithmConfig

__all__ = [
    "ACTION_SPACES",
    "LOOPS",
    "ActionSpace",
    "AlgorithmSpec",
    "Loop",
    "algorithm_spec",
    "available_algorithms",
    "get_algorithm_config",
    "load_target",
    "register_algorithm",
    "registered_algorithms",
]

#: ``off_policy``: collect into a replay buffer and train on draws from it.
#: ``on_policy``: train on each round of fresh rollouts (PPO).
#: ``offline``: train on a demonstration dataset, with no collection (IQL, SFT).
Loop = Literal["off_policy", "on_policy", "offline"]
LOOPS: tuple[Loop, ...] = ("off_policy", "on_policy", "offline")

#: What the actor outputs. ``latent``: the frozen base model's flow noise
#: (DSRL); ``latent_residual``: flow noise plus a bounded residual added after
#: the decode (RFS); ``edit``: a bounded edit of the base model's decoded chunk
#: (SAC-edit); ``full_chunk``: the whole action chunk, by training the base
#: model itself (PPO, IQL, SFT).
ActionSpace = Literal["latent", "latent_residual", "edit", "full_chunk"]
ACTION_SPACES: tuple[ActionSpace, ...] = (
    "latent",
    "latent_residual",
    "edit",
    "full_chunk",
)

#: The rollout policy each action space collects with.
_POLICY_FAMILY: dict[str, str] = {
    "latent": "steering",
    "latent_residual": "steering",
    "edit": "edit",
    "full_chunk": "flow",
}
#: The policy parts each action space composes (placement can put parts on different devices).
_POLICY_PARTS: dict[str, tuple[str, ...]] = {
    "latent": ("base", "actor"),
    "latent_residual": ("base", "actor"),
    "edit": ("base", "actor"),
    "full_chunk": ("policy",),
}


@dataclass(frozen=True)
class AlgorithmSpec:
    """One registered algorithm.

    Attributes:
        name: Registry name, matching ``algorithm.name`` and ``configs/algorithm/<name>.yaml``.
        loop: The training loop (:data:`LOOPS`).
        action_space: What the actor outputs (:data:`ACTION_SPACES`).
        config: ``"module:attr"`` of the ``BaseAlgorithmConfig`` subclass the
            ``algorithm`` group is checked against, or ``None`` when the
            algorithm has no settings in that group (PPO reads ``ppo:``).
        launcher: ``"module:attr"`` of the launcher (``runners/launchers.py``)
            that plans and builds the run.
        sections: Config sections, besides ``algorithm``, that hold the
            algorithm's settings; the Python constructors route keyword
            arguments to them in this order.
        schedules: The schedules (``sync``, ``overlapped``, ``async``) its loop implements.
        requires: Optional third-party modules it needs; when one is missing
            the algorithm is listed as unavailable.
    """

    name: str
    loop: Loop
    action_space: ActionSpace
    config: str | None
    launcher: str
    sections: tuple[str, ...] = ()
    schedules: tuple[str, ...] = ("sync", "overlapped")
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.loop not in LOOPS:
            raise ValueError(
                f"{self.name}: loop must be one of {LOOPS}, got {self.loop!r}"
            )
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"{self.name}: action_space must be one of {ACTION_SPACES}, got {self.action_space!r}"
            )
        for target in (self.config, self.launcher):
            if target is not None and ":" not in target:
                raise ValueError(
                    f"{self.name}: targets must be 'module:attr', got {target!r}"
                )
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "schedules", tuple(self.schedules))
        object.__setattr__(self, "requires", tuple(self.requires))

    @property
    def policy_family(self) -> str:
        """The rollout policy it collects with: ``steering``, ``edit`` or ``flow``."""
        return _POLICY_FAMILY[self.action_space]

    @property
    def family(self) -> str:
        """Today's ``algorithm.family`` value: ``steering``, ``edit``, ``on_policy`` or ``offline``."""
        if self.loop in ("on_policy", "offline"):
            return self.loop
        return "edit" if self.action_space == "edit" else "steering"

    @property
    def policy_parts(self) -> tuple[str, ...]:
        """The parts of its acting policy (``base`` and ``actor``, or one trainable ``policy``)."""
        return _POLICY_PARTS[self.action_space]

    @property
    def reads_base_model_inputs(self) -> bool:
        """Whether its acting policy feeds the simulator's mapped state, cameras and language to a frozen base model."""
        return self.action_space != "full_chunk"


_OFF_POLICY = "isaac_rl_posttraining.runners.launchers:off_policy"
_OFFLINE = "isaac_rl_posttraining.runners.launchers:offline"
_SAC_CONFIG = "isaac_rl_posttraining.algorithms.sac:SACConfig"
#: Both built-in loops run every schedule (on-policy async keeps the policy lag at most 1).
_ALL_SCHEDULES = ("sync", "overlapped", "async")

_ALGORITHMS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            "dsrl_sac",
            "off_policy",
            "latent",
            _SAC_CONFIG,
            _OFF_POLICY,
            ("dsrl",),
            schedules=_ALL_SCHEDULES,
        ),
        AlgorithmSpec(
            "rfs",
            "off_policy",
            "latent_residual",
            "isaac_rl_posttraining.algorithms.rfs:RFSConfig",
            _OFF_POLICY,
            ("rfs", "dsrl"),
            schedules=_ALL_SCHEDULES,
        ),
        AlgorithmSpec(
            "sac_edit",
            "off_policy",
            "edit",
            "isaac_rl_posttraining.algorithms.sac_edit:SACEditConfig",
            _OFF_POLICY,
            ("direct_action",),
            schedules=_ALL_SCHEDULES,
        ),
        AlgorithmSpec(
            "ppo",
            "on_policy",
            "full_chunk",
            None,
            "isaac_rl_posttraining.runners.launchers:flow_sde_ppo",
            ("ppo",),
            schedules=_ALL_SCHEDULES,
        ),
        # dsrl_ppo trains the same steering actor as dsrl_sac with PPO: the steering
        # setup sizes that actor through the SAC config, and the run discards the SAC.
        AlgorithmSpec(
            "dsrl_ppo",
            "on_policy",
            "latent",
            _SAC_CONFIG,
            "isaac_rl_posttraining.runners.launchers:dsrl_ppo",
            ("ppo", "dsrl"),
            schedules=_ALL_SCHEDULES,
        ),
        AlgorithmSpec(
            "iql",
            "offline",
            "full_chunk",
            "isaac_rl_posttraining.algorithms.iql:IQLConfig",
            _OFFLINE,
            schedules=("sync",),
        ),
        AlgorithmSpec(
            "sft",
            "offline",
            "full_chunk",
            "isaac_rl_posttraining.algorithms.sft:SFTConfig",
            _OFFLINE,
            schedules=("sync",),
        ),
    )
}


def register_algorithm(
    name: str,
    target: str | None = None,
    requires: tuple[str, ...] = (),
    override: bool = False,
    *,
    loop: Loop = "off_policy",
    action_space: ActionSpace = "latent",
    launcher: str | None = None,
    sections: tuple[str, ...] = (),
    schedules: tuple[str, ...] = ("sync", "overlapped"),
) -> AlgorithmSpec:
    """Register an algorithm; nothing is imported until it is used.

    Args:
        name: Algorithm name, matching the ``algorithm.name`` config field.
        target: ``"module:attr"`` of its ``BaseAlgorithmConfig`` subclass
            (``None``: no ``algorithm`` group settings).
        requires: Optional modules it needs at construction time.
        override: Allow replacing an existing registration; without it a
            duplicate name raises, so two integrations cannot silently shadow
            each other.
        loop: ``off_policy``, ``on_policy`` or ``offline``.
        action_space: ``latent``, ``latent_residual``, ``edit`` or ``full_chunk``.
        launcher: ``"module:attr"`` of its launcher; defaults to the
            off-policy launcher for an off-policy algorithm.
        sections: Extra config sections its constructor keywords route to.
        schedules: The schedules its loop implements.

    Returns:
        The registered :class:`AlgorithmSpec`.
    """
    if name in _ALGORITHMS and not override:
        raise ValueError(
            f"Algorithm {name!r} is already registered (pass override=True to replace it)"
        )
    if launcher is None:
        if loop != "off_policy":
            raise ValueError(
                f"Algorithm {name!r}: an {loop} algorithm must name its launcher"
            )
        launcher = _OFF_POLICY
    spec = AlgorithmSpec(
        name=name,
        loop=loop,
        action_space=action_space,
        config=target,
        launcher=launcher,
        sections=sections,
        schedules=schedules,
        requires=tuple(requires),
    )
    _ALGORITHMS[name] = spec
    return spec


def _missing_requirements(spec: AlgorithmSpec) -> list[str]:
    """The algorithm's optional requirements that are not installed."""
    missing = []
    for module in spec.requires:
        try:
            found = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def registered_algorithms() -> dict[str, AlgorithmSpec]:
    """Every registered algorithm by name, installed or not."""
    return dict(_ALGORITHMS)


def available_algorithms() -> list[str]:
    """Names of the registered algorithms whose optional dependencies are installed."""
    return [
        name for name, spec in _ALGORITHMS.items() if not _missing_requirements(spec)
    ]


def algorithm_spec(name: str) -> AlgorithmSpec:
    """The registered :class:`AlgorithmSpec` of ``name``.

    Raises:
        KeyError: If ``name`` is not registered.
        ImportError: If its optional dependencies are missing.
    """
    if name not in _ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm {name!r}; available: {available_algorithms()}"
        )
    spec = _ALGORITHMS[name]
    missing = _missing_requirements(spec)
    if missing:
        raise ImportError(
            f"Algorithm {name!r} requires optional dependencies that are not installed: {missing}"
        )
    return spec


def load_target(target: str) -> Any:
    """Import ``"module:attr"`` and return the attribute."""
    module_name, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def get_algorithm_config(name: str) -> type[BaseAlgorithmConfig]:
    """Resolve an algorithm's config class by name.

    Returns:
        The ``BaseAlgorithmConfig`` subclass (not an instance); construct it
        with the composed config values, wire its runtime slots, then call
        ``config.create()`` for the algorithm.

    Raises:
        KeyError: If ``name`` is not registered, or it has no ``algorithm`` group config.
        ImportError: If its optional dependencies are missing.
    """
    spec = algorithm_spec(name)
    if spec.config is None:
        raise KeyError(
            f"Algorithm {name!r} has no algorithm-group config class; its settings live in "
            f"{', '.join(spec.sections) or 'its launcher'}"
        )
    return load_target(spec.config)
