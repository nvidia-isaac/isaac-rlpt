# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainable flow policies: the contract PPO fine-tunes a base model through.

A frozen :class:`~isaac_rl_posttraining.policies.base.BaseModelPolicy` is a black
box the steering algorithms sample around. Fine-tuning needs more from a model:
the on-policy loop samples action chunks with a tractable likelihood, ships
what the learner needs to recompute that likelihood, and updates the model's
own weights on the learner ranks. This module holds that contract and the
plumbing every base model shares:

* :class:`TrainablePolicy` -- what the rollout worker and the PPO learner call:
  build model inputs from frames, states and an instruction; ``act`` with one
  stochastic denoising step; ``decode`` the chunk; pack the inputs for
  transport; re-evaluate a stored chain in ``forward``.
* :class:`TrainablePolicySpec` -- a picklable recipe that loads one base model's
  adapter on any process (learner rank or rollout worker), selected by name
  through the registry (:func:`get_trainable_policy`).
* :data:`STOCHASTIC_STEP_KEY`, :class:`FlowSDEConfig`, :class:`FlowSDESample` --
  the flow-SDE exploration vocabulary the adapters share.

The adapters live next to their model stack -- ``gr00t/flow_sde.py`` (N1.5),
``gr00t/flow_sde_n17.py`` (N1.7), ``pi0/flow_sde.py`` (pi0 / pi0.5) -- and
subclass :class:`~isaac_rl_posttraining.policies.flow_sde.FlowSDEActorCritic`,
which owns the sampling math. A base model contributes only how it encodes its
inputs, predicts a velocity at a chain state, pools features for the value
head, and decodes; see ``docs/adding_a_base_model.md``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

__all__ = [
    "STOCHASTIC_STEP_KEY",
    "FlowSDEConfig",
    "FlowSDESample",
    "TrainablePolicy",
    "TrainablePolicySpec",
    "available_trainable_policies",
    "get_trainable_policy",
    "register_trainable_policy",
    "resolve_trainable_policy_name",
]

#: Name of the per-sample denoising-step index carried inside a learner observation.
STOCHASTIC_STEP_KEY = "stochastic_step"


@dataclass(frozen=True)
class FlowSDEConfig:
    """Flow-SDE exploration and value-head settings (RLinf's ``rl_head_config``).

    Attributes:
        noise_level: Scale of the SDE transition noise at the stochastic step.
        denoising_steps: Euler steps ``K`` per decode; the stochastic step is
            drawn uniformly from ``range(K)`` for every decision batch.
        value_hidden_sizes: Hidden widths of the value MLP.
        keep_dropout: Keep the model's dropout layers, active only in
            ``train()`` mode, as offline fine-tuning does. On-policy RL removes
            them so behavior and learner passes agree.
    """

    noise_level: float = 0.5
    denoising_steps: int = 4
    value_hidden_sizes: tuple[int, ...] = (1024, 512, 256)
    keep_dropout: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_level) or self.noise_level <= 0.0:
            raise ValueError("noise_level must be finite and positive")
        if isinstance(self.denoising_steps, bool) or not isinstance(self.denoising_steps, int):
            raise TypeError("denoising_steps must be a positive integer")
        if self.denoising_steps < 1:
            raise ValueError("denoising_steps must be a positive integer")
        if not self.value_hidden_sizes or any(int(width) < 1 for width in self.value_hidden_sizes):
            raise ValueError("value_hidden_sizes must contain positive widths")


@dataclass(frozen=True)
class FlowSDESample:
    """One batched decode: the chunk, its denoising chain, log-density and value.

    Attributes:
        actions: Normalized action chunk ``[B, H, D]`` (the last chain entry).
        chains: Denoising chain ``[B, K + 1, H, D]`` from the initial noise to
            the action; the learner re-evaluates the stochastic step from it.
        log_prob: Chunk log-probability ``[B]`` (float32); zero for a fully
            deterministic decode.
        value: Value estimate ``[B]`` (float32).
        stochastic_step: The noised step index, ``None`` for a deterministic decode.
        cached_features: Raw outputs of the frozen part of this decode (see
            :attr:`TrainablePolicy.cached_feature_names`), which a learner can
            consume instead of re-running it.
    """

    actions: torch.Tensor
    chains: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    stochastic_step: int | None
    cached_features: Mapping[str, torch.Tensor] = field(default_factory=dict)


@runtime_checkable
class TrainablePolicy(Protocol):
    """A policy the on-policy loop fine-tunes: the rollout side and the learner side of one model.

    Rollout side (one copy per rollout worker): :meth:`build_inputs` turns a
    batch of camera frames, state parts and the task instruction into model
    inputs; :meth:`act` decodes them with one stochastic denoising step and
    returns the chain, its log-probability and the value; :meth:`decode` maps
    the normalized chunk to raw environment actions; :meth:`pack_inputs`
    splits the inputs into per-sample records the transition carries (with
    :meth:`learner_inputs` reducing them to what training still needs once
    the frozen part's outputs travel as cached features).

    Learner side (one copy per learner rank): the module is called as
    ``policy(observations, actions, cached_features=None)`` and returns
    ``(log_prob, entropy, value)``, each ``[B]`` float32, for the stored chains
    under the current weights; ``observations`` are the unpacked records plus
    the stochastic step index under :data:`STOCHASTIC_STEP_KEY`.

    ``frozen_submodules`` names the dotted paths (relative to the module) that
    never train; the runtime keeps them whole on every rank and leaves them
    out of published policy states and checkpoints. ``cached_feature_names``
    lists the frozen part's outputs a decision can ship in place of the inputs
    they came from (empty when the base model has none worth caching), and
    ``requires_cached_features`` says the raw inputs have no per-sample layout
    to ship at all, so decisions must travel as cached features.

    Every adapter class also provides two class methods the launcher uses
    before any weights load: ``spec_options(base_model)`` extracts and validates
    the base model's own fields from the ``base_model`` config section, and
    ``load(checkpoint, *, device, flow_sde, value_head_seed, **options)``
    builds the adapter on one process.
    """

    frozen_submodules: tuple[str, ...]
    cached_feature_names: tuple[str, ...]
    requires_cached_features: bool
    video_keys: Sequence[str]
    state_keys: Sequence[str]
    action_window: int
    denoising_steps: int

    def build_inputs(
        self,
        frames: Mapping[str, np.ndarray],
        states: Mapping[str, np.ndarray],
        instruction: str,
    ) -> dict[str, torch.Tensor]: ...

    def pack_inputs(
        self, inputs: Mapping[str, torch.Tensor], batch_size: int
    ) -> list[dict[str, np.ndarray]]: ...

    def unpack_inputs(
        self, batch: Mapping[str, Any], device: torch.device | str
    ) -> dict[str, torch.Tensor]: ...

    def learner_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]: ...

    def act(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        stochastic_step: int | None,
        noise: torch.Tensor | None = None,
    ) -> FlowSDESample: ...

    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inputs: Mapping[str, torch.Tensor],
        states: Mapping[str, np.ndarray],
    ) -> np.ndarray: ...

    def __call__(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        cached_features: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True) -> Any: ...

    def eval(self) -> Any: ...


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

#: name -> (module, class) of every trainable adapter. Modules are imported
#: lazily by get_trainable_policy; the model stacks themselves load inside
#: the adapters' ``load`` class methods.
_TRAINABLE_POLICIES: dict[str, tuple[str, str]] = {
    "gr00t_n15": ("isaac_rl_posttraining.policies.gr00t.flow_sde", "Gr00tFlowSDEActorCritic"),
    "gr00t_n17": ("isaac_rl_posttraining.policies.gr00t.flow_sde_n17", "Gr00tN17FlowSDEActorCritic"),
    "pi0": ("isaac_rl_posttraining.policies.pi0.flow_sde", "Pi0FlowSDEActorCritic"),
}

#: Optional modules an adapter needs to load its model.
_OPTIONAL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "pi0": ("openpi",),
}

#: The base-model name that covers both GR00T generations; the checkpoint's
#: ``config.json`` says which adapter it needs.
_GR00T_FAMILY = "gr00t"


def register_trainable_policy(
    name: str,
    target: str,
    requires: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register a trainable adapter under a name (the shape of ``register_base_model``).

    Args:
        name: Adapter name, matching the ``base_model.name`` config field.
        target: ``"module:Class"`` of the adapter class; imported lazily.
        requires: Optional third-party module names the adapter needs to load
            its model; when any is missing :func:`get_trainable_policy` raises
            ``ImportError``.
        override: Allow replacing an existing registration.
    """
    if name in _TRAINABLE_POLICIES and not override:
        raise ValueError(
            f"trainable policy {name!r} is already registered (pass override=True to replace it)"
        )
    if ":" not in target:
        raise ValueError(f"target must be 'module:Class', got {target!r}")
    module, attr = target.split(":", 1)
    _TRAINABLE_POLICIES[name] = (module, attr)
    if requires:
        _OPTIONAL_REQUIREMENTS[name] = tuple(requires)
    else:
        _OPTIONAL_REQUIREMENTS.pop(name, None)


def _missing_requirements(name: str) -> list[str]:
    missing = []
    for module in _OPTIONAL_REQUIREMENTS.get(name, ()):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
    return missing


def available_trainable_policies() -> list[str]:
    """Adapter names whose optional model stacks are installed."""
    return [name for name in _TRAINABLE_POLICIES if not _missing_requirements(name)]


def get_trainable_policy(name: str) -> type[Any]:
    """Resolve a trainable adapter class by name (importing its module on first use).

    Raises:
        KeyError: If ``name`` is not registered.
        ImportError: If the adapter's optional model stack is not installed.
    """
    if name not in _TRAINABLE_POLICIES:
        raise KeyError(
            f"unknown trainable policy {name!r}; registered: {sorted(_TRAINABLE_POLICIES)}"
        )
    missing = _missing_requirements(name)
    if missing:
        raise ImportError(
            f"trainable policy {name!r} requires optional dependencies that are not installed: {missing}"
        )
    module_name, class_name = _TRAINABLE_POLICIES[name]
    return getattr(importlib.import_module(module_name), class_name)


def resolve_trainable_policy_name(base_model_name: str, checkpoint: str) -> str:
    """The adapter a ``base_model.name`` selects.

    Registered names select themselves. ``base_model=gr00t`` covers both GR00T
    generations, so it is resolved from the ``model_type`` the checkpoint's
    ``config.json`` declares; that needs a local checkpoint directory, so a
    Hugging Face id or a not-yet-staged path must name the generation
    explicitly (``base_model=gr00t_n15`` / ``base_model=gr00t_n17``).
    """
    if base_model_name in _TRAINABLE_POLICIES:
        return base_model_name
    if base_model_name == _GR00T_FAMILY:
        if not os.path.isdir(checkpoint):
            raise ValueError(
                f"base_model={_GR00T_FAMILY} names both GR00T generations and the checkpoint "
                f"{checkpoint!r} is not a local directory whose config.json could tell them apart; "
                "select base_model=gr00t_n15 or base_model=gr00t_n17"
            )
        from isaac_rl_posttraining.policies.gr00t.n15 import is_n15_checkpoint

        return "gr00t_n15" if is_n15_checkpoint(checkpoint) else "gr00t_n17"
    raise KeyError(
        f"base model {base_model_name!r} has no trainable adapter; registered: "
        f"{sorted(_TRAINABLE_POLICIES)} (and {_GR00T_FAMILY!r}, resolved from the checkpoint)"
    )


# ----------------------------------------------------------------------
# Loading recipe
# ----------------------------------------------------------------------


def _current_device() -> str:
    return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainablePolicySpec:
    """How to load one trainable policy on any process.

    Picklable and small: the learner ranks and the rollout workers each build
    their own copy from it. ``options`` holds the base model's own loading
    fields (an N1.5 data config and embodiment tag, an openpi config, ...) as
    extracted by the adapter's ``spec_options``.

    Attributes:
        base_model: Registered adapter name (see :func:`get_trainable_policy`).
        checkpoint: Checkpoint directory (or model id) the adapter loads.
        flow_sde: Exploration and value-head settings.
        value_head_seed: Seed of the fresh value head, so every rank and
            worker starts from identical weights.
        options: Base-model-specific keyword arguments of the adapter's ``load``.
    """

    base_model: str
    checkpoint: str
    flow_sde: FlowSDEConfig
    value_head_seed: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint must be non-empty")
        if isinstance(self.value_head_seed, bool) or not isinstance(self.value_head_seed, int):
            raise TypeError("value_head_seed must be an integer")
        if self.base_model not in _TRAINABLE_POLICIES:
            raise KeyError(
                f"unknown trainable policy {self.base_model!r}; registered: {sorted(_TRAINABLE_POLICIES)}"
            )

    @property
    def policy_class(self) -> type[Any]:
        """The adapter class, resolved without loading any weights."""
        return get_trainable_policy(self.base_model)

    @property
    def frozen_submodules(self) -> tuple[str, ...]:
        """The adapter's frozen parts (dotted paths relative to the module)."""
        return tuple(self.policy_class.frozen_submodules)

    def build(self, device: str | None = None) -> Any:
        """Load the adapter on ``device`` (the process's current CUDA device when ``None``)."""
        target = _current_device() if device is None else device
        return self.policy_class.load(
            self.checkpoint,
            device=target,
            flow_sde=self.flow_sde,
            value_head_seed=self.value_head_seed,
            **dict(self.options),
        )
