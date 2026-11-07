# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm base class: component registry, checkpointing, trainer hooks."""

from __future__ import annotations

import abc
import copy
import dataclasses
import json
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, fields, make_dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv
from omegaconf import MISSING
from torch import nn

from isaac_rl_posttraining.algorithms.constants import (
    OPTIMIZER_COMPONENT_SUFFIX,
    TRAINING_STATE_FILE,
)
from isaac_rl_posttraining.buffers.replay import ReplayBuffer
from isaac_rl_posttraining.buffers.samplers import Sampler
from isaac_rl_posttraining.loggers.base import Histogram, Logger
from isaac_rl_posttraining.modules.base.actor import BaseActor

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.learner import LearnerBatch

__all__ = [
    "RUNTIME_INJECTED",
    "BaseAlgorithm",
    "BaseAlgorithmConfig",
    "LearnerModule",
    "LearnerOptimizer",
]

_RUNTIME_INJECTED_KEY = "runtime_injected"
#: Field metadata marking a config slot the entry point fills with a live
#: object; :meth:`BaseAlgorithmConfig.yaml_schema` leaves such fields out.
RUNTIME_INJECTED = {_RUNTIME_INJECTED_KEY: True}


@dataclass(frozen=True)
class LearnerModule:
    """A network the learner runtime places, checkpoints and (``publish``) sends to the rollout workers."""

    module: nn.Module
    publish: bool = False


@dataclass(frozen=True)
class LearnerOptimizer:
    """An optimizer the learner runtime checkpoints, with the name of the learner module it steps."""

    optimizer: torch.optim.Optimizer
    module: str


@dataclass
class BaseAlgorithmConfig(abc.ABC):
    """Runtime wiring shared by all algorithm configs.

    These fields hold live objects the entry point injects at setup time; they
    are never set from YAML. Hydra validates the hyperparameters of a subclass
    against :meth:`yaml_schema`, which is this dataclass minus these slots.

    Attributes:
        name: Registry name of the algorithm (``algorithms/registry.py``), set
            by its ``configs/algorithm/<name>.yaml``; the entry point routes on it.
        family: Learner family the entry point dispatches on (``steering``,
            ``edit``, ``on_policy`` or ``offline``), set by the same yaml.
        env: Environment (vectorized or not) the target entropy is derived from.
        actor: Policy network.
        critic: Value network the algorithm builds its ensemble from.
        buffer: The replay buffer of the robot's own episodes.
        sampler: How the learner's main batch is drawn from replay.
        demonstrations: A second buffer of demonstrations, for the terms
            that learn from experts only.
        logger: Metric logger.
    """

    name: str = MISSING
    family: str = MISSING
    # Injected at runtime, not read from YAML.
    env: Env | VectorEnv | None = field(default=None, metadata=RUNTIME_INJECTED)
    actor: BaseActor | None = field(default=None, metadata=RUNTIME_INJECTED)
    critic: nn.Module | None = field(default=None, metadata=RUNTIME_INJECTED)
    buffer: ReplayBuffer | None = field(default=None, metadata=RUNTIME_INJECTED)
    sampler: Sampler | None = field(default=None, metadata=RUNTIME_INJECTED)
    demonstrations: ReplayBuffer | None = field(default=None, metadata=RUNTIME_INJECTED)
    logger: Logger | None = field(default=None, metadata=RUNTIME_INJECTED)

    @classmethod
    def yaml_schema(cls) -> type:
        """Dataclass of the YAML-settable fields, the node Hydra validates against.

        The runtime-injected slots are live Python objects OmegaConf cannot
        represent, so they are left out; every other field keeps its type and
        default, so unknown keys and mistyped hyperparameters still fail at
        composition time.
        """
        hints = get_type_hints(cls)
        yaml_fields = []
        for config_field in fields(cls):
            if _RUNTIME_INJECTED_KEY in config_field.metadata:
                continue
            yaml_fields.append(
                (
                    config_field.name,
                    hints[config_field.name],
                    field(default=config_field.default, default_factory=config_field.default_factory),
                )
            )
        return make_dataclass(f"{cls.__name__}Schema", yaml_fields)

    @property
    def algorithm_class(self) -> type:
        """Concrete algorithm class this config instantiates."""
        return BaseAlgorithm

    def create(self) -> "BaseAlgorithm":
        """Instantiate the algorithm described by this config."""
        algorithm_class = self.algorithm_class
        if algorithm_class is None:
            raise ValueError(f"algorithm_class not defined for {self.__class__.__name__}")
        return algorithm_class(self)


class BaseAlgorithm(abc.ABC):
    """Common scaffolding for policy-optimization algorithms.

    Subclasses register their stateful pieces in ``component_names`` (models,
    target networks, optimizers, learned coefficients) and declare which of
    them the learner runtime manages itself (:meth:`learner_modules`,
    :meth:`learner_optimizers`); :meth:`learner_state` carries the rest.
    """

    #: Plain-tensor components that belong to the training state rather than
    #: the model (learned coefficients such as an entropy temperature): they
    #: are skipped when optimizer state is not loaded and mirrored as scalars
    #: into the JSON training state.
    training_state_tensors: tuple[str, ...] = ()

    #: Attributes only the controller uses (replay access, logging); a
    #: :meth:`learner_replica` leaves them out.
    controller_attributes: tuple[str, ...] = ("buffer", "sampler", "logger")

    def __init__(self, config: BaseAlgorithmConfig) -> None:
        self.config = config
        self.component_names: list[str] = []
        self.step_counter = 0
        self.logger = config.logger

    @abc.abstractmethod
    def train_step(self, rollout_step: int = None) -> dict[str, Any]:
        """Run one training step and return its metrics; the controller logs them."""

    def learner_batch(self) -> LearnerBatch:
        """Draw the replay batches one ``train_step`` call consumes, packed for the learner runtime.

        A controller that keeps the replay buffer away from the learner calls
        this where the buffer lives and ships the result; the learner hands it
        back to ``train_step(batches=...)``.
        """
        raise NotImplementedError(f"{type(self).__name__} draws no learner batches")

    def learner_replica(self) -> BaseAlgorithm:
        """A shallow copy for a learner rank in another process: no replay buffer, samplers, logger or environment.

        Networks and optimizers are shared with this object until the copy is
        pickled to the rank.
        """
        replica = copy.copy(self)
        for name in self.controller_attributes:
            if hasattr(replica, name):
                setattr(replica, name, None)
        replica.config = dataclasses.replace(
            self.config, env=None, buffer=None, sampler=None, demonstrations=None, logger=None
        )
        return replica

    @property
    def respects_step_schedules(self) -> bool:
        """Whether updates take the collected environment steps (``train_step(rollout_step=...)``)."""
        return False

    def learner_modules(self) -> dict[str, LearnerModule]:
        """The networks a learner runtime places and checkpoints, by component name.

        Exactly one is marked ``publish``: the policy the rollout workers run.
        """
        raise NotImplementedError(f"{type(self).__name__} declares no learner modules")

    def learner_optimizers(self) -> dict[str, LearnerOptimizer]:
        """The optimizers of :meth:`learner_modules`, by component name."""
        raise NotImplementedError(f"{type(self).__name__} declares no learner optimizers")

    def learner_managed_components(self) -> tuple[str, ...]:
        """Component names the learner runtime checkpoints itself (modules and their optimizers)."""
        return (*self.learner_modules(), *self.learner_optimizers())

    def replicated_parameters(self) -> tuple[torch.Tensor, ...]:
        """Trainable tensors outside :meth:`learner_modules` (a learned temperature, for example).

        On several learner ranks their gradients are averaged like the modules'.
        """
        managed = set(self.learner_managed_components())
        tensors = []
        for name in self.component_names:
            component = getattr(self, name, None)
            if name not in managed and torch.is_tensor(component) and component.requires_grad:
                tensors.append(component)
        return tuple(tensors)

    def learner_state(self, *, managed: Collection[str]) -> dict[str, np.ndarray]:
        """The training state a learner runtime does not manage itself, as arrays.

        ``managed`` names the components the runtime places and checkpoints on
        its own (modules and their optimizers). Every other registered
        component -- learned coefficients, their optimizers, auxiliary modules
        -- plus the step counter is serialized here so a runtime checkpoint
        restores the algorithm completely. A component the configuration
        switched off (``None``, such as the temperature of a fixed entropy
        coefficient) has no state to carry, as in :meth:`save`.
        """
        state: dict[str, np.ndarray] = {
            "step_counter": np.asarray(self.step_counter, dtype=np.int64)
        }
        for name in self.component_names:
            if name in managed:
                continue
            component = getattr(self, name)
            if component is not None:
                state.update(_component_arrays(name, component))
        return state

    def load_learner_state(
        self, state: Mapping[str, np.ndarray], *, managed: Collection[str]
    ) -> None:
        """Restore :meth:`learner_state` in place (the rule :meth:`load` follows)."""
        for name in self.component_names:
            if name in managed:
                continue
            component = getattr(self, name)
            if component is not None:
                _restore_component_arrays(name, component, state)
        self.step_counter = int(state["step_counter"])

    def on_evaluation(self, success_rate: float) -> dict[str, float]:
        """Hook the trainer runs after each evaluation.

        Receives the evaluation's success rate; any metrics returned are logged
        under the ``eval`` prefix. The default reacts to nothing and returns no
        metrics.
        """
        return {}

    def target_update_count(self, env_steps: int) -> int | None:
        """Cumulative learner-update target after ``env_steps`` collected steps.

        The trainer sizes its next training burst from this value; ``None``
        (the default) keeps the collected-decision burst size.
        """
        return None

    def observe_rewards(
        self,
        rewards: Any,
        dones: Any,
        truncateds: Any,
        episode_ids: Any,
    ) -> None:
        """Hook the trainer runs with the decisions just stored, in buffer order, before replay sampling.

        Each decision comes with its reward, end flags and episode id; decisions
        of several environments interleave. The default ignores the stream.
        """

    def log_metrics(
        self,
        metrics: dict[str, float | Histogram],
        *,
        step: int,
        prefix: str | None = None,
    ) -> None:
        """Send one metrics dict to the injected logger; a no-op without one."""
        if self.logger is not None:
            self.logger.log(metrics, step=step, prefix=prefix)

    def load(
        self,
        load_dir: str,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        """Restore a checkpoint in the per-component layout (``<name>.pt`` + ``training_state.json``).

        This is the layout older runs wrote; the runtime loop writes the
        learner group's checkpoint instead (``utils.training_utils.load_checkpoint``
        reads both). Every registered component is restored with strict
        shapes; optimizers and the training-state tensors come along unless
        ``load_optimizer_state=False``.

        Module components are restored IN PLACE (``load_state_dict`` into the
        existing object) so that every external reference to them (the
        trainer's eval actor, rollout collectors) sees the loaded weights.
        Non-module components keep the replace semantics.
        """
        for component in self.component_names:
            if not hasattr(self, component):
                continue
            if not load_optimizer_state and (
                component.endswith(OPTIMIZER_COMPONENT_SUFFIX) or component in self.training_state_tensors
            ):
                continue
            self._load_component(component, load_dir)
        self._load_training_state(load_dir, load_optimizer_state=load_optimizer_state)

    def _load_training_state(
        self, load_dir: str, *, load_optimizer_state: bool
    ) -> None:
        """Restore the step counter and, with optimizer state, the training-state tensors."""
        with open(os.path.join(load_dir, TRAINING_STATE_FILE), "r") as f:
            training_state = json.load(f)
        self.step_counter = training_state["step"]
        if not load_optimizer_state:
            return
        for name in self.training_state_tensors:
            tensor = getattr(self, name)
            if training_state[name] is None or tensor is None:
                continue
            # Keep the existing storage and device. Replacing ``.data`` with a
            # CPU tensor breaks the live CUDA optimizer reference on the first
            # update of the coefficient after resume.
            with torch.no_grad():
                tensor.fill_(training_state[name])

    def _load_component(self, component: str, load_dir: str) -> None:
        """Restore one per-component file (see :meth:`load`)."""
        file_path = os.path.join(load_dir, f"{component}.pt")
        # The per-component layout pickles whole modules and optimizers, so full unpickling is required.
        loaded_component = torch.load(  # NOSONAR
            file_path, map_location="cpu", weights_only=False
        )

        current_component = getattr(self, component)
        if hasattr(loaded_component, "state_dict") and hasattr(
            current_component, "load_state_dict"
        ):
            current_component.load_state_dict(loaded_component.state_dict())
        elif (
            torch.is_tensor(loaded_component)
            and torch.is_tensor(current_component)
            and loaded_component.shape == current_component.shape
        ):
            # Plain-tensor components restore in place too: replacing the
            # object would orphan it from any optimizer that holds a
            # reference (silently freezing a learned coefficient on
            # resumed runs).
            with torch.no_grad():
                current_component.data.copy_(loaded_component.data)
        else:
            setattr(self, component, _to_device_of(loaded_component, current_component))


def _to_device_of(loaded_component: Any, current_component: Any) -> Any:
    """Move a replaced component onto the device of the one it replaces, if both have parameters."""
    if hasattr(loaded_component, "parameters") and hasattr(
        current_component, "parameters"
    ):
        try:
            device = next(current_component.parameters()).device
            loaded_component = loaded_component.to(device)
        except StopIteration:
            pass
    return loaded_component


def _component_arrays(name: str, component: Any) -> dict[str, np.ndarray]:
    """Serialize one component for :meth:`BaseAlgorithm.learner_state`."""
    state: dict[str, np.ndarray] = {}
    if isinstance(component, nn.Module):
        for key, value in component.state_dict().items():
            state[f"{name}/{key}"] = _as_array(value)
    elif torch.is_tensor(component):
        state[name] = _as_array(component)
    elif isinstance(component, torch.optim.Optimizer):
        for index, parameter_state in component.state_dict()["state"].items():
            for key, value in parameter_state.items():
                state[f"{name}/{index}/{key}"] = _as_array(value)
    else:
        raise TypeError(
            f"component {name!r} ({type(component).__name__}) cannot be serialized"
        )
    return state


def _restore_component_arrays(
    name: str, component: Any, state: Mapping[str, np.ndarray]
) -> None:
    """Restore one component for :meth:`BaseAlgorithm.load_learner_state`."""
    prefix = f"{name}/"
    if isinstance(component, nn.Module):
        component.load_state_dict(
            {
                key[len(prefix) :]: torch.as_tensor(value)
                for key, value in state.items()
                if key.startswith(prefix)
            },
            strict=True,
        )
    elif torch.is_tensor(component):
        if name not in state:
            # The checkpoint was written by a configuration without this
            # component (a fixed entropy coefficient carries no temperature).
            raise ValueError(
                f"the checkpoint has no {name!r} but this algorithm configuration "
                "trains one; evaluate or resume it with the algorithm settings it "
                "was trained with (e.g. the same algorithm.ent_coef)"
            )
        with torch.no_grad():
            component.copy_(torch.as_tensor(state[name]))
    elif isinstance(component, torch.optim.Optimizer):
        _restore_optimizer_arrays(component, prefix, state)
    else:
        raise TypeError(
            f"component {name!r} ({type(component).__name__}) cannot be restored"
        )


def _restore_optimizer_arrays(
    optimizer: torch.optim.Optimizer, prefix: str, state: Mapping[str, np.ndarray]
) -> None:
    restored: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        index_text, field_name = key[len(prefix) :].split("/", 1)
        restored.setdefault(int(index_text), {})[field_name] = torch.as_tensor(value)
    optimizer.load_state_dict(
        {
            "state": restored,
            "param_groups": optimizer.state_dict()["param_groups"],
        }
    )


def _as_array(value: Any) -> np.ndarray:
    return torch.as_tensor(value).detach().cpu().numpy()
