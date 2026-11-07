# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run RLPT from Python: ``launch(algorithm, env)`` returns a :class:`~isaac_rl_posttraining.run.Run`.

::

    from isaac_rl_posttraining import launch, train
    from isaac_rl_posttraining.algorithms import DSRL

    run = launch(DSRL(base="gr00t_n17", base_ckpt=ckpt), env="arena/drill_lift")
    run.loop()                      # or: train(DSRL(...), env=...)

Both compose ``train.yaml`` with the same overrides ``scripts/train.py`` would
get (the algorithm's and the environment's, then ``placement``, ``schedule``
and the caller's ``overrides``), so the Python and the command-line run are
the same run. The older ``train(env, policy, algorithm, ...)`` form still
works and warns.
"""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from isaac_rl_posttraining.algorithms.api import Algorithm, override_value
from isaac_rl_posttraining.algorithms.registry import registered_algorithms
from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.configs.normalize import normalize_train_config
from isaac_rl_posttraining.envs.providers import register_simulator_entry
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry
from isaac_rl_posttraining.policies import available_base_models, register_base_model
from isaac_rl_posttraining.policies.base import BaseModelPolicy
from isaac_rl_posttraining.run import Run
from isaac_rl_posttraining.runners.build import (
    build_run,
    selected_algorithm,
    train_plan,
)
from isaac_rl_posttraining.runners.run_placement import placement_overrides
from isaac_rl_posttraining.runtimes.placement import Placement
from isaac_rl_posttraining.training.recording import PersistSpec
from isaac_rl_posttraining.utils.config_utils import refuse_removed_overrides

__all__ = [
    "Env",
    "Run",
    "compose_train_config",
    "launch",
    "launch_overrides",
    "place_learner",
    "train",
    "train_overrides",
]

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


class Env:
    """The environments a run trains on.

    ``name`` is ``"<simulator>/<task>"`` (the simulator segment picks the
    registered entry). ``simulator`` selects the simulator config group when
    it differs from the default (``"libero_10"``); ``num_envs`` and
    ``max_episode_steps`` set ``training.num_envs`` and ``env.max_episode_steps``;
    ``options`` set ``env.<key>``.
    """

    def __init__(
        self,
        name: str,
        *,
        simulator: str | None = None,
        num_envs: int | None = None,
        max_episode_steps: int | None = None,
        **options: Any,
    ) -> None:
        self.name = name
        self.simulator = simulator
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.options = dict(options)

    def overrides(self) -> list[str]:
        """The overrides selecting these environments."""
        overrides = [] if self.simulator is None else [f"simulator={self.simulator}"]
        overrides.append(f"env.env_name={override_value(self.name)}")
        if self.num_envs is not None:
            overrides.append(f"training.num_envs={self.num_envs}")
        if self.max_episode_steps is not None:
            overrides.append(f"env.max_episode_steps={self.max_episode_steps}")
        return overrides + [
            f"env.{key}={override_value(value)}" for key, value in self.options.items()
        ]


EnvLike: TypeAlias = str | Env | SimulatorEntry | type[SimulatorEntry]
AlgorithmLike: TypeAlias = Algorithm | str


def _env_overrides(env: EnvLike) -> list[str]:
    if isinstance(env, Env):
        return env.overrides()
    if isinstance(env, str):
        return [f"env.env_name={override_value(env)}"]
    register_simulator_entry(env.name, env, override=True)
    return [f"env.env_name={override_value(env.name)}"]


def compose_train_config(overrides: Sequence[str]) -> DictConfig:
    """Compose ``train.yaml`` with ``overrides`` as ``scripts/train.py`` does, in its normalized form.

    Normalizing fills what the registry and the simulator group declare
    (``algorithm.family``, ``env.max_episode_steps``) and maps renamed keys,
    exactly as the run does before it reads the config.
    """
    refuse_removed_overrides(overrides)
    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return normalize_train_config(
            compose(config_name="train", overrides=list(overrides))
        )


def launch_overrides(
    algorithm: AlgorithmLike,
    env: EnvLike,
    runtime: str = "native",
    placement: Placement | None = None,
    schedule: str = "sync",
    max_policy_lag: int | None = None,
    overrides: Sequence[str] = (),
    persist: str | PersistSpec | None = None,
) -> list[str]:
    """The overrides :func:`launch` composes, in order (the caller's ``overrides`` last)."""
    algorithm = (
        algorithm if isinstance(algorithm, Algorithm) else Algorithm.named(algorithm)
    )
    composed = [*algorithm.overrides(), f"runtime={runtime}", *_env_overrides(env)]
    if placement is not None:
        composed += placement_overrides(placement)
    if schedule != "sync":
        composed.append(f"schedule.kind={schedule}")
    if max_policy_lag is not None:
        composed.append(f"schedule.max_policy_lag={max_policy_lag}")
    if persist is not None:
        spec = persist if isinstance(persist, PersistSpec) else PersistSpec(persist)
        composed += [
            f"data.persist.path={override_value(spec.path)}",
            f"data.persist.every_episodes={override_value(spec.every_episodes)}",
            f"data.persist.keep_frames={override_value(spec.keep_frames)}",
        ]
    return composed + list(overrides)


def launch(
    algorithm: AlgorithmLike,
    env: EnvLike,
    runtime: str = "native",
    placement: Placement | None = None,
    schedule: str = "sync",
    max_policy_lag: int | None = None,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    persist: str | PersistSpec | None = None,
) -> Run:
    """Plan a run and return its handle; nothing loads until the run is driven.

    Args:
        algorithm: An algorithm object (``DSRL(base=..., base_ckpt=...)``) or a registered name.
        env: ``"<simulator>/<task>"``, an :class:`Env`, or a ``SimulatorEntry`` (class or instance).
        runtime: A name ``runtimes.available_runtimes()`` lists (``native``, ``rlinf``, ...).
        placement: Which role runs where (``runtimes.placement.Placement``); ``None`` leaves it to the planner.
        schedule: ``sync``, ``overlapped`` or ``async``.
        max_policy_lag: Largest policy lag training data may have (``None``: the schedule's own).
        overrides: Further Hydra overrides, applied last (``training.num_rollouts=...``).
        dry_run: Plan only; the returned run refuses to start.
        persist: Save every completed training episode under this directory (relative: under the
            run directory) as a loadable replay buffer; a :class:`~isaac_rl_posttraining.training.PersistSpec`
            also sets the episodes per part and whether camera frames are kept (``data.persist.*``).

    Raises:
        ValueError: If the plan's checks fail (data specs, placement, schedule), as on the command line.
    """
    composed = launch_overrides(
        algorithm, env, runtime, placement, schedule, max_policy_lag, overrides, persist
    )
    return Run(compose_train_config(composed), dry_run=dry_run)


def train_overrides(
    env: str | SimulatorEntry | type[SimulatorEntry],
    policy: str | type[BaseModelPolicy],
    algorithm: str,
    runtime: str,
) -> list[str]:
    """The overrides of the older ``train(env, policy, algorithm, runtime)`` form."""
    if isinstance(env, str):
        env_name = env
    else:
        register_simulator_entry(env.name, env, override=True)
        env_name = env.name
    if isinstance(policy, str):
        base_model = f"base_model={policy}"
    else:
        name = f"{policy.__module__}.{policy.__qualname__}"
        if name not in available_base_models():
            register_base_model(name, f"{policy.__module__}:{policy.__qualname__}")
        base_model = f"base_model.name={name}"
    return [
        f"algorithm={algorithm}",
        f"runtime={runtime}",
        f"env.env_name={env_name}",
        base_model,
    ]


def place_learner(cfg: DictConfig, gpus: int) -> None:
    """Map a GPU count onto the learner placement of a composed config (the older ``gpus=`` argument).

    An on-policy learner runs on the GPU (CPU when ``gpus=0``), and under
    ``runtime=rlinf`` the rollout workers start after the learner's GPUs.
    """
    if gpus < 0:
        raise ValueError(f"gpus must be a non-negative count, got {gpus}")
    if selected_algorithm(cfg).loop == "on_policy":
        cfg.learner.device = "cpu" if gpus == 0 else "gpu"
    if str(cfg.runtime.name) == "rlinf":
        cfg.runtime.rlinf.first_accelerator = gpus


def _legacy_train(
    env: str | SimulatorEntry | type[SimulatorEntry],
    policy: str | type[BaseModelPolicy],
    algorithm: str,
    runtime: str = "native",
    gpus: int | None = None,
    overrides: Sequence[str] = (),
) -> None:
    cfg = compose_train_config(
        [*train_overrides(env, policy, algorithm, runtime), *overrides]
    )
    if gpus is not None:
        place_learner(cfg, gpus)
    # The same launch-plan checks the CLI runs before any policy or simulator loads.
    train_plan(cfg)
    build_run(cfg)()


def _is_legacy_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "policy" in kwargs or "gpus" in kwargs:
        return True
    first = args[0] if args else kwargs.get("algorithm")
    return not isinstance(first, Algorithm) and not (
        isinstance(first, str) and first in registered_algorithms()
    )


def train(*args: Any, **kwargs: Any) -> None:
    """Launch a run and run its loop to the budget: ``train(...) == launch(...).loop()``.

    Takes :func:`launch`'s arguments. The older form
    ``train(env, policy, algorithm, runtime="native", gpus=None, overrides=())``
    still works and warns.
    """
    if _is_legacy_call(args, kwargs):
        warnings.warn(
            "train(env, policy, algorithm, ...) is deprecated; use train(DSRL(base=policy), env=env, ...) "
            "or launch(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        _legacy_train(*args, **kwargs)
        return
    launch(*args, **kwargs).loop()


class _CallableModule(types.ModuleType):
    """``from isaac_rl_posttraining import train`` can yield this module; calling it calls :func:`train`."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        train(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
