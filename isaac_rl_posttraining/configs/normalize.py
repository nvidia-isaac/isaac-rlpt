# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bring a composed ``train`` config to its canonical form before it is validated.

Each step is idempotent: fill ``algorithm.family`` from the algorithm
registry; refuse ``algorithm.*`` keys the algorithm's config does not have;
map renamed keys through :data:`CONFIG_ALIASES` onto their canonical keys
(with a one-line deprecation note), which are the only ones the runs read; and turn explicit
``placement`` GPU lists into RLInf's first accelerators.
The ``rlpt.episode_cap`` resolver registered here gives ``env.max_episode_steps``
its default: the task's cap from the simulator group.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import algorithm_spec, load_target
from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

__all__ = [
    "CONFIG_ALIASES",
    "ConfigAlias",
    "check_algorithm_keys",
    "normalize_train_config",
    "register_resolvers",
    "resolved_without_old_keys",
    "task_episode_cap",
]

_MISSING = object()


def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class ConfigAlias:
    """An older key (``old``) that maps onto its canonical key (``new``).

    ``old_default`` is the old key's value when nobody set it. A set old key
    moves onto the new key as ``to_new(value)`` (a set old key is a deliberate
    override, since recipes no longer set it); readers read the new key only.
    ``deprecated=False`` marks a supported synonym, applied without a note.
    """

    old: str
    new: str
    old_default: Any = None
    to_new: Callable[[Any], Any] = _identity
    deprecated: bool = True


_SERVING_TO_MODE = {"auto": "auto", "worker_local": "local", "shared": "served"}
_RLINF_BATCH = 32
_RLINF_WAIT_SECONDS = 0.005

#: Every renamed or relocated key and where it lives now.
CONFIG_ALIASES: tuple[ConfigAlias, ...] = (
    ConfigAlias(
        "eval.eval_deterministic",
        "eval.action_selection",
        False,
        lambda value: "mean" if value else "sample",
    ),
    ConfigAlias("ppo.deterministic_evaluation", "ppo.deterministic_kernels", True),
    ConfigAlias(
        "runtime.rlinf.deterministic_evaluation",
        "runtime.rlinf.serve_evaluation_unbatched",
        True,
    ),
    ConfigAlias("runtime.num_rollout_workers", "placement.envs.workers"),
    ConfigAlias(
        "runtime.rlinf.policy_serving",
        "placement.policy.mode",
        "auto",
        lambda value: _SERVING_TO_MODE.get(value, value),
    ),
    ConfigAlias("runtime.rlinf.num_policy_servers", "placement.policy.replicas"),
    ConfigAlias(
        "runtime.rlinf.policy_batch_size",
        "placement.policy.max_batch",
        _RLINF_BATCH,
        lambda value: None if value == _RLINF_BATCH else value,
    ),
    ConfigAlias(
        "runtime.rlinf.policy_batch_wait_seconds",
        "placement.policy.max_wait_ms",
        _RLINF_WAIT_SECONDS,
        lambda value: None if value == _RLINF_WAIT_SECONDS else float(value) * 1000.0,
    ),
)

_noted: set[str] = set()


def _select(cfg: DictConfig, key: str) -> Any:
    return OmegaConf.select(cfg, key, default=_MISSING, throw_on_missing=False)


def _apply_aliases(cfg: DictConfig) -> None:
    for alias in CONFIG_ALIASES:
        old, new = _select(cfg, alias.old), _select(cfg, alias.new)
        if old is _MISSING or new is _MISSING or old == alias.old_default:
            continue
        value = alias.to_new(old)
        if value == new:
            continue
        if alias.deprecated and alias.old not in _noted:
            _noted.add(alias.old)
            logger.warning(f"{alias.old} is deprecated; use {alias.new}={value} (applied)")
        OmegaConf.update(cfg, alias.new, value, force_add=True)


def check_algorithm_keys(algorithm: Mapping[str, Any], name: str) -> None:
    """Refuse ``algorithm.*`` keys the algorithm's config does not declare, before anything loads."""
    spec = algorithm_spec(name)
    declared = {"name", "family"}
    if spec.config is not None:
        declared |= {
            field.name
            for field in dataclasses.fields(load_target(spec.config).yaml_schema())
        }
    unknown = sorted(set(algorithm) - declared)
    if unknown:
        settings = sorted(declared - {"name", "family"})
        where = (
            f"its settings are {settings}"
            if settings
            else f"its settings live in {', '.join(spec.sections)}"
        )
        raise RemovedConfigKeyError(
            f"config keys this version does not accept: {', '.join(f'algorithm.{key}' for key in unknown)}: "
            f"not settings of algorithm {name!r}; {where}"
        )


def _fill_family(cfg: DictConfig) -> None:
    name = _select(cfg, "algorithm.name")
    if name is _MISSING or name is None:
        return
    spec = algorithm_spec(str(name))
    family = _select(cfg, "algorithm.family")
    if family not in (_MISSING, None, "???") and family != spec.family:
        raise ValueError(
            f"algorithm.family={family!r} disagrees with the registry ({name} is {spec.family!r}); "
            "the registry declares it, so drop algorithm.family"
        )
    OmegaConf.update(cfg, "algorithm.family", spec.family, force_add=True)
    algorithm = OmegaConf.to_container(cfg.algorithm, resolve=False)
    assert isinstance(algorithm, dict)
    check_algorithm_keys(algorithm, str(name))


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def task_episode_cap(caps: Any, env_name: str) -> int:
    """The episode cap of ``env_name``'s task from a simulator's ``max_episode_steps``.

    ``caps`` is one number for every task, or a map keyed by task id. The task
    is the part of ``env_name`` after ``/``; it matches a key exactly, or the
    one key it ends (ignoring case and separators: ``drill_lift`` is ``LMDrillLift``).
    """
    if isinstance(caps, int) and not isinstance(caps, bool):
        return caps
    if not isinstance(caps, Mapping) or not caps:
        raise ValueError(
            "the simulator group declares no max_episode_steps; set env.max_episode_steps "
            "(or select the simulator group with simulator=<group>)"
        )
    task = env_name.split("/", 1)[1] if "/" in env_name else ""
    if task in caps:
        return int(caps[task])
    matches = [
        key for key in caps if task and _squash(str(key)).endswith(_squash(task))
    ]
    if len(matches) == 1:
        return int(caps[matches[0]])
    if len(caps) == 1:
        return int(next(iter(caps.values())))
    raise ValueError(
        f"env.max_episode_steps is unset and task {task!r} of env.env_name={env_name!r} names none of "
        f"the simulator's tasks {sorted(caps)}; set env.max_episode_steps or env.env_name=<simulator>/<task>"
    )


def _episode_cap_resolver(caps: Any, env_name: str) -> int:
    if isinstance(caps, DictConfig):
        caps = OmegaConf.to_container(caps, resolve=True)
    return task_episode_cap(caps, str(env_name))


def register_resolvers() -> None:
    """Register ``${rlpt.episode_cap:<caps>,<env_name>}``, which ``train.yaml``'s episode cap reads (idempotent)."""
    OmegaConf.register_new_resolver(
        "rlpt.episode_cap", _episode_cap_resolver, replace=True
    )


def _first_of_contiguous(devices: Any, key: str) -> int:
    indices = []
    for device in devices:
        node, _, index = str(device).rpartition(":")
        if node not in ("", "0", "node0") or not index.isdigit():
            raise ValueError(
                f"runtime=rlinf places {key} on node 0 by index; got {device!r}"
            )
        indices.append(int(index))
    if not indices or sorted(indices) != list(
        range(min(indices), min(indices) + len(indices))
    ):
        raise ValueError(
            f"runtime=rlinf needs {key} to be contiguous GPUs, got {list(devices)}"
        )
    return min(indices)


def _lower_rlinf_gpus(cfg: DictConfig) -> None:
    """Explicit learner and rollout-worker GPU lists become RLInf's first accelerators."""
    if _select(cfg, "runtime.name") != "rlinf":
        return
    for key, target in (
        ("placement.learner.gpus", "runtime.rlinf.learner_first_accelerator"),
        ("placement.envs.gpus", "runtime.rlinf.first_accelerator"),
    ):
        devices = _select(cfg, key)
        if devices is _MISSING or devices is None or isinstance(devices, int):
            continue
        OmegaConf.update(cfg, target, _first_of_contiguous(devices, key))


def normalize_train_config(cfg: DictConfig) -> DictConfig:
    """A canonical copy of a composed ``train`` config (``cfg`` itself is left as it is)."""
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)
    _fill_family(cfg)
    _apply_aliases(cfg)
    _lower_rlinf_gpus(cfg)
    return cfg


def resolved_without_old_keys(cfg: DictConfig) -> dict[str, Any]:
    """``cfg`` normalized and resolved, without the deprecated spellings the run never reads (what the dry run prints)."""
    resolved = OmegaConf.to_container(normalize_train_config(cfg), resolve=True)
    assert isinstance(resolved, dict)
    for alias in CONFIG_ALIASES:
        if not alias.deprecated:
            continue
        *parents, leaf = alias.old.split(".")
        section: Any = resolved
        for key in parents:
            section = section.get(key) if isinstance(section, dict) else None
        if isinstance(section, dict):
            section.pop(leaf, None)
    return resolved
