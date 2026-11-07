# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launch plan's checks: run them all, collect every problem, report them together.

A dry run is meant to work like a type check: each check compares two
things the config states (the simulator's observation with the policy's
inputs, the GPUs one role asks for with another's, the evaluation episodes
with the evaluation environments) and either passes, fails with a message
that names the keys to change, or stays unverified because one side is known
only once something loads. :class:`PlanChecks` runs the checks one after the
other so a failure does not hide the next one, and summarizes them in the
``plan.checks`` line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.configs import evaluation_enabled
from isaac_rl_posttraining.data.spec import SpecComparison, SpecMismatch
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry, requested_layout
from isaac_rl_posttraining.runners.simulator_entries import (
    required_phases,
    shares_training_workers,
)
from isaac_rl_posttraining.runtimes.base import RuntimeFeatures
from isaac_rl_posttraining.runtimes.placement import PlacementPlan

if TYPE_CHECKING:
    from isaac_rl_posttraining.runners.launchers import LoopPlan

__all__ = [
    "PlanCheckError",
    "PlanChecks",
    "check_base_model_servable",
    "check_base_model_steerable",
    "check_learner_budget",
    "check_rollout_workers",
    "check_runtime_installed",
    "check_served_policy",
    "check_server_replicas",
    "native_eval_only",
    "native_only_error",
]

T = TypeVar("T")


class PlanCheckError(ValueError):
    """Several launch-plan checks failed; the message lists every one."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        listed = "\n".join(
            f"  {number}. {error}" for number, error in enumerate(self.errors, start=1)
        )
        super().__init__(
            f"the launch plan found {len(self.errors)} problems:\n{listed}"
        )


class PlanChecks:
    """What the plan checked, what it could not verify (with the reason), and what failed."""

    def __init__(self) -> None:
        self.checked: list[str] = []
        self.unverified: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[Exception] = []

    def run(self, name: str, check: Callable[[], T]) -> T | None:
        """Run one check; a ``ValueError`` it raises is recorded and ``None`` returned."""
        try:
            result = check()
        except ValueError as error:
            # Two checks that read the same broken setting report it once.
            if all(str(error) != str(seen) for seen in self.errors):
                self.errors.append(error)
            return None
        self.checked.append(name)
        return result

    def add(self, comparison: SpecComparison, context: str) -> None:
        """Record a data-spec comparison: its checks, its unverified sizes and, as one error, its mismatches."""
        self.checked.extend(comparison.checks)
        self.unverified.extend(comparison.unverified)
        if comparison.mismatches:
            self.errors.append(
                SpecMismatch(f"{context}: " + "; ".join(comparison.mismatches))
            )

    def summary(self) -> str:
        """The ``plan.checks`` line: counts, then every unverified item and every error."""
        parts = [
            f"{len(self.checked)} checked",
            f"{len(self.unverified)} unverified",
            f"{len(self.errors)} error(s)",
        ]
        line = ", ".join(parts)
        if self.unverified:
            line += " | unverified: " + "; ".join(dict.fromkeys(self.unverified))
        if self.warnings:
            line += " | warnings: " + "; ".join(self.warnings)
        if self.errors:
            line += " | errors: " + "; ".join(str(error) for error in self.errors)
        return line

    def raise_errors(self) -> None:
        """Raise the one failure as it was raised, or every failure together."""
        if len(self.errors) == 1:
            raise self.errors[0]
        if self.errors:
            raise PlanCheckError(self.errors) from self.errors[0]


def check_rollout_workers(cfg: DictConfig) -> None:
    """Every rollout worker of ``placement.envs.workers`` owns at least one of ``training.num_envs``.

    How the environments split beyond that is the simulator entry's to say
    (Isaac Lab needs an even split; one environment per worker splits any way).
    """
    workers = OmegaConf.select(cfg, "placement.envs.workers")
    if workers is None or int(cfg.training.num_rollouts) == 0:
        return
    if int(workers) > int(cfg.training.num_envs):
        raise ValueError(
            f"placement.envs.workers={int(workers)} exceeds training.num_envs={int(cfg.training.num_envs)}; "
            "every rollout worker needs at least one environment"
        )


def check_learner_budget(cfg: DictConfig) -> str | None:
    """A warning when an off-policy budget can end before its first learner update; ``None`` otherwise.

    Updates start once ``algorithm.learning_starts`` env steps are collected;
    with ``training.train_after_episode`` they wait for the next episode end,
    which can take one episode per environment more (a whole round when
    rollouts are counted by episode).
    """
    budget = int(cfg.training.num_rollouts)
    starts = int(cfg.algorithm.learning_starts)
    if budget == 0:
        return None
    if budget <= starts:
        return (
            f"training.num_rollouts={budget} ends before algorithm.learning_starts={starts}, so the learner never "
            "updates; raise training.num_rollouts or lower algorithm.learning_starts"
        )
    episode_steps = OmegaConf.select(cfg, "env.max_episode_steps")
    if not bool(cfg.training.train_after_episode) or episode_steps is None:
        return None
    num_envs = int(cfg.training.num_envs)
    latest = starts + num_envs * int(episode_steps)
    if budget >= latest:
        return None
    return (
        f"training.num_rollouts={budget} may end before the first learner update, which can come as late as "
        f"{latest} env steps (algorithm.learning_starts={starts} plus one episode per environment, "
        f"{num_envs} x {int(episode_steps)}); raise training.num_rollouts to {latest} to be sure of one"
    )


def check_server_replicas(
    cfg: DictConfig, plan: PlacementPlan, entry: SimulatorEntry
) -> None:
    """An explicit replica count leaves no policy server without an environment to serve.

    The servers answer the training environments and the evaluation
    environments that do not share the training workers. A count the planner
    chose is not checked here; the planner sizes it from the GPUs.
    """
    phases = required_phases(cfg)
    environments = int(cfg.training.num_envs) if "train" in phases else 0
    if "eval" in phases and not shares_training_workers(cfg, entry):
        environments += int(cfg.eval.eval_num_envs)
    for role in plan.services:
        section = (
            "placement.parts.base" if role.parts == ("base",) else "placement.policy"
        )
        if (
            OmegaConf.select(cfg, f"{section}.replicas") is None
            or role.processes <= environments
        ):
            continue
        raise ValueError(
            f"{section}.replicas={role.processes} policy servers of {'+'.join(role.parts)} for {environments} "
            f"environment(s) (training and evaluation) leave servers idle; set {section}.replicas<={environments} "
            "or raise training.num_envs"
        )


def check_served_policy(
    algorithm: str, loop: LoopPlan, plan: PlacementPlan, features: RuntimeFeatures
) -> None:
    """A served policy needs rollout workers that can act through a policy server, on a runtime that serves them.

    The workers' spec must accept a served policy and the runtime must serve
    batched workers when they are batched. Only a placement that serves the
    whole policy is checked here; serving the base model alone is checked by
    the planner against the policy's parts.
    """
    from isaac_rl_posttraining.runtimes.serving.engine import SERVING_ENGINES

    for role in plan.services:
        served = role.served
        engine = SERVING_ENGINES.get(served.engine) if served is not None else None
        if (
            served is not None
            and engine is not None
            and served.gpus_per_replica > engine.max_gpus_per_replica
        ):
            raise ValueError(
                f"placement.policy.gpus_per_replica={served.gpus_per_replica}, but the {served.engine} engine runs "
                f"a replica on {engine.max_gpus_per_replica} GPU (it does not shard the model); "
                "set placement.policy.gpus_per_replica=1, or use more replicas"
            )
    if not plan.services or plan.base_served_alone:
        return
    if not loop.served_policy:
        raise ValueError(
            f"placement.policy.mode=served, but the rollout workers of algorithm={algorithm} keep a "
            "copy of the policy each (their worker spec does not support a served policy); "
            "set placement.policy.mode=local"
        )
    if loop.batched_workers and not features.served_batched_collectors:
        raise ValueError(
            f"placement.policy.mode=served, but runtime={features.name} cannot serve rollout workers "
            f"that drive several environments in one simulator ({loop.layout}); "
            "set placement.policy.mode=local or select runtime=native"
        )


def check_base_model_servable(cfg: DictConfig, plan: PlacementPlan) -> str | None:
    """A served placement needs a base model that declares it may be served (``BaseModelPolicy.servable``).

    A stateful single-session base model (DreamZero's served-model client, whose
    ``start_episode()`` takes no session) keeps one episode per instance, so
    several rollout workers cannot share it. Returns a note when the base model's
    class does not import here, so the check stays unverified.
    """
    from isaac_rl_posttraining.policies import base_model_servable
    from isaac_rl_posttraining.runners.run_placement import placement_config_of
    from isaac_rl_posttraining.runtimes import policy_serving_mode

    # Only a placement that asks for serving: auto shares an in-process server, which one session can use.
    if not plan.services or policy_serving_mode(placement_config_of(cfg))[0] not in (
        "served",
        "base",
    ):
        return None

    name = str(OmegaConf.select(cfg, "base_model.name", default=""))
    if not name:
        return None
    servable = base_model_servable(name)
    if servable is None:
        return (
            f"whether base model {name} may be served: its class does not import here"
        )
    if not servable:
        section = (
            "placement.parts.base" if plan.base_served_alone else "placement.policy"
        )
        raise ValueError(
            f"base_model={name} keeps one episode's state per instance, so rollout workers cannot share it "
            f"through a policy server; set {section}.mode=local"
        )
    return None


def evaluation_group_workers(
    cfg: DictConfig, entry: SimulatorEntry, loop: str
) -> int | None:
    """Rollout workers of an evaluation group open next to the training group (``None``: none is).

    Off-policy: the evaluator's own workers unless it evaluates on the training
    workers. A run that only evaluates opens no training group, and on-policy
    loops close the training workers before evaluation workers replace them.
    """
    if loop != "off_policy" or "train" not in required_phases(cfg):
        return None
    if not evaluation_enabled(cfg) or shares_training_workers(cfg, entry):
        return None
    return requested_layout(entry, int(cfg.eval.eval_num_envs)).workers


def check_runtime_installed(cfg: DictConfig) -> str | None:
    """Why the selected runtime cannot load on this machine, or ``None`` when it can.

    A dry run may plan on a machine without the runtime's extras, so the plan
    lists the reason as unverified; the launch requires it before anything loads.
    """
    runtime = str(cfg.runtime.name)
    if runtime == "rlinf":
        from isaac_rl_posttraining.compat.rlinf import is_available

        if not is_available():
            return "runtime=rlinf needs the rlinf package, which is not installed; install the [rlinf] extra (uv sync --extra rlinf)"
    return None


def native_eval_only(cfg: DictConfig) -> bool:
    """An eval-only run of the frozen policy: no training rollouts, the base policy evaluated."""
    return int(cfg.training.num_rollouts) == 0 and bool(
        OmegaConf.select(cfg, "eval.eval_base", default=False)
    )


def native_only_error(base_model_name: str) -> ValueError:
    """The refusal of a training run on a base model that cannot take steering latents."""
    return ValueError(
        f"base_model={base_model_name} is native-only: it cannot take steering latents, so it cannot be trained. "
        "Select a base model that can be steered, or evaluate the frozen policy with "
        "training.num_rollouts=0 eval.eval_base=true eval.eval_on_first_step=true"
    )


def check_base_model_steerable(cfg: DictConfig) -> str | None:
    """An off-policy training run needs a base model that takes steering latents (``BaseModelPolicy.supports_steering``).

    Returns a note when the base model's class does not say without loading, so the check stays unverified.
    """
    from isaac_rl_posttraining.policies import base_model_steerable

    name = str(OmegaConf.select(cfg, "base_model.name", default=""))
    if not name or native_eval_only(cfg):
        return None
    steerable = base_model_steerable(name)
    if steerable is None:
        return f"whether base model {name} takes steering latents: its class does not say without loading"
    if not steerable:
        raise native_only_error(name)
    return None
