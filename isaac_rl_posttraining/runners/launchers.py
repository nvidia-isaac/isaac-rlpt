# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launcher of each training loop: what its run will do, and the run itself.

A registered algorithm names its launcher (``algorithms/registry.py``); the
entry point calls :meth:`Launcher.plan` for the dry run and
:meth:`Launcher.build` for the run, so a new algorithm with its own loop
brings its own launcher and routes without core edits.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from omegaconf import DictConfig

from isaac_rl_posttraining.configs import evaluation_enabled
from isaac_rl_posttraining.configs.schema import PPOTrainingConfig
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry, requested_layout
from isaac_rl_posttraining.runners import (
    dsrl_ppo_launch,
    flow_sde_ppo_launch,
    off_policy_launch,
    offline_launch,
    run_schedule,
)
from isaac_rl_posttraining.runners.ppo_launch import (
    ppo_settings,
    rank_minibatch_size,
    reject_flow_sde_keys,
)
from isaac_rl_posttraining.runners.run_placement import native_runs_processes
from isaac_rl_posttraining.runners.simulator_entries import (
    environment_spec,
    exported_server_url,
    require_base_model_support,
    require_evaluation_shape,
    require_simulator_entry,
    resolve_simulator_entry,
    shares_training_workers,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.buffers import ReplayBuffer
    from isaac_rl_posttraining.runners.off_policy_session import OffPolicySession
    from isaac_rl_posttraining.runners.on_policy_session import OnPolicySession
    from isaac_rl_posttraining.runtimes import PolicyState
    from isaac_rl_posttraining.training.telemetry import ParquetTelemetrySink

__all__ = [
    "Launcher",
    "LoopPlan",
    "StepSession",
    "dsrl_ppo",
    "entry_layout",
    "flow_sde_ppo",
    "off_policy",
    "offline",
]


@dataclass(frozen=True)
class LoopPlan:
    """What a loop's launcher resolves from the config alone.

    ``batched_workers``: each rollout worker drives several environments in
    one simulator (a ``BatchedWorkerSpec``). ``served_policy``: the rollout
    workers can act through policy-server replicas
    (``placement.policy.mode=served``) instead of a copy of their own.
    ``rollout_workers``: how many training rollout workers the run starts
    (``None``: the launcher does not say; the placement plan decides).
    """

    entry: SimulatorEntry
    layout: str
    evaluation_mode: str
    budget: str
    learner_ranks: int
    batched_workers: bool = False
    served_policy: bool = True
    rollout_workers: int | None = None


class StepSession(Protocol):
    """A run opened for step-wise control; :class:`~isaac_rl_posttraining.run.Run` drives it.

    ``train(steps)`` returns the last update's metrics and publishes the new
    weights; ``buffer`` raises for a loop without a replay buffer.
    """

    run_dir: str

    @property
    def telemetry(self) -> ParquetTelemetrySink | None: ...

    @property
    def buffer(self) -> ReplayBuffer: ...

    def collect(self, rounds: int = 1) -> Mapping[str, float]: ...

    def train(self, steps: int) -> Mapping[str, float]: ...

    def evaluate(self) -> Mapping[str, float]: ...

    def publish(self) -> PolicyState: ...

    def save(self, directory: str) -> None: ...

    def load(self, directory: str) -> None: ...

    def close(self) -> None: ...


class Launcher(Protocol):
    """Plans and builds the run of one training loop from a validated config.

    A launcher may also offer ``open_session(cfg)`` returning a
    :class:`StepSession`, the run opened for step-wise control.
    """

    def plan(self, cfg: DictConfig) -> LoopPlan: ...

    def build(self, cfg: DictConfig) -> Callable[[], str | None]:
        """The run as a no-argument callable; it returns the run directory (``None`` when it has none)."""
        ...


def training_workers(
    entry: SimulatorEntry, num_envs: int, num_servers: int, workers: int | None = None
) -> int:
    """How many rollout workers the run starts: ``placement.envs.workers``, else the entry's layout."""
    if workers is not None:
        return int(workers)
    return requested_layout(entry, num_envs, num_servers=num_servers).workers


def entry_layout(
    entry: SimulatorEntry, num_envs: int, num_servers: int, workers: int | None = None
) -> str:
    """``<workers> rollout worker(s) x <envs> environment(s)`` for the plan."""
    layout = requested_layout(entry, num_envs, num_servers=num_servers, workers=workers)
    count = training_workers(entry, num_envs, num_servers, workers)
    environments_per_worker = num_envs // count
    batched = ", batched" if layout.batched else ""
    servers = f" on {num_servers} simulator servers" if num_servers > 1 else ""
    return f"{count} rollout worker(s) x {environments_per_worker} environment(s){batched}{servers}"


def _batched(cfg: DictConfig, entry: SimulatorEntry, workers: int | None) -> bool:
    """Whether the steering families' rollout workers each drive several environments in one simulator."""
    return requested_layout(
        entry,
        int(cfg.training.num_envs),
        num_servers=_training_servers(cfg, entry),
        workers=workers,
    ).batched


def _training_servers(cfg: DictConfig, entry: SimulatorEntry) -> int:
    """How many servers the launcher exported for the training environments.

    One for an in-process simulator, and one when the address is not exported
    yet (the plan names it as a launch input; the launch requires it).
    """
    if entry.server_url_variables is None or int(cfg.training.num_rollouts) == 0:
        return 1
    return environment_spec(
        cfg,
        num_envs=int(cfg.training.num_envs),
        server_url=exported_server_url(entry, "train"),
    ).num_servers


def _require_off_policy_learner(cfg: DictConfig) -> None:
    """The off-policy learner trains in the controller process, or on ranks of a runtime with a distributed learner."""
    distributed = str(cfg.runtime.name) == "rlinf" or native_runs_processes(cfg)
    if int(cfg.placement.learner.ranks) != 1 and not distributed:
        raise ValueError(
            f"{cfg.algorithm.name} trains in the controller process on runtime {cfg.runtime.name!r}; "
            f"placement.learner.ranks must be 1 (got {cfg.placement.learner.ranks}), "
            "or select runtime=rlinf (or runtime=native with runtime.native.processes=auto|ray) "
            "for several learner ranks"
        )
    run_schedule.check_run_config(cfg, on_policy=False)
    if str(cfg.learner.device) != "auto":
        raise ValueError(
            "learner.device applies to the on-policy learner group; the off-policy learner uses "
            "CUDA when available (keep learner.device=auto)"
        )


def _evaluation_mode(cfg: DictConfig, entry: SimulatorEntry) -> str:
    """The entry's evaluation mode as this run uses it: a shared entry that cannot share evaluates on its own workers."""
    if not evaluation_enabled(cfg):
        return "disabled"
    if entry.evaluation == "shared" and not shares_training_workers(cfg, entry):
        return "dedicated"
    return entry.evaluation


class _OffPolicy:
    """The replay loop of the steering and edit algorithms (DSRL, RFS, SAC-edit)."""

    def plan(self, cfg: DictConfig) -> LoopPlan:
        _require_off_policy_learner(cfg)
        entry = require_simulator_entry(cfg)
        servers = _training_servers(cfg, entry)
        workers = cfg.placement.envs.workers
        num_envs = int(cfg.training.num_envs)
        layout = entry_layout(entry, num_envs, servers, workers)
        evaluation_mode = _evaluation_mode(cfg, entry)
        eval_freq = (
            cfg.eval.eval_freq
            if cfg.eval.eval_freq is not None
            else int(cfg.training.num_rollouts) // 100
        )
        budget = (
            f"{cfg.training.num_rollouts} env steps, learning_starts={cfg.algorithm.learning_starts}, "
            f"eval_freq={eval_freq} ({cfg.eval.eval_num_episodes} episodes), "
            f"save_interval={cfg.training.save_interval}, retention={cfg.training.checkpoint_retention}"
        )
        # steering_worker_spec lets both worker kinds act through a policy server.
        return LoopPlan(
            entry,
            layout,
            evaluation_mode,
            budget,
            int(cfg.placement.learner.ranks),
            batched_workers=_batched(cfg, entry, workers),
            rollout_workers=training_workers(entry, num_envs, servers, workers),
        )

    def build(self, cfg: DictConfig) -> Callable[[], str | None]:
        _require_off_policy_learner(cfg)
        require_evaluation_shape(cfg, resolve_simulator_entry(cfg))
        return functools.partial(off_policy_launch.run, cfg)

    def open_session(self, cfg: DictConfig) -> OffPolicySession:
        return off_policy_launch.open_session(cfg)


def _on_policy_budget(ppo: PPOTrainingConfig, learner_ranks: int) -> str:
    if ppo.eval_only:
        source = (
            ppo.resume_from if ppo.resume_from is not None else "the base checkpoint"
        )
        return f"evaluation only: {ppo.eval_episodes_per_environment} episode(s) per environment of {source}"
    rank_minibatch = rank_minibatch_size(ppo, learner_ranks)
    micro_batch = (
        rank_minibatch if ppo.micro_batch_size is None else int(ppo.micro_batch_size)
    )
    return (
        f"{ppo.iterations} iterations x {ppo.episodes_per_environment} episode(s) per environment, "
        f"eval_every={ppo.eval_every} ({ppo.eval_episodes_per_environment} episode(s) per environment), "
        f"checkpoint_every={ppo.checkpoint_every}, retention={ppo.checkpoint_retention}, "
        f"global batch {ppo.global_batch_size} (rank minibatch {rank_minibatch}, micro {micro_batch})"
    )


class _FlowSdePPO:
    """PPO over a trainable base model's flow head (GR00T flow-SDE)."""

    def plan(self, cfg: DictConfig) -> LoopPlan:
        run_schedule.check_run_config(cfg, on_policy=True)
        plan = flow_sde_ppo_launch.build_plan(cfg, require_servers=False)
        layout = (
            f"{plan.num_rollout_workers} rollout worker(s) x {plan.envs_per_worker} scene(s), "
            f"{plan.max_episode_steps}-step episodes"
        )
        workers = flow_sde_ppo_launch.rollout_worker_spec(plan)
        # The flow-SDE workers evaluate on the training workers with a deterministic decode.
        return LoopPlan(
            plan.entry,
            layout,
            "shared",
            _on_policy_budget(plan.ppo, plan.num_learner_ranks),
            plan.num_learner_ranks,
            batched_workers=True,
            served_policy=workers.supports_served_policy,
            rollout_workers=plan.num_rollout_workers,
        )

    def build(self, cfg: DictConfig) -> Callable[[], str | None]:
        return functools.partial(
            flow_sde_ppo_launch.run, cfg, flow_sde_ppo_launch.build_plan(cfg)
        )

    def open_session(self, cfg: DictConfig) -> OnPolicySession:
        return flow_sde_ppo_launch.open_session(cfg, flow_sde_ppo_launch.build_plan(cfg))


class _DsrlPPO:
    """PPO over the steering actor of a frozen base model."""

    def plan(self, cfg: DictConfig) -> LoopPlan:
        run_schedule.check_run_config(cfg, on_policy=True)
        dsrl_ppo_launch.reject_off_policy_eval_keys(cfg)
        ppo = ppo_settings(cfg)
        reject_flow_sde_keys(ppo)
        learner_ranks = int(cfg.placement.learner.ranks)
        entry = resolve_simulator_entry(cfg)
        require_base_model_support(cfg, entry)
        servers = _training_servers(cfg, entry)
        workers = cfg.placement.envs.workers
        num_envs = int(cfg.training.num_envs)
        layout = entry_layout(entry, num_envs, servers, workers)
        evaluation_mode = (
            "fresh_workers" if ppo.isolate_evaluation_workers else "shared"
        )
        return LoopPlan(
            entry,
            layout,
            evaluation_mode,
            _on_policy_budget(ppo, learner_ranks),
            learner_ranks,
            batched_workers=_batched(cfg, entry, workers),
            rollout_workers=training_workers(entry, num_envs, servers, workers),
        )

    def build(self, cfg: DictConfig) -> Callable[[], str | None]:
        return functools.partial(dsrl_ppo_launch.run, cfg)

    def open_session(self, cfg: DictConfig) -> OnPolicySession:
        return dsrl_ppo_launch.open_session(cfg)


class _Offline:
    """The demonstration loop of the offline algorithms (IQL, SFT): no collection, one learner in this process."""

    def plan(self, cfg: DictConfig) -> LoopPlan:
        entry, layout, evaluation_mode, budget = offline_launch.describe(cfg)
        return LoopPlan(entry, layout, evaluation_mode, budget, 1, served_policy=False)

    def build(self, cfg: DictConfig) -> Callable[[], str | None]:
        offline_launch.describe(cfg)
        return functools.partial(offline_launch.run, cfg)


off_policy: Launcher = _OffPolicy()
offline: Launcher = _Offline()
flow_sde_ppo: Launcher = _FlowSdePPO()
dsrl_ppo: Launcher = _DsrlPPO()
