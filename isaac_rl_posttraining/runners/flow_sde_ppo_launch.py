# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run PPO fine-tuning of a flow-matching policy checkpoint on an Isaac Lab task.

RLinf's stack-cube recipe (``configs/algorithm/ppo.yaml``) over this
repository's runtime layer: learner ranks train the checkpoint's action and
value heads, rollout workers each host one Isaac Lab simulation with several
scenes and a copy of the model, and the frozen part of the model (a
vision-language backbone) stays frozen. The on-policy controller runs in this
process. ``base_model.name`` selects the trainable adapter -- GR00T N1.5
(``gr00t_n15``, the validated reference), GR00T N1.7 (``gr00t_n17``) or pi0
(``pi0``) -- through :mod:`isaac_rl_posttraining.policies.trainable`; the rest
of the run is the same.

    scripts/train.py algorithm=ppo base_model.ckpt=/path/to/RLinf-Gr00t-SFT-Stack-cube

runs the recipe's placement (``runtime=rlinf``: four FSDP learner ranks and
four rollout workers of 16 scenes on one node), while

    scripts/train.py algorithm=ppo base_model.ckpt=... \\
        runtime=native placement.learner.ranks=1 placement.envs.workers=1 training.num_envs=16

keeps the learner and one rollout worker on a single GPU. ``ppo.eval_only=true
placement.learner.ranks=0`` scores the base checkpoint as the workers load it (or the
policy snapshot of ``ppo.resume_from=<iteration dir>``) without a learner. Every checkpoint
directory (``iteration_<n>`` and ``final``) holds the learner state
(``learner.pt``, one file per rank), a ``policy_state.pt`` snapshot of the
published weights and a ``manifest.json``; ``evaluation_summary.json`` next to
them keeps every evaluation. Metrics go to ``ppo_metrics.jsonl`` in the run
directory and to Weights & Biases.
``ISAAC_RL_POSTTRAINING_DRYRUN=1`` prints the resolved plan and exits before
Isaac Sim or GR00T is imported.
"""

from __future__ import annotations

import functools
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.ppo import PPO
from isaac_rl_posttraining.configs import (
    RuntimeConfig,
    TrainConfig,
    selected_algorithm_name,
)
from isaac_rl_posttraining.configs.schema import PPOTrainingConfig
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    SimulatorEntry,
    requested_layout,
)
from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger, wandb_run_dir
from isaac_rl_posttraining.policies.gr00t.flow_sde_policy import FlowSDEPolicySpec
from isaac_rl_posttraining.policies.trainable import (
    FlowSDEConfig,
    TrainablePolicySpec,
    get_trainable_policy,
    resolve_trainable_policy_name,
)
from isaac_rl_posttraining.runners.off_policy_launch import new_run_output_dir, sigterm_as_interrupt
from isaac_rl_posttraining.runners.on_policy_session import OnPolicySession
from isaac_rl_posttraining.runners.ppo_launch import (
    checkpoint_store,
    close_resources,
    evaluation_sink,
    local_learner_device,
    metric_sink,
    on_policy_config_from,
    open_on_policy_source,
    ppo_config_from,
    rank_minibatch_size,
    reject_steering_keys,
)
from isaac_rl_posttraining.runners.run_placement import (
    RolloutShape,
    native_runs_processes,
    open_runtime,
)
from isaac_rl_posttraining.runners.run_schedule import schedule_spec
from isaac_rl_posttraining.runners.simulator_entries import (
    exported_server_url,
    phase_server_url,
    require_base_model_support,
    resolve_simulator_entry,
)
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    SparseOutcomeReward,
    load_policy_snapshot,
    open_learner_group,
)
from isaac_rl_posttraining.runtimes.watchdog import on_frozen_rollout
from isaac_rl_posttraining.training import (
    DataSource,
    OnPolicyConfig,
    OnPolicyTrainer,
    ScheduleSpec,
    checkpoint_iteration,
)
from isaac_rl_posttraining.training.checkpoints import save_aborted_checkpoint
from isaac_rl_posttraining.training.demonstrations import open_demonstration_source
from isaac_rl_posttraining.utils.config_utils import convert_hydra_to_dataclass
from isaac_rl_posttraining.utils.logging_utils import setup_loguru_logging

#: env.env_name of an Isaac Lab task is the provider prefix plus the registered task id.
ENV_NAME_PREFIX = "isaaclab/"
#: Kit start-up is not fully reliable on a shared node, so a worker whose simulation
#: dies replaces it and redoes the episode this many times before giving up.
EPISODE_ATTEMPTS = 3


@dataclass(frozen=True)
class LaunchPlan:
    """Everything the run needs, validated before any process starts."""

    env_name: str
    task_id: str
    entry: SimulatorEntry
    trainable: TrainablePolicySpec
    frozen_submodules: tuple[str, ...]
    action_window: int
    image_resolution: int
    action_dim: int
    max_episode_steps: int
    server_url: str | None
    instruction: str
    episode_attempts: int
    num_envs: int
    num_rollout_workers: int
    num_learner_ranks: int
    learner_device: str
    seed: int
    ppo: PPOTrainingConfig
    runtime: RuntimeConfig

    @property
    def learnerless(self) -> bool:
        return self.num_learner_ranks == 0

    @property
    def envs_per_worker(self) -> int:
        return self.num_envs // self.num_rollout_workers

    @property
    def chunks_per_episode(self) -> int:
        return self.max_episode_steps // self.action_window

    @property
    def environment_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            env_name=self.env_name,
            num_envs=self.num_envs,
            max_episode_steps=self.max_episode_steps,
            seed=self.seed,
            server_url=self.server_url,
            action_dim=self.action_dim,
            image_resolution=self.image_resolution,
        )

    @property
    def flow_sde(self) -> FlowSDEConfig:
        return self.trainable.flow_sde

    @property
    def denoising_steps(self) -> int:
        return self.trainable.flow_sde.denoising_steps


def _task_id(cfg: DictConfig) -> str:
    """The Isaac Lab task id behind ``env.env_name``, checked against the composed simulator group.

    G1 Factory's server chooses the task, so its ``env.env_name`` only routes to the entry and labels the run.
    """
    env_name = str(cfg.env.env_name)
    if str(cfg.simulator.provider) == "g1_arena":
        return env_name
    if not env_name.startswith(ENV_NAME_PREFIX):
        raise ValueError(
            f"env.env_name must be '{ENV_NAME_PREFIX}<Isaac Lab task id>', got {env_name!r}"
        )
    task_id = env_name[len(ENV_NAME_PREFIX) :]
    if str(cfg.simulator.provider) != "isaaclab":
        raise ValueError(
            f"the composed simulator group has provider {cfg.simulator.provider!r}; "
            "select an Isaac Lab simulator group (simulator=isaaclab_<task>)"
        )
    tasks = [str(task) for task in cfg.simulator.tasks]
    if task_id not in tasks:
        raise ValueError(
            f"{task_id!r} is not a task of the composed simulator group (tasks: {tasks}); "
            "select the simulator=<group> that lists it"
        )
    return task_id


def _require_base_model_fields(base_model: Any) -> None:
    for name in ("ckpt", "denoising_steps", "action_horizon"):
        if getattr(base_model, name) is None:
            raise ValueError(f"base_model.{name} is required by the flow-SDE PPO entry")


def _check_whole_chunks(max_episode_steps: int, action_window: int) -> None:
    """The workers run the episode cap as whole action chunks."""
    if max_episode_steps % action_window:
        raise ValueError(
            f"env.max_episode_steps={max_episode_steps} is not a whole number of "
            f"base_model.action_horizon={action_window}-step action chunks"
        )


def _check_flow_sde_options(
    ppo: PPOTrainingConfig, adapter: Any, adapter_name: str
) -> None:
    if ppo.noise_level is None:
        raise ValueError("ppo.noise_level (flow-SDE transition noise) is required")
    if ppo.cache_features and not adapter.cached_feature_names:
        raise ValueError(f"base_model={adapter_name} has no cacheable features; set ppo.cache_features=false")
    if adapter.requires_cached_features and not ppo.cache_features:
        raise ValueError(
            f"base_model={adapter_name} decisions can only travel as cached features; set ppo.cache_features=true"
        )


def _check_run_mode(ppo: PPOTrainingConfig, num_learner_ranks: int) -> None:
    if num_learner_ranks == 0 and not ppo.eval_only:
        raise ValueError("placement.learner.ranks=0 (no learner) is only valid with ppo.eval_only=true")
    if ppo.eval_only and ppo.eval_episodes_per_environment < 1:
        raise ValueError("ppo.eval_only=true needs ppo.eval_episodes_per_environment >= 1")
    if ppo.resume_from is not None and not Path(ppo.resume_from).is_absolute():
        raise ValueError("ppo.resume_from must be an absolute checkpoint directory (iteration_<n>, final or aborted)")


def _check_placement(
    runtime: RuntimeConfig,
    num_learner_ranks: int,
    num_rollout_workers: int,
    num_envs: int,
    *,
    processes: bool = False,
) -> None:
    """``processes``: the runtime runs its roles in their own processes (rlinf, or native on Ray)."""
    if runtime.name != "rlinf" and not processes and num_learner_ranks > 1:
        raise ValueError(
            f"runtime={runtime.name} runs one in-process learner rank; set "
            "placement.learner.ranks=1 (0 with ppo.eval_only=true), runtime.native.processes=auto, "
            "or runtime=rlinf for FSDP ranks"
        )
    if num_envs % num_rollout_workers:
        raise ValueError(
            f"training.num_envs={num_envs} must be a multiple of "
            f"placement.envs.workers={num_rollout_workers}: every worker hosts one "
            "simulation with the same number of scenes"
        )


def _simulator_geometry(cfg: DictConfig) -> dict[str, int]:
    geometry = {}
    for name in ("image_resolution", "action_dim"):
        value = OmegaConf.select(cfg, f"simulator.{name}")
        if value is None:
            raise ValueError(f"simulator.{name} is required by the Isaac Lab flow-SDE workers")
        geometry[name] = int(value)
    return geometry


def build_plan(cfg: DictConfig, *, require_servers: bool = True) -> LaunchPlan:
    """Validate a composed ``algorithm=ppo`` config against the schema and the placement rules.

    ``require_servers=False`` (the dry run) leaves a served simulator's
    unexported address unset instead of failing; the launch requires it.
    """
    cfg_dc = convert_hydra_to_dataclass(cfg, TrainConfig)
    if cfg_dc.ppo is None:
        raise ValueError("algorithm=ppo composes the ppo section; it is missing")
    ppo = cfg_dc.ppo
    learner = cfg_dc.learner
    learner_ranks = cfg_dc.placement.learner.ranks
    reject_steering_keys(ppo)
    task_id = _task_id(cfg)
    base_model = cfg_dc.base_model
    _require_base_model_fields(base_model)
    # The base model's adapter validates its own loading fields before any weights load.
    adapter_name = resolve_trainable_policy_name(
        str(base_model.name), str(base_model.ckpt)
    )
    adapter = get_trainable_policy(adapter_name)
    adapter_options = adapter.spec_options(base_model)
    action_window = int(base_model.action_horizon)
    if action_window < 1:
        raise ValueError("base_model.action_horizon must be a positive integer")
    max_episode_steps = int(cfg_dc.env.max_episode_steps)
    _check_whole_chunks(max_episode_steps, action_window)
    _check_flow_sde_options(ppo, adapter, adapter_name)
    if ppo.frozen_submodules is None:
        frozen_submodules = tuple(adapter.frozen_submodules)
    else:
        frozen_submodules = tuple(str(path) for path in ppo.frozen_submodules)
    trainable = TrainablePolicySpec(
        base_model=adapter_name,
        checkpoint=str(base_model.ckpt),
        flow_sde=FlowSDEConfig(
            noise_level=float(ppo.noise_level),
            denoising_steps=int(base_model.denoising_steps),
        ),
        value_head_seed=int(cfg_dc.training.seed),
        options=adapter_options,
    )
    _check_run_mode(ppo, learner_ranks)
    # The learner config validates the batch geometry even for a learner-less evaluation.
    ppo_config_from(ppo, num_learner_ranks=learner_ranks, seed=cfg_dc.training.seed)

    runtime = cfg_dc.runtime
    num_envs = int(cfg_dc.training.num_envs)
    geometry = _simulator_geometry(cfg)
    entry = resolve_simulator_entry(cfg)
    require_base_model_support(cfg, entry)
    server_url = phase_server_url(entry, "train") if require_servers else exported_server_url(entry, "train")
    environment = EnvironmentSpec(
        env_name=str(cfg.env.env_name),
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        seed=int(cfg_dc.training.seed),
        server_url=server_url,
        action_dim=geometry["action_dim"],
        image_resolution=geometry["image_resolution"],
    )
    workers = cfg_dc.placement.envs.workers
    # The entry checks the count against its servers; unset, it picks one worker per server.
    num_rollout_workers = requested_layout(
        entry, num_envs, num_servers=environment.num_servers, workers=None if workers is None else int(workers)
    ).workers
    _check_placement(
        runtime,
        learner_ranks,
        num_rollout_workers,
        num_envs,
        processes=native_runs_processes(cfg),
    )
    task_instruction = base_model.task_instruction
    instruction = entry.instruction(environment) if task_instruction is None else str(task_instruction)
    return LaunchPlan(
        env_name=str(cfg.env.env_name),
        task_id=task_id,
        entry=entry,
        trainable=trainable,
        frozen_submodules=frozen_submodules,
        action_window=action_window,
        image_resolution=geometry["image_resolution"],
        action_dim=geometry["action_dim"],
        max_episode_steps=max_episode_steps,
        server_url=server_url,
        instruction=instruction,
        # A worker can replace an in-process simulation, not a remote simulator server.
        episode_attempts=EPISODE_ATTEMPTS if server_url is None else 1,
        num_envs=num_envs,
        num_rollout_workers=num_rollout_workers,
        num_learner_ranks=learner_ranks,
        learner_device=learner.device,
        seed=int(cfg_dc.training.seed),
        ppo=ppo,
        runtime=runtime,
    )



def rollout_worker_spec(plan: LaunchPlan) -> BatchedWorkerSpec:
    """The rollout workers of a plan: each hosts its scenes in one simulation of the entry.

    The policy is conditioned on the task's language. It is a copy in every
    worker, or policy-server replicas every worker sends its scenes to
    (``placement.policy.mode=served``); a server answers with the weight
    version it acted with, and the learner's weights reach it between batches.
    """
    return BatchedWorkerSpec(
        entry=plan.entry,
        environment=plan.environment_spec,
        phase="train",
        scenes=True,
        policy=FlowSDEPolicySpec(
            spec=plan.trainable,
            action_window=plan.action_window,
            instruction=plan.instruction,
            base_seed=plan.seed,
            cache_features=bool(plan.ppo.cache_features),
        ),
        chunks_per_episode=plan.chunks_per_episode,
        # RLinf's sparse task reward: one when a chunk ends the episode successfully.
        reward_transform=SparseOutcomeReward(success_reward=1.0, failure_reward=0.0, step_reward=0.0),
        # The next decision's own value serves as the bootstrap value, so no
        # decision costs a second backbone pass.
        bootstrap_values="reuse_next_decision",
        bootstrap_at_time_limit=bool(plan.ppo.bootstrap_at_time_limit),
        episode_attempts=plan.episode_attempts,
        supports_served_policy=True,
    )


def trainer_config(plan: LaunchPlan, schedule: ScheduleSpec | None = None) -> OnPolicyConfig:
    """The loop's schedule for the plan; a learner-less evaluation has no learner state to checkpoint."""
    config = on_policy_config_from(plan.ppo, seed=plan.seed, schedule=schedule or "sync")
    return replace(config, checkpoint_every=0) if plan.learnerless else config


def _resume_trainer(
    trainer: OnPolicyTrainer, plan: LaunchPlan, resume_from: str
) -> None:
    if plan.learnerless:
        # Log the snapshot's score under the iteration that produced it.
        trainer.iteration = checkpoint_iteration(resume_from)
    else:
        trainer.resume(resume_from)
        logger.info(
            f"resumed from {resume_from}; continuing at iteration {trainer.iteration + 1}"
        )


def open_session(cfg: DictConfig, plan: LaunchPlan) -> OnPolicySession:
    """Open the learner, the rollout workers and their data source, and the trainer over them."""
    ppo = plan.ppo
    exp_name, output_dir_name = new_run_output_dir(cfg)
    output_dir = Path(output_dir_name)
    setup_loguru_logging(log_level=str(cfg.logging.log_level), output_dir=str(output_dir))
    (output_dir / "ppo_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    rank_minibatch = rank_minibatch_size(ppo, plan.num_learner_ranks)
    micro_batch = rank_minibatch if ppo.micro_batch_size is None else int(ppo.micro_batch_size)
    logger.info(
        f"PPO on {plan.task_id}: {plan.num_rollout_workers} rollout workers x "
        f"{plan.envs_per_worker} scenes ({plan.num_envs} environments), "
        f"{ppo.episodes_per_environment} episodes per scene and iteration of up to "
        f"{plan.max_episode_steps} steps; {plan.num_learner_ranks} learner ranks, global batch "
        f"{ppo.global_batch_size} (rank minibatch {rank_minibatch}, micro "
        f"{micro_batch}), {ppo.update_epochs} update epochs"
        + ("; cached backbone features" if ppo.cache_features else "")
        + f"; {schedule_spec(cfg, on_policy=True).kind} schedule"
        + ("; evaluation only, no learner" if plan.learnerless else "")
    )
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=bool(cfg.logging.wandb_offline),
        project=str(cfg.logging.wandb_project),
        entity=cfg.logging.wandb_entity,
        log_dir=wandb_run_dir(cfg.logging.wandb_log_dir_base, str(cfg.logging.wandb_name)),
        prefix=cfg.logging.wandb_prefix,
    )
    ppo_config = ppo_config_from(ppo, num_learner_ranks=plan.num_learner_ranks, seed=plan.seed)
    # Demonstration co-training, when the config asks for it: the anchor's
    # action window is the one the policy executes, so the supervised target
    # and the emitted action share a geometry. The demonstrations are drawn
    # here, with the controller, as every other buffer in this repo is; each
    # rank is shipped only the ones it scores.
    demonstrations, source = open_demonstration_source(
        ppo_config, action_horizon=plan.action_window, trainable=plan.trainable
    )
    if demonstrations is not None:
        logger.info(
            f"Co-training on {ppo.cotrain_dataset_path}: {len(demonstrations.buffer)} "
            f"demonstrations anchor the RL loss, "
            f"{ppo.cotrain_batch_size} per co-training batch, weighted {ppo.cotrain_beta}"
        )
    algorithm = PPO(ppo_config, demonstrations=demonstrations)
    # Every learner rank loads its own copy of the checkpoint onto its pinned device.
    learner_spec = algorithm.learner(
        plan.trainable.build,
        frozen_submodules=plan.frozen_submodules,
        demonstration_source=source,
    )
    rollout_spec = rollout_worker_spec(plan)
    shape = RolloutShape.of_spec(rollout_spec, plan.num_rollout_workers)
    runtime = open_runtime(cfg, shape)
    learner_group = None
    rollout_group = None
    train_source: DataSource | None = None
    telemetry_sink = None
    try:
        if not plan.learnerless:
            started = time.perf_counter()
            learner_group = open_learner_group(
                runtime,
                learner_spec,
                num_learner_ranks=plan.num_learner_ranks,
                name="gr00t_ppo_learner",
                local_device=local_learner_device(plan.learner_device),
            )
            logger.info(f"learner group ready after {time.perf_counter() - started:.0f}s")
        started = time.perf_counter()
        rollout_group = runtime.open_rollout_group(
            rollout_spec,
            num_rollout_workers=plan.num_rollout_workers,
            num_envs=plan.num_envs,
            name="isaaclab_ppo_rollout",
        )
        logger.info(f"rollout group ready after {time.perf_counter() - started:.0f}s")
        # Without a learner, a checkpoint's published weights come from its policy snapshot.
        initial_policy_state = None
        if plan.learnerless and ppo.resume_from is not None:
            initial_policy_state = load_policy_snapshot(ppo.resume_from)
            logger.info(f"evaluating the policy snapshot of {ppo.resume_from}")
        store, fingerprints = checkpoint_store(
            output_dir, ppo, selected_algorithm_name(cfg)
        )
        telemetry_sink, train_source = open_on_policy_source(
            cfg,
            output_dir,
            rollout_group,
            episodes_per_environment=int(ppo.episodes_per_environment),
            reward_channel=rollout_spec.reward_channel,
            shape=shape,
        )
        session = OnPolicySession(
            trainer=OnPolicyTrainer(
                train_source=train_source,
                learner_group=learner_group,
                algorithm=algorithm,
                config=trainer_config(plan, schedule_spec(cfg, on_policy=True)),
                metric_sink=metric_sink(output_dir, wandb_logger),
                # Without a learner or a snapshot the workers score the checkpoint they loaded.
                evaluation_sink=evaluation_sink(
                    output_dir,
                    base_checkpoint=(
                        str(plan.trainable.checkpoint)
                        if plan.learnerless and ppo.resume_from is None
                        else None
                    ),
                ),
                initial_policy_state=initial_policy_state,
                # A learner-less run saves no checkpoint (checkpoint_every=0) but still writes evaluation_summary.json.
                checkpoint_store=store,
                fingerprint_sink=fingerprints,
                reward_channel=rollout_spec.reward_channel,
            ),
            run_dir=str(output_dir),
            resources=(learner_group, runtime),
            wandb_logger=wandb_logger,
            telemetry=telemetry_sink,
        )
        if ppo.resume_from is not None:
            _resume_trainer(session.trainer, plan, ppo.resume_from)
    except BaseException as exc:
        # The data source, once open, owns the rollout workers.
        workers = rollout_group if train_source is None else train_source
        close_resources((workers, learner_group, runtime, telemetry_sink), wandb_logger, exc)
        raise
    return session


def run(cfg: DictConfig, plan: LaunchPlan) -> str:
    """Train (or evaluate) one flow-SDE PPO run; returns its run directory."""
    session = open_session(cfg, plan)
    trainer = session.trainer
    try:
        if plan.ppo.eval_only:
            trainer.evaluate()
        else:
            try:
                with sigterm_as_interrupt(), on_frozen_rollout(functools.partial(save_aborted_checkpoint, trainer)):
                    trainer.run()
            except KeyboardInterrupt:
                trainer.save_checkpoint("interrupted")
                raise
            except Exception as exc:
                save_aborted_checkpoint(trainer, exc)
                raise
    finally:
        session.close(sys.exc_info()[1])
    return session.run_dir
