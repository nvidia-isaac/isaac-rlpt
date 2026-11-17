# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run PPO over a steering actor on LIBERO or G1 Factory through any runtime."""

from __future__ import annotations

import copy
import functools
import gc
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining import configs as configs_package
from isaac_rl_posttraining.algorithms.ppo import PPO, DSRLPPOActorCritic
from isaac_rl_posttraining.configs import selected_algorithm_name
from isaac_rl_posttraining.configs.schema import PPOTrainingConfig
from isaac_rl_posttraining.envs.simulator_entry import (
    Phase,
    SimulatorEntry,
    requested_layout,
)
from isaac_rl_posttraining.runners.off_policy_launch import (
    OffPolicyComponents,
    setup_dsrl_training,
    sigterm_as_interrupt,
)
from isaac_rl_posttraining.runners.on_policy_session import OnPolicySession
from isaac_rl_posttraining.runners.ppo_launch import (
    checkpoint_store,
    close_resources,
    evaluation_sink,
    learner_settings,
    local_learner_device,
    metric_sink,
    on_policy_config_from,
    open_on_policy_source,
    ppo_config_from,
    ppo_settings,
    reject_flow_sde_keys,
)
from isaac_rl_posttraining.runners.run_placement import RolloutShape, open_runtime
from isaac_rl_posttraining.runners.run_schedule import check_run_config, schedule_spec
from isaac_rl_posttraining.runners.simulator_entries import (
    environment_spec,
    phase_server_url,
    require_base_model_support,
    resolve_simulator_entry,
)
from isaac_rl_posttraining.runtimes import (
    SparseOutcomeReward,
    open_learner_group,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.policy_families import decode_batch_size_for
from isaac_rl_posttraining.runtimes.steering_rollouts import (
    resolve_image_keys,
    steering_policy_spec,
    steering_worker_spec,
)
from isaac_rl_posttraining.runtimes.watchdog import on_frozen_rollout
from isaac_rl_posttraining.training import DataSource, OnPolicyTrainer
from isaac_rl_posttraining.training.checkpoints import save_aborted_checkpoint
from isaac_rl_posttraining.utils.training_utils import actor_fingerprint


def dsrl_ppo_worker_spec(
    cfg: DictConfig,
    components: OffPolicyComponents,
    actor_critic: DSRLPPOActorCritic,
    entry: SimulatorEntry,
    ppo: PPOTrainingConfig,
    *,
    phase: Phase,
) -> Any:
    """Attach the simulator's environments and the PPO steering policy for one phase.

    A remote simulator serves one client, so evaluation needs its own server
    unless the run only evaluates.
    """
    evaluation = phase == "eval"
    server_url = phase_server_url(entry, phase)
    if (
        evaluation
        and not ppo.eval_only
        and server_url is not None
        and server_url == phase_server_url(entry, "train")
    ):
        raise ValueError(
            f"{entry.name} PPO evaluation requires a server separate from the training server"
        )
    environment = environment_spec(
        cfg, num_envs=int(cfg.training.num_envs), server_url=server_url
    )
    policy = steering_policy_spec(
        cfg,
        actor=actor_critic,
        base_model=components.base_model,
        controller_env=components.env,
        latent_env=components.latent_env,
        action_modulator=None,
        steering_key_mapping=dict(entry.steering_key_mapping(environment)),
        dino_image_keys=resolve_image_keys(cfg, entry, environment).keys,
        decode_batch_size=decode_batch_size_for(cfg, entry, phase=phase),
        phase=phase,
        deterministic_inference=evaluation and bool(ppo.deterministic_kernels),
        sample_evaluation_actions=evaluation and bool(ppo.sample_evaluation_actions),
    )
    return steering_worker_spec(
        entry,
        environment,
        phase,
        policy=policy,
        action_exec_len=int(cfg.dsrl.action_exec_len),
        reward_transform=SparseOutcomeReward(
            success_reward=float(cfg.env.success_reward),
            failure_reward=float(cfg.env.failure_reward),
            step_reward=float(cfg.env.step_reward),
        ),
        configured_workers=cfg.placement.envs.workers,
    )


def _drop_demonstrations(cfg: DictConfig) -> None:
    """PPO learns from fresh episodes only, so demonstration settings are ignored."""
    if (
        cfg.buffer.demonstrations is not None
        or cfg.buffer.demonstration_fraction is not None
    ):
        logger.warning(
            "algorithm=dsrl_ppo trains on fresh episodes only; ignoring "
            "buffer.demonstrations and buffer.demonstration_fraction"
        )
        cfg.buffer.demonstrations = None
        cfg.buffer.demonstration_fraction = None


def _write_initial_fingerprint(
    actor_critic: DSRLPPOActorCritic, output_dir: Path
) -> None:
    initial_fingerprint = actor_fingerprint(actor_critic)
    if initial_fingerprint is not None:
        logger.info(f"Initial PPO actor-critic fingerprint: {initial_fingerprint}")
        (output_dir / "initial_actor_critic_fingerprint.txt").write_text(
            initial_fingerprint + "\n",
            encoding="utf-8",
        )


#: ``eval.*`` keys an on-policy run sets so the off-policy evaluator stays off, and their values.
_OFF_POLICY_EVALUATOR_OFF: dict[str, object] = {
    "eval_freq": 0,
    "eval_on_first_step": False,
    "eval_record_video": False,
}


def reject_off_policy_eval_keys(cfg: DictConfig) -> None:
    """Check ``eval.*`` for an on-policy run, which evaluates through ``ppo.eval_*`` instead.

    The run switches the off-policy evaluator off (``eval.eval_freq=0``,
    ``eval.eval_on_first_step=false``, ``eval.eval_record_video=false``), so a
    different, non-default value of those keys fails here rather than being
    overwritten; any other ``eval.*`` key changed from its ``train.yaml``
    default is reported as ignored.
    """
    defaults = OmegaConf.load(Path(configs_package.__file__).parent / "train.yaml").eval
    changed = sorted(key for key, value in cfg.eval.items() if value != defaults.get(key))
    overwritten = [
        key for key in changed if key in _OFF_POLICY_EVALUATOR_OFF and cfg.eval[key] != _OFF_POLICY_EVALUATOR_OFF[key]
    ]
    if overwritten:
        raise ValueError(
            "algorithm=dsrl_ppo evaluates through ppo.eval_every, "
            "ppo.eval_episodes_per_environment and ppo.eval_seed and keeps the off-policy "
            f"evaluator off; remove eval.{', eval.'.join(overwritten)} (or set "
            + ", ".join(f"eval.{key}={_OFF_POLICY_EVALUATOR_OFF[key]}" for key in overwritten)
            + ")"
        )
    ignored = [key for key in changed if key not in _OFF_POLICY_EVALUATOR_OFF]
    if ignored:
        logger.warning(
            f"algorithm=dsrl_ppo ignores eval.{', eval.'.join(ignored)} (off-policy evaluator "
            "settings); its evaluation follows ppo.eval_*"
        )


def resume_trainer(trainer: OnPolicyTrainer, resume_from: str | Path) -> None:
    """Restore the learner, the iteration and the env-step counters from a checkpoint."""
    checkpoint = Path(resume_from).expanduser().resolve()
    trainer.resume(checkpoint)
    logger.info(f"Resumed from {checkpoint}; continuing at iteration {trainer.iteration + 1}")


def open_session(cfg: DictConfig) -> OnPolicySession:
    """Build the steering actor-critic, open the learner, the rollout workers and their data source, and the trainer."""
    reject_off_policy_eval_keys(cfg)
    check_run_config(cfg, on_policy=True)
    for key, value in _OFF_POLICY_EVALUATOR_OFF.items():
        cfg.eval[key] = value
    ppo = ppo_settings(cfg)
    reject_flow_sde_keys(ppo)
    learner = learner_settings(cfg)
    learner_ranks = int(cfg.placement.learner.ranks)
    if learner_ranks < 1:
        raise ValueError("algorithm=dsrl_ppo needs placement.learner.ranks >= 1")
    seed = int(cfg.training.seed)
    _drop_demonstrations(cfg)
    components = setup_dsrl_training(cfg)
    wandb_logger = components.wandb_logger
    output_dir = Path(components.save_dir).parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ppo_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    actor_critic = DSRLPPOActorCritic(
        copy.deepcopy(components.actor).cpu(),
        copy.deepcopy(components.v_net).cpu(),
    )
    _write_initial_fingerprint(actor_critic, output_dir)
    algorithm = PPO(
        ppo_config_from(ppo, num_learner_ranks=learner_ranks, seed=seed)
    )
    learner_spec = algorithm.learner(functools.partial(copy.deepcopy, actor_critic))
    entry = resolve_simulator_entry(cfg)
    require_base_model_support(cfg, entry)
    isolate_evaluation = bool(ppo.isolate_evaluation_workers)
    initial_phase: Literal["train", "eval"] = (
        "eval" if ppo.eval_episodes_per_environment > 0 else "train"
    )
    rollout_specs = {
        "train": dsrl_ppo_worker_spec(
            cfg, components, actor_critic, entry, ppo, phase="train"
        )
    }
    rollout_specs["eval"] = (
        dsrl_ppo_worker_spec(cfg, components, actor_critic, entry, ppo, phase="eval")
        if isolate_evaluation and ppo.eval_episodes_per_environment > 0
        else rollout_specs["train"]
    )
    # The off-policy setup also built the SAC learner and replay buffer PPO does
    # not use; drop them with the sizing environment before the runtime opens.
    components.env.close()
    del components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_workers = resolve_num_rollout_workers(
        rollout_specs[initial_phase],
        num_envs=int(cfg.training.num_envs),
        configured=cfg.placement.envs.workers,
        batched_workers=requested_layout(
            entry,
            int(cfg.training.num_envs),
            num_servers=rollout_specs["train"].environment.num_servers,
        ).workers,
    )
    shape = RolloutShape.of_spec(rollout_specs["train"], num_workers)
    runtime = open_runtime(cfg, shape)
    learner_group = None
    rollout_group = None
    train_source: DataSource | None = None
    telemetry_sink = None
    try:
        learner_group = open_learner_group(
            runtime,
            learner_spec,
            num_learner_ranks=learner_ranks,
            name="ppo_learner",
            local_device=local_learner_device(learner.device),
        )

        def open_phase_group(phase: str):
            return runtime.open_rollout_group(
                rollout_specs[phase],
                num_rollout_workers=num_workers,
                num_envs=int(cfg.training.num_envs),
                name=f"ppo_{phase}_rollouts",
            )

        rollout_group = open_phase_group(initial_phase)
        logger.info(
            f"PPO topology: runtime={runtime.name}, "
            f"envs={cfg.training.num_envs}, rollout_workers={num_workers}, "
            f"learner_ranks={learner_group.num_learner_ranks}"
        )
        store, fingerprints = checkpoint_store(
            output_dir, ppo, selected_algorithm_name(cfg)
        )
        telemetry_sink, train_source = open_on_policy_source(
            cfg,
            output_dir,
            rollout_group,
            episodes_per_environment=int(ppo.episodes_per_environment),
            reward_channel=rollout_specs["train"].reward_channel,
            shape=shape,
        )
        session = OnPolicySession(
            trainer=OnPolicyTrainer(
                train_source=train_source,
                learner_group=learner_group,
                algorithm=algorithm,
                config=on_policy_config_from(ppo, seed=seed, schedule=schedule_spec(cfg, on_policy=True)),
                metric_sink=metric_sink(output_dir, wandb_logger),
                evaluation_sink=evaluation_sink(output_dir),
                open_rollout_group=(open_phase_group if isolate_evaluation else None),
                rollout_phase=initial_phase,
                checkpoint_store=store,
                fingerprint_sink=fingerprints,
                reward_channel=rollout_specs["train"].reward_channel,
            ),
            run_dir=str(output_dir),
            resources=(learner_group, runtime),
            wandb_logger=wandb_logger,
            telemetry=telemetry_sink,
        )
        if ppo.resume_from:
            resume_trainer(session.trainer, ppo.resume_from)
    except BaseException as exc:
        # The data source, once open, owns the rollout workers.
        workers = rollout_group if train_source is None else train_source
        close_resources((workers, learner_group, runtime, telemetry_sink), wandb_logger, exc)
        raise
    return session


def run(cfg: DictConfig) -> str:
    """Train (or evaluate) one DSRL-PPO run; returns its run directory."""
    session = open_session(cfg)
    trainer = session.trainer
    try:
        if ppo_settings(cfg).eval_only:
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
