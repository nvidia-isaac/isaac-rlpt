# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An open off-policy run: rollout groups, the learner, the evaluator and the trainer over them.

:func:`open_off_policy_session` builds what the off-policy loop drives; the
launcher runs :meth:`OffPolicyTrainer.run` on it, and a Python
:class:`~isaac_rl_posttraining.run.Run` drives its pieces step by step
(collect, train, publish, evaluate, save).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.configs import evaluation_enabled
from isaac_rl_posttraining.envs.simulator_entry import (
    Phase,
    SimulatorEntry,
    requested_layout,
)
from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.runners.off_policy_evaluation import open_evaluator
from isaac_rl_posttraining.runners.off_policy_models import policy_family
from isaac_rl_posttraining.runners.ppo_launch import log_namespaced_metrics
from isaac_rl_posttraining.runners.run_placement import RolloutShape, open_runtime, resolved_placement
from isaac_rl_posttraining.runners.run_schedule import (
    open_replay_learner_group,
    schedule_spec,
)
from isaac_rl_posttraining.runners.simulator_entries import (
    environment_spec,
    phase_server_url,
    require_base_model_support,
)
from isaac_rl_posttraining.runtimes import (
    PolicyState,
    RolloutRuntime,
    RolloutWorkerSpec,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.learner import LearnerGroup
from isaac_rl_posttraining.runtimes.policy_families import (
    rollout_worker_spec,
)
from isaac_rl_posttraining.training import (
    CheckpointStore,
    DataSource,
    FingerprintSink,
    OffPolicyConfig,
    OffPolicyTrainer,
    open_run_telemetry,
)
from isaac_rl_posttraining.training.data_source import rollout_count_by
from isaac_rl_posttraining.training.off_policy import observe_reward_stream
from isaac_rl_posttraining.training.prefetch import learner_updates
from isaac_rl_posttraining.training.recording import persist_from_config

if TYPE_CHECKING:
    from isaac_rl_posttraining.buffers import ReplayBuffer
    from isaac_rl_posttraining.runners.off_policy_launch import OffPolicyComponents
    from isaac_rl_posttraining.training.evaluation import RolloutEvaluator
    from isaac_rl_posttraining.training.telemetry import ParquetTelemetrySink

__all__ = [
    "OffPolicySession",
    "open_off_policy_session",
    "rollout_worker_spec_for",
]


def rollout_worker_spec_for(
    cfg: DictConfig,
    components: OffPolicyComponents,
    entry: SimulatorEntry,
    *,
    phase: Phase,
    num_envs: int,
    configured_workers: int | None,
) -> RolloutWorkerSpec:
    """Compose the entry's environments with the algorithm's policy family for one phase."""
    require_base_model_support(cfg, entry)
    return rollout_worker_spec(
        policy_family(components.algorithm_name),
        cfg,
        entry,
        environment_spec(
            cfg, num_envs=num_envs, server_url=phase_server_url(entry, phase)
        ),
        actor=components.actor,
        base_model=components.base_model,
        controller_env=components.env,
        latent_env=components.latent_env,
        action_modulator=components.action_modulator,
        phase=phase,
        configured_workers=configured_workers,
    )


def _off_policy_config(
    cfg: DictConfig,
    entry: SimulatorEntry,
    *,
    learning_starts: int,
    save_interval: int,
    save_dir: str,
) -> OffPolicyConfig:
    """The off-policy loop's schedule from the ``training``, ``eval`` and ``buffer`` sections."""
    eval_freq = (
        cfg.eval.eval_freq
        if cfg.eval.eval_freq is not None
        else cfg.training.num_rollouts // 100
    )
    return OffPolicyConfig(
        num_rollouts=int(cfg.training.num_rollouts),
        learning_starts=learning_starts,
        train_after_episode=bool(
            OmegaConf.select(cfg.training, "train_after_episode", default=False)
        ),
        max_updates_per_burst=OmegaConf.select(
            cfg.training, "max_updates_per_burst", default=None
        ),
        eval_freq=int(eval_freq),
        eval_on_first_step=bool(
            OmegaConf.select(cfg.eval, "eval_on_first_step", default=False)
        ),
        # A shared simulator must not be interrupted mid-episode by evaluation.
        eval_at_episode_boundary=entry.evaluation == "shared"
        or bool(OmegaConf.select(cfg.eval, "eval_at_episode_boundary", default=False)),
        save_interval=save_interval,
        save_buffer_on_exit=bool(
            OmegaConf.select(cfg.buffer, "save_buffer_on_exit", default=False)
        ),
        save_buffer_every=int(
            OmegaConf.select(cfg.buffer, "save_buffer_every", default=0)
        ),
        save_buffer_dir=os.path.join(
            os.path.dirname(save_dir),
            OmegaConf.select(cfg.buffer, "save_buffer_dir", default="replay_buffers"),
        ),
        schedule=schedule_spec(cfg),
    )


@dataclass
class OffPolicySession:
    """The pieces of an open off-policy run and step-wise access to them.

    The step methods keep their own policy version and step count; a session
    is driven either by :attr:`trainer` (``trainer.run()``) or step by step,
    not both. ``release`` (when set) frees the rest of the run on :meth:`close`.
    """

    trainer: OffPolicyTrainer
    train_source: DataSource
    learner_group: LearnerGroup
    evaluator: RolloutEvaluator | None
    learning_starts: int
    run_dir: str
    telemetry: ParquetTelemetrySink | None = None
    env_steps: int = 0
    policy_version: int = 0
    policy_state: PolicyState | None = field(default=None, repr=False)
    release: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def buffer(self) -> ReplayBuffer:
        """The replay buffer the rollouts fill."""
        return self.trainer.buffer

    def publish(self) -> PolicyState:
        """Send the learner's current weights to the rollout workers (a new policy version)."""
        self.policy_version += 1
        self.policy_state = self.learner_group.policy_state(version=self.policy_version)
        self.train_source.publish(self.policy_state)
        return self.policy_state

    def collect(self, rounds: int = 1) -> Mapping[str, float]:
        """Collect ``rounds`` rounds into the replay buffer, handing their rewards to the learner; returns the last round's metrics."""
        if self.policy_state is None:
            self.publish()
        metrics: Mapping[str, float] = {}
        for _ in range(rounds):
            metrics = self.train_source.run(
                can_train=self.env_steps >= self.learning_starts
            )
            observe_reward_stream(
                self.learner_group.hooks(),
                self.train_source.last_transitions,
                self.trainer.buffer,
            )
            self.env_steps = int(metrics["total_steps"])
        return metrics

    def train(self, steps: int) -> Mapping[str, float]:
        """Run ``steps`` learner updates on replay draws, then publish.

        Returns the last update's metrics averaged over learner ranks, the values the trainer logs.
        """
        metrics: Mapping[str, float] = {}
        for metrics in learner_updates(self.learner_group, self.trainer.draw_batch, self.env_steps, steps):
            self.train_source.log_learner(
                self.learner_group.hooks().gradient_steps(),
                self.env_steps,
                metrics,
                publish_version=self.policy_version,
            )
        if steps > 0:
            self.publish()
        return metrics

    def evaluate(self) -> Mapping[str, float]:
        """Score the last published weights on the evaluation workers."""
        if self.evaluator is None:
            raise RuntimeError(
                "this run has no evaluation (eval.eval_freq=0 and eval.eval_on_first_step=false)"
            )
        state = self.policy_state if self.policy_state is not None else self.publish()
        return self.evaluator.evaluate(state, env_steps=self.env_steps)

    def save(self, directory: str) -> None:
        """Write the learner's checkpoint to ``directory``."""
        self.learner_group.save_checkpoint(os.path.abspath(directory))

    def load(self, directory: str) -> None:
        """Load a learner checkpoint from ``directory`` and publish it."""
        self.learner_group.load_checkpoint(os.path.abspath(directory))
        self.publish()

    def close(self) -> None:
        """Close the evaluator, the data source, the learner group and the telemetry, then ``release`` the rest of the run."""
        try:
            if self.evaluator is not None:
                self.evaluator.close()
            # Writes the episodes still waiting for persistence and closes the rollout workers.
            self.train_source.close()
            self.learner_group.close()
            if self.telemetry is not None:
                self.telemetry.close()
        finally:
            if self.release is not None:
                self.release()


def open_off_policy_session(
    cfg: DictConfig,
    components: OffPolicyComponents,
    entry: SimulatorEntry,
    runtimes: list[RolloutRuntime],
) -> OffPolicySession:
    """Open the rollout groups, the local learner group and the evaluator, and build the trainer over them."""
    algorithm = components.algorithm
    buffer = components.buffer
    device = components.device
    wandb_logger = components.wandb_logger
    save_dir = components.save_dir
    configured_workers = cfg.placement.envs.workers
    num_envs = int(cfg.training.num_envs)
    eval_num_envs = int(cfg.eval.eval_num_envs)
    train_spec = rollout_worker_spec_for(
        cfg,
        components,
        entry,
        phase="train",
        num_envs=num_envs,
        configured_workers=configured_workers,
    )
    eval_spec = None
    if evaluation_enabled(cfg):
        eval_spec = rollout_worker_spec_for(
            cfg,
            components,
            entry,
            phase="eval",
            num_envs=eval_num_envs,
            configured_workers=None,
        )
    # The controller's local environment only sized the models; a remote
    # simulator serves one client, so it closes before the workers connect.
    components.env.close()
    components.env = None

    layout = requested_layout(
        entry,
        num_envs,
        num_servers=train_spec.environment.num_servers,
        workers=configured_workers,
    )
    num_workers = resolve_num_rollout_workers(
        train_spec,
        num_envs=num_envs,
        configured=configured_workers,
        batched_workers=layout.workers,
    )
    shape = RolloutShape.of_spec(train_spec, num_workers)
    runtime = open_runtime(cfg, shape)
    runtimes.append(runtime)
    train_group = runtime.open_rollout_group(
        train_spec,
        num_rollout_workers=num_workers,
        num_envs=num_envs,
        name="training_rollouts",
    )
    telemetry = open_run_telemetry(os.path.dirname(save_dir), config=cfg, placement=resolved_placement(cfg, shape))
    train_source = DataSource(
        train_group,
        InProcessBufferSink(buffer),
        request_count=1,
        count_by=rollout_count_by(str(cfg.training.rollout_count_by)),
        device=device,
        reward_channel=train_spec.reward_channel,
        reward_aggregation=str(cfg.buffer.reward_aggregation),
        telemetry=telemetry,
        persist=persist_from_config(OmegaConf.select(cfg, "data.persist"), os.path.dirname(save_dir)),
        max_policy_lag=schedule_spec(cfg).staleness_bound,
    )
    logger.info(
        f"Rollout runtime: {runtime.name}; {num_workers} worker(s), {num_envs} environment(s); "
        f"evaluation={entry.evaluation}"
    )
    learner_group = open_replay_learner_group(cfg, runtime, algorithm, device)
    evaluator = None
    if eval_spec is not None:
        evaluator = open_evaluator(
            cfg,
            entry,
            runtime,
            train_group,
            eval_spec,
            eval_num_envs=eval_num_envs,
            experiment_logger=wandb_logger,
            reward_channel=eval_spec.reward_channel,
            reward_aggregation=str(cfg.buffer.reward_aggregation),
            output_dir=os.path.dirname(save_dir),
            telemetry=train_source.recorder.child("eval"),
        )
    learning_starts = int(algorithm.config.learning_starts)
    trainer = OffPolicyTrainer(
        train_source=train_source,
        learner_group=learner_group,
        buffer=buffer,
        draw_batch=algorithm.learner_batch,
        evaluator=evaluator,
        config=_off_policy_config(
            cfg,
            entry,
            learning_starts=learning_starts,
            save_interval=components.save_interval,
            save_dir=save_dir,
        ),
        metric_sink=functools.partial(log_namespaced_metrics, wandb_logger),
        fingerprint_sink=FingerprintSink(save_dir),
        checkpoint_store=CheckpointStore(
            save_dir,
            retention=str(cfg.training.checkpoint_retention),
            artifact_logger=wandb_logger,
            algorithm=components.algorithm_name,
        ),
    )
    return OffPolicySession(
        trainer=trainer,
        train_source=train_source,
        learner_group=learner_group,
        evaluator=evaluator,
        learning_starts=learning_starts,
        run_dir=os.path.dirname(save_dir),
        telemetry=telemetry,
    )
