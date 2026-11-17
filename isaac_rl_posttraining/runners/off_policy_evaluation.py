# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-group construction and evaluation-only off-policy execution."""

from __future__ import annotations

import os

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry
from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.runners.ppo_launch import log_namespaced_metrics
from isaac_rl_posttraining.runners.run_placement import open_runtime
from isaac_rl_posttraining.runners.simulator_entries import environment_spec
from isaac_rl_posttraining.runtimes import (
    RolloutGroup,
    RolloutRuntime,
    RolloutWorkerSpec,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors
from isaac_rl_posttraining.runtimes.steering_rollouts import resolve_image_keys
from isaac_rl_posttraining.training import RolloutEvaluator, RolloutRecorder
from isaac_rl_posttraining.training.checkpoints import CheckpointStore, FingerprintSink


def open_evaluator(
    cfg: DictConfig,
    entry: SimulatorEntry,
    runtime: RolloutRuntime,
    train_group: RolloutGroup | None,
    eval_spec: RolloutWorkerSpec,
    *,
    eval_num_envs: int,
    experiment_logger: Logger,
    reward_channel: str,
    reward_aggregation: str,
    output_dir: str,
    telemetry: RolloutRecorder | None = None,
) -> RolloutEvaluator:
    """Build the evaluator using the simulator entry's declared lifecycle."""
    eval_workers = resolve_num_rollout_workers(
        eval_spec,
        num_envs=eval_num_envs,
        configured=None,
    )

    def open_evaluation_group() -> RolloutGroup:
        return runtime.open_rollout_group(
            eval_spec,
            num_rollout_workers=eval_workers,
            num_envs=eval_num_envs,
            name="evaluation_rollouts",
        )

    # Next to the run's checkpoints, whatever the logger's own directory is.
    video_dir = os.path.join(output_dir, "evaluation_videos")
    settings = {
        "num_episodes": int(cfg.eval.eval_num_episodes),
        "max_decisions": int(
            OmegaConf.select(cfg, "eval.eval_max_steps", default=1000)
        ),
        "reset_seed": cfg.eval.reset_seed,
        "record_video": bool(cfg.eval.eval_record_video),
        "video_dir": video_dir,
        "image_keys": list(
            resolve_image_keys(
                cfg,
                entry,
                environment_spec(cfg, num_envs=1, server_url=None),
            ).keys
        ),
        "video_fps": int(OmegaConf.select(cfg, "eval.video_fps", default=20)),
        "logger": experiment_logger,
        "reward_channel": reward_channel,
        "reward_aggregation": reward_aggregation,
        "episode_record_path": os.path.join(
            output_dir,
            "evaluation_episodes.jsonl",
        ),
        "telemetry": telemetry,
    }
    if entry.evaluation == "dedicated":
        return RolloutEvaluator(group=open_evaluation_group(), **settings)
    if entry.evaluation == "fresh_workers":
        return RolloutEvaluator(open_group=open_evaluation_group, **settings)
    if entry.evaluation == "shared":
        # The training workers evaluate only when they host the evaluation's environment count.
        if train_group is None or train_group.num_envs != eval_num_envs:
            return RolloutEvaluator(group=open_evaluation_group(), **settings)
        return RolloutEvaluator(group=train_group, shared_group=True, **settings)
    raise ValueError(f"unknown evaluation mode {entry.evaluation!r}")


def run_evaluation_only(
    cfg: DictConfig,
    entry: SimulatorEntry,
    eval_spec: RolloutWorkerSpec,
    *,
    eval_num_envs: int,
    runtimes: list[RolloutRuntime],
    actor: nn.Module,
    optimizer_updates: int,
    experiment_logger: Logger,
    output_dir: str,
    save_dir: str,
) -> None:
    """Evaluate without opening training workers, replay collection, or a learner.

    The score lands where a training run's does: ``evaluation_summary.json``
    under ``save_dir`` (what the launchers read) and an ``eval`` digest of the
    scored weights in ``fingerprints.jsonl``, matchable against the ``load``
    digest and the checkpoint's own ``save`` event.
    """
    runtime = open_runtime(cfg)
    runtimes.append(runtime)
    evaluator = open_evaluator(
        cfg,
        entry,
        runtime,
        None,
        eval_spec,
        eval_num_envs=eval_num_envs,
        experiment_logger=experiment_logger,
        reward_channel=eval_spec.reward_channel,
        reward_aggregation=str(cfg.buffer.reward_aggregation),
        output_dir=output_dir,
    )
    try:
        state = policy_state_from_tensors(actor.state_dict(), version=1)
        metrics = dict(evaluator.evaluate(state))
        metrics["simulator_steps"] = 0.0
        metrics["optimizer_updates"] = float(optimizer_updates)
        log_namespaced_metrics(
            experiment_logger,
            {f"eval/{name}": value for name, value in metrics.items()},
            0,
        )
        CheckpointStore(save_dir).write_evaluation_summary([{"step": 0, **metrics}])
        FingerprintSink(save_dir).record(
            "eval", step=0, state=state, success_rate=float(metrics["success_rate"])
        )
        logger.info(
            f"Evaluation-only run completed: success_rate={metrics['success_rate']:.1%} "
            f"over {int(metrics['num_eval_episodes'])} episodes"
        )
    finally:
        evaluator.close()
