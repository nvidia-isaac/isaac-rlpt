# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-to-runtime plumbing shared by the on-policy runs.

Both on-policy algorithms compose the ``ppo`` section (``configs/algorithm/``):
``dsrl_ppo`` trains a steering actor over a frozen policy and ``ppo``
fine-tunes GR00T's own action head. This module turns that section and the
``learner`` placement into the algorithm's :class:`PPOConfig` and the loop's
:class:`OnPolicyConfig`, and provides the metric and evaluation-manifest sinks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.ppo import PPOConfig
from isaac_rl_posttraining.configs.schema import LearnerConfig, PPOTrainingConfig
from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.runners.run_placement import RolloutShape, resolved_placement
from isaac_rl_posttraining.runtimes import PolicyState, RolloutGroup
from isaac_rl_posttraining.training import (
    ITERATION_CHECKPOINT_PREFIX,
    CheckpointStore,
    ChunkTrajectory,
    DataSource,
    FingerprintSink,
    OnPolicyConfig,
)
from isaac_rl_posttraining.training.recording import persist_from_config
from isaac_rl_posttraining.training.schedule import ScheduleSpec
from isaac_rl_posttraining.training.telemetry import (
    ParquetTelemetrySink,
    open_run_telemetry,
)
from isaac_rl_posttraining.utils.compat import add_note

__all__ = [
    "checkpoint_store",
    "close_resources",
    "evaluation_sink",
    "learner_settings",
    "local_learner_device",
    "metric_sink",
    "on_policy_config_from",
    "open_on_policy_source",
    "ppo_config_from",
    "ppo_settings",
    "rank_minibatch_size",
    "reject_flow_sde_keys",
    "reject_steering_keys",
]


def ppo_settings(cfg: DictConfig) -> PPOTrainingConfig:
    """Build the typed ``ppo`` section from a composed config."""
    if OmegaConf.select(cfg, "ppo") is None:
        raise ValueError(
            "the ppo section is missing; the on-policy algorithms (algorithm=ppo, "
            "algorithm=dsrl_ppo) compose it"
        )
    return PPOTrainingConfig(**OmegaConf.to_container(cfg.ppo, resolve=True))


def learner_settings(cfg: DictConfig) -> LearnerConfig:
    """Build the typed ``learner`` section from a composed config."""
    return LearnerConfig(**OmegaConf.to_container(cfg.learner, resolve=True))


def rank_minibatch_size(ppo: PPOTrainingConfig, num_learner_ranks: int) -> int:
    """Rank-local PPO minibatch: the global batch split evenly over the learner ranks.

    Learner-less evaluation (``placement.learner.ranks=0``) keeps the whole global batch.
    """
    ranks = max(int(num_learner_ranks), 1)
    if ppo.global_batch_size % ranks:
        raise ValueError(
            f"ppo.global_batch_size={ppo.global_batch_size} must be a multiple of "
            f"placement.learner.ranks={num_learner_ranks}"
        )
    return int(ppo.global_batch_size) // ranks


def ppo_config_from(ppo: PPOTrainingConfig, *, num_learner_ranks: int, seed: int) -> PPOConfig:
    """The algorithm's hyperparameters from the ``ppo`` section (advantages and update)."""
    if len(ppo.adam_betas) != 2:
        raise ValueError(f"ppo.adam_betas must hold two values, got {list(ppo.adam_betas)}")
    return PPOConfig(
        gamma=float(ppo.gamma),
        gae_lambda=float(ppo.gae_lambda),
        discount_per=ppo.discount_per,
        bootstrap_at_time_limit=bool(ppo.bootstrap_at_time_limit),
        learning_rate=float(ppo.learning_rate),
        value_learning_rate=(
            None if ppo.value_learning_rate is None else float(ppo.value_learning_rate)
        ),
        optimizer=ppo.optimizer,
        adam_betas=(float(ppo.adam_betas[0]), float(ppo.adam_betas[1])),
        optimizer_eps=float(ppo.optimizer_eps),
        weight_decay=float(ppo.weight_decay),
        clip_ratio=float(ppo.clip_ratio),
        value_clip=None if ppo.value_clip is None else float(ppo.value_clip),
        value_loss=ppo.value_loss,
        huber_delta=float(ppo.huber_delta),
        value_loss_coef=float(ppo.value_loss_coef),
        entropy_coef=float(ppo.entropy_coef),
        epochs=int(ppo.update_epochs),
        minibatch_size=rank_minibatch_size(ppo, num_learner_ranks),
        micro_batch_size=None if ppo.micro_batch_size is None else int(ppo.micro_batch_size),
        max_grad_norm=None if ppo.max_grad_norm is None else float(ppo.max_grad_norm),
        target_kl=None if ppo.target_kl is None else float(ppo.target_kl),
        normalize_advantages=True,
        seed=seed,
        cotrain_beta=float(ppo.cotrain_beta),
        cotrain_batch_size=int(ppo.cotrain_batch_size),
        cotrain_dataset_path=ppo.cotrain_dataset_path,
    )


def on_policy_config_from(
    ppo: PPOTrainingConfig, *, seed: int, schedule: ScheduleSpec | str = "sync"
) -> OnPolicyConfig:
    """The loop's iteration, evaluation, and checkpoint schedule from the ``ppo`` section and the run's schedule."""
    return OnPolicyConfig(
        iterations=int(ppo.iterations),
        episodes_per_environment=int(ppo.episodes_per_environment),
        eval_episodes_per_environment=int(ppo.eval_episodes_per_environment),
        eval_every=int(ppo.eval_every),
        checkpoint_every=int(ppo.checkpoint_every),
        seed=seed,
        eval_seed=int(ppo.eval_seed),
        schedule=schedule,
    )


def checkpoint_store(
    output_dir: Path, ppo: PPOTrainingConfig, algorithm: str
) -> tuple[CheckpointStore, FingerprintSink]:
    """The run's ``checkpoints/`` store (tagged by iteration, manifests naming ``algorithm``) and its fingerprint log."""
    directory = str(output_dir / "checkpoints")
    store = CheckpointStore(
        directory,
        retention=str(ppo.checkpoint_retention),
        tag_prefix=ITERATION_CHECKPOINT_PREFIX,
        algorithm=algorithm,
    )
    return store, FingerprintSink(directory)


def local_learner_device(value: str) -> str:
    """Resolve ``learner.device`` for the one-rank in-process learner group."""
    if value not in ("auto", "cpu", "gpu"):
        raise ValueError("learner.device must be auto, cpu, or gpu")
    if value == "cpu":
        return "cpu"
    if value == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("local PPO learner requested a GPU but CUDA is unavailable")
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def reject_flow_sde_keys(ppo: PPOTrainingConfig) -> None:
    """The steering actor has no flow-SDE head, so its recipe must leave those keys off."""
    if ppo.noise_level is not None or ppo.cache_features or ppo.frozen_submodules:
        raise ValueError(
            "ppo.noise_level, ppo.cache_features and ppo.frozen_submodules configure GR00T "
            "flow-SDE fine-tuning (algorithm=ppo); algorithm=dsrl_ppo does not read them, "
            "so leave them null / false / []"
        )


def reject_steering_keys(ppo: PPOTrainingConfig) -> None:
    """The flow-SDE workers evaluate on the training workers with a deterministic decode."""
    if ppo.isolate_evaluation_workers or ppo.sample_evaluation_actions:
        raise ValueError(
            "ppo.isolate_evaluation_workers and ppo.sample_evaluation_actions configure "
            "algorithm=dsrl_ppo; the Isaac Lab flow-SDE workers evaluate on the training "
            "workers with a deterministic decode, so set both to false"
        )


def log_namespaced_metrics(wandb_logger: Logger, metrics: Mapping[str, float], iteration: int) -> None:
    """Log ``<namespace>/<name>`` metrics under their own W&B namespaces (train, eval, learner, system)."""
    grouped: dict[str, dict[str, float]] = {}
    for name, value in metrics.items():
        namespace, _, metric = name.partition("/")
        if not metric:
            raise ValueError(f"PPO metric {name!r} has no namespace")
        grouped.setdefault(namespace, {})[metric] = value
    for namespace, values in grouped.items():
        wandb_logger.log(values, step=iteration, prefix=namespace)


def metric_sink(
    output_dir: Path, wandb_logger: Logger
) -> Callable[[Mapping[str, float], int], None]:
    """Append every iteration's metrics to ``ppo_metrics.jsonl`` and forward them to W&B."""
    metrics_path = output_dir / "ppo_metrics.jsonl"

    def write(metrics: Mapping[str, float], iteration: int) -> None:
        record = {"iteration": iteration, **dict(metrics)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        log_namespaced_metrics(wandb_logger, metrics, iteration)
        summary = " ".join(
            f"{name}={value:.4g}"
            for name, value in metrics.items()
            if name
            in {
                "train/success_rate",
                "eval/success_rate",
                "system/env_steps_per_second",
            }
        )
        logger.info(f"PPO iteration {iteration}: {summary}")
        if "eval/success_rate" in metrics:
            # The same line the off-policy trainer prints, so one grep reads every algorithm.
            logger.info(f"[EVAL] Success Rate: {metrics['eval/success_rate']:.1%}")

    return write


def _policy_digest(policy_state: PolicyState) -> str:
    digest = hashlib.sha256()
    for name in sorted(policy_state.tensors):
        digest.update(name.encode())
        digest.update(policy_state.tensors[name].tobytes())
    return digest.hexdigest()


#: Files of a checkpoint directory whose contents the base-policy fingerprint covers.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def checkpoint_fingerprint(checkpoint: str) -> str | None:
    """SHA-256 over the weight files at the top of a checkpoint directory (names and contents).

    ``None`` when ``checkpoint`` is not a local directory (a model id) or holds no weight file.
    """
    root = Path(checkpoint).expanduser()
    if not root.is_dir():
        return None
    files = sorted(
        item for item in root.iterdir() if item.is_file() and item.suffix in _WEIGHT_SUFFIXES
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)
    return digest.hexdigest()


def evaluation_sink(
    output_dir: Path, *, base_checkpoint: str | None = None
) -> Callable[[Sequence[ChunkTrajectory], int, PolicyState | None], None]:
    """Append every evaluation's per-episode manifest to ``ppo_evaluation_episodes.jsonl``.

    ``policy_sha256`` fingerprints the scored weights: the published policy
    state (``policy_source: published``), or, without one, the workers' start-up
    weights from ``base_checkpoint`` (``policy_source: base_checkpoint``, a
    digest of its weight files, computed once).
    """
    path = output_dir / "ppo_evaluation_episodes.jsonl"
    base_sha256: list[str | None] = []

    def write(
        trajectories: Sequence[ChunkTrajectory],
        iteration: int,
        policy_state: PolicyState | None,
    ) -> None:
        source = None
        if policy_state is not None:
            policy_sha256, source = _policy_digest(policy_state), "published"
        elif base_checkpoint is not None:
            if not base_sha256:
                base_sha256.append(checkpoint_fingerprint(base_checkpoint))
            policy_sha256, source = base_sha256[0], "base_checkpoint"
        else:
            policy_sha256 = None
        episode_counts: dict[int, int] = {}
        episodes = []
        for trajectory in trajectories:
            environment_index = int(trajectory.environment_index)
            episode_index = episode_counts.get(environment_index, 0)
            episode_counts[environment_index] = episode_index + 1
            episodes.append(
                {
                    "environment_index": environment_index,
                    "episode_index": episode_index,
                    "reset_seed": trajectory.reset_seed,
                    "reset_seed_scope": trajectory.reset_seed_scope,
                    "success": bool(trajectory.success),
                    "env_steps": int(trajectory.env_steps),
                    "return": float(trajectory.rewards.sum()),
                    "initial_observation_sha256": trajectory.initial_observation_sha256,
                    "initial_observation_sha256_by_key": dict(
                        trajectory.initial_observation_sha256_by_key
                    ),
                    "initial_raw_observation_sha256": trajectory.initial_raw_observation_sha256,
                    "initial_raw_observation_sha256_by_key": dict(
                        trajectory.initial_raw_observation_sha256_by_key
                    ),
                    "first_execution_action_sha256": trajectory.first_execution_action_sha256,
                    "execution_action_sha256": trajectory.execution_action_sha256,
                    "observation_sha256_by_decision": trajectory.observation_sha256_by_decision,
                    "execution_action_sha256_by_decision": (
                        trajectory.execution_action_sha256_by_decision
                    ),
                }
            )
        record = {
            "schema_version": 4,
            "iteration": iteration,
            "policy_version": None if policy_state is None else int(policy_state.version),
            "policy_sha256": policy_sha256,
            "policy_source": source,
            "base_checkpoint": base_checkpoint if source == "base_checkpoint" else None,
            "episodes": episodes,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    return write


def open_on_policy_source(
    cfg: DictConfig,
    output_dir: Path,
    group: RolloutGroup,
    *,
    episodes_per_environment: int,
    reward_channel: str,
    shape: RolloutShape | None = None,
) -> tuple[ParquetTelemetrySink, DataSource]:
    """The run's telemetry sink (``<output_dir>/telemetry``) and the data source the on-policy trainer collects through.

    The data source buffers nothing; it records lineage and telemetry and saves
    the episodes when ``data.persist.path`` is set.
    """
    sink = open_run_telemetry(output_dir, config=cfg, placement=resolved_placement(cfg, shape))
    source = DataSource(
        group,
        None,
        request_count=episodes_per_environment,
        count_by="episode",
        reward_channel=reward_channel,
        # A decision's reward is the sum over its executed chunk, as the trajectories count it.
        reward_aggregation="sum",
        telemetry=sink,
        persist=persist_from_config(OmegaConf.select(cfg, "data.persist"), output_dir),
    )
    return sink, source


def close_resources(resources: tuple[Any, ...], wandb_logger: Any, primary_error: BaseException | None) -> None:
    """Finish W&B, then close every opened resource (``None`` entries are skipped); a failure is noted or raised."""
    errors = []
    # Ending W&B first sends its last rows before the slower runtime shutdown.
    try:
        wandb_logger.finish()
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    if errors:
        message = f"{len(errors)} PPO runtime resource(s) failed to close"
        if primary_error is not None:
            add_note(primary_error, f"{message}: {errors[0]!r}")
        else:
            raise RuntimeError(message) from errors[0]
