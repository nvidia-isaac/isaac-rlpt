# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting of the off-policy loop: buffer statistics, progress bar and per-episode summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

from isaac_rl_posttraining.buffers import ReplayBuffer

__all__ = [
    "buffer_stats",
    "print_detailed_eval_metrics",
    "print_detailed_metrics",
    "rate",
    "update_progress_bar",
]


def buffer_stats(buffer: ReplayBuffer) -> dict[str, float]:
    """Buffer size, episode counts and the decision-reward statistics per channel, from the buffer's running records."""
    stats: dict[str, float] = {"buffer/total_size": float(len(buffer))}
    for key, value in buffer.get_statistics().items():
        stats[f"buffer/{key}"] = float(value)
    for channel, record in buffer.reward_statistics().items():
        # The learning channel keeps the reference names; the others are named after their channel.
        name = (
            "total" if channel == buffer.reward_channel else channel.replace("/", "_")
        )
        stats[f"buffer/avg_{name}_reward"] = float(record["mean"])
        stats[f"buffer/min_{name}_reward"] = float(record["min"])
        stats[f"buffer/max_{name}_reward"] = float(record["max"])
        stats[f"buffer/std_{name}_reward"] = float(record["std"])
    return stats


def rate(steps: int, seconds: float) -> float:
    return steps / seconds if seconds > 0.0 else 0.0


def update_progress_bar(
    pbar: Any,
    buffer_size: int,
    learning_starts: int,
    can_train: bool,
    rollout_metrics: Mapping[str, float] | None,
    train_metrics: Mapping[str, float] | None,
    steps_this_iter: int,
) -> None:
    """Refresh the tqdm postfix with training/rollout status."""
    postfix: dict[str, Any] = {"buffer_size": buffer_size}
    postfix["status"] = (
        "training" if can_train else f"collecting ({buffer_size}/{learning_starts})"
    )
    if rollout_metrics:
        for key in ("ep_avg_reward", "ep_avg_progress_reward", "success_rate"):
            if key in rollout_metrics:
                postfix[key] = f"{rollout_metrics[key]:.2f}"
    if train_metrics:
        for key in ("q_values_mean", "actor_loss", "ent_coef", "prior_kl"):
            if key in train_metrics:
                postfix[key] = f"{train_metrics[key]:.4f}"
    pbar.set_postfix(postfix)
    pbar.update(steps_this_iter if steps_this_iter > 0 else 1)


def print_detailed_metrics(
    episodes: int,
    rollout_metrics: Mapping[str, float] | None,
    train_metrics: Mapping[str, float] | None,
) -> None:
    """Log a per-episode summary of rollout and training metrics."""
    logger.info(f"\n--- Episode {episodes} Metrics ---")
    if rollout_metrics:
        logger.info(f"Average Reward: {rollout_metrics.get('ep_avg_reward', 0):.3f}")
        logger.info(
            f"Min/Max Reward: {rollout_metrics.get('ep_min_reward', 0):.3f} / "
            f"{rollout_metrics.get('ep_max_reward', 0):.3f}"
        )
        logger.info(f"Success Rate: {rollout_metrics.get('success_rate', 0):.1%}")
        if "ep_avg_progress_reward" in rollout_metrics:
            logger.info(
                f"Avg Progress (Relabeled) Reward: {rollout_metrics['ep_avg_progress_reward']:.3f}"
            )
        if "ep_avg_env_reward" in rollout_metrics:
            logger.info(f"Avg Env Reward: {rollout_metrics['ep_avg_env_reward']:.3f}")
    if train_metrics:
        logger.info(f"[TRAIN] Critic Loss: {train_metrics.get('critic_loss', 0):.6f}")
        logger.info(f"[TRAIN] Actor Loss: {train_metrics.get('actor_loss', 0):.6f}")
        logger.info(f"[TRAIN] Entropy Coef: {train_metrics.get('ent_coef', 0):.6f}")
        # The gap between the actor's chosen-action Q and the data-action
        # Q is the standard Q-overestimation drift indicator.
        actor_q = train_metrics.get("actor_chosen_q_mean", 0.0)
        data_q = train_metrics.get("q_values_mean", 0.0)
        logger.info(
            f"[TRAIN] Q-gap (actor_q - data_q): {actor_q - data_q:+.4f} "
            f"(actor_q={actor_q:.3f} data_q={data_q:.3f})"
        )
        if "kl_div_mean" in train_metrics:
            logger.info(
                f"[TRAIN] KL(current||snapshot): {train_metrics['kl_div_mean']:.6f}"
            )
    logger.info("---")


def print_detailed_eval_metrics(eval_metrics: Mapping[str, float]) -> None:
    """Log an evaluation summary."""
    logger.info(f"[EVAL] Avg Reward: {eval_metrics.get('avg_reward', 0):.3f}")
    logger.info(f"[EVAL] Std Reward: {eval_metrics.get('std_reward', 0):.3f}")
    logger.info(f"[EVAL] Min Reward: {eval_metrics.get('min_reward', 0):.3f}")
    logger.info(f"[EVAL] Max Reward: {eval_metrics.get('max_reward', 0):.3f}")
    logger.info(f"[EVAL] Success Rate: {eval_metrics.get('success_rate', 0):.1%}")
    logger.info(f"[EVAL] Avg Steps: {eval_metrics.get('avg_steps', 0):.1f}")
    logger.info("---")
