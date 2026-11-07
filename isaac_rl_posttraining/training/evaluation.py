# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a published policy state on a rollout group.

The evaluator scores whole episodes collected in ``eval`` mode: success is
read at the terminal decision only (a transient mid-episode flag counts for
nothing), every episode is capped at a number of decisions so a wedged
simulator cannot stall the run, and one extra episode can be recorded as a
video from the frames the worker captures. Base-policy evaluation is a
property of the group's policy (``native=True``), not of the evaluator.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from isaac_rl_posttraining.buffers.replay import RewardAggregation
from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.rollouts.video import (
    detect_image_keys,
    stack_camera_frames,
    write_video,
)
from isaac_rl_posttraining.runtimes.base import (
    CollectedDecision,
    PolicyState,
    RolloutGroup,
    RolloutRequest,
    RolloutResult,
)
from isaac_rl_posttraining.training.on_policy_metrics import (
    drain_policy_service_metrics,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.training.recording import RolloutRecorder

__all__ = ["EpisodeOutcome", "RolloutEvaluator", "episode_outcomes", "summarize_episodes"]


@dataclass(frozen=True)
class EpisodeOutcome:
    """One scored evaluation episode."""

    environment_index: int
    episode_id: int | str
    total_reward: float
    decisions: int
    success: bool
    reset_seed: int | None
    reset_seed_scope: str | None
    policy_version: int | None
    policy_seed: int | None
    case_seed: int | None
    round_index: int | None
    initial_state_fingerprint: str | None
    terminal_reason: str | None
    task_metrics: Mapping[str, bool | int | float | str | None]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _terminal_task_metrics(
    terminal_extra: Mapping[str, object],
) -> dict[str, bool | int | float | str | None]:
    """The terminal decision's task metrics, checked to be JSON scalars."""
    raw_task_metrics = terminal_extra.get("task_metrics", {})
    if not isinstance(raw_task_metrics, Mapping):
        raise TypeError("terminal info task_metrics must be a mapping")
    task_metrics: dict[str, bool | int | float | str | None] = {}
    for name, value in raw_task_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("task metric names must be non-empty strings")
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(
            value, (bool, int, float, str)
        ):
            raise TypeError(
                f"task metric {name!r} must be a JSON scalar, got {type(value).__name__}"
            )
        task_metrics[name] = value
    return task_metrics


def _scored_episode(
    decision: CollectedDecision,
    episode_id: int | str,
    initial_extra: Mapping[str, object],
    rewards: list[float],
) -> EpisodeOutcome:
    """Score one episode from its ending decision and its first decision's metadata."""
    transition = decision.transition
    terminal_extra = transition.metadata.extra
    task_metrics = _terminal_task_metrics(terminal_extra)
    fingerprint_value = initial_extra.get(
        "initial_state_fingerprint",
        initial_extra.get("raw_observation_sha256"),
    )
    return EpisodeOutcome(
        environment_index=decision.environment_index,
        episode_id=episode_id,
        total_reward=float(sum(rewards)),
        decisions=len(rewards),
        success=transition.succeeded,
        reset_seed=transition.metadata.reset_seed,
        reset_seed_scope=transition.metadata.reset_seed_scope,
        policy_version=transition.metadata.policy_version,
        policy_seed=_optional_int(initial_extra.get("policy_seed")),
        case_seed=_optional_int(initial_extra.get("case_seed")),
        round_index=_optional_int(initial_extra.get("round_index")),
        initial_state_fingerprint=_optional_str(fingerprint_value),
        terminal_reason=_optional_str(terminal_extra.get("terminal_reason")),
        task_metrics=task_metrics,
    )


def episode_outcomes(
    results: Sequence[RolloutResult],
    *,
    reward_channel: str = "env",
    reward_aggregation: RewardAggregation = "last",
) -> list[EpisodeOutcome]:
    """Split collected decisions into complete episodes, scoring success at the end only.

    ``total_reward`` is the episode's return on ``reward_channel``: the sum
    over its decisions of each decision's reward under ``reward_aggregation``,
    the same number the learner trains on. Success is the ending transition's
    flag, which the worker already scores as a failure when its own decision
    cap ended the episode.
    """
    running: dict[int, list[float]] = {}
    initial_metadata: dict[int, tuple[int | str, Mapping[str, object]]] = {}
    outcomes: list[EpisodeOutcome] = []
    for result in sorted(results, key=lambda item: item.worker_index):
        for decision in result.decisions:
            transition = decision.transition
            rewards = running.setdefault(decision.environment_index, [])
            if decision.environment_index not in initial_metadata:
                initial_metadata[decision.environment_index] = (
                    transition.episode_id,
                    dict(transition.metadata.extra),
                )
            rewards.append(transition.decision_reward(reward_channel, aggregation=reward_aggregation))
            if not transition.ended:
                continue
            episode_id, initial_extra = initial_metadata.pop(
                decision.environment_index
            )
            outcomes.append(
                _scored_episode(decision, episode_id, initial_extra, rewards)
            )
            running.pop(decision.environment_index)
    if running:
        raise RuntimeError(
            f"evaluation returned unfinished episodes for environments {sorted(running)}"
        )
    return outcomes


def summarize_episodes(outcomes: Sequence[EpisodeOutcome]) -> dict[str, float]:
    """Aggregate scored episodes into the evaluation metrics contract."""
    if not outcomes:
        raise ValueError("cannot summarize an evaluation with no episodes")
    rewards = np.asarray([outcome.total_reward for outcome in outcomes], dtype=np.float64)
    decisions = np.asarray([outcome.decisions for outcome in outcomes], dtype=np.float64)
    successes = np.asarray([outcome.success for outcome in outcomes], dtype=np.float64)
    metrics = {
        "avg_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "avg_steps": float(decisions.mean()),
        "success_rate": float(successes.mean()),
        "num_eval_episodes": float(len(outcomes)),
    }
    boolean_task_metrics: dict[str, list[bool]] = {}
    for outcome in outcomes:
        for name, value in outcome.task_metrics.items():
            if isinstance(value, bool):
                boolean_task_metrics.setdefault(name, []).append(value)
    for name, values in sorted(boolean_task_metrics.items()):
        metrics[f"task/{name}_rate"] = float(np.mean(values))
        metrics[f"task/{name}_count"] = float(len(values))
    logger.info(f"Evaluation over {len(outcomes)} episodes:")
    logger.info(f"  Average Reward: {metrics['avg_reward']:.3f} +/- {metrics['std_reward']:.3f}")
    logger.info(f"  Success Rate: {metrics['success_rate']:.1%}")
    logger.info(f"  Average Steps: {metrics['avg_steps']:.1f}")
    return metrics


def _worker_time_summary(results: Sequence[RolloutResult]) -> str:
    """The workers' timings summed by name: ``*_seconds`` entries as seconds, ``*_calls`` as counts."""
    totals: dict[str, float] = {}
    for result in results:
        for name, value in result.timings.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    if not totals:
        return ""
    parts = []
    for name, value in sorted(totals.items()):
        if name.endswith("_calls"):
            parts.append(f"{name}={value:.0f}")
        else:
            parts.append(f"{name}={value:.0f}s")
    return "; worker time: " + ", ".join(parts)


class RolloutEvaluator:
    """Score policy states on a persistent rollout group or on fresh workers per call.

    Args:
        group: A persistent evaluation group (a dedicated simulator, or the
            training group when the simulator is shared and evaluations are
            boundary-gated). Exactly one of ``group`` / ``open_group``.
        open_group: Opens a fresh group for one evaluation and closes it
            afterwards, for simulators whose evaluation workers are rebuilt
            every time.
        shared_group: ``group`` is the training group. Its data source
            publishes every policy state, so the evaluator only requests
            ``eval`` decisions and checks the workers acted with the state it
            was given; publishing it again would be rejected as a stale
            version.
        num_episodes: Exact episodes to score. It must be at least the group's
            environment count and an exact multiple of it.
        max_decisions: Per-episode decision cap; a capped episode is a failure.
        reset_seed: Seeds the evaluation resets per environment and episode;
            ``None`` keeps evaluation unseeded.
        record_video: Record one extra episode (single-environment groups
            only) after the scored pass.
        video_dir: Directory the MP4 is written into.
        image_keys: Camera keys stacked into the video; detected by name when
            ``None``.
        video_fps: Framerate of the saved video.
        logger: Optional experiment logger that receives the video.
        reward_channel: The reward channel the reported episode returns sum.
        reward_aggregation: How a decision's per-step rewards of that channel
            count in the return (``last`` or ``sum``); the learner's setting.
        telemetry: Records every evaluation round in the run's telemetry.
    """

    def __init__(
        self,
        *,
        group: RolloutGroup | None = None,
        open_group: Callable[[], RolloutGroup] | None = None,
        shared_group: bool = False,
        num_episodes: int,
        max_decisions: int,
        reset_seed: int | None = None,
        record_video: bool = False,
        video_dir: str | None = None,
        image_keys: Sequence[str] | None = None,
        video_fps: int = 20,
        logger: Logger | None = None,
        reward_channel: str = "env",
        reward_aggregation: RewardAggregation = "last",
        episode_record_path: str | None = None,
        telemetry: RolloutRecorder | None = None,
    ) -> None:
        if (group is None) == (open_group is None):
            raise ValueError("provide exactly one of group or open_group")
        if shared_group and group is None:
            raise ValueError("shared_group requires the training group as group")
        if num_episodes < 1:
            raise ValueError("num_episodes must be a positive integer")
        if max_decisions < 1:
            raise ValueError("max_decisions must be a positive integer")
        if record_video and video_dir is None:
            raise ValueError("record_video requires a video_dir")
        self._group = group
        self._open_group = open_group
        self._shared_group = bool(shared_group)
        self._group_version = 0
        self.num_episodes = int(num_episodes)
        self.max_decisions = int(max_decisions)
        self.reset_seed = reset_seed
        self.record_video = bool(record_video)
        self.video_dir = video_dir
        self.image_keys = None if image_keys is None else list(image_keys)
        self.video_fps = int(video_fps)
        self.logger = logger
        self.reward_channel = reward_channel
        if reward_aggregation not in ("sum", "last"):
            raise ValueError(f"reward_aggregation must be 'sum' or 'last', got {reward_aggregation!r}")
        self.reward_aggregation = reward_aggregation
        self.episode_record_path = episode_record_path
        self.telemetry = telemetry
        self._evaluation_index = 0

    def evaluate(self, policy_state: PolicyState, *, env_steps: int = 0) -> dict[str, float]:
        """Score ``policy_state`` on the evaluation workers, publishing it first unless the group is shared.

        ``env_steps`` is the run's environment step count, the step the video row is logged at.
        """
        if self._open_group is not None:
            group = self._open_group()
            try:
                group.set_policy_state(policy_state)
                return self._evaluate_on(group, policy_state, env_steps)
            finally:
                group.close()
        assert self._group is not None
        if self._shared_group:
            return self._evaluate_on(self._group, policy_state, env_steps)
        if policy_state.version > self._group_version:
            self._group.set_policy_state(policy_state)
            self._group_version = policy_state.version
        metrics = self._evaluate_on(self._group, policy_state, env_steps)
        # Nothing reads this group's policy-server counters; clear them so they stay bounded.
        drain_policy_service_metrics(self._group)
        return metrics

    @staticmethod
    def _log_worker_time(results: Sequence[RolloutResult], wall_seconds: float) -> None:
        """Where the evaluation's time went, summed over workers, so a slow run can be attributed."""
        logger.info(f"[RolloutEvaluator] collect wall={wall_seconds:.0f}s" + _worker_time_summary(results))

    def _evaluate_on(self, group: RolloutGroup, policy_state: PolicyState, env_steps: int) -> dict[str, float]:
        if self.num_episodes < group.num_envs:
            raise ValueError(
                f"eval_num_episodes ({self.num_episodes}) must be at least "
                f"eval_num_envs ({group.num_envs})"
            )
        if self.num_episodes % group.num_envs != 0:
            raise ValueError(
                f"eval_num_episodes ({self.num_episodes}) must be an exact multiple "
                f"of eval_num_envs ({group.num_envs})"
            )
        rounds = self.num_episodes // group.num_envs
        if group.num_envs > 1:
            logger.info(
                f"[RolloutEvaluator] {group.num_envs} environments x {rounds} episodes = "
                f"{self.num_episodes} scored episodes"
            )
        request = RolloutRequest(
            episodes_per_environment=rounds,
            mode="eval",
            reset_seed=self.reset_seed,
            max_decisions_per_episode=self.max_decisions,
        )
        started, wall_start = time.perf_counter(), time.time()
        results = group.collect([request] * group.num_rollout_workers)
        self._log_worker_time(results, time.perf_counter() - started)
        self._check_versions(results, policy_state.version)
        if self.telemetry is not None:
            self.telemetry.ingest(results, mode="eval", wall_start=wall_start, publish_version=policy_state.version)
        outcomes = episode_outcomes(results, reward_channel=self.reward_channel, reward_aggregation=self.reward_aggregation)
        if len(outcomes) != self.num_episodes:
            raise RuntimeError(
                f"evaluation returned {len(outcomes)} episodes, expected exactly "
                f"{self.num_episodes}"
            )
        self._write_episode_records(outcomes, policy_state)
        for index, outcome in enumerate(outcomes, start=1):
            running = sum(item.success for item in outcomes[:index]) / index
            logger.info(
                f"[eval] episode {index}/{len(outcomes)} (env {outcome.environment_index}) "
                f"success={outcome.success} running SR={100.0 * running:.1f}%"
            )
        metrics = summarize_episodes(outcomes)
        if not self.record_video:
            return metrics
        if group.num_envs > 1:
            logger.info("[RolloutEvaluator] video recording skipped for batched evaluation")
            return metrics
        metrics.update(self._record_video(group, policy_state, env_steps))
        return metrics

    def _write_episode_records(
        self,
        outcomes: Sequence[EpisodeOutcome],
        policy_state: PolicyState,
    ) -> None:
        """Append compact, simulator-neutral episode provenance for this evaluation."""
        if self.episode_record_path is None:
            self._evaluation_index += 1
            return
        parent = os.path.dirname(os.path.abspath(self.episode_record_path))
        os.makedirs(parent, exist_ok=True)
        with open(self.episode_record_path, "a", encoding="utf-8") as stream:
            for outcome in outcomes:
                record = {
                    "evaluation_index": self._evaluation_index,
                    "policy_version": policy_state.version,
                    "environment_index": outcome.environment_index,
                    "episode_id": outcome.episode_id,
                    "reset_seed": outcome.reset_seed,
                    "reset_seed_scope": outcome.reset_seed_scope,
                    "policy_seed": outcome.policy_seed,
                    "case_seed": outcome.case_seed,
                    "round_index": outcome.round_index,
                    "initial_state_fingerprint": outcome.initial_state_fingerprint,
                    "terminal_reason": outcome.terminal_reason,
                    "decisions": outcome.decisions,
                    "total_reward": outcome.total_reward,
                    "success": outcome.success,
                    "task_metrics": dict(outcome.task_metrics),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._evaluation_index += 1

    @staticmethod
    def _check_versions(results: Sequence[RolloutResult], version: int) -> None:
        for result in results:
            if result.policy_version != version:
                raise RuntimeError(
                    f"evaluation worker {result.worker_index} acted with policy version "
                    f"{result.policy_version}, expected {version}"
                )

    def _record_video(self, group: RolloutGroup, policy_state: PolicyState, env_steps: int) -> dict[str, float]:
        """Play one more episode with frame capture and save it as an MP4."""
        request = RolloutRequest(
            episodes_per_environment=1,
            mode="eval",
            capture_frames=True,
            max_decisions_per_episode=self.max_decisions,
        )
        results = group.collect([request] * group.num_rollout_workers)
        self._check_versions(results, policy_state.version)
        (outcome,) = episode_outcomes(results, reward_channel=self.reward_channel, reward_aggregation=self.reward_aggregation)
        frames = [frame for result in results for frame in result.frames]
        saved = self._save_video(frames, env_steps)
        return {
            "video_reward": outcome.total_reward,
            "video_steps": float(outcome.decisions),
            "video_saved": float(saved),
            "video_success": float(outcome.success),
        }

    def _save_video(self, frames: Sequence[Mapping[str, np.ndarray]], env_steps: int) -> bool:
        if not frames:
            return False
        image_keys = (
            self.image_keys if self.image_keys is not None else detect_image_keys(list(frames[0]))
        )
        stacked = [stack_camera_frames(frame, image_keys) for frame in frames]
        bgr_frames = [frame for frame in stacked if frame is not None]
        if not bgr_frames:
            return False
        assert self.video_dir is not None
        os.makedirs(self.video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_dir, f"eval_{timestamp}_all_cameras.mp4")
        if self.logger is not None:
            rgb = np.stack([frame[..., ::-1].transpose(2, 0, 1) for frame in bgr_frames])
            self.logger.log_video(
                "eval_video",
                video=torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0),
                step=env_steps,
                prefix="eval",
            )
        write_video(bgr_frames, video_path, fps=self.video_fps)
        logger.success(f"Saved evaluation video: {video_path}")
        return True

    def close(self) -> None:
        if self._group is not None:
            self._group.close()
