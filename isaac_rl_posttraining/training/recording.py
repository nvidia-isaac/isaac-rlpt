# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lineage, telemetry rows and optional persistence for collected rollouts.

A :class:`RolloutRecorder` takes the results of one collection round, gives
every decision its lineage (data source, rollout worker, policy version and
policy lag at ingest, origin) and an episode id unique across the run, and
writes the ``transitions``, ``episodes`` and ``rounds`` telemetry rows. The
data source drives one for its own rounds; a trainer that collects without a
data source (the on-policy loop, an evaluator) can drive one directly::

    recorder = RolloutRecorder(sink, run_id="run", source="eval")
    recorder.ingest(results, mode="eval", wall_start=started, publish_version=version)

An :class:`EpisodePersister` streams completed episodes to disk in the saved
replay-buffer format, so collected data can be reloaded with ``load_buffer``.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from isaac_rl_posttraining.buffers.replay import ReplayBuffer, RewardAggregation
from isaac_rl_posttraining.buffers.storage import save_buffer_part
from isaac_rl_posttraining.data import Transition
from isaac_rl_posttraining.data.transitions import ORIGINS, Origin
from isaac_rl_posttraining.runtimes.base import RolloutResult
from isaac_rl_posttraining.training.telemetry import NullTelemetrySink, TelemetrySink

__all__ = [
    "DEFAULT_SOURCE",
    "EpisodePersister",
    "PersistSpec",
    "RolloutRecorder",
    "RunCounters",
    "persist_from_config",
]

#: The source name whose episode ids stay plain integers (today's numbering).
DEFAULT_SOURCE = "train"


@dataclass
class RunCounters:
    """Run-wide counters shared by every recorder of one run (train, eval, imagined data)."""

    env_steps: int = 0
    learner_step: int = 0
    publish_version: int = 0
    rounds: int = 0


@dataclass(frozen=True)
class PersistSpec:
    """Stream completed episodes to ``path`` as ``part-NNNNN`` saved buffers.

    Every ``every_episodes`` completed episodes become one part; the episodes
    still pending are written when the data source closes. Camera frames
    (``uint8 [H, W, 3]`` observation entries) are dropped unless
    ``keep_frames``. Reload everything with ``load_buffer(path)``.
    """

    path: str
    every_episodes: int = 100
    keep_frames: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, os.PathLike)) or not os.fspath(self.path):
            raise ValueError("PersistSpec.path must be a non-empty path")
        object.__setattr__(self, "path", os.fspath(self.path))
        if isinstance(self.every_episodes, bool) or not isinstance(self.every_episodes, int) or self.every_episodes < 1:
            raise ValueError(f"every_episodes must be a positive integer, got {self.every_episodes!r}")


def persist_from_config(section: Mapping[str, Any] | None, run_dir: str | os.PathLike[str]) -> PersistSpec | None:
    """The :class:`PersistSpec` of a run config's ``data.persist`` section; ``None`` when ``path`` is unset.

    A relative ``path`` is under ``run_dir``.
    """
    if section is None or section.get("path") is None:
        return None
    path = os.path.join(os.fspath(run_dir), os.fspath(section["path"]))
    return PersistSpec(
        path,
        every_episodes=int(section.get("every_episodes", 100)),
        keep_frames=bool(section.get("keep_frames", False)),
    )


def _frame_components(transition: Transition) -> set[str]:
    return {
        name
        for observation in (transition.observation, transition.next_observation)
        for name, value in observation.arrays().items()
        if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3
    }


class EpisodePersister:
    """Collect transitions per episode and write completed episodes as buffer parts."""

    def __init__(
        self, spec: PersistSpec, *, reward_channel: str = "env", reward_aggregation: RewardAggregation = "last"
    ) -> None:
        self.spec = spec
        self.reward_channel = reward_channel
        self.reward_aggregation = reward_aggregation
        self._open: dict[int | str, list[Transition]] = {}
        self._complete: list[list[Transition]] = []
        self.parts_written = 0
        self.episodes_written = 0

    def add(self, transition: Transition) -> None:
        episode = self._open.setdefault(transition.episode_id, [])
        episode.append(transition)
        if transition.ended:
            self._complete.append(self._open.pop(transition.episode_id))
            if len(self._complete) >= self.spec.every_episodes:
                self.flush()

    def flush(self) -> None:
        """Write the completed episodes not yet on disk as one part."""
        if not self._complete:
            return
        episodes, self._complete = self._complete, []
        transitions = [transition for episode in episodes for transition in episode]
        drop: set[str] = set()
        if not self.spec.keep_frames:
            for transition in transitions:
                drop |= _frame_components(transition)
        part = ReplayBuffer(
            len(transitions),
            drop_components=sorted(drop),
            reward_channel=self.reward_channel,
            reward_aggregation=self.reward_aggregation,
        )
        for transition in transitions:
            part.add(transition)
        save_buffer_part(part, self.spec.path)
        self.parts_written += 1
        self.episodes_written += len(episodes)

    def close(self) -> None:
        """Write what is complete; episodes still running are not persisted."""
        self.flush()


@dataclass
class _OpenEpisode:
    number: int
    environment_index: int
    first_version: int
    wall_start: float
    decisions: int = 0
    env_steps: int = 0
    total: float = 0.0
    succeeded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutRecorder:
    """Lineage and telemetry for the rounds of one data source.

    Args:
        telemetry: Where rows go; ``None`` records lineage only.
        run_id: The run's id in every row.
        source: The data source's name (``train``, ``eval``, ``imagined``, ...).
        origin: ``real`` for simulator or robot data, ``model`` for imagined data.
        episode_namespace: Prefix that makes episode ids unique across data
            sources feeding one buffer (ids become ``"<namespace>:<n>"``);
            defaults to plain integers for ``train`` and to ``source`` otherwise.
        reward_channel / reward_aggregation: How a decision's reward is read
            for the ``reward`` and ``return`` columns (as the learner reads it).
        counters: Run-wide counters to share with other recorders of the run.
        counts_env_steps: Whether this source's steps advance the run's env-step
            total (training data does, evaluation does not).
    """

    def __init__(
        self,
        telemetry: TelemetrySink | None = None,
        *,
        run_id: str = "",
        source: str = DEFAULT_SOURCE,
        origin: Origin = "real",
        episode_namespace: str | None = None,
        reward_channel: str = "env",
        reward_aggregation: RewardAggregation = "last",
        counters: RunCounters | None = None,
        counts_env_steps: bool = True,
    ) -> None:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        self.telemetry: TelemetrySink = telemetry if telemetry is not None else NullTelemetrySink()
        self.run_id = run_id
        self.source = source
        self.origin: Origin = origin
        self.episode_namespace = (
            episode_namespace if episode_namespace is not None else (None if source == DEFAULT_SOURCE else source)
        )
        self.reward_channel = reward_channel
        self.reward_aggregation: RewardAggregation = reward_aggregation
        self.counters = counters if counters is not None else RunCounters()
        self.counts_env_steps = counts_env_steps
        self._numbers: dict[tuple[int, int | str], int] = {}
        self._next_number = 0
        self._episodes: dict[int, _OpenEpisode] = {}
        #: Policy lag of the decisions of the last round (mean, max).
        self.last_policy_lag: tuple[float, int] = (0.0, 0)
        #: Whether each decision of the last round reaches the buffer (``ingest``'s ``keep``).
        self.last_kept: tuple[bool, ...] = ()

    def child(self, source: str, *, origin: Origin | None = None, counts_env_steps: bool = False) -> RolloutRecorder:
        """A recorder for another source of the same run (same sink, run id and counters)."""
        return RolloutRecorder(
            self.telemetry,
            run_id=self.run_id,
            source=source,
            origin=self.origin if origin is None else origin,
            reward_channel=self.reward_channel,
            reward_aggregation=self.reward_aggregation,
            counters=self.counters,
            counts_env_steps=counts_env_steps,
        )

    def episode_id(self, number: int) -> int | str:
        """The buffer's episode id of this source's ``number``-th episode."""
        return number if self.episode_namespace is None else f"{self.episode_namespace}:{number}"

    def ingest(
        self,
        results: Sequence[RolloutResult],
        *,
        mode: str,
        wall_start: float,
        wall_end: float | None = None,
        publish_version: int | None = None,
        queue_depth: int | None = None,
        keep: Callable[[Transition], bool] | None = None,
    ) -> list[tuple[int, Transition]]:
        """Tag every decision of one round with its lineage and record the round.

        Returns ``(environment_index, transition)`` pairs in worker order,
        each transition carrying its run-unique episode id. ``publish_version``
        is the newest version published when the round is ingested (the
        counters' when omitted); ``policy_lag`` is measured against it.
        ``wall_end`` is when the workers finished the round (the ingest time
        when omitted), so a round ingested late still records its own length.
        ``keep`` (called once per tagged decision, in order) marks the
        decisions that reach the buffer; ``last_kept`` holds its answers and
        the round's row counts the others (``dropped_stale``).
        """
        current = self.counters.publish_version if publish_version is None else publish_version
        received = time.time()
        tagged: list[tuple[int, Transition]] = []
        transition_rows: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        lags: list[int] = []
        kept: list[bool] = []
        round_steps = 0
        for result in sorted(results, key=lambda item: item.worker_index):
            count = max(len(result.decisions), 1)
            inference = _share(result.timings, "action_inference_seconds", count)
            environment = _share(result.timings, "environment_step_seconds", count)
            for collected in result.decisions:
                lag = max(current - result.policy_version, 0)
                transition, number = self._tag(
                    collected.environment_index, collected.transition, result.worker_index, lag
                )
                tagged.append((collected.environment_index, transition))
                kept.append(True if keep is None else keep(transition))
                lags.append(lag)
                round_steps += transition.steps
                if self.counts_env_steps:
                    self.counters.env_steps += transition.steps
                transition_rows.append(
                    self._transition_row(
                        transition, number, mode, received, inference, environment, result.policy_version
                    )
                )
                ended = self._advance_episode(number, collected.environment_index, transition, result.policy_version)
                if ended is not None:
                    episode_rows.append(ended)
        self.last_policy_lag = (float(np.mean(lags)) if lags else 0.0, max(lags, default=0))
        self.last_kept = tuple(kept)
        self.telemetry.write_transitions(transition_rows)
        self.telemetry.write_episodes(episode_rows)
        self.telemetry.write_rounds(
            [
                {
                    "run_id": self.run_id,
                    "round_index": self.counters.rounds,
                    "mode": mode,
                    "wall_start": float(wall_start),
                    "wall_end": received if wall_end is None else float(wall_end),
                    "env_steps": round_steps,
                    "decisions": len(tagged),
                    "episodes": len(episode_rows),
                    "publish_version": current,
                    "learner_step": self.counters.learner_step,
                    "policy_lag_mean": self.last_policy_lag[0],
                    "policy_lag_max": self.last_policy_lag[1],
                    "queue_depth": queue_depth,
                    "dropped_stale": None if keep is None else kept.count(False),
                }
            ]
        )
        self.counters.rounds += 1
        return tagged

    def log_learner(
        self,
        step: int,
        metrics: Mapping[str, Any],
        *,
        env_steps: int | None = None,
        publish_version: int | None = None,
        wall_time: float | None = None,
    ) -> None:
        """Record one learner update (non-numeric and non-finite metrics are left out)."""
        self.counters.learner_step = int(step)
        scalars = {}
        for name, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                scalars[str(name)] = number
        self.telemetry.write_learner(
            [
                {
                    "run_id": self.run_id,
                    "learner_step": int(step),
                    "env_steps": self.counters.env_steps if env_steps is None else int(env_steps),
                    "wall_time": time.time() if wall_time is None else float(wall_time),
                    "publish_version": publish_version,
                    "metrics": scalars,
                }
            ]
        )

    def _tag(
        self, environment_index: int, transition: Transition, worker_index: int, lag: int
    ) -> tuple[Transition, int]:
        """The transition with its lineage and run-unique episode id, and the episode's number in this source."""
        key = (environment_index, transition.episode_id)
        number = self._numbers.get(key)
        if number is None:
            number = self._next_number
            self._next_number += 1
            self._numbers[key] = number
        if transition.ended:
            del self._numbers[key]
        metadata = replace(transition.metadata, source=self.source, worker_index=worker_index, policy_lag=lag)
        tagged = replace(transition, episode_id=self.episode_id(number), origin=self.origin, metadata=metadata)
        return tagged, number

    def _transition_row(
        self,
        transition: Transition,
        number: int,
        mode: str,
        received: float,
        inference: float | None,
        environment: float | None,
        result_version: int,
    ) -> dict[str, Any]:
        metadata = transition.metadata
        channels = None
        if len(transition.rewards) > 1:
            channels = {name: float(values.sum()) for name, values in transition.rewards.items()}
        version = result_version if metadata.policy_version is None else metadata.policy_version
        return {
            "run_id": self.run_id,
            "source": self.source,
            "origin": transition.origin,
            "worker_index": metadata.worker_index,
            "environment_index": metadata.environment_index if metadata.environment_index is not None else -1,
            "episode_id": number,
            "decision_index": metadata.decision_index if metadata.decision_index is not None else -1,
            "steps": transition.steps,
            "env_steps": self.counters.env_steps,
            "policy_version": version,
            "policy_lag": metadata.policy_lag,
            "mode": mode,
            "reward": self._reward(transition),
            "reward_env": float(transition.rewards["env"].sum()),
            "reward_channels": channels,
            "terminated": bool(transition.terminated[-1]),
            "truncated": bool(transition.truncated[-1]),
            "success": transition.succeeded,
            "t_obs": metadata.wall_time if metadata.wall_time is not None else received,
            "t_received": received,
            "inference_seconds": inference,
            "env_seconds": environment,
            "network_seconds": None,
        }

    def _reward(self, transition: Transition) -> float:
        channel = self.reward_channel if self.reward_channel in transition.rewards else "env"
        return transition.decision_reward(channel, aggregation=self.reward_aggregation)

    def _advance_episode(
        self, number: int, environment_index: int, transition: Transition, result_version: int
    ) -> dict[str, Any] | None:
        """Account one decision to its episode; the episode's row when it ended."""
        version = result_version if transition.metadata.policy_version is None else transition.metadata.policy_version
        t_obs = transition.metadata.wall_time if transition.metadata.wall_time is not None else time.time()
        episode = self._episodes.get(number)
        if episode is None:
            episode = _OpenEpisode(number, environment_index, version, t_obs)
            self._episodes[number] = episode
        episode.decisions += 1
        episode.env_steps += transition.steps
        episode.total += self._reward(transition)
        episode.succeeded = episode.succeeded or transition.succeeded
        if not transition.ended:
            return None
        del self._episodes[number]
        if transition.metadata.extra.get("decision_cap_reached"):
            reason = "decision_cap"
        elif transition.terminated[-1]:
            reason = "terminated"
        else:
            reason = "truncated"
        return {
            "run_id": self.run_id,
            "source": self.source,
            "episode_id": number,
            "environment_index": environment_index,
            "length_decisions": episode.decisions,
            "length_env_steps": episode.env_steps,
            "return": episode.total,
            "success": episode.succeeded,
            "terminal_reason": reason,
            "policy_version_first": episode.first_version,
            "policy_version_last": version,
            "wall_start": episode.wall_start,
            "wall_end": t_obs,
        }


def _share(timings: Mapping[str, float], name: str, count: int) -> float | None:
    """A worker's round total of one timing, shared evenly over its decisions."""
    if name not in timings:
        return None
    return float(timings[name]) / count
