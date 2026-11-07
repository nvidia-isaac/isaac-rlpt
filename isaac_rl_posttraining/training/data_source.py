# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data source: collect rounds from a rollout group into a replay buffer, with lineage and telemetry.

A :class:`DataSource` submits collection rounds to a rollout group, stamps
every decision with its lineage (source name, rollout worker, policy version
and lag, origin, a run-unique episode id), pushes it into the replay buffer,
records telemetry rows, and optionally streams completed episodes to disk::

    source = DataSource(group, InProcessBufferSink(buffer),
                        telemetry=open_run_telemetry(run_dir), run_id="run")
    imagined = DataSource(model_group, InProcessBufferSink(buffer),
                          source="imagined", origin="model")   # ids "imagined:<n>"
    episodes = DataSource(group, None, count_by="episode")
    episodes.run(can_train=True, reset_seed=7)   # on-policy: the round is in episodes.last_results

``RuntimeDataSource`` is the previous name of the same class.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from isaac_rl_posttraining.buffers.replay import RewardAggregation
from isaac_rl_posttraining.data import Transition
from isaac_rl_posttraining.data.transitions import Origin
from isaac_rl_posttraining.rollouts.episode_tracker import EpisodeTracker
from isaac_rl_posttraining.rollouts.transition_sink import TransitionSink
from isaac_rl_posttraining.runtimes.base import (
    DecisionMode,
    PolicyState,
    RolloutGroup,
    RolloutRequest,
    RolloutResult,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.concurrency import AsyncCall
from isaac_rl_posttraining.training.recording import (
    DEFAULT_SOURCE,
    EpisodePersister,
    PersistSpec,
    RolloutRecorder,
    RunCounters,
)
from isaac_rl_posttraining.training.telemetry import TelemetrySink

__all__ = [
    "DataSource",
    "PendingCollection",
    "PersistSpec",
    "RuntimeDataSource",
    "collector_metrics",
    "rollout_count_by",
]


@dataclass(frozen=True)
class PendingCollection:
    """A collection submitted by :meth:`DataSource.start` and not yet ingested.

    ``policy_version`` is the workers' version at submission; the results carry
    it whatever is published while they are in flight. ``started`` is the
    submission time on the wall clock (``time.time()``), the same clock the
    call's ``finished_at`` is stamped with where the round ran.
    """

    call: AsyncCall[tuple[RolloutResult, ...]]
    policy_version: int
    mode: DecisionMode
    started: float

    def duration(self) -> float:
        """Seconds from the submission to the round's end on the workers, however late it was ingested."""
        finished_at = self.call.finished_at()
        if finished_at is None:
            raise RuntimeError("the round's duration is known once its result has been retrieved")
        return finished_at - self.started


def collector_metrics(timings: Mapping[str, float]) -> dict[str, float]:
    """Per-decision and per-step collector costs from the workers' summed timings.

    Workers report seconds per phase plus how many policy calls and simulator
    step calls they made (``policy_calls``, ``environment_step_calls``; every
    step call advances all of a worker's environments by one step). A batched
    decision is one policy call for all active environments, so dividing by
    calls gives the cost of one decision and one lockstep simulator step -- the
    numbers a layout comparison is made of. Phases a worker did not time are
    left out rather than reported as zero.
    """
    metrics: dict[str, float] = {}
    policy_calls = timings["policy_calls"] if "policy_calls" in timings else 0.0
    step_calls = timings["environment_step_calls"] if "environment_step_calls" in timings else 0.0
    if policy_calls > 0.0:
        metrics["collector/decode_s_per_decision"] = timings["action_inference_seconds"] / policy_calls
        cycle = timings["action_inference_seconds"]
        if "environment_step_seconds" in timings:
            cycle += timings["environment_step_seconds"]
        # Workers before the rename report the next-observation pass as successor_inference_seconds.
        for phase in ("next_observation_inference_seconds", "successor_inference_seconds"):
            if phase in timings:
                cycle += timings[phase]
                break
        metrics["collector/cycle_s_per_decision"] = cycle / policy_calls
    if step_calls > 0.0:
        metrics["collector/env_step_s_per_step"] = timings["environment_step_seconds"] / step_calls
    if "environment_reset_seconds" in timings:
        metrics["collector/env_reset_s"] = timings["environment_reset_seconds"]
    return metrics


class DataSource:
    """Ingest remote rollout results into the controller-owned replay buffer.

    The runtime moves serialized result records, never the replay-buffer
    object itself.
    This preserves one authoritative sampling and mutation point for DSRL,
    RFS, and direct-action SAC while rollout workers remain independently
    placeable. An on-policy loop passes ``sink=None``: nothing is buffered, and
    it builds its trajectories from ``last_results``. ``reward_channel`` names the reward channel the collection
    metrics (episode returns) are summed from, each decision counted under
    ``reward_aggregation`` as the learner sees it.

    Lineage and data products:
        source: This data source's name. Episode ids stay plain integers for
            ``"train"`` and become ``"<source>:<n>"`` otherwise, so several
            data sources can feed one buffer.
        origin: ``"real"`` (simulator or robot) or ``"model"`` (imagined).
        telemetry: Sink for the ``transitions``, ``episodes``, ``rounds`` and
            ``learner`` rows (see :mod:`~isaac_rl_posttraining.training.telemetry`);
            ``None`` records nothing. The caller owns (and closes) the sink.
        run_id: The run id written in every telemetry row (by default the
            sink's ``run_id``, the run directory's name, when it has one).
        persist: Stream completed episodes to disk (:class:`PersistSpec`).
        counters: Run-wide counters to share with other recorders of the run.

    Staleness: with ``max_policy_lag`` set, a decision whose policy is more
    than that many versions behind the learner (``learner_version`` of
    :meth:`run` / :meth:`finish`) is dropped before the buffer, and so is the
    rest of its episode (the buffer stores episodes without gaps). Dropped
    decisions still count towards the episode metrics, the telemetry and
    persistence; ``dropped_stale_last_run`` / ``dropped_stale_total`` count them.
    """

    def __init__(
        self,
        group: RolloutGroup,
        sink: TransitionSink | None,
        *,
        request_count: int = 1,
        count_by: Literal["decision", "episode"] = "decision",
        device: torch.device | None = None,
        reward_channel: str = "env",
        reward_aggregation: RewardAggregation = "last",
        source: str = DEFAULT_SOURCE,
        origin: Origin = "real",
        telemetry: TelemetrySink | None = None,
        run_id: str | None = None,
        persist: PersistSpec | None = None,
        counters: RunCounters | None = None,
        max_policy_lag: int | None = None,
    ) -> None:
        if isinstance(request_count, bool) or not isinstance(request_count, int):
            raise TypeError("request_count must be a positive integer")
        if request_count < 1:
            raise ValueError("request_count must be a positive integer")
        if count_by not in ("decision", "episode"):
            raise ValueError("count_by must be 'decision' or 'episode'")
        if not isinstance(reward_channel, str) or not reward_channel:
            raise ValueError("reward_channel must be a non-empty string")
        if reward_aggregation not in ("sum", "last"):
            raise ValueError(f"reward_aggregation must be 'sum' or 'last', got {reward_aggregation!r}")
        if max_policy_lag is not None and (
            isinstance(max_policy_lag, bool) or not isinstance(max_policy_lag, int) or max_policy_lag < 0
        ):
            raise ValueError("max_policy_lag must be a non-negative integer or None")
        self.group = group
        self.max_policy_lag = max_policy_lag
        self.dropped_stale_last_run = 0
        self.dropped_stale_total = 0
        #: Episodes that lost a decision to staleness; their remaining decisions are dropped too.
        self._cut_episodes: set[int | str] = set()
        self._sink = sink
        self.request_count = request_count
        self.count_by = count_by
        self.device = device if device is not None else torch.device("cpu")
        self.reward_channel = reward_channel
        self.reward_aggregation = reward_aggregation
        self.num_envs = group.num_envs
        self.episode_tracker = EpisodeTracker(self.num_envs)
        self.total_episodes = 0
        #: Decisions the last round put into the buffer (stale ones excluded).
        self.decisions_collected_last_run = 0
        self.last_timings: dict[str, float] = {}
        #: The transitions the last round put into the buffer, in arrival order.
        self.last_transitions: tuple[Transition, ...] = ()
        #: The last round's results as the workers returned them (before lineage tagging).
        self.last_results: tuple[RolloutResult, ...] = ()
        self._policy_version = 0
        self._weights: WeightsRef | None = None
        self.recorder = RolloutRecorder(
            telemetry,
            run_id=run_id if run_id is not None else str(getattr(telemetry, "run_id", "")),
            source=source,
            origin=origin,
            reward_channel=reward_channel,
            reward_aggregation=reward_aggregation,
            counters=counters,
        )
        self._persister = (
            None
            if persist is None
            else EpisodePersister(persist, reward_channel=reward_channel, reward_aggregation=reward_aggregation)
        )
        self._in_flight: PendingCollection | None = None
        self._closed = False

    @property
    def source(self) -> str:
        return self.recorder.source

    @property
    def origin(self) -> Origin:
        return self.recorder.origin

    @property
    def policy_version(self) -> int:
        """The version of the weights the rollout workers hold."""
        return self._policy_version

    def run(
        self, can_train: bool, *, learner_version: int | None = None, reset_seed: int | None = None
    ) -> dict[str, float]:
        """Collect one round on the calling thread and ingest it.

        ``learner_version`` is the learner's newest published version (by
        default the one the workers hold); policy lag is measured against it.
        ``reset_seed`` seeds the round's resets (episode-counted rounds only).
        """
        self._require_open()
        mode = _decision_mode(can_train)
        started = time.time()
        results = self.group.collect(self._requests(mode, reset_seed))
        return self._ingest(
            results,
            expected_version=self._policy_version,
            learner_version=learner_version,
            mode=mode,
            started=started,
        )

    def start(self, can_train: bool, *, reset_seed: int | None = None) -> PendingCollection:
        """Submit one round and return without waiting for it (``reset_seed`` as in :meth:`run`).

        The controller trains on the previous round meanwhile and ingests this
        one with :meth:`finish`. The workers keep the weights they hold now until
        the round ends: a :meth:`publish` issued while it is in flight applies
        afterwards, on every runtime.
        """
        self._require_open()
        if self._in_flight is not None:
            raise RuntimeError("a collection is already in flight; finish it before starting another")
        mode = _decision_mode(can_train)
        started = time.time()  # before the submission, so the round's end can never precede its start
        pending = PendingCollection(
            call=self.group.collect_async(self._requests(mode, reset_seed)),
            policy_version=self._policy_version,
            mode=mode,
            started=started,
        )
        self._in_flight = pending
        return pending

    def finish(self, pending: PendingCollection, *, learner_version: int | None = None) -> dict[str, float]:
        """Wait for the round submitted by :meth:`start` and ingest it (``learner_version`` as in :meth:`run`)."""
        self._require_open()
        if pending is not self._in_flight:
            raise RuntimeError("finish() takes the collection returned by the last start()")
        self._in_flight = None
        results = pending.call.result()
        return self._ingest(
            results,
            expected_version=pending.policy_version,
            learner_version=learner_version,
            mode=pending.mode,
            started=pending.started,
            finished=pending.call.finished_at(),
        )

    def _requests(self, mode: DecisionMode, reset_seed: int | None) -> list[RolloutRequest]:
        if self.count_by == "decision":
            if reset_seed is not None:
                raise ValueError("reset_seed needs a data source that counts rounds in episodes (count_by='episode')")
            request = RolloutRequest(
                decisions_per_environment=self.request_count,
                mode=mode,
            )
        else:
            request = RolloutRequest(
                episodes_per_environment=self.request_count,
                mode=mode,
                reset_seed=reset_seed,
            )
        return [request] * self.group.num_rollout_workers

    def _ingest(
        self,
        results: Sequence[RolloutResult],
        *,
        expected_version: int,
        learner_version: int | None,
        mode: DecisionMode,
        started: float,
        finished: float | None = None,
    ) -> dict[str, float]:
        """Push the round's transitions into the sink; ``expected_version`` is the version the results must carry."""
        self.decisions_collected_last_run = 0
        timings: dict[str, float] = {}
        for result in results:
            if result.worker_index >= self.group.num_rollout_workers:
                raise RuntimeError(
                    f"runtime returned worker {result.worker_index} for a "
                    f"{self.group.num_rollout_workers}-worker group"
                )
            if result.policy_version != expected_version:
                raise RuntimeError(
                    f"worker {result.worker_index} returned policy version "
                    f"{result.policy_version}; controller expects "
                    f"{expected_version}"
                )
            for collected in result.decisions:
                if collected.environment_index >= self.num_envs:
                    raise RuntimeError(
                        f"runtime returned environment {collected.environment_index} for a "
                        f"{self.num_envs}-environment group"
                    )
        for result in sorted(results, key=lambda item: item.worker_index):
            for name, seconds in result.timings.items():
                timings[name] = timings.get(name, 0.0) + float(seconds)

        tagged = self.recorder.ingest(
            results,
            mode=mode,
            wall_start=started,
            wall_end=finished,
            publish_version=self._policy_version if learner_version is None else learner_version,
            keep=None if self.max_policy_lag is None else self._keep,
        )
        kept = self.recorder.last_kept if self.max_policy_lag is not None else (True,) * len(tagged)
        ingested: list[Transition] = []
        self.dropped_stale_last_run = 0
        for (environment_index, transition), keep in zip(tagged, kept, strict=True):
            if not keep:
                self.dropped_stale_last_run += 1
                if self._sink is not None:  # a cut episode never ends in the buffer
                    self._sink.discard_open(transition.episode_id)
            else:
                if self._sink is not None:
                    self._sink.push(transition)
                ingested.append(transition)
                self.decisions_collected_last_run += 1
            if self._persister is not None:
                self._persister.add(transition)
            self.episode_tracker.add_step(
                environment_index,
                transition.decision_reward(self.reward_channel, aggregation=self.reward_aggregation),
                transition.metadata.extra,
                bool(transition.terminated[-1]),
                bool(transition.truncated[-1]),
                transition.steps,
                success=transition.succeeded,
            )
            if transition.ended:
                self.episode_tracker.end_episode(environment_index)
                self.total_episodes += 1
        self.dropped_stale_total += self.dropped_stale_last_run
        self.last_results = tuple(results)
        self.last_transitions = tuple(ingested)
        self.last_timings = timings
        metrics = dict(self.episode_tracker.get_metrics())
        metrics.update(collector_metrics(timings))
        return metrics

    def _keep(self, transition: Transition) -> bool:
        """Whether a decision reaches the buffer: not too stale, and its episode not cut before."""
        episode = transition.episode_id
        lag = transition.metadata.policy_lag
        stale = self.max_policy_lag is not None and lag is not None and lag > self.max_policy_lag
        if not stale and episode not in self._cut_episodes:
            return True
        if transition.ended:
            self._cut_episodes.discard(episode)
        else:
            self._cut_episodes.add(episode)
        return False

    def log_learner(
        self,
        step: int,
        env_steps: int | None,
        metrics: Mapping[str, float],
        *,
        publish_version: int | None = None,
    ) -> None:
        """Record one learner update in the ``learner`` telemetry table (the trainer calls this)."""
        self.recorder.log_learner(step, metrics, env_steps=env_steps, publish_version=publish_version)

    def set_weights(self, weights: WeightsRef) -> None:
        """Send published weights to every rollout worker.

        Later results must carry this version, which is how stale workers
        are caught.
        """
        self._require_open()
        if weights.version <= self._policy_version:
            raise ValueError(
                f"policy state version must advance beyond {self._policy_version}, "
                f"got {weights.version}"
            )
        self._send(weights)
        self._policy_version = weights.version
        self._weights = weights
        self.recorder.counters.publish_version = weights.version

    def replace_group(self, open_group: Callable[[], RolloutGroup]) -> None:
        """Close the rollout workers, open their replacement with ``open_group`` and send it the last published weights.

        The old workers close first, because a remote simulator serves one
        client; this is how a loop that isolates evaluation in its own
        simulator processes switches between training and evaluation workers.
        """
        self._require_open()
        if self._in_flight is not None:
            raise RuntimeError("a collection is in flight; finish it before replacing the rollout workers")
        old = self.group
        try:
            old.close()
        # The old workers are released on a best-effort basis; the run continues on the new ones.
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"rollout group raised during cleanup: {exc!r}", RuntimeWarning, stacklevel=2)
        group = open_group()
        if (group.num_rollout_workers, group.num_envs) != (old.num_rollout_workers, old.num_envs):
            group.close()
            raise RuntimeError(
                "replacement rollout topology changed from "
                f"{old.num_rollout_workers} workers/{old.num_envs} envs to "
                f"{group.num_rollout_workers} workers/{group.num_envs} envs"
            )
        self.group = group
        if self._weights is not None:
            self._send(self._weights)

    def _send(self, weights: WeightsRef) -> None:
        versions = self.group.set_weights(weights)
        if versions != (weights.version,) * self.group.num_rollout_workers:
            raise RuntimeError(
                f"rollout workers acknowledged policy versions {versions}, "
                f"expected {weights.version} on every worker"
            )

    def publish(self, state: PolicyState) -> None:
        """Send a policy state to every rollout worker (:meth:`set_weights` with the state inline)."""
        self.set_weights(WeightsRef.of(state))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime data source is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Workers are never closed under a running collection; a round still in
        # flight is waited for and dropped, and its failure reported after the close.
        in_flight_error: Exception | None = None
        if self._in_flight is not None:
            pending, self._in_flight = self._in_flight, None
            try:
                pending.call.result()
            # The workers are closed first, then the failure is raised.
            except Exception as exc:  # noqa: BLE001
                in_flight_error = exc
        try:
            if self._persister is not None:
                self._persister.close()
            self.recorder.telemetry.flush()
        finally:
            self.group.close()
        if in_flight_error is not None:
            raise RuntimeError(
                "the collection in flight failed while the data source was closing"
            ) from in_flight_error


#: The previous name of :class:`DataSource`.
RuntimeDataSource = DataSource


def rollout_count_by(name: str) -> Literal["decision", "episode"]:
    """How a data source counts a collection request, from ``training.rollout_count_by`` (``step`` or ``episode``)."""
    if name in ("step", "decision"):
        return "decision"
    if name == "episode":
        return "episode"
    raise ValueError("training.rollout_count_by must be 'step' or 'episode'")


def _decision_mode(can_train: bool) -> DecisionMode:
    return "train" if can_train else "warmup"
