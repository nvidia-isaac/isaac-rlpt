# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The off-policy training loop over runtime rollout and learner groups.

The controller owns the replay buffer and the schedule; a rollout group
collects, a learner group updates, and an evaluator scores published policy
states. The schedule (warm-up gate, burst sizing, boundary-gated evaluation,
save-after-eval, retention) is the DSRL reference recipe's.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from loguru import logger
from tqdm import tqdm

from isaac_rl_posttraining.buffers import ReplayBuffer, save_buffer
from isaac_rl_posttraining.data import Transition
from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerHooks,
    LearnerStepResult,
    publish_weights,
)
from isaac_rl_posttraining.training.checkpoints import CheckpointStore, FingerprintSink
from isaac_rl_posttraining.training.data_source import PendingCollection
from isaac_rl_posttraining.training.on_policy_metrics import drain_policy_service_metrics
from isaac_rl_posttraining.training.off_policy_reporting import (
    buffer_stats,
    print_detailed_eval_metrics,
    print_detailed_metrics,
    rate,
    update_progress_bar,
)
from isaac_rl_posttraining.training.prefetch import learner_updates
from isaac_rl_posttraining.training.schedule import ScheduleSpec, as_schedule_spec

__all__ = [
    "MetricSink",
    "OffPolicyConfig",
    "OffPolicyDataSource",
    "OffPolicyTrainer",
    "PolicyEvaluator",
    "ReplayLearnerGroup",
    "observe_reward_stream",
]

def observe_reward_stream(
    hooks: LearnerHooks, transitions: Sequence[Transition], buffer: ReplayBuffer
) -> None:
    """Hand the decisions just stored to the learner, with the rewards its batches will read."""
    if transitions:
        hooks.observe_rewards(
            [t.decision_reward(buffer.reward_channel, aggregation=buffer.reward_aggregation) for t in transitions],
            [bool(t.terminated[-1]) for t in transitions],
            [bool(t.truncated[-1]) for t in transitions],
            [t.episode_id for t in transitions],
        )


#: Receives already-namespaced metrics (``eval/…``, ``train/…``, ``buffer/…``)
#: with the environment step they belong to.
MetricSink = Callable[[Mapping[str, float], int], None]


class OffPolicyDataSource(Protocol):
    """The collection seam the trainer drives (:class:`~isaac_rl_posttraining.training.data_source.DataSource`).

    ``run`` collects and ingests one round on the calling thread; ``start`` and
    ``finish`` split that into a submission and a later ingest, which is how the
    overlapped and async schedules run a round during the learner's burst.
    ``learner_version`` is the newest published version: the data source drops
    decisions lagging it by more than its ``max_policy_lag`` (counted in
    ``dropped_stale_*``); ``last_transitions`` and ``decisions_collected_last_run``
    are what reached the buffer.
    ``set_weights`` sends weights to the rollout workers; ``log_learner``
    records one learner update in the run's telemetry.
    """

    total_episodes: int
    decisions_collected_last_run: int
    last_transitions: tuple[Transition, ...]
    max_policy_lag: int | None
    dropped_stale_last_run: int
    dropped_stale_total: int

    def run(self, can_train: bool, *, learner_version: int | None = None) -> Mapping[str, float]: ...

    def start(self, can_train: bool) -> PendingCollection: ...

    def finish(self, pending: PendingCollection, *, learner_version: int | None = None) -> Mapping[str, float]: ...

    def set_weights(self, weights: WeightsRef) -> None: ...

    def log_learner(
        self,
        step: int,
        env_steps: int | None,
        metrics: Mapping[str, float],
        *,
        publish_version: int | None = None,
    ) -> None: ...


class ReplayLearnerGroup(Protocol):
    """A learner group hosting a :class:`ReplayLearner` on every rank.

    ``num_learner_ranks`` (one when absent) is how many rank-local slices each
    draw is cut into; ``publish(version)`` (optional) returns the weights by
    reference, otherwise ``policy_state`` is wrapped inline.
    """

    def train_step(self, batches: Sequence[LearnerBatch]) -> tuple[LearnerStepResult, ...]: ...

    def policy_state(self, *, version: int) -> PolicyState: ...

    def save_checkpoint(self, directory: str) -> None: ...

    def hooks(self) -> LearnerHooks: ...


class PolicyEvaluator(Protocol):
    """Scores one published policy state; the metrics always carry ``success_rate``."""

    def evaluate(self, policy_state: PolicyState, *, env_steps: int = 0) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class OffPolicyConfig:
    """Schedule of one off-policy run.

    ``num_rollouts`` is the environment-step budget; ``learning_starts`` gates
    updates on collected steps; bursts count actual stored decisions, including
    partial action chunks. ``eval_freq``
    and ``save_interval`` are in environment steps (``0`` / ``None`` disable),
    and ``eval_at_episode_boundary`` defers a due evaluation (and the save
    that follows it) to the next completed episode.

    ``schedule`` (a :class:`~isaac_rl_posttraining.training.schedule.ScheduleSpec`
    or its name) is when
    collection and updates happen:

    * ``sync`` (the reference recipe) collects a round, trains on it, then publishes.
    * ``overlapped`` starts the next round before the burst, so the burst runs
      while the workers collect on the weights published after the previous
      burst (a policy lag of one); the publish after the burst waits for that round.
    * ``async`` re-submits a round as soon as the previous one returns, trains
      on what arrived, and publishes once after each burst (or every
      ``publish_every`` updates when set); new weights reach the workers when their
      round ends, so a round's data lags one publish (more with ``publish_every``);
      transitions beyond ``max_policy_lag`` are dropped.

    An iteration that ends in an evaluation, and the last one of the budget,
    start no round ahead: the evaluator scores the newest weights on idle workers.
    """

    num_rollouts: int
    learning_starts: int
    train_after_episode: bool = False
    max_updates_per_burst: int | None = None
    eval_freq: int = 0
    eval_on_first_step: bool = False
    eval_at_episode_boundary: bool = False
    save_interval: int | None = None
    save_buffer_on_exit: bool = False
    save_buffer_every: int = 0
    save_buffer_dir: str = "replay_buffers"
    schedule: ScheduleSpec | str = "sync"

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", as_schedule_spec(self.schedule))
        if self.num_rollouts < 0:
            raise ValueError("num_rollouts must be non-negative")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if self.max_updates_per_burst is not None and self.max_updates_per_burst < 1:
            raise ValueError(
                f"max_updates_per_burst must be >= 1 or None, got {self.max_updates_per_burst}"
            )
        if self.eval_freq < 0:
            raise ValueError("eval_freq must be non-negative")
        if self.save_interval is not None and self.save_interval < 0:
            raise ValueError("save_interval must be non-negative or None")
        if self.save_buffer_every < 0:
            raise ValueError("save_buffer_every must be non-negative")

    @property
    def schedule_spec(self) -> ScheduleSpec:
        """``schedule`` as a :class:`ScheduleSpec` (normalized at construction)."""
        assert isinstance(self.schedule, ScheduleSpec)
        return self.schedule


@dataclass
class _LoopState:
    """Bookkeeping carried across iterations of :meth:`OffPolicyTrainer.run`."""

    schedule: ScheduleSpec
    last_eval_steps: int = 0
    last_save_steps: int = 0
    current_rollouts: int = 0
    pending_episode_updates: int = 0
    training_started_logged: bool = False
    pending: PendingCollection | None = None
    #: Updates since the last publish (async with ``publish_every`` set).
    updates_since_publish: int = 0


class OffPolicyTrainer:
    """The off-policy loop over runtime rollout and learner groups.

    Each iteration collects one round of decisions through ``train_source``,
    draws the learner's batches with ``draw_batch`` (the algorithm's samplers
    over the controller-owned ``buffer``; one slice per learner rank), runs the
    scheduled updates on ``learner_group``, publishes the new policy state to
    the workers, and handles evaluation, checkpointing, and buffer persistence.
    Under the overlapped and async schedules the next round is already
    collecting while the updates run (:class:`OffPolicyConfig`). An async
    schedule's ``max_policy_lag`` is enforced by the data source, which must
    have been opened with the same bound.
    """

    def __init__(
        self,
        *,
        train_source: OffPolicyDataSource,
        learner_group: ReplayLearnerGroup,
        buffer: ReplayBuffer,
        draw_batch: Callable[[], LearnerBatch],
        evaluator: PolicyEvaluator | None,
        config: OffPolicyConfig,
        metric_sink: MetricSink | None = None,
        fingerprint_sink: FingerprintSink | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        schedule = config.schedule_spec
        bound = schedule.staleness_bound
        source_bound = train_source.max_policy_lag
        if bound != source_bound:
            raise ValueError(
                f"the {schedule.kind} schedule drops data beyond max_policy_lag={bound}, but the data "
                f"source drops beyond {source_bound}; open it with max_policy_lag={bound}"
            )
        self.train_source = train_source
        self.learner_group = learner_group
        self.buffer = buffer
        self.draw_batch = draw_batch
        self.evaluator = evaluator
        self.config = config
        self.metric_sink = metric_sink
        self.fingerprint_sink = fingerprint_sink
        self.checkpoint_store = checkpoint_store

        if config.eval_at_episode_boundary:
            logger.info("Evaluation restricted to episode boundaries (eval_at_episode_boundary=True)")
        if config.save_buffer_on_exit or config.save_buffer_every > 0:
            os.makedirs(config.save_buffer_dir, exist_ok=True)
            logger.info(
                f"Buffer saving enabled: on_exit={config.save_buffer_on_exit}, "
                f"every={config.save_buffer_every} steps"
            )
        if config.train_after_episode:
            logger.info("Training bursts run after episode completion only.")
        if config.max_updates_per_burst is not None:
            logger.info(f"Training phases capped at {config.max_updates_per_burst} updates each.")
        if schedule.kind == "overlapped":
            logger.info("Overlapped schedule: the next collection round starts before each training burst.")
        elif schedule.kind == "async":
            every = schedule.updates_per_publish
            cadence = "once per training burst" if every is None else f"every {every} update(s)"
            logger.info(
                "Async schedule: rounds are re-submitted as soon as they return; weights are "
                f"published {cadence}; max_policy_lag={schedule.max_policy_lag} publishes."
            )

        self._eval_pending = False
        self._save_pending = False
        self._eval_history: dict[int, float] = {}
        self._eval_records: list[dict[str, float | int]] = []
        self._last_buffer_save_steps = 0
        self._policy_version = 0
        self._weights: WeightsRef | None = None
        self._policy_state: PolicyState | None = None
        #: Newest version the rollout workers hold.
        self._delivered_version = 0
        self.total_episodes_completed = 0
        self.previous_total_episodes = 0
        self.total_env_steps = 0

    @property
    def hooks(self) -> LearnerHooks:
        """The learner group's typed algorithm hooks."""
        return self.learner_group.hooks()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the training loop until the environment-step budget is spent."""
        max_env_steps = self.config.num_rollouts
        # Workers act on published weights from the first decision on, so a
        # zero warm-up cannot silently start from worker-local initialization.
        self._publish()

        if self.config.eval_on_first_step and self.evaluator is not None:
            self._run_initial_evaluation()

        with tqdm(total=max_env_steps, desc="Training Rollouts", unit="step") as pbar:
            loop = _LoopState(schedule=self.config.schedule_spec)
            while self.total_env_steps < max_env_steps:
                self._run_iteration(loop, pbar)
            self._finish_run(loop)

    def _run_initial_evaluation(self) -> None:
        logger.info("Running initial evaluation before training...")
        eval_metrics = self._with_evaluation_counters(self.evaluate())
        self._sink(eval_metrics, prefix="eval")
        print_detailed_eval_metrics(eval_metrics)
        self._record_evaluation(0, eval_metrics)

    def _run_iteration(self, loop: _LoopState, pbar: tqdm) -> None:
        """One collect / train / publish / evaluate / save round."""
        max_env_steps = self.config.num_rollouts
        learning_starts = self.config.learning_starts
        iteration_started = time.perf_counter()
        rollout_metrics, collect_wall_s, collect_wait_s, policy_lag = self._collect(
            loop, iteration_started
        )
        if loop.schedule.kind == "async":
            # No round is in flight now: hand the workers the newest weights before the next one.
            self._deliver_newest()
        # Before any draw, so the learner scales this round's rewards as it reads them.
        observe_reward_stream(self.hooks, self.train_source.last_transitions, self.buffer)

        buffer_size = len(self.buffer)
        self.total_episodes_completed = self.train_source.total_episodes

        prev_steps = self.total_env_steps
        self.total_env_steps = int(rollout_metrics["total_steps"])
        steps_this_iter = self.total_env_steps - prev_steps
        # Re-check with the fresh step count so a burst that crosses
        # learning_starts trains in the same iteration.
        can_train = self.total_env_steps >= learning_starts

        episodes_actually_completed = (
            self.total_episodes_completed > self.previous_total_episodes
        )
        self.previous_total_episodes = self.total_episodes_completed

        if can_train and not loop.training_started_logged:
            logger.info(
                f"Env steps {self.total_env_steps} >= learning_starts ({learning_starts}); training begins"
            )
            loop.training_started_logged = True

        train_updates = self._burst_size(
            loop, prev_steps, can_train, episodes_actually_completed, steps_this_iter
        )
        should_eval_now = self._schedule_evaluation(loop, episodes_actually_completed)

        # Overlapped and async: the next round collects while this burst
        # runs. Never past the budget, and never into an evaluation: the
        # evaluator must find idle workers and the newest weights (an
        # in-process collection would also share the evaluation's re-seeded
        # random generators).
        if (
            loop.schedule.overlaps_collection
            and self.total_env_steps < max_env_steps
            and not should_eval_now
        ):
            loop.pending = self.train_source.start(can_train=can_train)

        train_metrics, train_wall_s = self._train_burst(loop, train_updates)
        # Under overlapped the publish waits for the round in flight, so
        # its time is the overlap's slack and is kept apart from the burst.
        publish_started = time.perf_counter()
        if train_updates > 0 and loop.schedule.updates_per_publish is None:
            # Async keeps the round in flight on its weights; they reach the workers when it ends.
            self._publish(deliver=loop.schedule.kind != "async" or loop.pending is None)
        publish_wall_s = time.perf_counter() - publish_started
        iteration_wall_s = time.perf_counter() - iteration_started

        # Wall-clock split of every iteration: how long collecting took,
        # how long the controller waited for it, how long the burst and
        # the publish took, and the two rates that follow: the collection
        # rate (steps over the round's own wall time, the layout metric)
        # and the run's rate (steps over the whole iteration, what a
        # schedule changes). Logged each iteration (not only on episode
        # ends) so a comparison has one sample per round.
        system_metrics = {
            "collect_wall_s": collect_wall_s,
            "collect_wait_s": collect_wait_s,
            "train_wall_s": train_wall_s,
            "publish_wall_s": publish_wall_s,
            "iteration_wall_s": iteration_wall_s,
            "train_updates": float(train_updates),
            "policy_lag": float(policy_lag),
            "env_steps_per_s": rate(steps_this_iter, collect_wall_s),
            "iteration_env_steps_per_s": rate(steps_this_iter, iteration_wall_s),
        }
        if loop.schedule.staleness_bound is not None:
            system_metrics["dropped_stale_decisions"] = float(self.train_source.dropped_stale_last_run)
            system_metrics["dropped_stale_decisions_total"] = float(self.train_source.dropped_stale_total)
        self._sink(system_metrics, prefix="system")
        if served := drain_policy_service_metrics(getattr(self.train_source, "group", None)):
            self._sink(served)
        if train_metrics is not None:
            self._sink({**train_metrics, "updates": float(self.hooks.gradient_steps())}, prefix="learner")
        if self.metric_sink is not None and episodes_actually_completed:
            self._sink(buffer_stats(self.buffer))
            self._sink(rollout_metrics, prefix="train")

        loop.current_rollouts += 1
        update_progress_bar(
            pbar,
            buffer_size,
            learning_starts,
            can_train,
            rollout_metrics,
            train_metrics,
            steps_this_iter,
        )

        if should_eval_now:
            self._run_scheduled_evaluation(loop)
            loop.last_eval_steps = self.total_env_steps

        self._maybe_save_checkpoint(loop, episodes_actually_completed)
        self._maybe_save_buffer()

        if episodes_actually_completed:
            print_detailed_metrics(
                self.total_episodes_completed, rollout_metrics, train_metrics
            )

    def _collect(
        self, loop: _LoopState, iteration_started: float
    ) -> tuple[Mapping[str, float], float, float, int]:
        """Run (or finish) this iteration's collection round; returns metrics, wall, wait and policy lag."""
        # Gate on collected env steps: workers explore with random
        # latents until training can start.
        can_train = self.total_env_steps >= self.config.learning_starts
        pending = loop.pending
        if pending is None:
            rollout_metrics = self.train_source.run(can_train=can_train, learner_version=self._policy_version)
            collect_wall_s = time.perf_counter() - iteration_started
            return rollout_metrics, collect_wall_s, collect_wall_s, self._policy_version - self._delivered_version
        # The round started before the last burst; its wall time is its
        # own (submission to the workers' completion), the wait what
        # the controller still spent on it here.
        rollout_metrics = self.train_source.finish(pending, learner_version=self._policy_version)
        collect_wall_s = pending.duration()
        collect_wait_s = time.perf_counter() - iteration_started
        policy_lag = self._policy_version - pending.policy_version
        loop.pending = None
        return rollout_metrics, collect_wall_s, collect_wait_s, policy_lag

    def _decisions_ingested(self) -> int:
        """Decisions of the last round that reached the buffer."""
        return self.train_source.decisions_collected_last_run

    def _burst_size(
        self,
        loop: _LoopState,
        prev_steps: int,
        can_train: bool,
        episodes_actually_completed: bool,
        steps_this_iter: int,
    ) -> int:
        """Number of learner updates this iteration (episode cadence, learner target, cap)."""
        train_updates = 0
        if self.config.train_after_episode:
            # Capped bursts include warm-up credit. Count stored
            # decisions exactly, including short terminal chunks.
            if (
                self.config.max_updates_per_burst is not None
                or prev_steps >= self.config.learning_starts
            ):
                loop.pending_episode_updates += self._decisions_ingested()
            if can_train and episodes_actually_completed and steps_this_iter > 0:
                train_updates = loop.pending_episode_updates
                loop.pending_episode_updates = 0
        elif can_train:
            train_updates = self._decisions_ingested()

        # An algorithm may request an exact cumulative learner-update
        # count, overriding the collected-decision burst.
        update_cadence_open = (
            not self.config.train_after_episode or episodes_actually_completed
        )
        if can_train and update_cadence_open:
            target_updates = self.hooks.target_update_count(self.total_env_steps)
            if target_updates is not None:
                train_updates = max(0, int(target_updates) - self.hooks.gradient_steps())
        return self._cap_burst(train_updates)

    def _cap_burst(self, train_updates: int) -> int:
        """Apply ``max_updates_per_burst``."""
        cap = self.config.max_updates_per_burst
        if cap is not None and train_updates > cap:
            logger.info(
                f"Training capped at {cap} updates (uncapped rule: {train_updates})"
            )
            return cap
        return train_updates

    def _schedule_evaluation(
        self, loop: _LoopState, episodes_actually_completed: bool
    ) -> bool:
        """Mark an evaluation pending at frequency boundaries; return whether it runs now.

        Executes immediately, or at the next episode boundary when the simulator
        is shared / remote. Decided before the burst because the lookahead depends on it.
        """
        if (
            self.evaluator is not None
            and self.config.eval_freq > 0
            and self.total_env_steps - loop.last_eval_steps >= self.config.eval_freq
        ):
            if self.config.eval_at_episode_boundary and not self._eval_pending:
                logger.info(
                    f"[eval] scheduled at step {self.total_env_steps}, waiting for an episode boundary"
                )
            self._eval_pending = True
        return self._eval_pending and (
            not self.config.eval_at_episode_boundary or episodes_actually_completed
        )

    def _train_burst(
        self, loop: _LoopState, train_updates: int
    ) -> tuple[Mapping[str, float] | None, float]:
        """Run ``train_updates`` learner steps; returns the last step's metrics and the burst wall time."""
        train_metrics = None
        train_started = time.perf_counter()
        if train_updates > 0:
            if self.config.train_after_episode:
                logger.info(f"Training {train_updates} updates")
            for train_metrics in learner_updates(
                self.learner_group, self.draw_batch, self.total_env_steps, train_updates
            ):
                self.train_source.log_learner(
                    self.hooks.gradient_steps(),
                    self.total_env_steps,
                    train_metrics,
                    publish_version=self._policy_version,
                )
                if loop.schedule.updates_per_publish is not None:
                    loop.updates_since_publish += 1
                    if loop.updates_since_publish >= loop.schedule.updates_per_publish:
                        # The round in flight keeps its weights; these reach the workers when it ends.
                        self._publish(deliver=loop.pending is None)
                        loop.updates_since_publish = 0
        return train_metrics, time.perf_counter() - train_started

    def _flush_updates(self, loop: _LoopState) -> None:
        """Async: publish pending updates for evaluations and checkpoints; a round in flight keeps its weights."""
        if loop.updates_since_publish > 0:
            self._publish(deliver=loop.pending is None)
            loop.updates_since_publish = 0
        elif loop.pending is None:
            self._deliver_newest()

    def _maybe_save_checkpoint(
        self, loop: _LoopState, episodes_actually_completed: bool
    ) -> None:
        """Periodic checkpointing (by env steps).

        Saves defer to the same execution gate as evaluation and run AFTER the
        eval block, with no training in between, so a checkpoint that shares a
        step tag with an eval contains exactly the weights that eval measured.
        """
        if (
            self.checkpoint_store is not None
            and self.config.save_interval is not None
            and self.config.save_interval > 0
            and self.total_env_steps - loop.last_save_steps >= self.config.save_interval
        ):
            self._save_pending = True
        should_save_now = self._save_pending and (
            not self.config.eval_at_episode_boundary or episodes_actually_completed
        )
        if not should_save_now:
            return
        assert self.checkpoint_store is not None
        if loop.schedule.kind == "async":
            self._flush_updates(loop)
        eval_sr_here = self._eval_history.get(self.total_env_steps)
        if eval_sr_here is not None:
            logger.info(
                f"[save] checkpoint-{self.total_env_steps} weights are exactly the "
                f"weights evaluated at this step (sr={eval_sr_here})"
            )
        saved = self._save_checkpoint(
            tag=self.total_env_steps,
            metadata={
                "env_steps": self.total_env_steps,
                "rollouts_completed": loop.current_rollouts,
                "episodes": self.total_episodes_completed,
                "algo_step": self.hooks.gradient_steps(),
                "eval_success_rate": eval_sr_here,
            },
        )
        loop.last_save_steps = self.total_env_steps
        self._save_pending = False
        if saved:
            self._record_fingerprint("save", checkpoint=str(self.total_env_steps))
            self.checkpoint_store.apply_retention(self._eval_history)

    def _maybe_save_buffer(self) -> None:
        """Periodic buffer saving (overwrites the same directory)."""
        if (
            self.config.save_buffer_every > 0
            and self.total_env_steps - self._last_buffer_save_steps
            >= self.config.save_buffer_every
        ):
            self._save_buffer(tag="latest", use_timestamp=False)
            self._last_buffer_save_steps = self.total_env_steps

    def _finish_run(self, loop: _LoopState) -> None:
        """Final evaluation, final checkpoint and final buffer once the budget is spent."""
        if loop.schedule.kind == "async":
            self._flush_updates(loop)
        if self.config.num_rollouts > 0 and self.hooks.gradient_steps() == 0:
            cause = (f"{loop.pending_episode_updates} updates waited for an episode to end (training.train_after_episode=true) and none ended; raise training.num_rollouts" if loop.pending_episode_updates
                     else f"updates are credited only for rounds that start at or after learning_starts ({self.config.learning_starts}); raise training.num_rollouts or lower algorithm.learning_starts")
            logger.warning(f"The budget ended after {self.total_env_steps} env steps with no learner update: {cause}.")
        # The end of the budget is a boundary too: whatever was collected
        # since the last evaluation is scored now.
        if (
            self.evaluator is not None
            and self.config.eval_freq > 0
            and self.total_env_steps > loop.last_eval_steps
        ):
            logger.info(
                f"Budget spent; final evaluation at step {self.total_env_steps} (last one at {loop.last_eval_steps})"
            )
            self._run_scheduled_evaluation(loop)

        if self.checkpoint_store is not None:
            self._record_fingerprint("save", checkpoint="final")
            self._save_checkpoint(
                tag="final",
                metadata={
                    "env_steps": self.total_env_steps,
                    "rollouts_completed": loop.current_rollouts,
                    "episodes": self.total_episodes_completed,
                    "algo_step": self.hooks.gradient_steps(),
                },
            )
        if self.config.save_buffer_on_exit:
            self._save_buffer(tag="final", use_timestamp=True)

    # ------------------------------------------------------------------
    # Policy publishing and learner hooks
    # ------------------------------------------------------------------

    def _publish(self, *, deliver: bool = True) -> None:
        """Publish the learner's weights as the next version; ``deliver`` also sends them to the workers."""
        self._policy_version += 1
        self._weights = publish_weights(self.learner_group, self._policy_version)
        self._policy_state = self._weights.resolve()
        if deliver:
            self._deliver_newest()

    def _deliver_newest(self) -> None:
        """Send the newest published weights to the rollout workers if they hold older ones."""
        if self._weights is None or self._weights.version <= self._delivered_version:
            return
        self.train_source.set_weights(self._weights)
        self._delivered_version = self._weights.version

    # ------------------------------------------------------------------
    # Evaluation and reporting
    # ------------------------------------------------------------------

    def evaluate(self) -> Mapping[str, float]:
        """Score the last published policy state on the evaluator."""
        if self.evaluator is None:
            raise RuntimeError("evaluate() called on a trainer built without an evaluator")
        if self._policy_state is None:
            raise RuntimeError("no policy state has been published yet; run() publishes first")
        return self.evaluator.evaluate(self._policy_state, env_steps=self.total_env_steps)

    def _run_scheduled_evaluation(self, loop: _LoopState) -> None:
        """Evaluate at ``total_env_steps``: log, record, notify the learner."""
        if loop.schedule.kind == "async":
            self._flush_updates(loop)
        logger.info(
            f"Starting evaluation at step {self.total_env_steps} "
            f"(episode {self.total_episodes_completed})"
        )
        eval_metrics = self._with_evaluation_counters(self.evaluate())
        self._sink(eval_metrics, prefix="eval")
        print_detailed_eval_metrics(eval_metrics)
        success_rate = self._record_evaluation(self.total_env_steps, eval_metrics)
        algorithm_metrics = self.hooks.on_evaluation(success_rate)
        if algorithm_metrics:
            self._sink(algorithm_metrics, prefix="eval")
        self._record_fingerprint("eval", success_rate=success_rate)
        self._eval_pending = False

    def _with_evaluation_counters(
        self, eval_metrics: Mapping[str, float]
    ) -> dict[str, float]:
        """Attach the exact training progress at which an evaluation ran."""
        metrics = dict(eval_metrics)
        metrics["simulator_steps"] = float(self.total_env_steps)
        metrics["optimizer_updates"] = float(self.hooks.gradient_steps())
        return metrics

    def _record_evaluation(self, step: int, eval_metrics: Mapping[str, float]) -> float:
        """Keep the evaluation for retention and the summary file; returns its success rate."""
        success_rate = float(eval_metrics["success_rate"])
        self._eval_history[step] = success_rate
        record: dict[str, float | int] = {"step": int(step)}
        for name, value in eval_metrics.items():
            record[name] = value.item() if isinstance(value, np.generic) else value
        self._eval_records.append(record)
        if self.checkpoint_store is not None:
            self.checkpoint_store.write_evaluation_summary(self._eval_records)
        return success_rate

    def _sink(self, metrics: Mapping[str, float], prefix: str | None = None) -> None:
        if self.metric_sink is None:
            return
        if prefix is None:
            namespaced = {key: float(value) for key, value in metrics.items()}
        else:
            namespaced = {f"{prefix}/{key}": float(value) for key, value in metrics.items()}
        self.metric_sink(namespaced, self.total_env_steps)

    def _record_fingerprint(self, event: str, **extra: Any) -> None:
        if self.fingerprint_sink is None:
            return
        assert self._policy_state is not None
        self.fingerprint_sink.record(
            event, step=self.total_env_steps, state=self._policy_state, **extra
        )

    def _save_checkpoint(self, tag: int | str, metadata: Mapping[str, Any]) -> bool:
        assert self.checkpoint_store is not None and self._policy_state is not None
        return self.checkpoint_store.save(
            tag,
            learner_group=self.learner_group,
            policy_state=self._policy_state,
            metadata=metadata,
        )

    def save_checkpoint(self, tag: str) -> None:
        """Publish the learner's current weights and write them under ``tag`` (an interrupted run, for example)."""
        if self.checkpoint_store is None:
            raise RuntimeError("save_checkpoint() called on a trainer built without a checkpoint store")
        if self._policy_state is None:
            raise RuntimeError("no policy state has been published yet; run() publishes first")
        # Signals land mid-burst, after updates the post-burst publish has not reached yet.
        self._publish(deliver=False)
        self._save_checkpoint(
            tag,
            metadata={
                "env_steps": self.total_env_steps,
                "episodes": self.total_episodes_completed,
                "algo_step": self.hooks.gradient_steps(),
            },
        )

    # ------------------------------------------------------------------
    # Buffer statistics and persistence
    # ------------------------------------------------------------------

    def _save_buffer(self, tag: str, *, use_timestamp: bool) -> None:
        """Save the replay buffer as a directory (timestamped when unique)."""
        if len(self.buffer) == 0:
            logger.warning("Buffer is empty, skipping save")
            return
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.config.save_buffer_dir, f"buffer_{tag}_{timestamp}")
        else:
            save_path = os.path.join(self.config.save_buffer_dir, f"buffer_{tag}")
        logger.info(
            f"Saving buffer to {save_path} (size={len(self.buffer)}, episodes={self.total_episodes_completed}, "
            f"env_steps={self.total_env_steps})"
        )
        try:
            save_buffer(self.buffer, save_path)
            logger.info(f"Buffer saved to {save_path}")
        except OSError as exc:
            logger.error(f"Failed to save buffer: {exc}")

    def save_buffer_on_signal(self) -> None:
        """Save the buffer after Ctrl-C or SIGTERM when ``save_buffer_on_exit`` is set."""
        if self.config.save_buffer_on_exit and len(self.buffer) > 0:
            self._save_buffer(tag="signal_exit", use_timestamp=True)
