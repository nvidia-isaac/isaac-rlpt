# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The on-policy training loop over runtime rollout and learner groups.

Every iteration collects a fixed number of episodes per environment with the
current policy, hands the trajectories to the algorithm (which turns them
into one learner batch per rank), runs the update on the learner group,
publishes the new weights to the workers, and evaluates and checkpoints on
the configured schedule. The algorithm is pluggable: PPO supplies GAE and the
clipped update, another on-policy method supplies its own batches and learner.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    RolloutGroup,
    RolloutInfrastructureError,
    RolloutRequest,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.concurrency import AsyncCall, train_step_async
from isaac_rl_posttraining.runtimes.learner import (
    LearnerGroup,
    publish_weights,
)
from isaac_rl_posttraining.training.checkpoints import CheckpointStore, FingerprintSink
from isaac_rl_posttraining.training.data_source import DataSource, PendingCollection
from isaac_rl_posttraining.training.on_policy_config import (
    ITERATION_CHECKPOINT_PREFIX,
    OnPolicyAlgorithm,
    OnPolicyConfig,
    checkpoint_iteration,
    checkpoint_manifest,
    training_reset_seed,
)
from isaac_rl_posttraining.training.on_policy_metrics import (
    learner_metrics,
    policy_service_metrics,
    trajectory_metrics,
)
from isaac_rl_posttraining.training.trajectories import (
    ChunkTrajectory,
    trajectories_from_results,
)
from isaac_rl_posttraining.utils.compat import add_note

__all__ = [
    "ITERATION_CHECKPOINT_PREFIX",
    "OnPolicyAlgorithm",
    "OnPolicyConfig",
    "OnPolicyTrainer",
    "checkpoint_iteration",
    "training_reset_seed",
]

_ROLLOUT_REFRESH_METRIC = "system/rollout_refresh_seconds"


@dataclass(frozen=True)
class _CollectedRound:
    """A training round :meth:`OnPolicyTrainer.collect` gathered: its episodes, metrics and timings."""

    trajectories: tuple[ChunkTrajectory, ...]
    metrics: Mapping[str, float]
    steps: int
    seconds: float
    refresh_seconds: float
    timing_metrics: Mapping[str, float]
    started: float


MetricSink = Callable[[Mapping[str, float], int], None]
#: Receives every evaluation's episodes, the iteration and the published policy
#: state; ``None`` when nothing was published and the workers scored the
#: weights they loaded at start-up (a learner-less run without a snapshot).
EvaluationSink = Callable[[Sequence[ChunkTrajectory], int, PolicyState | None], None]
OpenRolloutGroup = Callable[[Literal["train", "eval"]], RolloutGroup]


class OnPolicyTrainer:
    """Controller-side on-policy loop shared by every rollout runtime.

    ``algorithm`` supplies the batches (PPO: GAE, shuffling, sharding) and the
    learner worker the ``learner_group`` hosts; the loop itself owns collection,
    publishing, evaluation, and checkpoints. Training rounds go through
    ``train_source``, a :class:`DataSource` without a buffer (``sink=None``)
    that counts rounds in episodes, ``config.episodes_per_environment`` per
    environment; it gives the episodes their lineage, writes the run's
    telemetry tables (evaluations under the ``eval`` source) and, with
    ``persist``, saves them. ``checkpoint_store`` (tagged by iteration)
    receives the periodic and final checkpoints and the evaluation summary;
    ``fingerprint_sink`` records the weights each evaluation and checkpoint used.

    :meth:`run` drives the configured schedule; :meth:`collect`, :meth:`train`,
    :meth:`evaluate` and :meth:`publish` drive one step at a time.

    ``learner_group`` may be ``None`` for evaluation-only use: :meth:`evaluate`
    then scores ``initial_policy_state`` (for example a snapshot loaded with
    :func:`~isaac_rl_posttraining.runtimes.policy_snapshot.load_policy_snapshot`),
    or the weights the rollout workers loaded at start-up when no state is given.
    """

    def __init__(
        self,
        *,
        train_source: DataSource,
        learner_group: LearnerGroup | None,
        algorithm: OnPolicyAlgorithm,
        config: OnPolicyConfig,
        metric_sink: MetricSink,
        evaluation_sink: EvaluationSink | None = None,
        open_rollout_group: OpenRolloutGroup | None = None,
        rollout_phase: Literal["train", "eval"] = "train",
        initial_policy_state: PolicyState | None = None,
        checkpoint_store: CheckpointStore | None = None,
        fingerprint_sink: FingerprintSink | None = None,
        reward_channel: str = "env",
    ) -> None:
        if rollout_phase not in ("train", "eval"):
            raise ValueError("rollout_phase must be 'train' or 'eval'")
        if not isinstance(reward_channel, str) or not reward_channel:
            raise ValueError("reward_channel must be a non-empty string")
        if config.checkpoint_every and checkpoint_store is None:
            raise ValueError("checkpoint_every requires a checkpoint_store")
        if checkpoint_store is not None and checkpoint_store.tag_prefix != ITERATION_CHECKPOINT_PREFIX:
            raise ValueError(
                f"the on-policy checkpoint store must tag by iteration "
                f"(tag_prefix={ITERATION_CHECKPOINT_PREFIX!r}), got {checkpoint_store.tag_prefix!r}"
            )
        if train_source.count_by != "episode" or train_source.request_count != config.episodes_per_environment:
            raise ValueError(
                f"the on-policy loop collects {config.episodes_per_environment} episode(s) per environment "
                f"a round; open its data source with count_by='episode', "
                f"request_count={config.episodes_per_environment}"
            )
        if learner_group is not None and initial_policy_state is not None:
            raise ValueError(
                "initial_policy_state is for evaluation without a learner; a learner "
                "group publishes its own weights (resume() restores a checkpoint)"
            )
        self.train_source = train_source
        self.learner_group = learner_group
        self.algorithm = algorithm
        self.config = config
        self.metric_sink = metric_sink
        self._evaluation_sink = evaluation_sink
        self._open_rollout_group = open_rollout_group
        self._rollout_phase = rollout_phase
        self.checkpoint_store = checkpoint_store
        self.fingerprint_sink = fingerprint_sink
        self.reward_channel = reward_channel
        self._eval_telemetry = train_source.recorder.child("eval")
        self._rng = np.random.default_rng(config.seed)
        self._policy_state: PolicyState | None = None
        self._weights: WeightsRef | None = None
        self._initial_policy_state = initial_policy_state
        self.policy_version = 0
        #: Async: updates not yet published (with ``publish_every`` set).
        self._unpublished_updates = 0
        self.env_steps = 0
        #: Iterations completed so far; a resumed run continues from here.
        self.iteration = 0
        self._eval_history: dict[int, float] = {}
        self._eval_records: list[dict[str, float | int]] = []
        #: The round :meth:`collect` gathered and :meth:`train` has not updated on yet.
        self._collected: _CollectedRound | None = None
        #: The newest learner update started; a save waits for it so it never reads a half-applied step.
        self._update_in_flight: AsyncCall[tuple[Any, ...]] | None = None

    @property
    def rollout_group(self) -> RolloutGroup:
        """The rollout workers the data source collects with."""
        return self.train_source.group

    # ------------------------------------------------------------------
    # Policy publishing and rollout groups
    # ------------------------------------------------------------------
    def _synchronize_policy(self) -> None:
        if self._policy_state is None:
            # Nothing published yet: the workers serve the weights they loaded.
            return
        assert self._weights is not None
        self.train_source.set_weights(self._weights)

    def _publish_policy(self, *, synchronize: bool = True) -> float:
        """Publish the current weights as the next version; returns seconds spent.

        The weights come from the learner group, or from ``initial_policy_state``
        for a learner-less evaluation. Without either the workers keep the
        weights they loaded at start-up and nothing is sent. ``synchronize=False``
        leaves the workers on their weights until :meth:`_synchronize_newest`.
        """
        started = time.perf_counter()
        if self.learner_group is not None:
            self.policy_version += 1
            self._weights = publish_weights(self.learner_group, self.policy_version)
            self._policy_state = self._weights.resolve()
        elif self._initial_policy_state is not None and self._policy_state is None:
            self.policy_version += 1
            self._policy_state = replace(
                self._initial_policy_state, version=self.policy_version
            )
            self._weights = WeightsRef.of(self._policy_state)
        else:
            return 0.0
        if synchronize:
            self._synchronize_policy()
        return time.perf_counter() - started

    def _synchronize_newest(self) -> None:
        """Send the newest published weights to the workers if they hold older ones (no round in flight)."""
        if self._policy_state is not None and self.train_source.policy_version < self.policy_version:
            self._synchronize_policy()

    def _replace_rollout_group(self, phase: Literal["train", "eval"]) -> None:
        open_group = self._open_rollout_group
        assert open_group is not None
        self.train_source.replace_group(lambda: open_group(phase))
        self._rollout_phase = phase
        self._synchronize_newest()

    def _ensure_rollout_phase(self, phase: Literal["train", "eval"]) -> float:
        """Replace simulator workers when train/eval process isolation is enabled."""
        if self._open_rollout_group is None or self._rollout_phase == phase:
            return 0.0
        started = time.perf_counter()
        self._replace_rollout_group(phase)
        return time.perf_counter() - started

    def _restart_rollout_phase(self, phase: Literal["train", "eval"]) -> float:
        """Replace a failed group while preserving policy and environment identity."""
        if self._open_rollout_group is None:
            raise RuntimeError("rollout restart requires open_rollout_group")
        started = time.perf_counter()
        self._replace_rollout_group(phase)
        return time.perf_counter() - started

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def _start_collect(
        self,
        episodes_per_environment: int,
        *,
        train: bool,
        reset_seed: int | None,
    ) -> PendingCollection:
        reset_stats = getattr(self.rollout_group, "reset_policy_server_stats", None)
        if callable(reset_stats):
            reset_stats()
        if train:
            return self.train_source.start(True, reset_seed=reset_seed)
        request = RolloutRequest(
            episodes_per_environment=episodes_per_environment,
            mode="eval",
            reset_seed=reset_seed,
        )
        started = time.time()
        call = self.rollout_group.collect_async((request,) * self.rollout_group.num_rollout_workers)
        return PendingCollection(
            call=call,
            policy_version=self.train_source.policy_version,
            mode=request.mode,
            started=started,
        )

    def _finish_collect(
        self,
        pending: PendingCollection,
        episodes_per_environment: int,
    ) -> tuple[tuple[ChunkTrajectory, ...], float, int, dict[str, float]]:
        results = pending.call.result()
        rollout_seconds = time.time() - pending.started
        if pending.mode == "train":
            # Lineage, telemetry and persistence of training rounds belong to the data source.
            self.train_source.finish(pending, learner_version=self.policy_version)
        else:
            for result in results:
                if result.policy_version != pending.policy_version:
                    raise RuntimeError(
                        f"worker {result.worker_index} collected with policy version "
                        f"{result.policy_version}, expected {pending.policy_version}"
                    )
            self._eval_telemetry.ingest(
                results,
                mode=pending.mode,
                wall_start=pending.started,
                wall_end=pending.call.finished_at(),
                publish_version=self.policy_version,
            )
        steps = sum(result.env_steps for result in results)
        trajectory_started = time.perf_counter()
        trajectories = trajectories_from_results(
            results,
            gamma=self.algorithm.gamma,
            require_behavior_metadata=pending.mode == "train",
            discount_per=self.algorithm.discount_per,
            reward_channel=self.reward_channel,
        )
        trajectory_seconds = time.perf_counter() - trajectory_started
        elapsed = time.time() - pending.started
        expected = self.rollout_group.num_envs * episodes_per_environment
        if len(trajectories) != expected:
            raise RuntimeError(
                f"runtime returned {len(trajectories)} episodes, expected {expected}"
            )
        timing_metrics = {
            "system/rollout_seconds": rollout_seconds,
            "system/trajectory_build_seconds": trajectory_seconds,
        }
        worker_timings: dict[str, list[float]] = {}
        for result in results:
            for name, seconds in result.timings.items():
                worker_timings.setdefault(name, []).append(float(seconds))
        for name, seconds in worker_timings.items():
            timing_metrics[f"rollout_workers/{name}_sum"] = float(sum(seconds))
            timing_metrics[f"rollout_workers/{name}_max"] = float(max(seconds))
        timing_metrics.update(policy_service_metrics(self.rollout_group))
        return trajectories, elapsed, steps, timing_metrics

    def _collect(
        self,
        episodes_per_environment: int,
        *,
        train: bool,
        reset_seed: int | None = None,
    ) -> tuple[tuple[ChunkTrajectory, ...], float, int, dict[str, float]]:
        pending = self._start_collect(
            episodes_per_environment, train=train, reset_seed=reset_seed
        )
        return self._finish_collect(pending, episodes_per_environment)

    def _training_reset_seed(self, iteration: int) -> int:
        return training_reset_seed(self.config.seed, iteration)

    def _evaluation_due(self, iteration: int) -> bool:
        return (
            self.config.eval_episodes_per_environment > 0
            and iteration % self.config.eval_every == 0
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, iteration: int, *, refresh_seconds: float = 0.0) -> dict[str, float]:
        self._synchronize_newest()
        retries = 0
        try:
            trajectories, seconds, steps, timing_metrics = self._collect(
                self.config.eval_episodes_per_environment,
                train=False,
                reset_seed=self.config.eval_seed,
            )
        except RolloutInfrastructureError as initial_error:
            if self._open_rollout_group is None:
                raise
            warnings.warn(
                "evaluation rollout infrastructure failed; restarting the isolated "
                "group and replaying the same fixed seeds once",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                refresh_seconds += self._restart_rollout_phase("eval")
                trajectories, seconds, steps, timing_metrics = self._collect(
                    self.config.eval_episodes_per_environment,
                    train=False,
                    reset_seed=self.config.eval_seed,
                )
            except Exception as retry_error:
                add_note(retry_error, 
                    f"initial evaluation infrastructure error: {initial_error!r}"
                )
                raise
            retries = 1
        if self._evaluation_sink is not None:
            self._evaluation_sink(trajectories, iteration, self._policy_state)
        metrics = trajectory_metrics(trajectories, "eval")
        self._record_evaluation(iteration, metrics)
        metrics.update(
            {
                "eval/seconds": seconds,
                "eval/env_steps_per_second": steps / max(seconds, 1e-9),
                _ROLLOUT_REFRESH_METRIC: refresh_seconds,
                "system/eval_retries": float(retries),
            }
        )
        metrics.update(timing_metrics)
        self.metric_sink(metrics, iteration)
        return metrics

    def _record_evaluation(self, iteration: int, metrics: Mapping[str, float]) -> None:
        """Keep the evaluation for retention and the summary file, and fingerprint the weights it scored."""
        self._eval_history[iteration] = metrics["eval/success_rate"]
        record: dict[str, float | int] = {"iteration": iteration}
        for name, value in metrics.items():
            record[name.removeprefix("eval/")] = float(value)
        self._eval_records.append(record)
        if self.checkpoint_store is not None:
            self.checkpoint_store.write_evaluation_summary(self._eval_records)
        if self.fingerprint_sink is not None and self._policy_state is not None:
            self.fingerprint_sink.record(
                "eval", step=iteration, state=self._policy_state, success_rate=metrics["eval/success_rate"]
            )

    def evaluate(self) -> dict[str, float]:
        """Run the fixed evaluation set once on the newest weights (published first if none are yet).

        The metrics are logged under the current iteration: ``0`` for a fresh
        policy, the checkpoint's iteration after :meth:`resume`; they are also returned.
        """
        if not self.config.eval_episodes_per_environment:
            raise ValueError("evaluation requires eval_episodes_per_environment > 0")
        refresh_seconds = self._ensure_rollout_phase("eval")
        if self._policy_state is None:
            self._publish_policy()
        return self._evaluate(self.iteration, refresh_seconds=refresh_seconds)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def resume(self, checkpoint_dir: str | Path) -> None:
        """Restore the learner from a checkpoint (``iteration_<n>``, ``final`` or ``aborted``) and continue after it."""
        if self.learner_group is None:
            raise RuntimeError(
                "resume() restores a learner; without one, pass the checkpoint's "
                "policy snapshot as initial_policy_state instead"
            )
        directory = Path(checkpoint_dir)
        if not directory.is_absolute():
            raise ValueError("checkpoint_dir must be absolute")
        self.iteration = checkpoint_iteration(directory)
        self.env_steps = int(checkpoint_manifest(directory).get("env_steps", 0))
        self.learner_group.load_checkpoint(str(directory))

    def _checkpoint(self, tag: int | str, iteration: int) -> None:
        """Write the learner and the published weights under ``tag``, then prune by retention."""
        assert self.checkpoint_store is not None
        assert self.learner_group is not None
        assert self._policy_state is not None
        if not self.checkpoint_store.save(
            tag,
            learner_group=self.learner_group,
            policy_state=self._policy_state,
            metadata={
                "iteration": iteration,
                "env_steps": self.env_steps,
                "eval_success_rate": self._eval_history.get(iteration),
            },
        ):
            return
        if self.fingerprint_sink is not None:
            self.fingerprint_sink.record("save", step=iteration, state=self._policy_state, checkpoint=str(tag))
        self.checkpoint_store.apply_retention(self._eval_history)

    def save_checkpoint(self, tag: str) -> None:
        """Publish the learner's current weights and write both under ``tag`` (an aborted run, for example)."""
        if self.checkpoint_store is None or self.learner_group is None or self._policy_state is None:
            raise RuntimeError("nothing to checkpoint: no checkpoint store, learner, or published weights")
        if self._update_in_flight is not None:
            try:
                self._update_in_flight.result()
            except Exception as exc:  # noqa: BLE001 - the caller is already handling a failure
                warnings.warn(f"the in-flight learner update failed ({exc!r}); saving the state it left", stacklevel=2)
            self._update_in_flight = None
        self._publish_policy(synchronize=False)
        self._unpublished_updates = 0
        self._checkpoint(tag, self.iteration)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _start_update(
        self,
        trajectories: Sequence[ChunkTrajectory],
        metrics: dict[str, float],
    ) -> AsyncCall[tuple[Any, ...]]:
        assert self.learner_group is not None
        batch_started = time.perf_counter()
        batches, batch_metrics = self.algorithm.build_batches(
            trajectories,
            num_learner_ranks=self.learner_group.num_learner_ranks,
            rng=self._rng,
        )
        metrics["system/batch_build_seconds"] = time.perf_counter() - batch_started
        metrics.update({f"batch/{name}": value for name, value in batch_metrics.items()})
        self._update_in_flight = train_step_async(self.learner_group, batches)
        return self._update_in_flight

    def publish(self) -> PolicyState:
        """Publish the learner's current weights to the rollout workers as the next version."""
        if self.learner_group is None:
            raise RuntimeError("publishing needs a learner group")
        self._publish_policy()
        assert self._policy_state is not None
        return self._policy_state

    def collect(self) -> Mapping[str, float]:
        """Collect the next iteration's episodes with the newest weights; :meth:`train` updates on them.

        Returns the round's episode metrics (``train/...``).
        """
        if self.learner_group is None:
            raise RuntimeError("training needs a learner group; use evaluate() without one")
        if self._collected is not None:
            raise RuntimeError("the episodes of the last collect() are not trained on yet; call train() first")
        if self._policy_state is None:
            self._publish_policy()
        started = time.perf_counter()
        refresh_seconds = self._ensure_rollout_phase("train")
        trajectories, seconds, steps, timing_metrics = self._collect(
            self.config.episodes_per_environment,
            train=True,
            reset_seed=self._training_reset_seed(self.iteration + 1),
        )
        metrics = trajectory_metrics(trajectories, "train")
        self._collected = _CollectedRound(
            trajectories, metrics, steps, seconds, refresh_seconds, timing_metrics, started
        )
        return dict(metrics)

    def train(self) -> Mapping[str, float]:
        """Update on the episodes of the last :meth:`collect`, publish, and log the iteration; returns its metrics."""
        collected, self._collected = self._collected, None
        if collected is None:
            raise RuntimeError("train() updates on the episodes of the last collect(); call collect() first")
        metrics = dict(collected.metrics)
        update = self._start_update(collected.trajectories, metrics)
        train_started = time.perf_counter()
        results = update.result()
        metrics["system/train_seconds"] = time.perf_counter() - train_started
        publish_seconds = self._publish_policy()
        metrics.update(learner_metrics(results))
        metrics.update(
            {
                "system/collect_seconds": collected.seconds,
                "system/publish_seconds": publish_seconds,
                _ROLLOUT_REFRESH_METRIC: collected.refresh_seconds,
            }
        )
        metrics.update(collected.timing_metrics)
        self._log_iteration(
            self.iteration + 1, metrics, steps=collected.steps, iteration_started=collected.started
        )
        return metrics

    def _log_iteration(
        self,
        iteration: int,
        metrics: dict[str, float],
        *,
        steps: int,
        iteration_started: float,
    ) -> None:
        self.env_steps += steps
        total_seconds = time.perf_counter() - iteration_started
        metrics.update(
            {
                "system/iteration_seconds": total_seconds,
                "system/env_steps": float(self.env_steps),
                "system/env_steps_per_second": steps / max(total_seconds, 1e-9),
                "system/policy_version": float(self.policy_version),
            }
        )
        self.metric_sink(metrics, iteration)
        update = {name.removeprefix("learner/"): value for name, value in metrics.items() if name.startswith("learner/")}
        self.train_source.log_learner(iteration, self.env_steps, update, publish_version=self.policy_version)
        self.iteration = iteration

    def _after_iteration(self, iteration: int) -> None:
        """Checkpoint and evaluate when the schedule says so."""
        checkpoint_due = bool(self.config.checkpoint_every) and iteration % self.config.checkpoint_every == 0
        if self._unpublished_updates and (
            checkpoint_due or self._evaluation_due(iteration) or iteration == self.config.iterations
        ):
            # Async: checkpoints and evaluations see the newest weights.
            self._publish_policy(synchronize=False)
            self._unpublished_updates = 0
        if checkpoint_due:
            self._checkpoint(iteration, iteration)
        if self._evaluation_due(iteration):
            refresh_seconds = self._ensure_rollout_phase("eval")
            self._evaluate(iteration, refresh_seconds=refresh_seconds)

    def _finish_iteration(
        self,
        iteration: int,
        metrics: dict[str, float],
        *,
        steps: int,
        iteration_started: float,
    ) -> None:
        self._log_iteration(iteration, metrics, steps=steps, iteration_started=iteration_started)
        self._after_iteration(iteration)

    def run(self) -> None:
        """Train from the current iteration to ``config.iterations``."""
        if self.learner_group is None:
            raise RuntimeError("training needs a learner group; use evaluate() without one")
        self._publish_policy()
        if self.config.eval_episodes_per_environment and self.iteration == 0:
            refresh_seconds = self._ensure_rollout_phase("eval")
            self._evaluate(0, refresh_seconds=refresh_seconds)
        kind = self.config.schedule_spec.kind
        if kind == "overlapped":
            self._run_overlapped()
        elif kind == "async":
            self._run_async()
        else:
            self._run_serial()
        if self.checkpoint_store is not None:
            self._checkpoint("final", self.iteration)

    def _run_serial(self) -> None:
        for iteration in range(self.iteration + 1, self.config.iterations + 1):
            self.collect()
            self.train()
            self._after_iteration(iteration)

    def _run_overlapped(self) -> None:
        """Pipelined loop: the next iteration's rollouts run while the learner updates.

        Per iteration: wait for the collection in flight (or start one now),
        start the update, submit the next collection with the policy that
        produced this data, wait for the update, publish. Workers apply the new
        weights once their in-flight collection has finished, so the update of
        iteration ``k`` trains on data of the policy after iteration ``k - 2``
        (``system/policy_lag`` = 1). Iterations followed by an evaluation start
        the next collection only after the evaluation, so evaluations always
        score the newest policy on idle workers.
        """
        pending: PendingCollection | None = None
        for iteration in range(self.iteration + 1, self.config.iterations + 1):
            iteration_started = time.perf_counter()
            refresh_seconds = self._ensure_rollout_phase("train")
            if pending is None:
                pending = self._start_collect(
                    self.config.episodes_per_environment,
                    train=True,
                    reset_seed=self._training_reset_seed(iteration),
                )
            wait_started = time.perf_counter()
            trajectories, collect_seconds, steps, timing_metrics = self._finish_collect(
                pending, self.config.episodes_per_environment
            )
            collect_wait_seconds = time.perf_counter() - wait_started
            policy_lag = self.policy_version - pending.policy_version
            pending = None
            metrics = trajectory_metrics(trajectories, "train")
            update = self._start_update(trajectories, metrics)
            next_iteration = iteration + 1
            if next_iteration <= self.config.iterations and not self._evaluation_due(
                iteration
            ):
                pending = self._start_collect(
                    self.config.episodes_per_environment,
                    train=True,
                    reset_seed=self._training_reset_seed(next_iteration),
                )
            train_started = time.perf_counter()
            results = update.result()
            metrics["system/train_seconds"] = time.perf_counter() - train_started
            metrics["system/publish_seconds"] = self._publish_policy()
            metrics.update(learner_metrics(results))
            metrics.update(
                {
                    "system/collect_seconds": collect_seconds,
                    "system/collect_wait_seconds": collect_wait_seconds,
                    "system/policy_lag": float(policy_lag),
                    _ROLLOUT_REFRESH_METRIC: refresh_seconds,
                }
            )
            metrics.update(timing_metrics)
            self._finish_iteration(
                iteration, metrics, steps=steps, iteration_started=iteration_started
            )

    def _run_async(self) -> None:
        """Async loop: the next collection is submitted as soon as the previous one returns.

        Per iteration: wait for the collection in flight, hand the workers the
        newest published weights (no collection is running at that moment),
        submit the next iteration's collection, then update on the data that
        arrived and publish (after every update, or every ``publish_every``). Published weights
        reach the workers when their collection ends, so data lags the policy
        by at most one version (checked against ``max_policy_lag``).
        Iterations followed by an evaluation submit nothing ahead.
        """
        schedule = self.config.schedule_spec
        bound = schedule.policy_lag_bound
        pending: PendingCollection | None = None
        for iteration in range(self.iteration + 1, self.config.iterations + 1):
            iteration_started = time.perf_counter()
            refresh_seconds = self._ensure_rollout_phase("train")
            if pending is None:
                self._synchronize_newest()
                pending = self._start_collect(
                    self.config.episodes_per_environment,
                    train=True,
                    reset_seed=self._training_reset_seed(iteration),
                )
            wait_started = time.perf_counter()
            trajectories, collect_seconds, steps, timing_metrics = self._finish_collect(
                pending, self.config.episodes_per_environment
            )
            collect_wait_seconds = time.perf_counter() - wait_started
            policy_lag = self.policy_version - pending.policy_version
            if bound is not None and policy_lag > bound:
                raise RuntimeError(
                    f"iteration {iteration} collected with policy version {pending.policy_version}, "
                    f"{policy_lag} behind the learner (max_policy_lag={bound})"
                )
            pending = None
            next_iteration = iteration + 1
            if next_iteration <= self.config.iterations and not self._evaluation_due(iteration):
                self._synchronize_newest()
                pending = self._start_collect(
                    self.config.episodes_per_environment,
                    train=True,
                    reset_seed=self._training_reset_seed(next_iteration),
                )
            metrics = trajectory_metrics(trajectories, "train")
            update = self._start_update(trajectories, metrics)
            train_started = time.perf_counter()
            results = update.result()
            metrics["system/train_seconds"] = time.perf_counter() - train_started
            self._unpublished_updates += 1
            publish_seconds = 0.0
            if self._unpublished_updates >= (schedule.updates_per_publish or 1):
                publish_seconds = self._publish_policy(synchronize=pending is None)
                self._unpublished_updates = 0
            metrics.update(learner_metrics(results))
            metrics.update(
                {
                    "system/collect_seconds": collect_seconds,
                    "system/collect_wait_seconds": collect_wait_seconds,
                    "system/publish_seconds": publish_seconds,
                    "system/policy_lag": float(policy_lag),
                    _ROLLOUT_REFRESH_METRIC: refresh_seconds,
                }
            )
            metrics.update(timing_metrics)
            self._finish_iteration(
                iteration, metrics, steps=steps, iteration_started=iteration_started
            )
