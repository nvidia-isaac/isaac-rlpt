# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout worker for one simulator process that owns a batch of environments."""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from loguru import logger

from isaac_rl_posttraining.data import TransitionMetadata
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import (
    SEED_MODULUS,
    CollectedDecision,
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
    _require_positive_integer,
)
from isaac_rl_posttraining.runtimes.batched_chunks import (
    ChunkOutcome,
    EpisodeStart,
    chain_next_values,
    unbatch_observation,
)
from isaac_rl_posttraining.runtimes.batched_options import (
    BOOTSTRAP_VALUES,
    INACTIVE_ACTIONS,
    BatchedEnvironment,
    BootstrapValues,
    InactiveAction,
    ObservationCaptureControl,
    RestartableBatchedEnvironment,
    SuccessorValues,
    accepts_active_mask,
    resolve_bootstrap_values,
)
from isaac_rl_posttraining.runtimes.gym_worker import (
    PolicyRef,
    check_policy_ref,
    worker_policy,
)
from isaac_rl_posttraining.runtimes.policy import (
    BatchedActionChunkPolicy,
    DecisionMode,
    RolloutPolicy,
)
from isaac_rl_posttraining.runtimes.policy_adapters import (
    as_rollout_policy,
    first_value,
)
from isaac_rl_posttraining.runtimes.rollout_records import (
    NEXT_OBSERVATION_TIMING,
    RewardTransform,
    decision_transition,
    learning_reward_channel,
    mirror_renamed_timings,
    timed,
)
from isaac_rl_posttraining.utils.compat import add_note
from isaac_rl_posttraining.utils.hashing import digest_value, digest_value_by_key

__all__ = [
    "BOOTSTRAP_VALUES",
    "INACTIVE_ACTIONS",
    "BatchedEnvironment",
    "BatchedEpisodeRolloutWorker",
    "BatchedWorkerSpec",
    "BootstrapValues",
    "InactiveAction",
    "RestartableBatchedEnvironment",
    "SuccessorValues",
    "resolve_bootstrap_values",
]


@dataclass(frozen=True)
class BatchedWorkerSpec:
    """One natively batched simulator per worker, played as fixed-horizon episodes.

    A picklable recipe the runtime calls with the worker's context; the
    simulator entry builds the worker's batched environment there. ``scenes``
    selects ``entry.make_scenes`` over ``entry.make_environment``: the flow
    policy family plays all of a worker's scenes in one simulation, the
    steering families take the entry's batched environment of the phase.

    ``episode_attempts`` above one redoes an episode whose environment raised a
    :class:`RolloutInfrastructureError` after ``env.restart()``; an episode
    starts from a full reset, so repeating it changes nothing but the scene
    draws (nothing at all with a reset seed).

    ``policy`` is a policy spec the worker builds its own copy from, or the
    handle of a policy service it connects to (routed by worker index, one
    session per environment); ``supports_served_policy`` says the spec's
    policy may be served that way. ``bootstrap_at_time_limit`` is the
    learner's setting, checked here against ``bootstrap_values``.
    ``successor_values`` is the old name of ``bootstrap_values``.
    """

    entry: SimulatorEntry
    environment: EnvironmentSpec
    phase: Phase
    policy: PolicyRef
    chunks_per_episode: int
    scenes: bool = False
    reward_transform: RewardTransform | None = None
    successor_values: str | None = None
    episode_attempts: int = 1
    bootstrap_values: BootstrapValues = "evaluate_next_observation"
    bootstrap_at_time_limit: bool = False
    inactive_action: InactiveAction = "zeros"
    supports_served_policy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SimulatorEntry):
            raise TypeError("entry must be a SimulatorEntry")
        if not isinstance(self.environment, EnvironmentSpec):
            raise TypeError("environment must be an EnvironmentSpec")
        if self.phase not in ("train", "eval"):
            raise ValueError("phase must be 'train' or 'eval'")
        check_policy_ref(self.policy)
        if not isinstance(self.scenes, bool):
            raise TypeError("scenes must be a boolean")
        if self.chunks_per_episode < 1:
            raise ValueError("chunks_per_episode must be positive")
        object.__setattr__(
            self,
            "bootstrap_values",
            resolve_bootstrap_values(self.bootstrap_values, self.successor_values),
        )
        object.__setattr__(self, "successor_values", None)
        _require_positive_integer(self.episode_attempts, name="episode_attempts")
        if not isinstance(self.bootstrap_at_time_limit, bool):
            raise TypeError("bootstrap_at_time_limit must be a boolean")
        if (
            self.bootstrap_at_time_limit
            and self.bootstrap_values == "reuse_next_decision"
        ):
            raise ValueError(
                "bootstrap_at_time_limit=true needs bootstrap_values="
                "'evaluate_next_observation': reusing the next decision's value "
                "leaves nothing to bootstrap a time-limit truncation from"
            )
        if self.inactive_action not in INACTIVE_ACTIONS:
            raise ValueError(f"inactive_action must be one of {INACTIVE_ACTIONS}")
        if not isinstance(self.supports_served_policy, bool):
            raise TypeError("supports_served_policy must be a boolean")

    def make_environment(self, context: RolloutWorkerContext) -> BatchedEnvironment:
        """The worker's batched environment, built by the simulator entry on the worker."""
        if self.scenes:
            return self.entry.make_scenes(self.environment, context)
        return self.entry.make_environment(self.environment, self.phase, context)

    @property
    def reward_channel(self) -> str:
        """The reward channel this spec's decisions are learned and scored from."""
        return learning_reward_channel(self.reward_transform)

    def __call__(self, context: RolloutWorkerContext):
        env = self.make_environment(context)
        try:
            policy = worker_policy(
                self.policy, context, route_index=context.worker_index
            )
            return BatchedEpisodeRolloutWorker(
                context,
                env,
                policy,
                chunks_per_episode=self.chunks_per_episode,
                reward_transform=self.reward_transform,
                bootstrap_values=self.bootstrap_values,
                episode_attempts=self.episode_attempts,
                inactive_action=self.inactive_action,
            )
        except BaseException as exc:
            try:
                env.close()
            except Exception as close_exc:  # noqa: BLE001
                add_note(exc, f"batched environment cleanup also failed: {close_exc!r}")
            raise


class BatchedEpisodeRolloutWorker:
    """Collect complete episodes from N lockstep environments in one simulator.

    Environments whose episode ended, or whose chunk ended early, execute
    ``inactive_action`` while the others finish; an environment that declares
    ``accepts_active_mask = True`` also receives ``step(actions, active=mask)``
    with the environments still executing their chunk.
    """

    def __init__(
        self,
        context: RolloutWorkerContext,
        env: BatchedEnvironment,
        policy: RolloutPolicy | BatchedActionChunkPolicy,
        *,
        chunks_per_episode: int,
        reward_transform: RewardTransform | None,
        bootstrap_values: str = "evaluate_next_observation",
        episode_attempts: int = 1,
        inactive_action: InactiveAction = "zeros",
        successor_values: str | None = None,
    ) -> None:
        if env.num_envs != len(context.environment_indices):
            raise ValueError(
                f"batched simulator hosts {env.num_envs} envs but worker owns "
                f"{len(context.environment_indices)}"
            )
        bootstrap_values = resolve_bootstrap_values(bootstrap_values, successor_values)
        rollout_policy = as_rollout_policy(policy)
        features = rollout_policy.features
        # Only valued decisions need the policy's value pass; a policy that
        # declares its decisions carry no value only encodes next observations.
        if (
            bootstrap_values == "evaluate_next_observation"
            and features.behavior_values is not False
            and not features.values_observations
        ):
            raise TypeError(
                "bootstrap_values='evaluate_next_observation' requires a policy that values "
                "next observations (encode_and_value_next_observations, formerly "
                "prepare_successors(...)) when its decisions carry behaviour values; "
                "a policy without a value head uses bootstrap_values='reuse_next_decision'"
            )
        if inactive_action not in INACTIVE_ACTIONS:
            raise ValueError(f"inactive_action must be one of {INACTIVE_ACTIONS}")
        if episode_attempts < 1:
            raise ValueError("episode_attempts must be a positive integer")
        if episode_attempts > 1 and not isinstance(env, RestartableBatchedEnvironment):
            raise TypeError(
                "episode_attempts above one requires an environment with restart()"
            )
        self.context = context
        self.env = env
        self.policy = rollout_policy
        self.chunks_per_episode = chunks_per_episode
        self.reward_transform = reward_transform
        self.bootstrap_values: BootstrapValues = bootstrap_values
        self.episode_attempts = episode_attempts
        self.inactive_action = inactive_action
        self._active_mask = accepts_active_mask(env)
        #: Each environment's last executed action, for ``inactive_action="hold"``.
        self._last_actions: np.ndarray | None = None
        self._policy_version = 0
        self._next_episode_id = 0
        self._closed = False
        self._decision_epoch: (
            Generator[list[CollectedDecision], DecisionMode, None] | None
        ) = None
        self._decision_timings: dict[str, float] = {}

    def collect(self, request: RolloutRequest) -> RolloutResult:
        if self._closed:
            raise RuntimeError("rollout worker is closed")
        if request.capture_frames:
            raise ValueError("batched episode workers do not capture video frames")
        if request.decisions_per_environment is not None:
            return self._collect_decisions(request)
        if self._decision_epoch is not None:
            raise ValueError(
                "Cannot switch to episode collection during a persistent decision stream"
            )
        assert request.episodes_per_environment is not None
        decision_limit = self.chunks_per_episode
        request_cap_forces_end = False
        if request.max_decisions_per_episode is not None:
            decision_limit = min(
                self.chunks_per_episode, request.max_decisions_per_episode
            )
            request_cap_forces_end = (
                request.max_decisions_per_episode < self.chunks_per_episode
            )
        decisions: list[CollectedDecision] = []
        timings: dict[str, float] = {}
        for episode_index in range(request.episodes_per_environment):
            seed = (
                None
                if request.reset_seed is None
                else (request.reset_seed + episode_index) % SEED_MODULUS
            )
            decisions.extend(
                self._collect_epoch_with_retries(
                    mode=request.mode,
                    reset_seed=seed,
                    timings=timings,
                    decision_limit=decision_limit,
                    request_cap_forces_end=request_cap_forces_end,
                )
            )
        env_steps = sum(item.transition.steps for item in decisions)
        return RolloutResult(
            worker_index=self.context.worker_index,
            decisions=tuple(decisions),
            env_steps=env_steps,
            episodes_completed=(
                request.episodes_per_environment * len(self.context.environment_indices)
            ),
            policy_version=self._policy_version,
            timings=mirror_renamed_timings(timings),
        )

    def _collect_epoch_with_retries(
        self,
        *,
        mode: DecisionMode,
        reset_seed: int | None,
        timings: dict[str, float],
        decision_limit: int,
        request_cap_forces_end: bool,
    ) -> list[CollectedDecision]:
        """Run one episode on every environment, replacing a crashed simulation and redoing it."""
        for attempt in range(1, self.episode_attempts + 1):
            try:
                return self._collect_epoch(
                    mode=mode,
                    reset_seed=reset_seed,
                    timings=timings,
                    decision_limit=decision_limit,
                    request_cap_forces_end=request_cap_forces_end,
                )
            except RolloutInfrastructureError as exc:
                if attempt == self.episode_attempts:
                    raise RolloutInfrastructureError(
                        f"worker {self.context.worker_index}: the environment failed in "
                        f"{self.episode_attempts} attempt(s) at one episode"
                    ) from exc
                logger.warning(
                    f"worker {self.context.worker_index}: environment failure during an "
                    f"episode (attempt {attempt}/{self.episode_attempts}): "
                    f"{str(exc).splitlines()[0]}; restarting it"
                )
                cast(RestartableBatchedEnvironment, self.env).restart()
        raise AssertionError("unreachable: episode_attempts is positive")

    def _collect_decisions(self, request: RolloutRequest) -> RolloutResult:
        """Pause after each batched decision so published weights apply next time."""
        if self.bootstrap_values != "evaluate_next_observation":
            raise ValueError(
                "Decision collection requires bootstrap_values='evaluate_next_observation' "
                "(formerly successor_values='policy')"
            )
        if self.episode_attempts != 1:
            raise ValueError(
                "Decision collection cannot retry already ingested episodes"
            )
        if (
            request.reset_seed is not None
            or request.max_decisions_per_episode is not None
        ):
            raise ValueError(
                "Persistent decision collection does not accept reset seeds or episode caps"
            )
        assert request.decisions_per_environment is not None
        decisions: list[CollectedDecision] = []
        previous_timings = dict(self._decision_timings)
        for _ in range(request.decisions_per_environment):
            batch = None
            if self._decision_epoch is not None:
                try:
                    batch = self._decision_epoch.send(request.mode)
                except StopIteration:
                    self._decision_epoch = None
            if self._decision_epoch is None:
                self._decision_epoch = self._iterate_epoch(
                    mode=request.mode,
                    reset_seed=None,
                    timings=self._decision_timings,
                    decision_limit=self.chunks_per_episode,
                    request_cap_forces_end=False,
                )
                batch = next(self._decision_epoch)
            assert batch is not None
            decisions.extend(batch)
        return RolloutResult(
            worker_index=self.context.worker_index,
            decisions=tuple(decisions),
            env_steps=sum(item.transition.steps for item in decisions),
            episodes_completed=sum(item.transition.ended for item in decisions),
            policy_version=self._policy_version,
            timings=mirror_renamed_timings(
                {
                    key: value - previous_timings.get(key, 0.0)
                    for key, value in self._decision_timings.items()
                }
            ),
        )

    def _collect_epoch(
        self,
        *,
        mode: DecisionMode,
        reset_seed: int | None,
        timings: dict[str, float],
        decision_limit: int,
        request_cap_forces_end: bool,
    ) -> list[CollectedDecision]:
        epoch = self._iterate_epoch(
            mode=mode,
            reset_seed=reset_seed,
            timings=timings,
            decision_limit=decision_limit,
            request_cap_forces_end=request_cap_forces_end,
        )
        decisions = next(epoch)
        while True:
            try:
                decisions.extend(epoch.send(mode))
            except StopIteration:
                break
        if self.bootstrap_values == "reuse_next_decision":
            return chain_next_values(decisions)
        return decisions

    def _iterate_epoch(
        self,
        *,
        mode: DecisionMode,
        reset_seed: int | None,
        timings: dict[str, float],
        decision_limit: int,
        request_cap_forces_end: bool,
    ) -> Generator[list[CollectedDecision], DecisionMode, None]:
        batch_size = self.env.num_envs
        with timed(timings, "environment_reset_seconds"):
            if reset_seed is None:
                observations, reset_info = self.env.reset()
            else:
                observations, reset_info = self.env.reset(seed=reset_seed)
            reset_info_by_environment = tuple(
                unbatch_observation(reset_info, index, batch_size)
                for index in range(batch_size)
            )
        session_ids = tuple(self.context.environment_indices)
        # One policy seed per session, derived once: what the policy is reset with is
        # what the first decision of the episode records.
        policy_seeds = tuple(
            None
            if reset_seed is None
            else (reset_seed + session_id * 1_000_003) % SEED_MODULUS
            for session_id in session_ids
        )
        self._reset_policy_sessions(session_ids, policy_seeds)
        episode_ids = tuple(
            range(self._next_episode_id, self._next_episode_id + batch_size)
        )
        self._next_episode_id += batch_size
        active: np.ndarray = np.ones(batch_size, dtype=bool)
        self._last_actions = None

        for chunk_index in range(decision_limit):
            active_indices = tuple(int(index) for index in np.flatnonzero(active))
            if not active_indices:
                break
            raw_observations = tuple(
                unbatch_observation(observations, index, batch_size)
                for index in active_indices
            )
            if chunk_index == 0:
                # Every environment is active at the first decision.
                episode = EpisodeStart(
                    reset_seed=reset_seed,
                    episode_ids=episode_ids,
                    policy_seeds=policy_seeds,
                    reset_info=reset_info_by_environment,
                    raw_observation_sha256=tuple(
                        digest_value(observation) for observation in raw_observations
                    ),
                    raw_observation_sha256_by_key=tuple(
                        digest_value_by_key(observation)
                        for observation in raw_observations
                    ),
                )
            chunks = self._action_chunks(
                raw_observations, active_indices, mode=mode, timings=timings
            )
            final_chunk = chunk_index + 1 == decision_limit
            outcome = self._execute_chunks(
                chunks,
                active_indices,
                observations,
                timings=timings,
                final_chunk=final_chunk,
            )
            observations = outcome.observations
            self._encode_next_observations(chunks, outcome, timings=timings)
            decisions = self._chunk_decisions(
                episode,
                outcome,
                chunks,
                raw_observations,
                chunk_index=chunk_index,
                cap_ended=request_cap_forces_end and final_chunk,
            )
            active &= ~(outcome.terminated | outcome.truncated)
            mode = yield decisions
        if active.any():
            raise RuntimeError("fixed-horizon batched rollout left episodes incomplete")

    def _reset_policy_sessions(
        self, session_ids: Sequence[int], policy_seeds: Sequence[int | None]
    ) -> None:
        self.policy.reset_episodes(session_ids, policy_seeds)

    def _action_chunks(
        self,
        raw_observations: Sequence[Mapping[str, Any]],
        active_indices: Sequence[int],
        *,
        mode: DecisionMode,
        timings: dict[str, float],
    ) -> tuple[ActionChunk, ...]:
        """One batched policy call for the active environments, checked for a common shape."""
        with timed(timings, "action_inference_seconds"):
            chunks = tuple(
                self.policy.action_chunks(
                    raw_observations,
                    mode=(mode,) * len(active_indices),
                    session_ids=tuple(
                        self.context.environment_indices[index]
                        for index in active_indices
                    ),
                )
            )
        timings["policy_calls"] = timings.get("policy_calls", 0.0) + 1.0
        if len(chunks) != len(active_indices):
            raise RuntimeError("batched policy returned the wrong number of chunks")
        execution_length = chunks[0].execution_length
        if any(chunk.execution_length != execution_length for chunk in chunks):
            raise ValueError("batched action chunks must share execution_length")
        if any(chunk.batch_size != 1 for chunk in chunks):
            raise ValueError("batched policy must return batch-one ActionChunks")
        # A policy server answers with the weights it acted with; they must be the published ones.
        versions = {chunk.policy_version for chunk in chunks}
        if versions != {self._policy_version}:
            raise RuntimeError(
                f"policy returned versions {sorted(versions)}, but worker serves "
                f"version {self._policy_version}"
            )
        return chunks

    def _execute_chunks(
        self,
        chunks: Sequence[ActionChunk],
        active_indices: Sequence[int],
        observations: Mapping[str, np.ndarray],
        *,
        timings: dict[str, float],
        final_chunk: bool,
    ) -> ChunkOutcome:
        """Step every active environment through its chunk in lockstep."""
        batch_size = self.env.num_envs
        execution_length = chunks[0].execution_length
        action_dim = chunks[0].action_dim
        outcome = ChunkOutcome.start(active_indices, observations, batch_size)
        observation_control = (
            cast(ObservationCaptureControl, self.env)
            if hasattr(self.env, "need_obs")
            else None
        )
        if self._last_actions is None or self._last_actions.shape != (
            batch_size,
            action_dim,
        ):
            self._last_actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        try:
            for tick in range(execution_length):
                if observation_control is not None:
                    observation_control.need_obs = tick + 1 == execution_length
                actions = self._inactive_actions(batch_size, action_dim)
                executing: np.ndarray = np.zeros(batch_size, dtype=bool)
                for local_index, environment_index in enumerate(active_indices):
                    if environment_index in outcome.still_executing:
                        actions[environment_index] = chunks[
                            local_index
                        ].execution_actions[0, tick]
                        executing[environment_index] = True
                self._last_actions = actions
                with timed(timings, "environment_step_seconds"):
                    if self._active_mask:
                        step = self.env.step(actions, active=executing)  # type: ignore[call-arg]
                    else:
                        step = self.env.step(actions)
                # One call advances every environment by one step.
                timings["environment_step_calls"] = (
                    timings.get("environment_step_calls", 0.0) + 1.0
                )
                outcome.record_step(*step)
                if not outcome.still_executing:
                    break
        finally:
            if observation_control is not None:
                observation_control.need_obs = True
        outcome.finish(final_chunk=final_chunk)
        return outcome

    def _inactive_actions(self, batch_size: int, action_dim: int) -> np.ndarray:
        """The commands of environments not executing a chunk this step."""
        if self.inactive_action == "hold":
            assert self._last_actions is not None
            return np.array(self._last_actions, copy=True)
        return np.zeros((batch_size, action_dim), dtype=np.float32)

    def _encode_next_observations(
        self,
        chunks: Sequence[ActionChunk],
        outcome: ChunkOutcome,
        *,
        timings: dict[str, float],
    ) -> None:
        """Encode the next observations replay stores, and value them for valued decisions."""
        if self.bootstrap_values != "evaluate_next_observation":
            return
        session_ids = self.context.environment_indices
        needs_value: list[int] = []
        needs_observation: list[int] = []
        for environment_index, chunk in zip(
            outcome.active_indices, chunks, strict=True
        ):
            if chunk.behavior_value is not None:
                needs_value.append(environment_index)
            elif chunk.learner_observation is not None:
                needs_observation.append(environment_index)
        if needs_value:
            # Every next observation is encoded, the terminal ones included: a
            # replay batch needs one observation layout throughout.
            if not self.policy.features.values_observations:
                raise TypeError(
                    "decisions carry behaviour values but the policy does not value "
                    "next observations; use bootstrap_values='reuse_next_decision'"
                )
            with timed(timings, NEXT_OBSERVATION_TIMING):
                prepared = self.policy.encode_and_value_next_observations(
                    tuple(outcome.successors[index] for index in needs_value),
                    session_ids=tuple(session_ids[index] for index in needs_value),
                )
            for environment_index, (prepared_observation, value) in zip(
                needs_value, prepared, strict=True
            ):
                outcome.replay_successors[environment_index] = prepared_observation
                # A terminal next observation is never bootstrapped; its value stays 0.
                if not outcome.terminated[environment_index]:
                    outcome.next_values[environment_index] = first_value(value)
        if needs_observation:
            with timed(timings, NEXT_OBSERVATION_TIMING):
                prepared_observations = self.policy.encode_observations(
                    tuple(outcome.successors[index] for index in needs_observation),
                    session_ids=tuple(
                        session_ids[index] for index in needs_observation
                    ),
                )
            if prepared_observations is None:
                raise TypeError(
                    "decisions carry learner observations but the policy does not "
                    "encode next observations (encode_observations)"
                )
            for environment_index, prepared_observation in zip(
                needs_observation, prepared_observations, strict=True
            ):
                outcome.replay_successors[environment_index] = prepared_observation

    def _chunk_decisions(
        self,
        episode: EpisodeStart,
        outcome: ChunkOutcome,
        chunks: Sequence[ActionChunk],
        raw_observations: Sequence[Mapping[str, Any]],
        *,
        chunk_index: int,
        cap_ended: bool,
    ) -> list[CollectedDecision]:
        """File one decision per active environment for the chunk just executed."""
        decisions: list[CollectedDecision] = []
        for local_index, environment_index in enumerate(outcome.active_indices):
            chunk = chunks[local_index]
            info = dict(outcome.terminal_info[environment_index])
            decision_cap_reached = (
                cap_ended
                and not outcome.terminated[environment_index]
                and environment_index in outcome.still_executing
            )
            learner_observation = chunk.learner_observation
            replay_observation = (
                raw_observations[local_index]
                if learner_observation is None
                else {
                    key: np.array(value[0], copy=True)
                    for key, value in learner_observation.items()
                }
            )
            if self.bootstrap_values == "evaluate_next_observation":
                replay_next_observation = outcome.replay_successors.get(
                    environment_index, outcome.successors[environment_index]
                )
                # A terminal successor is never bootstrapped, so its value is 0.
                next_value = (
                    outcome.next_values.get(environment_index, 0.0)
                    if chunk.behavior_value is not None
                    else None
                )
            else:
                # The successor is the next decision's own observation and
                # value, filled in once the episode is complete (see below).
                replay_next_observation = outcome.successors[environment_index]
                next_value = None
            global_index = self.context.environment_indices[environment_index]
            decisions.append(
                CollectedDecision(
                    environment_index=global_index,
                    transition=decision_transition(
                        chunk,
                        episode_id=episode.episode_ids[environment_index],
                        observation=replay_observation,
                        next_observation=replay_next_observation,
                        rewards=outcome.rewards[environment_index],
                        terminated=bool(outcome.terminated[environment_index]),
                        truncated=bool(outcome.truncated[environment_index]),
                        info=info,
                        decision_cap_reached=decision_cap_reached,
                        reward_transform=self.reward_transform,
                        metadata=TransitionMetadata(
                            step_in_episode=chunk_index * chunks[0].execution_length,
                            decision_index=chunk_index,
                            policy_version=chunk.policy_version,
                            environment_index=global_index,
                            reset_seed=episode.reset_seed,
                            reset_seed_scope="worker_batch",
                            wall_time=time.time(),
                            extra=episode.decision_extra(
                                environment_index,
                                info,
                                first_decision=chunk_index == 0,
                                decision_cap_reached=decision_cap_reached,
                            ),
                        ),
                        next_value=next_value,
                    ),
                    action_chunk=chunk,
                )
            )
        return decisions

    def set_policy_state(self, state: PolicyState) -> int:
        applied = self.policy.set_policy_state(state)
        if applied != state.version:
            raise RuntimeError(
                f"policy acknowledged version {applied}, expected {state.version}"
            )
        self._policy_version = applied
        return applied

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._decision_epoch is not None:
            self._decision_epoch.close()
            self._decision_epoch = None
        errors: list[Exception] = []
        try:
            self.policy.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        try:
            self.env.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} batched rollout resource(s) failed to close"
            ) from errors[0]
