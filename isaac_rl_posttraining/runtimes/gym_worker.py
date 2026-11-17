# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulator-neutral Gymnasium worker for runtime parity and integration."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np

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
    PolicyContext,
    PolicyState,
    RolloutRequest,
    RolloutResult,
    RolloutWorkerContext,
)
from isaac_rl_posttraining.runtimes.policy import (
    ActionChunkPolicy,
    DecisionMode,
    InfoValue,
    Observation,
    ObservationValue,
    RolloutPolicy,
)
from isaac_rl_posttraining.runtimes.policy_adapters import (
    as_rollout_policy,
    first_value,
)
from isaac_rl_posttraining.runtimes.rollout_records import (
    NEXT_OBSERVATION_TIMING,
    RewardTransform,
    SparseOutcomeReward,
    TransformedReward,
    decision_transition,
    image_frames,
    learning_reward_channel,
    mirror_renamed_timings,
    timed,
)
from isaac_rl_posttraining.runtimes.serving.handles import ServiceHandle
from isaac_rl_posttraining.runtimes.serving.local import connect_policy
from isaac_rl_posttraining.utils.compat import add_note
from isaac_rl_posttraining.utils.hashing import digest_value, digest_value_by_key

__all__ = [
    "ActionChunkPolicy",
    "ChunkSteppingEnvironment",
    "DecisionMode",
    "GymEnvironment",
    "GymRolloutWorker",
    "GymWorkerSpec",
    "Observation",
    "PolicyRef",
    "PolicySpec",
    "RewardTransform",
    "SparseOutcomeReward",
    "TransformedReward",
    "decision_transition",
    "image_frames",
    "learning_reward_channel",
    "resolve_num_rollout_workers",
    "worker_policy",
]


@runtime_checkable
class GymEnvironment(Protocol):
    """The unbatched Gymnasium surface required by :class:`GymRolloutWorker`."""

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[Observation, Mapping[str, InfoValue]]: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, Mapping[str, InfoValue]]: ...

    def close(self) -> None: ...


@runtime_checkable
class ChunkSteppingEnvironment(GymEnvironment, Protocol):
    """An environment that executes a whole ``[chunk, action_dim]`` chunk per call.

    ``step_chunk`` returns the observation after the last executed action,
    one reward per executed action, the terminal flags, and the last info.
    """

    def step_chunk(
        self, actions: np.ndarray
    ) -> tuple[Observation, Sequence[float], bool, bool, Mapping[str, InfoValue]]: ...


class GymRolloutWorker:
    """Execute action chunks against one persistent, unbatched environment.

    The policy owns algorithm-specific inference and decode.
    The environment owns simulator-specific observation and action semantics.
    This worker only enforces the common chunk protocol, receding-horizon
    execution, and transition accounting, which makes it usable under every
    runtime provider. With ``chunked_step`` the environment executes each
    chunk in one ``step_chunk`` call (one round trip to a remote simulator)
    instead of one ``step`` per action.

    The policy decides for ``session_id`` (0 for a policy of the worker's
    own; the environment index for a policy server shared by many workers).
    With ``fused_act_and_value``, a valued decision whose episode goes on
    asks for its bootstrap value and the next decision in one request.
    """

    def __init__(
        self,
        context: RolloutWorkerContext,
        env: GymEnvironment,
        policy: RolloutPolicy | ActionChunkPolicy,
        *,
        reward_transform: RewardTransform | None = None,
        chunked_step: bool = False,
        session_id: Hashable = 0,
        fused_act_and_value: bool = False,
    ) -> None:
        if len(context.environment_indices) != 1:
            raise ValueError(
                "GymRolloutWorker hosts exactly one unbatched environment; use a "
                "vectorized worker factory when assigning multiple environments"
            )
        if chunked_step and not isinstance(env, ChunkSteppingEnvironment):
            raise TypeError("chunked_step requires an environment with step_chunk(...)")
        self.policy = as_rollout_policy(policy)
        features = self.policy.features
        if features.behavior_values and not features.values_observations:
            raise TypeError(
                "a policy whose decisions carry behaviour values must value next "
                "observations (encode_and_value_next_observations) so time-limit "
                "transitions keep an unbiased bootstrap value"
            )
        if fused_act_and_value and not features.values_observations:
            raise TypeError(
                "fused_act_and_value needs a policy that values observations"
            )
        self.context = context
        self.env = env
        self.session_id = session_id
        self.reward_transform = reward_transform
        self.chunked_step = chunked_step
        self.fused_act_and_value = bool(fused_act_and_value)
        #: The next decision, already chosen by a fused request.
        self._next_chunk: ActionChunk | None = None
        self._observation: Observation | None = None
        self._episode_id = 0
        self._step_in_episode = 0
        self._decisions_in_episode = 0
        self._episode_reset_seed: int | None = None
        self._episode_reset_info: dict[str, Any] = {}
        self._episode_raw_digests: tuple[str, dict[str, str]] = ("", {})
        self._policy_version = 0
        self._closed = False

    @staticmethod
    def _unbatch_learner_observation(chunk: ActionChunk) -> Observation:
        learner_observation = chunk.learner_observation
        if learner_observation is None:
            raise ValueError("ActionChunk has no learner observation")
        result: dict[str, ObservationValue] = {}
        for key, value in learner_observation.items():
            if value.ndim < 1 or value.shape[0] != 1:
                raise ValueError(
                    f"learner observation {key!r} must have batch size 1, "
                    f"got {value.shape}"
                )
            result[key] = np.array(value[0], copy=True)
        return result

    def collect(self, request: RolloutRequest) -> RolloutResult:
        if self._closed:
            raise RuntimeError("rollout worker is closed")
        decisions: list[CollectedDecision] = []
        frames: list[dict[str, np.ndarray]] = []
        env_steps = 0
        episodes_completed = 0
        timings: dict[str, float] = {}
        if request.reset_seed is not None and self._observation is not None:
            raise RuntimeError("seeded collection must begin at an episode boundary")

        while True:
            if (
                request.decisions_per_environment is not None
                and len(decisions) >= request.decisions_per_environment
            ):
                break
            if (
                request.episodes_per_environment is not None
                and episodes_completed >= request.episodes_per_environment
            ):
                break
            decision = self._decide(
                request,
                episodes_completed,
                timings,
                frames,
                decides_again=request.decisions_per_environment is None
                or len(decisions) + 1 < request.decisions_per_environment,
            )
            decisions.append(decision)
            env_steps += decision.transition.steps
            if decision.transition.ended:
                episodes_completed += 1

        if self._next_chunk is not None:
            raise RuntimeError(
                "a fused request chose a decision the collection did not use"
            )
        return RolloutResult(
            worker_index=self.context.worker_index,
            decisions=tuple(decisions),
            env_steps=env_steps,
            episodes_completed=episodes_completed,
            policy_version=self._policy_version,
            timings=mirror_renamed_timings(timings),
            frames=tuple(frame for frame in frames if frame)
            if request.capture_frames
            else None,
        )

    def _decide(
        self,
        request: RolloutRequest,
        episodes_completed: int,
        timings: dict[str, float],
        frames: list[dict[str, np.ndarray]],
        *,
        decides_again: bool,
    ) -> CollectedDecision:
        """Act once from the current observation, resetting first at an episode boundary.

        ``decides_again`` says whether this collection decides again after
        this decision, so a fused request may choose that decision now.
        """
        if self._observation is None:
            self._start_episode(request.reset_seed, episodes_completed, timings)
        current = self._observation
        if current is None:
            raise RuntimeError("environment reset returned no observation")
        observation = copy.deepcopy(current)
        if request.capture_frames:
            frames.append(image_frames(observation))
        if self._next_chunk is None:
            with timed(timings, "action_inference_seconds"):
                chunk = self.policy.action_chunks(
                    (observation,), mode=(request.mode,), session_ids=(self.session_id,)
                )[0]
        else:
            chunk, self._next_chunk = self._next_chunk, None
        timings["policy_calls"] = timings.get("policy_calls", 0.0) + 1.0
        if chunk.batch_size != 1:
            raise ValueError(
                f"one GymRolloutWorker requires batch_size=1, got {chunk.batch_size}"
            )
        if chunk.policy_version != self._policy_version:
            raise RuntimeError(
                f"policy returned version {chunk.policy_version}, but worker serves "
                f"version {self._policy_version}"
            )
        replay_observation = (
            observation
            if chunk.learner_observation is None
            else self._unbatch_learner_observation(chunk)
        )

        next_observation, rewards, terminated, truncated, terminal_info = (
            self._execute_chunk(chunk, observation, timings)
        )
        self._decisions_in_episode += 1
        # The cap ends a non-terminating episode as a time-limit cut; the
        # transition scores it as a failure whatever the info says.
        decision_cap_reached = (
            request.max_decisions_per_episode is not None
            and not terminated
            and not truncated
            and self._decisions_in_episode >= request.max_decisions_per_episode
        )
        if decision_cap_reached:
            truncated = True
        extra = self._decision_extra(terminal_info, decision_cap_reached)
        fused_mode = (
            request.mode
            if self.fused_act_and_value
            and decides_again
            and not terminated
            and not truncated
            else None
        )
        replay_next_observation, next_value = self._replay_next_observation(
            chunk, next_observation, timings, fused_mode=fused_mode
        )

        transition = decision_transition(
            chunk,
            episode_id=self._episode_id,
            observation=replay_observation,
            next_observation=replay_next_observation,
            rewards=rewards,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=terminal_info,
            decision_cap_reached=decision_cap_reached,
            reward_transform=self.reward_transform,
            metadata=TransitionMetadata(
                step_in_episode=self._step_in_episode,
                decision_index=self._decisions_in_episode - 1,
                policy_version=chunk.policy_version,
                environment_index=self.context.environment_indices[0],
                reset_seed=self._episode_reset_seed,
                reset_seed_scope="environment",
                wall_time=time.time(),
                extra=extra,
            ),
            next_value=next_value,
        )
        self._step_in_episode += len(rewards)

        if terminated or truncated:
            if request.capture_frames:
                frames.append(image_frames(next_observation))
            self._episode_id += 1
            self._observation = None
            self._episode_reset_seed = None
            self._episode_reset_info = {}
        else:
            self._observation = next_observation
        return CollectedDecision(
            environment_index=self.context.environment_indices[0],
            transition=transition,
            action_chunk=chunk,
        )

    def _start_episode(
        self,
        reset_seed: int | None,
        episodes_completed: int,
        timings: dict[str, float],
    ) -> None:
        """Reset the environment and the policy; the first decision records how."""
        episode_seed = None
        with timed(timings, "environment_reset_seconds"):
            if reset_seed is None:
                self._observation, reset_info = self.env.reset()
            else:
                environment_index = self.context.environment_indices[0]
                episode_seed = (
                    reset_seed + environment_index * 1_000_003 + episodes_completed
                ) % SEED_MODULUS
                self._observation, reset_info = self.env.reset(seed=episode_seed)
            self._episode_reset_seed = episode_seed
            self._episode_reset_info = dict(reset_info)
        self._step_in_episode = 0
        self._decisions_in_episode = 0
        self._episode_raw_digests = (
            digest_value(self._observation),
            digest_value_by_key(self._observation),
        )
        self.policy.reset_episode(self.session_id, seed=episode_seed)

    def _execute_chunk(
        self, chunk: ActionChunk, observation: Observation, timings: dict[str, float]
    ) -> tuple[Observation, list[float], bool, bool, dict[str, InfoValue]]:
        """Step the environment through the chunk until it ends or the episode does."""
        rewards: list[float] = []
        terminal_info: dict[str, InfoValue] = {}
        terminated = False
        truncated = False
        next_observation = observation
        if self.chunked_step:
            # The constructor rejects chunked_step without step_chunk(...).
            chunk_env = cast(ChunkSteppingEnvironment, self.env)
            with timed(timings, "environment_step_seconds"):
                (
                    next_observation,
                    chunk_rewards,
                    terminated,
                    truncated,
                    info,
                ) = chunk_env.step_chunk(chunk.execution_actions[0])
            # A chunked call advances the environment by several steps.
            timings["environment_step_calls"] = timings.get(
                "environment_step_calls", 0.0
            ) + float(len(chunk_rewards))
            rewards.extend(float(reward) for reward in chunk_rewards)
            terminal_info = dict(info)
        else:
            for action in chunk.execution_actions[0]:
                with timed(timings, "environment_step_seconds"):
                    next_observation, reward, terminated, truncated, info = (
                        self.env.step(action)
                    )
                timings["environment_step_calls"] = (
                    timings.get("environment_step_calls", 0.0) + 1.0
                )
                rewards.append(float(reward))
                terminal_info = dict(info)
                if terminated or truncated:
                    break
        if not rewards:
            raise RuntimeError("an ActionChunk executed zero simulator steps")
        return next_observation, rewards, terminated, truncated, terminal_info

    def _decision_extra(
        self, terminal_info: Mapping[str, InfoValue], decision_cap_reached: bool
    ) -> dict[str, Any]:
        """The decision's metadata extras; an episode's first decision also records its reset."""
        first_decision = self._step_in_episode == 0
        extra: dict[str, Any] = {}
        if first_decision:
            extra.update(self._episode_reset_info)
        extra.update(terminal_info)
        if decision_cap_reached:
            extra["decision_cap_reached"] = True
            extra["terminal_reason"] = "decision_cap"
        if first_decision:
            extra["policy_seed"] = self._episode_reset_seed
            extra["raw_observation_sha256"] = self._episode_raw_digests[0]
            extra["raw_observation_sha256_by_key"] = self._episode_raw_digests[1]
        return extra

    def _replay_next_observation(
        self,
        chunk: ActionChunk,
        next_observation: Observation,
        timings: dict[str, float],
        *,
        fused_mode: DecisionMode | None,
    ) -> tuple[Observation, float | None]:
        """The next observation to replay and, for a valued decision, its bootstrap value.

        A decision with a learner observation replays the next observation in
        the learner's layout; one with a behaviour value also stores the value
        of that observation. With ``fused_mode`` set, the same request also
        chooses the next decision.
        """
        if chunk.learner_observation is None:
            return copy.deepcopy(next_observation), None
        session = (self.session_id,)
        if chunk.behavior_value is not None:
            if not self.policy.features.values_observations:
                raise TypeError(
                    "a policy whose decisions carry behaviour values must value next "
                    "observations (encode_and_value_next_observations)"
                )
            if fused_mode is not None:
                with timed(timings, "action_inference_seconds"):
                    self._next_chunk, encoded, value = self.policy.act_and_value(
                        (next_observation,), mode=(fused_mode,), session_ids=session
                    )[0]
                return encoded, first_value(value)
            with timed(timings, NEXT_OBSERVATION_TIMING):
                encoded, value = self.policy.encode_and_value_next_observations(
                    (next_observation,), session_ids=session
                )[0]
            return encoded, first_value(value)
        # The edit family's frozen reference decode runs here, so it gets its
        # own bucket in the evaluator's time attribution.
        with timed(timings, NEXT_OBSERVATION_TIMING):
            encoded_observations = self.policy.encode_observations(
                (next_observation,), session_ids=session
            )
        if encoded_observations is None:
            raise TypeError(
                "a policy whose decisions carry learner observations must encode "
                "next observations (encode_observations)"
            )
        return encoded_observations[0], None

    def set_policy_state(self, state: PolicyState) -> int:
        if self._closed:
            raise RuntimeError("rollout worker is closed")
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
        errors: list[Exception] = []
        try:
            self.policy.close()
        # Still close the simulator.
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        try:
            self.env.close()
        # Report after all cleanup.
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} Gym rollout worker resource(s) failed to close"
            ) from errors[0]


#: Builds the acting policy of a worker or policy server from its context;
#: policy families provide these (``SteeringPolicySpec``, ``EditPolicySpec``, ...).
PolicySpec = Callable[[PolicyContext], RolloutPolicy | ActionChunkPolicy]
#: Where a worker's policy comes from: a spec it builds a copy from, or a
#: policy service it connects to.
PolicyRef: TypeAlias = PolicySpec | ServiceHandle


def check_policy_ref(policy: object) -> None:
    """Fail at construction unless ``policy`` is a policy spec or a service handle."""
    if not isinstance(policy, ServiceHandle) and not callable(policy):
        raise TypeError("policy must be a policy spec (callable) or a ServiceHandle")


def worker_policy(
    policy: PolicyRef, context: RolloutWorkerContext, *, route_index: int
) -> RolloutPolicy:
    """The policy a worker acts with: a client of the service, or its own copy built from the spec.

    ``route_index`` picks the service replica (an environment index for an
    unbatched worker, the worker index for a batched one).
    """
    if isinstance(policy, ServiceHandle):
        return connect_policy(policy, route_index=route_index)
    return as_rollout_policy(
        policy(
            PolicyContext(
                policy_index=context.worker_index,
                num_policies=context.num_rollout_workers,
                device=context.device,
                placement="rollout_worker",
            )
        )
    )


def resolve_num_rollout_workers(
    spec: object,
    *,
    num_envs: int,
    configured: int | None,
    batched_workers: int = 1,
) -> int:
    """Resolve an ergonomic worker default from the rollout worker spec.

    A Gym spec hosts one environment per worker; a batched spec hosts them in
    ``batched_workers`` workers (one per simulator server; one by default).
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if (
        isinstance(batched_workers, bool)
        or not isinstance(batched_workers, int)
        or batched_workers < 1
    ):
        raise ValueError("batched_workers must be a positive integer")
    if configured is None:
        return num_envs if isinstance(spec, GymWorkerSpec) else batched_workers
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        raise ValueError("configured rollout workers must be a positive integer")
    if configured > num_envs:
        raise ValueError("num_rollout_workers cannot exceed num_envs")
    return configured


@dataclass(frozen=True)
class GymWorkerSpec:
    """One unbatched environment per worker behind an acting policy.

    A picklable recipe: the runtime ships it to each worker and calls it with
    the worker's context. The simulator entry builds the environment there
    (``entry.make_environment``); ``policy`` is either a policy spec the
    worker builds its own copy from, or the handle of a policy service the
    worker connects to (one session per environment). Neither side imports a
    runtime provider. ``supports_batched_policy`` says the spec's policy may
    be shared by many workers as a policy service.
    """

    entry: SimulatorEntry
    environment: EnvironmentSpec
    phase: Phase
    policy: PolicyRef
    reward_transform: RewardTransform | None = None
    supports_batched_policy: bool = False
    #: The environments execute a whole chunk per ``step_chunk`` call.
    chunked_step: bool = False
    #: Ask for a valued decision's bootstrap value and the next decision in one request.
    fused_act_and_value: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SimulatorEntry):
            raise TypeError("entry must be a SimulatorEntry")
        if not isinstance(self.environment, EnvironmentSpec):
            raise TypeError("environment must be an EnvironmentSpec")
        if self.phase not in ("train", "eval"):
            raise ValueError("phase must be 'train' or 'eval'")
        check_policy_ref(self.policy)
        if not isinstance(self.supports_batched_policy, bool):
            raise TypeError("supports_batched_policy must be a boolean")
        if not isinstance(self.chunked_step, bool):
            raise TypeError("chunked_step must be a boolean")
        if not isinstance(self.fused_act_and_value, bool):
            raise TypeError("fused_act_and_value must be a boolean")

    def make_environment(self, context: RolloutWorkerContext) -> GymEnvironment:
        """The worker's environment, built by the simulator entry on the worker."""
        return self.entry.make_environment(self.environment, self.phase, context)

    @property
    def reward_channel(self) -> str:
        """The reward channel this spec's decisions are learned and scored from."""
        return learning_reward_channel(self.reward_transform)

    def __call__(self, context: RolloutWorkerContext) -> GymRolloutWorker:
        if len(context.environment_indices) != 1:
            raise ValueError("a Gym rollout worker owns exactly one environment")
        environment_index = context.environment_indices[0]
        served = isinstance(self.policy, ServiceHandle)
        env = self.make_environment(context)
        try:
            return GymRolloutWorker(
                context,
                env,
                worker_policy(self.policy, context, route_index=environment_index),
                reward_transform=self.reward_transform,
                chunked_step=self.chunked_step,
                session_id=environment_index if served else 0,
                fused_act_and_value=self.fused_act_and_value,
            )
        except BaseException as exc:
            try:
                env.close()
            # Preserve primary failure.
            except Exception as close_exc:  # noqa: BLE001
                add_note(
                    exc,
                    "environment cleanup after worker-construction failure also "
                    f"failed: {close_exc!r}",
                )
            raise
