# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decision records: what a policy produced, what the simulator executed, and how they file into episodes.

A :class:`Transition` is one decision of a rollout worker: the observation the
policy saw, the :class:`LearnerAction` it chose, the :class:`EnvAction` the
simulator executed step by step, the per-step reward channels and ending
flags, and the observation after the last executed step. Complete
:class:`Episode` objects are what the replay buffer stores. The field mapping
of the legacy ``buffers.base.Transition`` converts both ways without loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from isaac_rl_posttraining.data.records import (
    Array,
    ArrayMap,
    DynamicObservation,
    Observation,
    Record,
    coerce_array,
    dim,
    require_finite,
)
from isaac_rl_posttraining.data.spec import ActionSpec

__all__ = [
    "ACTION_KINDS",
    "BEHAVIOR_SOURCES",
    "ORIGINS",
    "RESET_SEED_SCOPES",
    "ActionKind",
    "BehaviorSource",
    "EnvAction",
    "Episode",
    "EpisodeMetadata",
    "Intervention",
    "InterventionPhase",
    "LearnerAction",
    "Origin",
    "PolicyOutputs",
    "ResetSeedScope",
    "Transition",
    "TransitionMetadata",
    "episode_success",
]

#: What produced a decision: the learned policy, warm-up exploration, the
#: frozen policy acting natively, or an expert (demonstrations, interventions).
BehaviorSource = Literal["policy", "warmup", "native", "expert"]
BEHAVIOR_SOURCES: tuple[str, ...] = ("policy", "warmup", "native", "expert")

#: Where the dynamics of a decision came from: the real (or simulated) environment, or a learned model.
Origin = Literal["real", "model"]
ORIGINS: tuple[str, ...] = ("real", "model")

#: What a reset seed indexes: one environment's episodes, or one batched worker's.
ResetSeedScope = Literal["environment", "worker_batch"]
RESET_SEED_SCOPES: tuple[str, ...] = ("environment", "worker_batch")

#: The space the learner acts in: the frozen policy's flow-noise latent
#: (DSRL, RFS), a bounded edit of the reference chunk (SAC-edit), the
#: executed chunk itself (direct methods), or the denoising chain of a flow
#: policy fine-tuned with PPO (every intermediate chunk of the sampler).
ActionKind = Literal["latent", "edit", "chunk", "chain"]
ACTION_KINDS: tuple[str, ...] = ("latent", "edit", "chunk", "chain")


@dataclass(frozen=True)
class PolicyOutputs(Record):
    """Per-decision quantities the acting policy produced besides the action.

    Scalars are per decision; ``chunk_steps`` arrays are per executed step of
    the chunk; ``denoising_steps`` arrays are per denoising step of a flow
    policy; ``features`` are the activations of frozen sub-modules a learner
    reuses instead of recomputing them. Policies with more outputs subclass
    this record.
    """

    log_prob: Array[np.float32] | None = None
    value: Array[np.float32] | None = None
    next_value: Array[np.float32] | None = None
    entropy: Array[np.float32] | None = None
    action_mean: Array[np.float32, dim.chunk_steps, dim.act_dim] | None = None
    action_std: Array[np.float32, dim.chunk_steps, dim.act_dim] | None = None
    denoising_log_prob: Array[np.float32, dim.denoising_steps] | None = None
    features: ArrayMap | None = None

    def validate(self, bindings: Mapping[str, int] | None = None) -> None:
        super().validate(bindings)
        owner = type(self).__name__
        for spec in type(self).declared_fields():
            value = getattr(self, spec.name)
            if value is None:
                continue
            if spec.kind == "array":
                require_finite(owner, spec.name, value)
            elif spec.kind == "arrays":
                for key, entry in value.items():
                    require_finite(owner, f"{spec.name}[{key!r}]", entry)


@dataclass(frozen=True)
class LearnerAction:
    """What the actor emitted and the critic scores, in the learner's own action space."""

    kind: ActionKind
    value: np.ndarray

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}; known: {ACTION_KINDS}")
        value = coerce_array("LearnerAction", "value", self.value)
        if value.ndim < 1:
            raise ValueError("a learner action needs at least one dimension")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"a learner action must be floating point, got {value.dtype}")
        require_finite("LearnerAction", "value", value)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class EnvAction:
    """The per-step commands the environment executed for one decision.

    ``commands`` holds only the steps that ran (``[steps, act_dim]``);
    ``execution_length`` is how many the collector asked for, which exceeds
    ``steps`` when the episode ended inside the chunk.
    """

    commands: np.ndarray
    execution_length: int

    def __post_init__(self) -> None:
        commands = coerce_array("EnvAction", "commands", self.commands)
        if commands.ndim != 2 or commands.shape[0] < 1 or commands.shape[1] < 1:
            raise ValueError(f"commands must have shape [steps, act_dim] with both positive, got {commands.shape}")
        if isinstance(self.execution_length, bool) or not isinstance(self.execution_length, int):
            raise TypeError("execution_length must be an integer")
        if self.execution_length < commands.shape[0]:
            raise ValueError(
                f"execution_length {self.execution_length} is shorter than the {commands.shape[0]} executed steps"
            )
        require_finite("EnvAction", "commands", commands)
        object.__setattr__(self, "commands", commands)

    @property
    def steps(self) -> int:
        return int(self.commands.shape[0])

    @property
    def width(self) -> int:
        return int(self.commands.shape[1])

    def spec(self) -> ActionSpec:
        return ActionSpec(width=self.width, dtype=self.commands.dtype.name)


#: Where a replay window sits relative to an expert hand-off within an episode.
InterventionPhase = Literal[
    "none", "pre_intervention", "intervention", "post_intervention", "pre_and_post_intervention"
]


@dataclass(frozen=True)
class Intervention:
    """Which executed steps of a decision an expert took over, when interventions are recorded."""

    phase: InterventionPhase
    policy_tick_mask: np.ndarray | None = None
    expert_tick_mask: np.ndarray | None = None
    pre_intervention_tick_mask: np.ndarray | None = None
    post_intervention_tick_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "policy_tick_mask",
            "expert_tick_mask",
            "pre_intervention_tick_mask",
            "post_intervention_tick_mask",
        ):
            mask = getattr(self, name)
            if mask is not None:
                object.__setattr__(self, name, np.asarray(mask))


@dataclass(frozen=True)
class TransitionMetadata:
    """Bookkeeping of one decision that training does not read.

    ``extra`` is the environment's untyped ``info`` mapping as the worker
    received it; the typed fields on the transition are the readers' source
    of truth, ``extra`` stays for inspection and for replaying old buffers.
    ``source``, ``worker_index`` and ``policy_lag`` are the lineage the data
    source stamps at ingest: which data source took the decision in, which
    rollout worker collected it, and how many weight versions were published
    between acting and ingesting.
    """

    step_in_episode: int
    decision_index: int | None = None
    policy_version: int | None = None
    environment_index: int | None = None
    reset_seed: int | None = None
    reset_seed_scope: ResetSeedScope | None = None
    wall_time: float | None = None
    max_steps_in_episode: int | None = None
    intervention: Intervention | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    worker_index: int | None = None
    policy_lag: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "step_in_episode",
            "decision_index",
            "policy_version",
            "environment_index",
            "max_steps_in_episode",
            "worker_index",
            "policy_lag",
        ):
            value = getattr(self, name)
            if value is None and name != "step_in_episode":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.reset_seed_scope is not None and self.reset_seed_scope not in RESET_SEED_SCOPES:
            raise ValueError(f"unknown reset seed scope {self.reset_seed_scope!r}; known: {RESET_SEED_SCOPES}")
        if self.source is not None and (not isinstance(self.source, str) or not self.source):
            raise ValueError(f"source must be a non-empty string, got {self.source!r}")
        object.__setattr__(self, "extra", dict(self.extra))


def episode_success(info: Mapping[str, Any]) -> bool:
    """The task outcome an environment reports in the ``info`` of the step that ended the episode.

    Simulators write ``is_success`` or ``success``; ending an episode without
    either is an error rather than a failure, so a renamed flag cannot silently
    score every episode as failed.
    """
    for key in ("is_success", "success"):
        if key in info:
            return bool(info[key])
    raise ValueError("an episode ended without an 'is_success' or 'success' flag in info")


def _flags(name: str, value: Any, steps: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean array per executed step, got {array.dtype}")
    if array.shape != (steps,):
        raise ValueError(f"{name} must have shape [{steps}], got {list(array.shape)}")
    return array


_LEGACY_FIELDS: tuple[str, ...] = (
    "obs",
    "action",
    "reward",
    "next_obs",
    "done",
    "truncated",
    "episode_id",
    "step_in_episode",
    "max_steps_in_episode",
    "timestamp",
    "language_instruction",
    "info",
    "reward_sequence",
    "bootstrap_steps",
    "behavior_source",
    "intervention_phase",
    "policy_tick_mask",
    "expert_tick_mask",
    "pre_intervention_tick_mask",
    "post_intervention_tick_mask",
    "cached_features",
)

#: ``info`` keys the legacy transition carried that the typed records hold in their own fields.
_LEGACY_OUTPUT_KEYS: dict[str, str] = {
    "behavior_log_prob": "log_prob",
    "behavior_value": "value",
    "behavior_next_value": "next_value",
}


def _reward_channels(rewards: Any) -> tuple[dict[str, np.ndarray], int]:
    """The reward channels as float32 arrays and the executed step count they agree on."""
    if not isinstance(rewards, Mapping) or "env" not in rewards:
        raise ValueError(
            "rewards must be a mapping with the environment's 'env' channel"
        )
    channels: dict[str, np.ndarray] = {}
    for name, values in rewards.items():
        if not isinstance(name, str) or not name:
            raise ValueError("reward channel names must be non-empty strings")
        channels[name] = np.asarray(values, dtype=np.float32)
        require_finite("Transition", f"rewards[{name!r}]", channels[name])
    steps = int(channels["env"].shape[0]) if channels["env"].ndim == 1 else 0
    if steps < 1:
        raise ValueError("rewards['env'] must hold one reward per executed step")
    for name, values in channels.items():
        if values.shape != (steps,):
            raise ValueError(
                f"reward channel {name!r} must have shape [{steps}], got {list(values.shape)}"
            )
    return channels, steps


@dataclass(frozen=True)
class Transition:
    """One decision of one episode, as the replay buffer stores it.

    ``rewards`` maps a channel name to one float per executed step; ``"env"``
    is the environment's own reward and is always present. Shaped rewards are
    derived from the flags at sample time and relabeled rewards are added as
    new channels, so ``"env"`` is never rewritten. ``origin`` says whether the
    dynamics were real (a simulator or robot) or imagined by a learned model.
    """

    episode_id: int | str
    observation: Observation
    next_observation: Observation
    learner_action: LearnerAction | None
    env_action: EnvAction | None
    rewards: Mapping[str, np.ndarray]
    terminated: np.ndarray
    truncated: np.ndarray
    success: np.ndarray
    behavior_source: BehaviorSource
    metadata: TransitionMetadata
    policy_outputs: PolicyOutputs = field(default_factory=PolicyOutputs)
    origin: Origin = "real"

    def __post_init__(self) -> None:
        self._check_field_types()
        channels, steps = _reward_channels(self.rewards)
        object.__setattr__(self, "rewards", channels)
        object.__setattr__(
            self, "terminated", _flags("terminated", self.terminated, steps)
        )
        object.__setattr__(
            self, "truncated", _flags("truncated", self.truncated, steps)
        )
        object.__setattr__(self, "success", _flags("success", self.success, steps))
        if self.terminated[:-1].any() or self.truncated[:-1].any():
            raise ValueError(
                "only the last executed step of a decision may end the episode"
            )
        if self.env_action is not None and self.env_action.steps != steps:
            raise ValueError(
                f"env_action executed {self.env_action.steps} steps but {steps} rewards were recorded"
            )
        self.validate()

    def _check_field_types(self) -> None:
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, (int, str)):
            raise TypeError("episode_id must be an int or a string")
        if not isinstance(self.observation, Observation) or not isinstance(self.next_observation, Observation):
            raise TypeError("observation and next_observation must be Observation records")
        if self.learner_action is not None and not isinstance(self.learner_action, LearnerAction):
            raise TypeError("learner_action must be a LearnerAction or None")
        if self.env_action is not None and not isinstance(self.env_action, EnvAction):
            raise TypeError("env_action must be an EnvAction or None")
        if self.behavior_source not in BEHAVIOR_SOURCES:
            raise ValueError(f"unknown behavior source {self.behavior_source!r}; known: {BEHAVIOR_SOURCES}")
        if not isinstance(self.metadata, TransitionMetadata):
            raise TypeError("metadata must be a TransitionMetadata")
        if not isinstance(self.policy_outputs, PolicyOutputs):
            raise TypeError("policy_outputs must be a PolicyOutputs record")
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r}; known: {ORIGINS}")

    def validate(self) -> None:
        """Hook for consistency rules beyond the field checks; the base record has none left."""

    @property
    def steps(self) -> int:
        """Executed environment steps of this decision."""
        return int(self.rewards["env"].shape[0])

    @property
    def ended(self) -> bool:
        return bool(self.terminated[-1] or self.truncated[-1])

    @property
    def succeeded(self) -> bool:
        return bool(self.success.any())

    @property
    def outcome(self) -> Literal["ongoing", "success", "failure", "timeout"]:
        """How the episode stands after this decision."""
        if not self.ended:
            return "ongoing"
        if self.succeeded:
            return "success"
        if self.truncated[-1] and not self.terminated[-1]:
            return "timeout"
        return "failure"

    def reward(self, channel: str = "env") -> np.ndarray:
        if channel not in self.rewards:
            raise KeyError(f"no reward channel {channel!r}; channels: {sorted(self.rewards)}")
        return self.rewards[channel]

    def decision_reward(self, channel: str = "env", *, aggregation: Literal["sum", "last"] = "sum") -> float:
        """One number for the whole decision: the sum of its steps, or the last step's reward."""
        values = self.reward(channel)
        if aggregation == "sum":
            return float(values.sum())
        if aggregation == "last":
            return float(values[-1])
        raise ValueError(f"unknown aggregation {aggregation!r}; use 'sum' or 'last'")

    @classmethod
    def from_legacy_fields(
        cls,
        fields: Mapping[str, Any],
        *,
        kind: ActionKind,
        env_action: EnvAction | None = None,
    ) -> Transition:
        """Build a transition from the field mapping of a legacy ``buffers.base.Transition``.

        ``kind`` says what the legacy ``action`` was (the workers know: the
        chunk itself when the policy emitted no learner action, otherwise a
        latent or an edit). ``env_action`` supplies the executed commands when
        the caller still has the decision's action chunk; legacy fields of the
        latent and edit kinds do not carry them. The legacy per-step rewards
        are whatever the worker recorded, so they become the ``env`` channel;
        a shaping the worker applied in place cannot be undone here.
        """
        if kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {kind!r}; known: {ACTION_KINDS}")
        steps = int(fields["bootstrap_steps"])
        if steps < 1:
            raise ValueError("bootstrap_steps must be positive")
        sequence = fields["reward_sequence"]
        if sequence is None:
            if steps != 1:
                raise ValueError("a multi-step legacy transition needs its reward_sequence")
            sequence = [float(fields["reward"])]
        info = dict(fields["info"] or {})
        done = bool(fields["done"])
        truncated = bool(fields["truncated"])
        terminated_flags = np.zeros(steps, dtype=bool)
        truncated_flags = np.zeros(steps, dtype=bool)
        success_flags = np.zeros(steps, dtype=bool)
        terminated_flags[-1] = done and not truncated  # legacy buffers bootstrapped when both were set
        truncated_flags[-1] = truncated
        # A success flag reported mid-episode is transient; only the ending step scores.
        success_flags[-1] = episode_success(info) if (done or truncated) else False

        action = fields["action"]
        learner_action = LearnerAction(kind, action)
        if env_action is None and kind == "chunk":
            env_action = EnvAction(commands=action, execution_length=steps)

        instruction = fields["language_instruction"]
        observation = _legacy_observation(fields["obs"], instruction)
        next_observation = _legacy_observation(fields["next_obs"], instruction)

        outputs: dict[str, Any] = {
            record_key: info[info_key] for info_key, record_key in _LEGACY_OUTPUT_KEYS.items() if info_key in info
        }
        cached = fields["cached_features"]
        if cached is not None:
            outputs["features"] = cached
        policy_version = info.get("policy_version")
        reset_seed = info.get("reset_seed")
        intervention = _legacy_intervention(fields)
        metadata = TransitionMetadata(
            step_in_episode=int(fields["step_in_episode"]),
            policy_version=None if policy_version is None else int(policy_version),
            reset_seed=None if reset_seed is None else int(reset_seed),
            wall_time=fields["timestamp"],
            max_steps_in_episode=int(fields["max_steps_in_episode"]),
            intervention=intervention,
            extra=info,
        )
        return cls(
            episode_id=fields["episode_id"],
            observation=observation,
            next_observation=next_observation,
            learner_action=learner_action,
            env_action=env_action,
            rewards={"env": np.asarray(sequence, dtype=np.float32)},
            terminated=terminated_flags,
            truncated=truncated_flags,
            success=success_flags,
            behavior_source=str(fields["behavior_source"]),
            metadata=metadata,
            policy_outputs=PolicyOutputs(**outputs),
        )

    def _legacy_info(self) -> dict[str, Any]:
        """The legacy ``info`` dict: the metadata extras plus the outputs and ids legacy readers look up there."""
        info = dict(self.metadata.extra)
        for info_key, record_key in _LEGACY_OUTPUT_KEYS.items():
            value = getattr(self.policy_outputs, record_key)
            if value is not None and info_key not in info:
                info[info_key] = value
        if "policy_version" not in info and self.metadata.policy_version is not None:
            info["policy_version"] = self.metadata.policy_version
        if "reset_seed" not in info and self.metadata.reset_seed is not None:
            info["reset_seed"] = self.metadata.reset_seed
        # Legacy readers score an ended episode from its info flag, so a transition
        # built natively must carry its outcome there too.
        if self.ended and "is_success" not in info and "success" not in info:
            info["is_success"] = self.succeeded
        return info

    def to_legacy_fields(
        self, *, aggregation: Literal["sum", "last"] = "sum", reward_channel: str = "env"
    ) -> dict[str, Any]:
        """The field mapping a legacy ``buffers.base.Transition`` takes, for the buffers that still store it.

        ``aggregation`` reproduces the legacy scalar reward: the sum of the
        per-step rewards, or the last step's reward as the sparse outcome
        transform reported it.
        """
        if self.learner_action is None:
            if self.env_action is None:
                raise ValueError("a transition without a learner action needs its env_action to export")
            action: np.ndarray = self.env_action.commands
        else:
            action = self.learner_action.value
        info = self._legacy_info()
        instruction = self.observation.text().get("instruction")
        intervention = self.metadata.intervention
        fields: dict[str, Any] = {
            "obs": self.observation.arrays(),
            "action": action,
            "reward": self.decision_reward(reward_channel, aggregation=aggregation),
            "next_obs": self.next_observation.arrays(),
            "done": bool(self.terminated[-1]),
            "truncated": bool(self.truncated[-1]),
            "episode_id": self.episode_id,
            "step_in_episode": self.metadata.step_in_episode,
            "max_steps_in_episode": 0 if self.metadata.max_steps_in_episode is None else self.metadata.max_steps_in_episode,
            "timestamp": self.metadata.wall_time,
            "language_instruction": instruction,
            "info": info,
            "reward_sequence": np.asarray(self.reward(reward_channel), dtype=np.float32),
            "bootstrap_steps": self.steps,
            "behavior_source": self.behavior_source,
            "intervention_phase": "none" if intervention is None else intervention.phase,
            "policy_tick_mask": None if intervention is None else intervention.policy_tick_mask,
            "expert_tick_mask": None if intervention is None else intervention.expert_tick_mask,
            "pre_intervention_tick_mask": None if intervention is None else intervention.pre_intervention_tick_mask,
            "post_intervention_tick_mask": None if intervention is None else intervention.post_intervention_tick_mask,
            "cached_features": None if self.policy_outputs.features is None else dict(self.policy_outputs.features),
        }
        return fields


def _legacy_observation(mapping: Any, instruction: str | None) -> Observation:
    if not isinstance(mapping, Mapping):
        raise TypeError("legacy observations must be name-keyed mappings")
    observation = dict(mapping)
    if instruction is not None:
        observation["instruction"] = instruction
    return DynamicObservation.from_mapping(observation)


def _legacy_intervention(fields: Mapping[str, Any]) -> Intervention | None:
    masks = {
        name: fields[name]
        for name in (
            "policy_tick_mask",
            "expert_tick_mask",
            "pre_intervention_tick_mask",
            "post_intervention_tick_mask",
        )
    }
    if fields["intervention_phase"] == "none" and all(mask is None for mask in masks.values()):
        return None
    return Intervention(phase=fields["intervention_phase"], **masks)


@dataclass(frozen=True)
class EpisodeMetadata:
    """What is true of a whole episode: the task, its seed and initial state, who acted."""

    task: str | None = None
    instruction: str | None = None
    seed: int | None = None
    initial_state_id: int | None = None
    embodiment: str | None = None
    behavior_source: BehaviorSource | None = None
    policy_version_at_start: int | None = None

    def __post_init__(self) -> None:
        if self.behavior_source is not None and self.behavior_source not in BEHAVIOR_SOURCES:
            raise ValueError(f"unknown behavior source {self.behavior_source!r}; known: {BEHAVIOR_SOURCES}")


@dataclass(frozen=True)
class Episode:
    """The decisions of one episode in order, complete or still running."""

    id: int | str
    transitions: tuple[Transition, ...]
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError("an episode needs at least one transition")
        expected_step = transitions[0].metadata.step_in_episode
        if expected_step != 0:
            raise ValueError(f"an episode starts at step 0, got {expected_step}")
        for index, transition in enumerate(transitions):
            if transition.episode_id != self.id:
                raise ValueError(f"transition {index} belongs to episode {transition.episode_id!r}, not {self.id!r}")
            if transition.metadata.step_in_episode != expected_step:
                raise ValueError(
                    f"transition {index} starts at step {transition.metadata.step_in_episode}, expected {expected_step}"
                )
            if transition.ended and index != len(transitions) - 1:
                raise ValueError(f"transition {index} ends the episode but {len(transitions) - 1 - index} follow it")
            expected_step += transition.steps
        object.__setattr__(self, "transitions", transitions)

    @classmethod
    def from_transitions(cls, transitions: tuple[Transition, ...] | list[Transition], metadata: EpisodeMetadata | None = None) -> Episode:
        ordered = tuple(transitions)
        if not ordered:
            raise ValueError("an episode needs at least one transition")
        return cls(id=ordered[0].episode_id, transitions=ordered, metadata=metadata or EpisodeMetadata())

    @property
    def length(self) -> int:
        """Executed environment steps."""
        return sum(transition.steps for transition in self.transitions)

    @property
    def complete(self) -> bool:
        return self.transitions[-1].ended

    @property
    def outcome(self) -> Literal["ongoing", "success", "failure", "timeout"]:
        if any(transition.succeeded for transition in self.transitions):
            return "success" if self.complete else "ongoing"
        return self.transitions[-1].outcome

    def return_(self, channel: str = "env") -> float:
        """Undiscounted sum of one reward channel over the episode."""
        return float(sum(float(transition.reward(channel).sum()) for transition in self.transitions))
