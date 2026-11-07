# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent contracts for rollout execution runtimes."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from isaac_rl_posttraining.data import Transition
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.concurrency import AsyncCall
    from isaac_rl_posttraining.runtimes.learner import LearnerGroup, LearnerWorkerSpec
    from isaac_rl_posttraining.runtimes.placement import GpuRequest
    from isaac_rl_posttraining.runtimes.serving import PolicyService

__all__ = [
    "DECISION_MODES",
    "RAW_BIT_DTYPES",
    "ActorFailedError",
    "BaseMode",
    "CollectedDecision",
    "DecisionMode",
    "PolicyContext",
    "PolicyMode",
    "PolicyState",
    "RolloutGroup",
    "RolloutInfrastructureError",
    "RolloutRequest",
    "RolloutResult",
    "RolloutRuntime",
    "RolloutWorker",
    "RolloutWorkerContext",
    "RolloutWorkerSpec",
    "RuntimeFeatures",
    "WeightsRef",
    "check_policy_modes",
    "check_rollout_results",
    "check_runtime",
    "missing_capability",
    "partition_environments",
    "register_weights_resolver",
]


# Environment and policy seeds are 32-bit unsigned integers (gymnasium rejects anything
# else); every derived seed wraps at this modulus.
SEED_MODULUS = 2**32


class RolloutInfrastructureError(RuntimeError):
    """A transient worker or transport failure before a rollout completes."""


class ActorFailedError(RuntimeError):
    """A process of a multi-process runtime died, raised, or did not answer in time."""


#: What a decision is for. ``warmup`` explores before learning starts, ``train``
#: acts with the learned policy and records what learning needs, ``eval`` acts
#: with the learned (or frozen native) policy and records only outcomes.
DecisionMode = Literal["warmup", "train", "eval"]
DECISION_MODES: tuple[DecisionMode, ...] = ("warmup", "train", "eval")


def _require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def partition_environments(
    num_envs: int,
    num_rollout_workers: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign contiguous environment indices evenly across rollout workers."""
    num_envs = _require_positive_integer(num_envs, name="num_envs")
    num_rollout_workers = _require_positive_integer(
        num_rollout_workers,
        name="num_rollout_workers",
    )
    if num_rollout_workers > num_envs:
        raise ValueError(
            "num_rollout_workers cannot exceed num_envs because every worker must "
            "own at least one environment"
        )

    base, remainder = divmod(num_envs, num_rollout_workers)
    assignments: list[tuple[int, ...]] = []
    first = 0
    for worker_index in range(num_rollout_workers):
        count = base + (worker_index < remainder)
        assignments.append(tuple(range(first, first + count)))
        first += count
    return tuple(assignments)


#: ``placement.policy.mode`` as a runtime takes it: ``local`` (a copy in every
#: rollout worker), ``served`` (policy servers) or ``auto`` (the runtime decides).
PolicyMode = Literal["auto", "local", "served"]
#: ``placement.parts.base.mode`` as a runtime takes it: ``served`` serves a
#: steering or edit policy's base model alone while every worker runs its actor.
BaseMode = Literal["local", "served"]


def check_policy_modes(policy_mode: object, base_mode: object) -> None:
    """Refuse modes outside the placement words, and serving the whole policy and its base model alone at once."""
    if policy_mode not in ("auto", "local", "served"):
        raise ValueError("policy_mode must be 'auto', 'local', or 'served'")
    if base_mode not in ("local", "served"):
        raise ValueError("base_mode must be 'local' or 'served'")
    if policy_mode == "served" and base_mode == "served":
        raise ValueError(
            "serve either the whole policy (placement.policy.mode=served) or its base model "
            "alone (placement.parts.base.mode=served), not both"
        )


@dataclass(frozen=True)
class RuntimeFeatures:
    """What a runtime can do, as the placement planner and the runtime openers check it.

    Only flags something checks: ``policy_service`` hosts served policies;
    ``served_batched_collectors`` lets batched rollout workers call a served policy;
    ``distributed_learner`` runs FSDP learner ranks; ``async_collect`` runs a
    collection while the caller trains; ``weights_by_reference`` publishes
    weights once and lets receivers fetch them; ``serving_engines`` names the
    engines a policy server can run; ``served_parts`` serves a policy's base
    model alone while the actor runs in the rollout workers.
    ``standalone_policy_service`` is ``RolloutRuntime.open_policy_service``
    (a service opened apart from any rollout group).
    """

    name: str
    multi_node: bool
    policy_service: bool
    served_batched_collectors: bool
    distributed_learner: bool
    async_collect: bool
    weights_by_reference: bool
    serving_engines: tuple[str, ...] = ()
    served_parts: bool = False
    standalone_policy_service: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("runtime features need the runtime name")
        object.__setattr__(self, "serving_engines", tuple(self.serving_engines))
        if self.serving_engines and not self.policy_service:
            raise ValueError("serving engines need policy_service=True")
        if self.served_parts and not self.policy_service:
            raise ValueError("served policy parts need policy_service=True")
        if self.standalone_policy_service and not self.policy_service:
            raise ValueError("a standalone policy service needs policy_service=True")


@dataclass(frozen=True)
class RolloutWorkerContext:
    """Provider-neutral identity and environment ownership for one worker."""

    worker_index: int
    num_rollout_workers: int
    environment_indices: tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.worker_index, bool)
            or not isinstance(self.worker_index, int)
            or self.worker_index < 0
        ):
            raise ValueError("worker_index must be a non-negative integer")
        _require_positive_integer(
            self.num_rollout_workers,
            name="num_rollout_workers",
        )
        if self.worker_index >= self.num_rollout_workers:
            raise ValueError(
                "worker_index must be less than num_rollout_workers, got "
                f"{self.worker_index}/{self.num_rollout_workers}"
            )
        if not self.environment_indices:
            raise ValueError("every rollout worker must own at least one environment")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.environment_indices
        ):
            raise ValueError("environment_indices must contain non-negative integers")
        if len(set(self.environment_indices)) != len(self.environment_indices):
            raise ValueError("environment_indices must be unique within a worker")
        if not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True)
class PolicyContext:
    """Placement supplied when constructing one acting-policy instance."""

    policy_index: int
    num_policies: int
    device: str
    placement: Literal["rollout_worker", "policy_server"]

    def __post_init__(self) -> None:
        if (
            isinstance(self.policy_index, bool)
            or not isinstance(self.policy_index, int)
            or self.policy_index < 0
        ):
            raise ValueError("policy_index must be a non-negative integer")
        _require_positive_integer(self.num_policies, name="num_policies")
        if self.policy_index >= self.num_policies:
            raise ValueError(
                "policy_index must be less than num_policies, got "
                f"{self.policy_index}/{self.num_policies}"
            )
        if not self.device:
            raise ValueError("device must be non-empty")


#: Torch dtypes without a NumPy equivalent, and the NumPy dtype that carries their raw bits.
RAW_BIT_DTYPES: dict[str, type] = {"bfloat16": np.int16}


@dataclass(frozen=True)
class PolicyState:
    """Versioned, transport-safe actor parameters sent to rollout workers.

    ``dtypes`` names the original dtype of tensors whose array only carries raw
    bits: NumPy has no bfloat16, so a bfloat16 tensor travels as ``int16`` under
    ``dtypes[name] == "bfloat16"`` at half the size of a float32 copy.
    :mod:`isaac_rl_posttraining.runtimes.policy_snapshot` converts to and from
    torch tensors and saves or loads snapshots.
    """

    version: int
    tensors: Mapping[str, np.ndarray]
    dtypes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("policy-state version must be an integer")
        if self.version < 0:
            raise ValueError("policy-state version must be non-negative")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("policy-state tensor names must be non-empty strings")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(
                    f"policy-state tensor {name!r} must not use object dtype"
                )
            normalized[name] = np.ascontiguousarray(array).copy()
        object.__setattr__(self, "tensors", normalized)
        object.__setattr__(self, "dtypes", _raw_bit_dtypes(self.dtypes, normalized))


def _raw_bit_dtypes(
    dtypes: Mapping[str, str], tensors: Mapping[str, np.ndarray]
) -> dict[str, str]:
    """Check that every declared dtype names a known tensor stored in its raw-bit carrier."""
    checked: dict[str, str] = {}
    for name, dtype_name in dtypes.items():
        if name not in tensors:
            raise ValueError(f"policy-state dtype given for unknown tensor {name!r}")
        if dtype_name not in RAW_BIT_DTYPES:
            raise ValueError(
                f"policy-state dtype {dtype_name!r} for {name!r} is not a raw-bit "
                f"carrier (supported: {sorted(RAW_BIT_DTYPES)})"
            )
        if tensors[name].dtype != RAW_BIT_DTYPES[dtype_name]:
            raise TypeError(
                f"policy-state tensor {name!r} declared {dtype_name} must be stored "
                f"as {RAW_BIT_DTYPES[dtype_name].__name__}, got {tensors[name].dtype}"
            )
        checked[name] = dtype_name
    return checked


#: Resolvers that turn a :class:`WeightsRef` handle into weights, by handle type.
_WEIGHTS_RESOLVERS: dict[type, Callable[[Any], PolicyState]] = {}


def register_weights_resolver(
    handle_type: type, resolver: Callable[[Any], PolicyState]
) -> None:
    """Let :meth:`WeightsRef.resolve` fetch handles of ``handle_type`` (and subclasses)."""
    _WEIGHTS_RESOLVERS[handle_type] = resolver


@dataclass(frozen=True)
class WeightsRef:
    """A published weight version: inline weights, a handle to fetch them, or the version alone.

    ``inline`` carries the :class:`PolicyState` itself (in-process runs);
    ``handle`` points to weights published once (for example a Ray object
    reference) that each receiver resolves; neither set means the receiver
    only needs the version (a rollout worker whose policy is served elsewhere).
    ``parts`` names the policy parts the weights cover.
    """

    version: int
    inline: PolicyState | None = None
    handle: Any | None = None
    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("weights version must be an integer")
        if self.version < 0:
            raise ValueError("weights version must be non-negative")
        if self.inline is not None and self.handle is not None:
            raise ValueError("weights are either inline or behind a handle, not both")
        if self.inline is not None:
            if not isinstance(self.inline, PolicyState):
                raise TypeError("inline weights must be a PolicyState")
            if self.inline.version != self.version:
                raise ValueError(
                    f"inline weights carry version {self.inline.version}, "
                    f"the reference says {self.version}"
                )
        object.__setattr__(self, "parts", tuple(self.parts))
        if not all(isinstance(part, str) and part for part in self.parts):
            raise ValueError("weights parts must be non-empty strings")

    @classmethod
    def of(cls, state: PolicyState, *, parts: Sequence[str] = ()) -> WeightsRef:
        """An inline reference to ``state`` at its own version."""
        return cls(version=state.version, inline=state, parts=tuple(parts))

    def resolve(self) -> PolicyState:
        """The weights: inline ones, or the handle fetched by its registered resolver."""
        if self.inline is not None:
            return self.inline
        if self.handle is None:
            raise ValueError(
                f"weights version {self.version} carries no weights (version-only reference)"
            )
        for handle_type in type(self.handle).__mro__:
            resolver = _WEIGHTS_RESOLVERS.get(handle_type)
            if resolver is not None:
                state = resolver(self.handle)
                if state.version != self.version:
                    raise ValueError(
                        f"resolved weights carry version {state.version}, "
                        f"the reference says {self.version}"
                    )
                return state
        raise TypeError(
            f"no weights resolver registered for handle type "
            f"{type(self.handle).__name__}; register one with register_weights_resolver"
        )


@dataclass(frozen=True)
class RolloutRequest:
    """A bounded collection call with a per-environment budget.

    ``capture_frames`` asks the worker to return the image observations of the
    collected episodes for video recording; ``max_decisions_per_episode``
    truncates an episode that has not ended after that many decisions, so a
    non-terminating simulator cannot stall an evaluation.
    """

    decisions_per_environment: int | None = None
    episodes_per_environment: int | None = None
    mode: DecisionMode = "train"
    reset_seed: int | None = None
    capture_frames: bool = False
    max_decisions_per_episode: int | None = None

    def __post_init__(self) -> None:
        budgets = (self.decisions_per_environment, self.episodes_per_environment)
        if sum(value is not None for value in budgets) != 1:
            raise ValueError(
                "exactly one of decisions_per_environment or "
                "episodes_per_environment must be set"
            )
        value = (
            self.decisions_per_environment
            if self.decisions_per_environment is not None
            else self.episodes_per_environment
        )
        _require_positive_integer(value, name="rollout budget")
        if self.mode not in DECISION_MODES:
            raise ValueError(f"mode must be one of {DECISION_MODES}, got {self.mode!r}")
        if self.reset_seed is not None:
            if self.episodes_per_environment is None:
                raise ValueError("reset_seed requires an episode-bounded request")
            if (
                isinstance(self.reset_seed, bool)
                or not isinstance(self.reset_seed, int)
                or self.reset_seed < 0
            ):
                raise ValueError("reset_seed must be a non-negative integer")
        if not isinstance(self.capture_frames, bool):
            raise TypeError("capture_frames must be a boolean")
        if self.max_decisions_per_episode is not None:
            _require_positive_integer(
                self.max_decisions_per_episode, name="max_decisions_per_episode"
            )

    @property
    def can_train(self) -> bool:
        """Whether the learned policy acts (every mode but warm-up)."""
        return self.mode != "warmup"


@dataclass(frozen=True)
class CollectedDecision:
    """A replay transition and action provenance from one environment."""

    environment_index: int
    transition: Transition
    action_chunk: ActionChunk

    def __post_init__(self) -> None:
        if (
            isinstance(self.environment_index, bool)
            or not isinstance(self.environment_index, int)
            or self.environment_index < 0
        ):
            raise ValueError("environment_index must be a non-negative integer")


@dataclass(frozen=True)
class RolloutResult:
    """Serializable result of one worker's bounded collection call."""

    worker_index: int
    decisions: tuple[CollectedDecision, ...]
    env_steps: int
    episodes_completed: int
    policy_version: int
    timings: Mapping[str, float] = field(default_factory=dict)
    #: Image observations captured per decision (plus the terminal one) when the
    #: request asked for frames; each entry maps an image key to one frame.
    #: ``None`` when no frames were requested: RLInf's transport treats an empty
    #: tuple as a (zero-length) tensor field and rebuilds the dataclass with that
    #: field set to ``None`` while the tensors travel separately.
    frames: tuple[Mapping[str, np.ndarray], ...] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "worker_index",
            "env_steps",
            "episodes_completed",
            "policy_version",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        _validate_frames(self.frames or ())
        realized_steps = sum(decision.transition.steps for decision in self.decisions)
        if realized_steps != self.env_steps:
            raise ValueError(
                f"env_steps={self.env_steps} disagrees with the decisions' executed steps={realized_steps}"
            )
        # An empty mapping rides RLInf's channel as a tensor field: the skeleton in
        # transit carries None here (as ``frames`` does); the receiver restores it.
        if self.timings is not None:
            _validate_timings(self.timings)
        for decision in self.decisions:
            if decision.action_chunk.policy_version != self.policy_version:
                raise ValueError(
                    "every action chunk in a result must match the result "
                    "policy_version"
                )


def _validate_frames(frames: Sequence[Mapping[str, np.ndarray]]) -> None:
    for frame in frames:
        if not isinstance(frame, Mapping) or not frame:
            raise ValueError("captured frames must be non-empty mappings of images")
        for key, image in frame.items():
            if not isinstance(key, str) or not isinstance(image, np.ndarray):
                raise TypeError("captured frames map string keys to numpy images")


def _validate_timings(timings: Mapping[str, float]) -> None:
    for name, seconds in timings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("rollout timing names must be non-empty strings")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"rollout timing {name!r} must be finite and non-negative")


def check_rollout_results(
    results: Sequence[RolloutResult], assignments: Sequence[Sequence[int]]
) -> None:
    """Each result comes from its worker, in worker order, and covers only that worker's environments."""
    if len(results) != len(assignments):
        raise RuntimeError(
            f"{len(results)} rollout results came back for {len(assignments)} rollout workers"
        )
    for expected_worker_index, (result, environment_indices) in enumerate(
        zip(results, assignments, strict=True)
    ):
        if result.worker_index != expected_worker_index:
            raise RuntimeError(
                f"worker {expected_worker_index} returned result for worker "
                f"{result.worker_index}"
            )
        owned = set(environment_indices)
        unexpected = {
            decision.environment_index
            for decision in result.decisions
            if decision.environment_index not in owned
        }
        if unexpected:
            raise RuntimeError(
                f"worker {expected_worker_index} returned unowned environment "
                f"indices {sorted(unexpected)}"
            )


@runtime_checkable
class RolloutWorker(Protocol):
    """Persistent simulator environment(s) plus an acting policy."""

    def collect(self, request: RolloutRequest) -> RolloutResult: ...

    def set_policy_state(self, state: PolicyState) -> int: ...

    def close(self) -> None: ...


#: A rollout worker spec is a picklable recipe a runtime ships to each worker
#: and calls with the worker's context; the ``*WorkerSpec`` dataclasses of this
#: package carry only data (the simulator entry, the environment description,
#: the policy spec) and build the worker there.
RolloutWorkerSpec = Callable[[RolloutWorkerContext], RolloutWorker]


@runtime_checkable
class RolloutGroup(Protocol):
    """Provider-independent proxy for persistent rollout workers.

    ``collect_async`` starts a collection and returns at once;
    ``set_weights`` sends published weights by reference (``set_policy_state``
    sends a resolved state). Groups without a native version of either
    inherit them from :class:`~isaac_rl_posttraining.runtimes.concurrency.RolloutGroupBase`.
    """

    num_rollout_workers: int
    num_envs: int

    def collect(
        self,
        requests: Sequence[RolloutRequest],
    ) -> tuple[RolloutResult, ...]: ...

    def collect_async(
        self, requests: Sequence[RolloutRequest]
    ) -> AsyncCall[tuple[RolloutResult, ...]]: ...

    def set_weights(self, weights: WeightsRef) -> tuple[int, ...]: ...

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class RolloutRuntime(Protocol):
    """Execution provider selected independently of simulator and algorithm.

    Every runtime implements every method; one without a capability raises
    :class:`NotImplementedError` from it and says so in ``features``
    (``standalone_policy_service`` for :meth:`open_policy_service`,
    ``distributed_learner`` for :meth:`open_learner_group`).
    """

    name: str
    features: RuntimeFeatures

    def open_rollout_group(
        self,
        spec: RolloutWorkerSpec,
        *,
        num_rollout_workers: int,
        num_envs: int,
        name: str,
    ) -> RolloutGroup: ...

    def open_policy_service(
        self,
        spec: Callable[[PolicyContext], Any],
        *,
        name: str,
        replicas: int | None = None,
        gpus: GpuRequest = None,
        max_batch: int | None = None,
        max_wait_ms: float | None = None,
        engine: str = "builtin",
    ) -> PolicyService:
        """A policy service of its own, apart from any rollout group, planned on the GPUs this runtime sees.

        ``gpus`` lists GPU indices, one per replica (``replicas`` defaults to
        their count, else one). Act through ``service.client(i)``, or pass
        ``service.handle()`` as a worker spec's ``policy``.
        """
        ...

    def open_learner_group(
        self, spec: LearnerWorkerSpec, *, num_learner_ranks: int, name: str
    ) -> LearnerGroup:
        """Learner ranks (FSDP) for ``spec``."""
        ...

    def close(self) -> None: ...


def check_runtime(runtime: Any) -> RolloutRuntime:
    """``runtime`` if it implements every :class:`RolloutRuntime` member, else a ``TypeError`` naming the missing ones."""
    if isinstance(runtime, RolloutRuntime):
        return runtime
    members = [
        *RolloutRuntime.__annotations__,
        *(n for n in vars(RolloutRuntime) if not n.startswith("_")),
    ]
    missing = sorted({name for name in members if not hasattr(runtime, name)})
    raise TypeError(
        f"{type(runtime).__name__} is not a complete RolloutRuntime: it lacks {', '.join(missing)} "
        "(a runtime without a capability raises NotImplementedError from the method instead)"
    )


def missing_capability(runtime: str, what: str, feature: str) -> NotImplementedError:
    """The error of a runtime method whose capability the runtime does not have."""
    return NotImplementedError(
        f"runtime {runtime!r} cannot {what} (its features declare {feature}=False); "
        "choose a runtime that can (runtime=native)"
    )
