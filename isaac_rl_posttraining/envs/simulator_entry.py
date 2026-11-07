# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator entries: how one simulator or robot plugs into the runtime layer.

An entry is a class a simulator (or robot) integration subclasses. Its
instance travels to every rollout worker and builds the environment there,
so the core never imports a simulator and nothing heavy crosses a process
boundary. The policy family (steering, edit, flow) comes from the algorithm,
not from the entry, so one entry serves DSRL, RFS and PPO alike.

Two tiers of authors use it. Someone bringing a robot writes a ``name`` and
``make_environment`` and gets the defaults declared on the base class: one
environment per worker, evaluation on the same workers, any frozen-policy
base model, the conventional observation keys. A simulator engineer overrides
the pieces that differ for their simulator (a batched server, a dedicated
evaluation server, task-specific camera keys, the flow family's scenes).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from isaac_rl_posttraining.data.spec import ActionSpec, ObservationSpec
    from isaac_rl_posttraining.runtimes.base import RolloutWorkerContext
    from isaac_rl_posttraining.runtimes.batched_worker import BatchedEnvironment
    from isaac_rl_posttraining.runtimes.gym_worker import GymEnvironment

__all__ = [
    "CONVENTIONAL_STEERING_KEYS",
    "EnvironmentSpec",
    "EvaluationMode",
    "Phase",
    "SimulatorEntry",
    "WorkerLayout",
    "requested_layout",
    "server_addresses",
    "steering_keys_of",
]

Phase = Literal["train", "eval"]

#: Where evaluation episodes run: on a dedicated persistent group (a second
#: simulator server), on workers rebuilt for every evaluation, or on the
#: training group itself (boundary-gated) when it hosts ``eval.eval_num_envs``
#: environments, else on a persistent group of that size.
EvaluationMode = Literal["dedicated", "fresh_workers", "shared"]
_EVALUATION_MODES: tuple[str, ...] = ("dedicated", "fresh_workers", "shared")

#: The observation keys the steering policies read when an entry declares no
#: observation spec: a ``state`` vector, one ``image`` camera, no language entry.
CONVENTIONAL_STEERING_KEYS: Mapping[str, Any] = {"state": "state", "language": None, "image": ["image"]}


def steering_keys_of(declared: ObservationSpec, *, entry: str) -> dict[str, Any]:
    """The steering key mapping a declared observation implies.

    Its image arrays are the cameras and its one other array the state (the
    one named ``state`` when there are several); no language embedding.
    Raises ``ValueError`` naming ``entry`` when the declaration pins neither
    down, so the author overrides ``steering_key_mapping`` instead.
    """
    images = list(declared.images)
    others = [name for name in declared.arrays if name not in images]
    if not images:
        raise ValueError(
            f"{entry} declares an observation without a camera entry ({declared.describe()}); the steering "
            "policies read at least one image, so override steering_key_mapping()"
        )
    if len(others) == 1:
        state = others[0]
    elif "state" in others:
        state = "state"
    else:
        raise ValueError(
            f"{entry} declares an observation whose state entry is ambiguous ({others}); override "
            "steering_key_mapping() to name the entry the policy reads"
        )
    return {"state": state, "language": None, "image": images}


def server_addresses(server_url: str | None) -> tuple[str, ...]:
    """The simulator server addresses a launcher exported as one variable: one, or several comma-separated.

    Empty when the variable is unset (an in-process simulator, or a stub environment).
    """
    if server_url is None:
        return ()
    return tuple(part.strip() for part in server_url.split(","))


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything an entry needs to build the environments of one phase.

    ``server_url`` is what the launcher exported for a remote simulator: one
    address, or several comma-separated when the phase's environments are
    spread over that many servers (:meth:`server_addresses`); ``None`` for
    in-process simulators. ``action_dim`` and ``image_resolution`` are the
    client-side geometry a simulator declares in its config group (``None``
    when it declares none; an entry that needs them rejects that).
    """

    env_name: str
    num_envs: int
    max_episode_steps: int
    seed: int | None
    server_url: str | None = None
    action_dim: int | None = None
    image_resolution: int | None = None
    use_init_states: bool = True
    fixed_init_state_id: int | None = None

    def __post_init__(self) -> None:
        if not self.env_name:
            raise ValueError("env_name must be non-empty")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if any(not address for address in server_addresses(self.server_url)):
            raise ValueError(f"server_url must be one address or a comma-separated list of them, got {self.server_url!r}")

    def server_addresses(self) -> tuple[str, ...]:
        """The exported server addresses, one per simulator server; empty for in-process simulators."""
        return server_addresses(self.server_url)

    @property
    def num_servers(self) -> int:
        """How many simulator servers host this phase's environments (one when the simulator is in-process)."""
        return max(1, len(self.server_addresses()))


@dataclass(frozen=True)
class WorkerLayout:
    """How many rollout workers host the environments, and whether one worker steps them as a batch."""

    workers: int
    batched: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")


class SimulatorEntry(ABC):
    """One simulator's (or robot's) attachment to the runtime layer; subclass it.

    Required: ``name`` and :meth:`make_environment`. Everything else has a
    default below, each chosen so that a single robot works without knowing
    about workers, batching or evaluation modes; override what differs.
    Declaring :meth:`observation_spec` and :meth:`action_spec` (from an
    ``Observation`` subclass) buys the launch-time check against the policy
    and the derived :meth:`steering_key_mapping`.

    Class attributes:
        name: Registry name of the integration (``"g1_arena"``, ``"my_robot"``).
        evaluation: Where evaluation episodes run (:data:`EvaluationMode`).
            Default ``"shared"``: one physical robot or simulation evaluates on
            the workers it trains on, at episode boundaries (when
            ``eval.eval_num_envs`` differs from the training count, on
            evaluation workers of that size instead).
        server_url_variables: For remote simulators, the environment
            variables the launcher exports with the train and eval server
            addresses; default ``None`` (in-process, or the entry reads its own
            address from ``EnvironmentSpec.server_url``).
        base_models: Base models the environments are built for;
            default ``None`` accepts any base model.
        decode_batch_size: Cap on the frozen policy's batched decode inside a
            shared policy when one policy serves many environments; default
            ``None`` (no cap).
        supports_chunked_step: The unbatched environment executes a whole
            action chunk per ``step_chunk`` call (one server round trip);
            default ``False`` (one ``step`` per action).
        sized_from_declaration: The launcher sizes the learner from
            :meth:`observation_spec` and :meth:`action_spec` instead of
            building an environment (for simulators that are slow to start);
            both must be declared with fixed sizes. Default ``False``.
        simulates_on_gpu: Each rollout worker simulates or renders on a GPU
            (Isaac Lab, LIBERO's EGL cameras), so it needs a GPU even when its
            policy is served; default ``False`` (CPU, or a simulator behind a
            server).
    """

    name: ClassVar[str]
    evaluation: ClassVar[EvaluationMode] = "shared"
    server_url_variables: ClassVar[tuple[str, str] | None] = None
    base_models: ClassVar[tuple[str, ...] | None] = None
    decode_batch_size: ClassVar[int | None] = None
    supports_chunked_step: ClassVar[bool] = False
    simulates_on_gpu: ClassVar[bool] = False
    sized_from_declaration: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _check_declaration(cls)

    def __init__(self) -> None:
        if not isinstance(getattr(type(self), "name", None), str) or not type(self).name:
            raise TypeError(f"{type(self).__qualname__} must declare a non-empty class attribute `name`")

    # -- required -----------------------------------------------------------

    @abstractmethod
    def make_environment(
        self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext
    ) -> GymEnvironment | BatchedEnvironment:
        """Build the environment(s) of one rollout worker, on that worker.

        ``spec`` describes the phase's environments (name, count, episode cap,
        seed, server address, client-side geometry) and ``context`` says which
        environment indices this worker owns. Return an unbatched environment
        when :meth:`worker_layout` is unbatched, else one batched environment
        hosting every index in ``context.environment_indices``.
        """

    # -- overridable defaults ----------------------------------------------

    def worker_layout(
        self, num_envs: int, num_servers: int = 1, *, workers: int | None = None
    ) -> WorkerLayout:
        """Worker layout for ``num_envs`` environments on ``num_servers`` simulator servers.

        Default: one environment per worker, unbatched, on one simulator, which
        is what an in-process simulator or a single robot needs. A simulator
        whose server hosts many environments behind one client returns one
        batched worker, and one whose phase may be spread over several servers
        (``spec.num_servers``) returns one batched worker per server.
        ``workers`` is the rollout-worker count the placement asks for
        (``placement.envs.workers``); an entry that can split its environments
        over several batched workers honours it, others may ignore it (the
        unbatched default already gives each environment its own worker).
        """
        if num_servers != 1:
            raise ValueError(
                f"{self.name} builds its environments on one simulator; {num_servers} server addresses were exported"
            )
        return WorkerLayout(workers=num_envs, batched=False)

    def steering_key_mapping(self, spec: EnvironmentSpec) -> Mapping[str, Any]:
        """Which observation entries hold the state, the camera images and the language embedding.

        Default: derived from :meth:`observation_spec` when the entry declares
        one (its image arrays are the cameras, its one other array the state,
        no language embedding), else the conventional names ``state`` and
        ``image``. Override it when the declaration is ambiguous, when the
        policy reads a language embedding, or when the keys depend on the task
        (``spec.env_name``). The steering step fails with the missing key and
        this method's name when an environment's observation does not carry
        the mapped entries.
        """
        declared = self.observation_spec(spec)
        if declared is None:
            return dict(CONVENTIONAL_STEERING_KEYS)
        return steering_keys_of(declared, entry=self.name)

    def make_scenes(self, spec: EnvironmentSpec, context: RolloutWorkerContext) -> BatchedEnvironment:
        """Every scene assigned to a worker as one natively batched environment.

        The flow policy family (GR00T flow-SDE PPO) plays whole episodes on a
        worker's own simulation; only simulators that host many scenes in one
        process implement this.
        """
        raise NotImplementedError(
            f"{self.name} does not host a worker's scenes in one simulation; "
            "the flow policy family needs make_scenes() and instruction() on its entry"
        )

    def instruction(self, spec: EnvironmentSpec) -> str:
        """The task's language instruction, for policies conditioned on it (with :meth:`make_scenes`)."""
        raise NotImplementedError(f"{self.name} declares no task instruction; override instruction()")

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec | None:
        """The typed observation the environments of ``spec`` emit, or ``None`` when undeclared.

        Declaring it lets the launch compare what the policy reads with what
        the simulator emits before any worker starts, and lets the default
        :meth:`steering_key_mapping` name the state, camera and language
        entries from the declaration.
        """
        del spec
        return None

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec | None:
        """The per-step action the environments of ``spec`` accept, or ``None`` when undeclared."""
        del spec
        return None

    def __repr__(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}(name={self.name!r})"


def _check_declaration(cls: type[SimulatorEntry]) -> None:
    """Reject a subclass whose declared class attributes are malformed, at class definition."""
    own = cls.__dict__
    if "name" in own and (not isinstance(own["name"], str) or not own["name"]):
        raise TypeError(f"{cls.__qualname__}.name must be a non-empty string")
    if "evaluation" in own and own["evaluation"] not in _EVALUATION_MODES:
        raise ValueError(
            f"{cls.__qualname__}.evaluation must be one of {_EVALUATION_MODES}, got {own['evaluation']!r}"
        )
    if "backends" in own:
        raise TypeError(f"{cls.__qualname__}.backends was renamed to base_models")
    base_models = own.get("base_models")
    if base_models is not None and not tuple(base_models):
        raise ValueError(f"{cls.__qualname__}.base_models must list at least one base model or be None")
    variables = own.get("server_url_variables")
    if variables is not None and len(variables) != 2:
        raise ValueError(f"{cls.__qualname__}.server_url_variables must name the train and eval variables")
    size = own.get("decode_batch_size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise ValueError(f"{cls.__qualname__}.decode_batch_size must be a positive integer or None")
    for flag in ("supports_chunked_step", "sized_from_declaration", "simulates_on_gpu"):
        if flag in own and not isinstance(own[flag], bool):
            raise TypeError(f"{cls.__qualname__}.{flag} must be a boolean")


def requested_layout(
    entry: SimulatorEntry, num_envs: int, *, num_servers: int = 1, workers: int | None = None
) -> WorkerLayout:
    """``entry.worker_layout`` with the placement's worker count, for entries that accept one.

    An entry whose ``worker_layout`` predates the ``workers`` argument decides
    its layout alone, as before.
    """
    if workers is None or "workers" not in inspect.signature(entry.worker_layout).parameters:
        return entry.worker_layout(num_envs, num_servers=num_servers)
    return entry.worker_layout(num_envs, num_servers=num_servers, workers=workers)
