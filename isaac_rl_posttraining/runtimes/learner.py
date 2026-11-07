# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral contracts for optionally distributed learner updates."""

from __future__ import annotations

import copy
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    RolloutRuntime,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.gradient_averaging import (
    average_gradients_across_ranks,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

__all__ = [
    "LEARNER_CHECKPOINT_FILE",
    "LEARNER_HOOKS",
    "GroupLearnerHooks",
    "HookDispatcher",
    "LearnerBatch",
    "LearnerContext",
    "LearnerGroup",
    "LearnerHooks",
    "LearnerModuleManager",
    "LearnerStepResult",
    "LearnerWorker",
    "LearnerWorkerSpec",
    "LocalLearnerGroup",
    "LocalLearnerHooks",
    "average_gradients_across_ranks",
    "check_only_frozen_missing",
    "frozen_parameter_prefixes",
    "open_distributed_learner_group",
    "open_learner_group",
    "publish_weights",
    "run_hook",
    "without_prefixes",
]

#: File name of the local learner group's checkpoint payload.
LEARNER_CHECKPOINT_FILE = "learner.pt"


@dataclass(frozen=True)
class LearnerContext:
    """Identity of one learner process in a PyTorch process group.

    `learner_rank` is the stable public identity used by RLPT.
    `world_size` retains PyTorch's standard name for the size of this learner
    process group and never means the number of simulator environments.
    """

    learner_rank: int
    world_size: int
    device: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.learner_rank, bool)
            or not isinstance(self.learner_rank, int)
            or self.learner_rank < 0
        ):
            raise ValueError("learner_rank must be a non-negative integer")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("world_size must be a positive integer")
        if self.world_size < 1 or self.learner_rank >= self.world_size:
            raise ValueError(
                "learner_rank/world_size must satisfy 0 <= learner_rank < "
                f"world_size, got {self.learner_rank}/{self.world_size}"
            )
        if not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True)
class LearnerBatch:
    """Transport-safe, rank-local data consumed by an algorithm worker.

    ``env_steps`` is how many environment steps had been collected when the
    batch was drawn (``None`` when the controller does not say); algorithms
    with step-scheduled settings read it.
    """

    tensors: Mapping[str, np.ndarray | Mapping[str, np.ndarray]]
    samples: int
    env_steps: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("learner batch samples must be a positive integer")
        if self.samples < 1:
            raise ValueError("learner batch samples must be a positive integer")
        if self.env_steps is not None and (
            isinstance(self.env_steps, bool)
            or not isinstance(self.env_steps, int)
            or self.env_steps < 0
        ):
            raise ValueError(
                "learner batch env_steps must be a non-negative integer or None"
            )
        if not self.tensors:
            raise ValueError("learner batch tensors must be non-empty")
        normalized: dict[str, np.ndarray | Mapping[str, np.ndarray]] = {}
        for name, value in self.tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("learner batch names must be non-empty strings")
            if isinstance(value, Mapping):
                normalized[name] = _nested_batch_arrays(name, value)
            else:
                normalized[name] = _batch_array(value, label=repr(name))
        object.__setattr__(self, "tensors", normalized)

    def at_env_steps(self, env_steps: int) -> LearnerBatch:
        """This batch, stamped with the environment steps collected when it was drawn (no copy)."""
        stamped = copy.copy(self)
        object.__setattr__(stamped, "env_steps", env_steps)
        if (
            isinstance(env_steps, bool)
            or not isinstance(env_steps, int)
            or env_steps < 0
        ):
            raise ValueError("learner batch env_steps must be a non-negative integer")
        return stamped

    def split(self, num_ranks: int) -> tuple[LearnerBatch, ...]:
        """Cut a sample-major batch into ``num_ranks`` equal rank-local batches.

        Every array is sliced along its first (sample) axis, so rank ``r``
        trains on the ``r``-th slice of the same draw; each array's leading
        size and ``samples`` must divide evenly.
        """
        if (
            isinstance(num_ranks, bool)
            or not isinstance(num_ranks, int)
            or num_ranks < 1
        ):
            raise ValueError("num_ranks must be a positive integer")
        if num_ranks == 1:
            return (self,)
        if self.samples % num_ranks:
            raise ValueError(
                f"a learner batch of {self.samples} samples does not split evenly "
                f"over {num_ranks} learner ranks"
            )
        parts: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = [
            {} for _ in range(num_ranks)
        ]
        for name, value in self.tensors.items():
            if isinstance(value, Mapping):
                for part in parts:
                    part[name] = {}
                for nested_name, array in value.items():
                    for rank, piece in enumerate(
                        _rank_slices(array, num_ranks, label=f"{name}.{nested_name}")
                    ):
                        nested = _nested_part(parts[rank], name)
                        nested[nested_name] = piece
            else:
                for rank, piece in enumerate(
                    _rank_slices(value, num_ranks, label=name)
                ):
                    parts[rank][name] = piece
        return tuple(
            LearnerBatch(
                tensors=part,
                samples=self.samples // num_ranks,
                env_steps=self.env_steps,
            )
            for part in parts
        )


def _nested_part(
    part: dict[str, np.ndarray | dict[str, np.ndarray]], name: str
) -> dict[str, np.ndarray]:
    nested = part[name]
    assert isinstance(nested, dict)
    return nested


def _rank_slices(array: np.ndarray, num_ranks: int, *, label: str) -> list[np.ndarray]:
    if array.ndim == 0 or array.shape[0] % num_ranks:
        raise ValueError(
            f"learner batch tensor {label!r} with shape {array.shape} does not split "
            f"evenly over {num_ranks} learner ranks along its sample axis"
        )
    return list(np.split(array, num_ranks, axis=0))


def _batch_array(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"learner batch tensor {label} uses object dtype")
    return np.ascontiguousarray(array).copy()


def _nested_batch_arrays(
    name: str, value: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not value:
        raise ValueError(f"learner batch mapping {name!r} must be non-empty")
    nested: dict[str, np.ndarray] = {}
    for nested_name, nested_value in value.items():
        if not isinstance(nested_name, str) or not nested_name:
            raise ValueError("nested learner batch names must be non-empty strings")
        nested[nested_name] = _batch_array(nested_value, label=f"{name}.{nested_name}")
    return nested


@dataclass(frozen=True)
class LearnerStepResult:
    """Metrics and ordering metadata returned by one learner rank."""

    learner_rank: int
    step: int
    samples: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        for field_name in ("learner_rank", "step"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("samples must be a positive integer")
        if self.samples < 1:
            raise ValueError("samples must be a positive integer")
        normalized: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("learner metric names must be non-empty strings")
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"learner metric {name!r} must be finite")
            normalized[name] = scalar
        object.__setattr__(self, "metrics", normalized)


@runtime_checkable
class LearnerModuleManager(Protocol):
    """Prepare trainable modules without exposing a provider sharding API."""

    def prepare(
        self,
        module: nn.Module,
        *,
        name: str,
        publish: bool = False,
        frozen_submodules: Sequence[str] = (),
        shard: bool = True,
    ) -> nn.Module:
        """Place ``module`` for training and return the object to run.

        ``frozen_submodules`` are dotted paths of submodules that never train
        (for example a vision-language backbone). The runtime keeps them whole
        on every rank instead of sharding them, and leaves them out of the
        published policy state and of checkpoints, which then carry only what
        training can change. Every parameter under such a path must have
        ``requires_grad == False``.

        ``shard=False`` keeps the module whole on every rank and averages its
        gradients across ranks (data parallel without sharding) and returns
        the module itself; for small networks whose code calls methods other
        than ``forward``.
        """
        ...

    def register_optimizer(
        self,
        name: str,
        optimizer: Optimizer,
        *,
        module: nn.Module,
    ) -> None: ...


@runtime_checkable
class LearnerWorker(Protocol):
    """Algorithm-owned update logic hosted by a distributed runtime."""

    def train_step(self, batch: LearnerBatch) -> Mapping[str, float]: ...

    def checkpoint_state(self) -> Mapping[str, np.ndarray]: ...

    def load_checkpoint_state(self, state: Mapping[str, np.ndarray]) -> None: ...

    def close(self) -> None: ...


LearnerWorkerSpec = Callable[
    [LearnerContext, LearnerModuleManager],
    LearnerWorker,
]


@runtime_checkable
class LearnerGroup(Protocol):
    """Collective-safe proxy for a data-parallel learner process group.

    ``publish(version)`` exports the published module once (rank 0) and
    returns a :class:`WeightsRef` the rollout side resolves; ``hooks()`` is the
    typed channel for the algorithm's queries and notifications between updates.
    """

    num_learner_ranks: int

    def train_step(
        self,
        batches: Sequence[LearnerBatch],
    ) -> tuple[LearnerStepResult, ...]: ...

    def policy_state(self, *, version: int) -> PolicyState: ...

    def publish(self, version: int) -> WeightsRef: ...

    def hooks(self) -> LearnerHooks: ...

    def save_checkpoint(self, directory: str) -> None: ...

    def load_checkpoint(self, directory: str) -> None: ...

    def close(self) -> None: ...


#: Hook names a learner worker may implement; :func:`run_hook` falls back to
#: the defaults below for the optional ones a worker leaves out.
LEARNER_HOOKS = (
    "gradient_steps",
    "target_update_count",
    "observe_rewards",
    "on_evaluation",
)


def run_hook(worker: object, name: str, *args: Any) -> Any:
    """Run hook ``name`` on a learner worker (the rank-side half of :class:`LearnerHooks`).

    ``target_update_count`` defaults to ``None``, ``observe_rewards`` to a
    no-op and ``on_evaluation`` to no metrics; ``gradient_steps`` is required.
    """
    if name not in LEARNER_HOOKS:
        raise ValueError(f"unknown learner hook {name!r}; known: {LEARNER_HOOKS}")
    method = getattr(worker, name, None)
    if callable(method):
        return method(*args)
    if name == "gradient_steps":
        raise TypeError(
            f"learner worker {type(worker).__name__} does not report gradient_steps()"
        )
    return {} if name == "on_evaluation" else None


@runtime_checkable
class LearnerHooks(Protocol):
    """Algorithm queries and notifications the trainer sends between updates.

    Queries are answered by learner rank 0; notifications reach every rank.
    """

    def gradient_steps(self) -> int:
        """Updates the algorithm has run so far."""
        ...

    def target_update_count(self, env_steps: int) -> int | None:
        """The cumulative update count the algorithm wants at ``env_steps``, or ``None``."""
        ...

    def observe_rewards(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        truncateds: Sequence[bool],
        episode_ids: Sequence[Hashable],
    ) -> None:
        """The decisions just collected, in buffer order, before any batch is drawn."""
        ...

    def on_evaluation(self, success_rate: float | None) -> Mapping[str, float]:
        """An evaluation finished; returns metrics to log under ``eval``."""
        ...


def open_distributed_learner_group(
    runtime: RolloutRuntime,
    spec: LearnerWorkerSpec,
    *,
    num_learner_ranks: int,
    name: str,
) -> LearnerGroup:
    """The runtime's learner group of ``num_learner_ranks`` ranks, if its features declare ``distributed_learner``."""
    if not runtime.features.distributed_learner:
        raise ValueError(
            f"runtime {runtime.name!r} runs one learner rank in this process; ask the placement "
            "for several ranks (runtime=native runs them as Ray actors) or select runtime=rlinf"
        )
    return runtime.open_learner_group(
        spec,
        num_learner_ranks=num_learner_ranks,
        name=name,
    )


def frozen_parameter_prefixes(
    module: nn.Module,
    frozen_submodules: Sequence[str],
) -> tuple[str, ...]:
    """Resolve frozen submodule paths to state-dict key prefixes, checking they are frozen."""
    prefixes: list[str] = []
    for path in frozen_submodules:
        submodule = module.get_submodule(path)
        trainable = [
            parameter_name
            for parameter_name, parameter in submodule.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise ValueError(
                f"frozen submodule {path!r} has trainable parameters (e.g. "
                f"{trainable[0]!r}); only frozen submodules may be excluded from "
                "sharding, publishing, and checkpoints"
            )
        prefixes.append(f"{path}.")
    return tuple(prefixes)


def without_prefixes(
    state: Mapping[str, Any],
    prefixes: Sequence[str],
) -> dict[str, Any]:
    """A state dict without the entries under ``prefixes``."""
    if not prefixes:
        return dict(state)
    return {
        name: value
        for name, value in state.items()
        if not name.startswith(tuple(prefixes))
    }


def check_only_frozen_missing(
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    prefixes: Sequence[str],
) -> None:
    """A checkpoint saved without frozen submodules must miss only their keys."""
    if unexpected_keys:
        raise ValueError(
            f"learner checkpoint carries unexpected keys: {list(unexpected_keys)[:5]}"
        )
    stray = [key for key in missing_keys if not key.startswith(tuple(prefixes))]
    if stray:
        raise ValueError(
            "learner checkpoint is missing keys outside the frozen submodules "
            f"{list(prefixes)}: {stray[:5]}"
        )


class _LocalModuleManager:
    """Prepare and checkpoint modules for an in-process learner."""

    def __init__(self, device: str) -> None:
        import torch

        self.device = torch.device(device)
        self.modules: dict[str, nn.Module] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.frozen_prefixes: dict[str, tuple[str, ...]] = {}
        self.published: str | None = None

    def prepare(
        self,
        module: nn.Module,
        *,
        name: str,
        publish: bool = False,
        frozen_submodules: Sequence[str] = (),
        shard: bool = True,
    ) -> nn.Module:
        del shard  # one rank: nothing to shard or average
        if name in self.modules:
            raise ValueError(f"module {name!r} is already registered")
        prefixes = frozen_parameter_prefixes(module, frozen_submodules)
        prepared = module.to(self.device)
        self.modules[name] = prepared
        self.frozen_prefixes[name] = prefixes
        if publish:
            if self.published is not None:
                raise ValueError("only one local module may publish policy state")
            self.published = name
        return prepared

    def register_optimizer(
        self,
        name: str,
        optimizer: Optimizer,
        *,
        module: nn.Module,
    ) -> None:
        if name in self.optimizers:
            raise ValueError(f"optimizer {name!r} is already registered")
        if module not in self.modules.values():
            raise ValueError("optimizer module was not prepared by this manager")
        self.optimizers[name] = optimizer


class LocalLearnerGroup:
    """One in-process learner implementing the common learner-group contract."""

    num_learner_ranks = 1

    def __init__(self, spec: LearnerWorkerSpec, *, device: str) -> None:
        import torch

        self._torch = torch
        self._modules = _LocalModuleManager(device)
        self._worker = spec(
            LearnerContext(learner_rank=0, world_size=1, device=device),
            self._modules,
        )
        if not isinstance(self._worker, LearnerWorker):
            raise TypeError("learner spec returned an incompatible worker")
        if self._modules.published is None:
            raise ValueError("local learner did not register a published module")
        self._step = 0
        self._policy_version = -1
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("local learner group is closed")

    def train_step(
        self,
        batches: Sequence[LearnerBatch],
    ) -> tuple[LearnerStepResult, ...]:
        self._require_open()
        if len(batches) != 1:
            raise ValueError("local learner requires exactly one rank-local batch")
        batch = batches[0]
        metrics = self._worker.train_step(batch)
        self._step += 1
        return (
            LearnerStepResult(
                learner_rank=0,
                step=self._step,
                samples=batch.samples,
                metrics=metrics,
            ),
        )

    def hooks(self) -> LocalLearnerHooks:
        """The typed algorithm hooks of this learner."""
        return LocalLearnerHooks(self)

    def _run_hook(self, name: str, *args: Any) -> Any:
        self._require_open()
        return run_hook(self._worker, name, *args)

    def publish(self, version: int) -> WeightsRef:
        """Export the published module at ``version``, inline (one process)."""
        return WeightsRef.of(self.policy_state(version=version))

    def policy_state(self, *, version: int) -> PolicyState:
        self._require_open()
        if version <= self._policy_version:
            raise ValueError(
                f"policy-state version must advance beyond {self._policy_version}"
            )
        published = self._modules.published
        assert published is not None
        module = self._modules.modules[published]
        state = policy_state_from_tensors(
            module.state_dict(),
            version=version,
            exclude_prefixes=self._modules.frozen_prefixes[published],
        )
        self._policy_version = version
        return state

    def save_checkpoint(self, directory: str) -> None:
        self._require_open()
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError("local learner checkpoint directory must be absolute")
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self._step,
            "modules": {
                name: without_prefixes(
                    module.state_dict(), self._modules.frozen_prefixes[name]
                )
                for name, module in self._modules.modules.items()
            },
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self._modules.optimizers.items()
            },
            "algorithm": dict(self._worker.checkpoint_state()),
            "torch_rng": self._torch.get_rng_state(),
        }
        temporary = path / f"{LEARNER_CHECKPOINT_FILE}.tmp"
        self._torch.save(payload, temporary)
        temporary.replace(path / LEARNER_CHECKPOINT_FILE)

    def load_checkpoint(self, directory: str, *, restore_rng: bool = True) -> None:
        """Restore a checkpoint written by :meth:`save_checkpoint`.

        ``restore_rng=False`` loads the learner state but leaves the process's
        random stream alone, for a warm start whose seed must stay the run's own.
        """
        self._require_open()
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError("local learner checkpoint directory must be absolute")
        payload = self._torch.load(
            path / LEARNER_CHECKPOINT_FILE,
            map_location=self._modules.device,
            weights_only=False,
        )
        for name, module in self._modules.modules.items():
            prefixes = self._modules.frozen_prefixes[name]
            result = module.load_state_dict(
                payload["modules"][name], strict=not prefixes
            )
            if prefixes:
                check_only_frozen_missing(
                    result.missing_keys, result.unexpected_keys, prefixes
                )
        for name, optimizer in self._modules.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        self._worker.load_checkpoint_state(payload["algorithm"])
        if restore_rng:
            self._torch.set_rng_state(payload["torch_rng"].cpu())
        step = payload["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("local learner checkpoint step must be non-negative")
        self._step = step

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.close()


class HookDispatcher(Protocol):
    """The group side of :class:`LearnerHooks`: query rank 0, or notify every rank."""

    def query(self, name: str, *args: Any) -> Any: ...

    def notify(self, name: str, *args: Any) -> Any: ...


class GroupLearnerHooks:
    """:class:`LearnerHooks` over a group's dispatcher.

    Queries (``gradient_steps``, ``target_update_count``) are answered by
    rank 0; ``observe_rewards`` and ``on_evaluation`` reach every rank, and
    ``on_evaluation`` returns rank 0's metrics.
    """

    def __init__(self, dispatcher: HookDispatcher) -> None:
        self._dispatcher = dispatcher

    def gradient_steps(self) -> int:
        return int(self._dispatcher.query("gradient_steps"))

    def target_update_count(self, env_steps: int) -> int | None:
        target = self._dispatcher.query("target_update_count", env_steps)
        return None if target is None else int(target)

    def observe_rewards(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        truncateds: Sequence[bool],
        episode_ids: Sequence[Hashable],
    ) -> None:
        self._dispatcher.notify(
            "observe_rewards", rewards, dones, truncateds, episode_ids
        )

    def on_evaluation(self, success_rate: float | None) -> Mapping[str, float]:
        return dict(self._dispatcher.notify("on_evaluation", success_rate) or {})


class LocalLearnerHooks(GroupLearnerHooks):
    """:class:`LearnerHooks` of the one in-process learner rank."""

    def __init__(self, group: LocalLearnerGroup) -> None:
        super().__init__(_LocalDispatcher(group))


class _LocalDispatcher:
    def __init__(self, group: LocalLearnerGroup) -> None:
        self._group = group

    def query(self, name: str, *args: Any) -> Any:
        return self._group._run_hook(name, *args)

    def notify(self, name: str, *args: Any) -> Any:
        return self._group._run_hook(name, *args)


def publish_weights(group: Any, version: int) -> WeightsRef:
    """``group.publish(version)``, or its ``policy_state`` wrapped inline for a group without ``publish``."""
    publish = getattr(group, "publish", None)
    if callable(publish):
        return publish(version)
    return WeightsRef.of(group.policy_state(version=version))


def open_learner_group(
    runtime: RolloutRuntime,
    spec: LearnerWorkerSpec,
    *,
    num_learner_ranks: int,
    name: str,
    local_device: str,
) -> LearnerGroup:
    """The one-rank learner in this process, or a provider learner group for several ranks.

    Every runtime keeps a single learner rank in this process, as the
    off-policy loop does, even when its other roles run in their own processes.
    """
    if num_learner_ranks == 1:
        return LocalLearnerGroup(spec, device=local_device)
    return open_distributed_learner_group(
        runtime,
        spec,
        num_learner_ranks=num_learner_ranks,
        name=name,
    )
