# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host an off-policy algorithm behind the provider-neutral learner contract.

The replay buffer stays with the controller; the learner never samples it.
For every update the algorithm draws the batches of one ``train_step`` call
through its samplers (:meth:`~isaac_rl_posttraining.algorithms.base.BaseAlgorithm.learner_batch`)
and the controller ships them as one :class:`LearnerBatch`.
:class:`ReplayLearner` turns that batch back into per-gradient-step tensors
and runs the algorithm's ``train_step`` on them.
"""

from __future__ import annotations

import functools
from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import torch

from isaac_rl_posttraining.algorithms.base import BaseAlgorithm
from isaac_rl_posttraining.buffers.replay import BATCH_FIELDS, Batch
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerContext,
    LearnerModuleManager,
    LearnerWorkerSpec,
    LocalLearnerGroup,
    average_gradients_across_ranks,
)

__all__ = [
    "ReplayLearner",
    "load_runtime_checkpoint",
    "pack_batches",
    "replica_learner_spec",
    "unpack_batches",
]

#: One gradient step's batches by draw name (``replay``, ``expert``, ...).
StepBatches = dict[str, Batch]


def pack_batches(steps: Sequence[Mapping[str, Batch]]) -> LearnerBatch:
    """Pack the batches of one ``train_step`` call, one ``{draw: batch}`` mapping per gradient step.

    Keys are ``"<step>/<draw>/<field>"``; only the array fields of a batch
    travel and dict observations stay nested one level, so the learner
    rebuilds exactly the batches the algorithm drew.
    """
    if not steps:
        raise ValueError("a learner batch needs at least one gradient step")
    tensors: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
    samples = 0
    for step, draws in enumerate(steps):
        if not draws:
            raise ValueError(f"gradient step {step} has no batches")
        for name, batch in draws.items():
            if "/" in name:
                raise ValueError(f"draw names contain no '/': {name!r}")
            for field in BATCH_FIELDS:
                tensors[f"{step}/{name}/{field}"] = _as_arrays(batch[field])
            samples += int(np.shape(batch["reward"])[0])
    return LearnerBatch(tensors=tensors, samples=samples)


def _as_arrays(value: object) -> np.ndarray | dict[str, np.ndarray]:
    """One batch field as an array, or a dict of arrays for dict observations."""
    if isinstance(value, Mapping):
        return {key: np.asarray(entry) for key, entry in value.items()}
    return np.asarray(value)


def unpack_batches(batch: LearnerBatch, device: torch.device) -> list[StepBatches]:
    """Regroup a packed learner batch into ``{draw: batch}`` tensors per gradient step, on ``device``."""
    steps: dict[int, StepBatches] = {}
    for key, value in batch.tensors.items():
        step_text, name, field = key.split("/", 2)
        draw = steps.setdefault(int(step_text), {}).setdefault(name, {})
        if isinstance(value, Mapping):
            draw[field] = {entry_key: torch.as_tensor(entry, device=device) for entry_key, entry in value.items()}
        else:
            draw[field] = torch.as_tensor(value, device=device)
    if sorted(steps) != list(range(len(steps))):
        raise ValueError(f"learner batch steps must cover 0..N-1, got {sorted(steps)}")
    return [steps[step] for step in range(len(steps))]


class ReplayLearner:
    """Learner worker running one off-policy algorithm on controller-drawn batches.

    The algorithm declares the networks and optimizers the runtime manages
    (:meth:`~isaac_rl_posttraining.algorithms.base.BaseAlgorithm.learner_modules`,
    :meth:`~isaac_rl_posttraining.algorithms.base.BaseAlgorithm.learner_optimizers`);
    everything else travels in its learner state. On several learner ranks
    every rank holds a full copy of the networks and their gradients are
    averaged, so each rank trains on its slice of the same draw.
    """

    def __init__(
        self,
        context: LearnerContext,
        modules: LearnerModuleManager,
        algorithm: BaseAlgorithm,
    ) -> None:
        self.device = torch.device(context.device)
        algorithm_modules = algorithm.learner_modules()
        published = [name for name, entry in algorithm_modules.items() if entry.publish]
        if len(published) != 1:
            raise ValueError(
                f"{type(algorithm).__name__} must mark exactly one learner module as published, got {published}"
            )
        algorithm_device = next(algorithm_modules[published[0]].module.parameters()).device
        if _canonical(algorithm_device) != _canonical(self.device):
            raise ValueError(
                f"the algorithm lives on {algorithm_device} but the learner rank "
                f"runs on {context.device}"
            )
        self.algorithm = algorithm
        for name, entry in algorithm_modules.items():
            prepared = modules.prepare(entry.module, name=name, publish=entry.publish, shard=False)
            if prepared is not entry.module:
                raise TypeError(
                    f"the learner runtime replaced module {name!r}; off-policy learners keep "
                    "their networks whole on every rank (prepare(..., shard=False))"
                )
        for name, entry in algorithm.learner_optimizers().items():
            if entry.module not in algorithm_modules:
                raise ValueError(f"optimizer {name!r} steps unknown learner module {entry.module!r}")
            modules.register_optimizer(name, entry.optimizer, module=algorithm_modules[entry.module].module)
        if context.world_size > 1:
            # The module manager averages the modules' gradients; these are the algorithm's other tensors.
            in_modules = {
                id(parameter)
                for entry in algorithm_modules.values()
                for parameter in entry.module.parameters()
            }
            average_gradients_across_ranks(
                [
                    tensor
                    for tensor in algorithm.replicated_parameters()
                    if id(tensor) not in in_modules
                ]
            )
        self._managed = algorithm.learner_managed_components()
        # A component the checkpoint format cannot carry fails here, before
        # any training, rather than at the first save.
        self.checkpoint_state()

    def train_step(self, batch: LearnerBatch) -> Mapping[str, float]:
        rollout_step = batch.env_steps if self.algorithm.respects_step_schedules else None
        metrics = self.algorithm.train_step(
            rollout_step=rollout_step,
            batches=unpack_batches(batch, self.device),
        )
        scalars: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, (bool, int, float, np.integer, np.floating)):
                scalars[name] = float(value)
        return scalars

    def gradient_steps(self) -> int:
        return int(self.algorithm.step_counter)

    def target_update_count(self, env_steps: int) -> int | None:
        return self.algorithm.target_update_count(env_steps)

    def observe_rewards(
        self,
        rewards: list[float],
        dones: list[bool],
        truncateds: list[bool],
        episode_ids: list[Hashable],
    ) -> None:
        self.algorithm.observe_rewards(rewards, dones, truncateds, episode_ids)

    def on_evaluation(self, success_rate: float | None) -> dict[str, float]:
        return dict(self.algorithm.on_evaluation(success_rate))

    def checkpoint_state(self) -> Mapping[str, np.ndarray]:
        return self.algorithm.learner_state(managed=self._managed)

    def load_checkpoint_state(self, state: Mapping[str, np.ndarray]) -> None:
        self.algorithm.load_learner_state(state, managed=self._managed)

    def close(self) -> None:
        return None


def replica_learner_spec(algorithm: BaseAlgorithm) -> LearnerWorkerSpec:
    """A picklable learner spec for ranks in other processes.

    Each rank unpickles its own copy of ``algorithm`` without the controller's
    replay buffer, samplers and logger
    (:meth:`~isaac_rl_posttraining.algorithms.base.BaseAlgorithm.learner_replica`)
    and trains it on the batches the controller draws.
    """
    return functools.partial(_replica_learner, algorithm.learner_replica())


def _replica_learner(
    algorithm: BaseAlgorithm, context: LearnerContext, modules: LearnerModuleManager
) -> ReplayLearner:
    return ReplayLearner(context, modules, algorithm)


def _canonical(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def load_runtime_checkpoint(algorithm: BaseAlgorithm, directory: str) -> None:
    """Restore ``algorithm`` in place from a checkpoint the runtime loop wrote.

    The local learner group owns the checkpoint layout, so the modules,
    optimizers and learner state come back through the same registration a
    training run uses. The process's random stream is left alone: a warm start
    or a re-evaluation keeps the run's own seed.
    """
    published = next(entry.module for entry in algorithm.learner_modules().values() if entry.publish)
    device = next(published.parameters()).device
    group = LocalLearnerGroup(functools.partial(ReplayLearner, algorithm=algorithm), device=str(device))
    try:
        group.load_checkpoint(directory, restore_rng=False)
    finally:
        group.close()
