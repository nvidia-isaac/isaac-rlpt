# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An algorithm's learner on one FSDP rank, independent of the process host.

The host (RLInf's worker, or the native runtime's Ray actor) mixes this class
into its worker type and supplies ``_rank`` and ``_world_size``; the PyTorch
process group is initialised here from the standard ``MASTER_ADDR`` /
``MASTER_PORT`` / ``RANK`` / ``WORLD_SIZE`` environment: NCCL on CUDA ranks,
gloo on CPU ranks.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from isaac_rl_posttraining.runtimes.base import PolicyState
from isaac_rl_posttraining.runtimes.fsdp.modules import FSDPModuleManager
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerContext,
    LearnerStepResult,
    LearnerWorker,
    LearnerWorkerSpec,
    run_hook,
)

__all__ = ["FSDPLearnerWorker"]

_NOT_INITIALIZED = "FSDP learner worker is not initialized"


class FSDPLearnerWorker:
    """An algorithm's learner on one FSDP rank, hosted by RLInf's worker or a native Ray actor.

    ``device`` is ``cuda`` (the one GPU the host isolated for this process,
    NCCL collectives) or ``cpu`` (gloo collectives).
    """

    # Provided by the host.
    _rank: int
    _world_size: int

    def __init__(
        self,
        spec: LearnerWorkerSpec,
        mixed_precision: str,
        seed: int,
        device: str = "cuda",
        collective_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__()
        if device not in ("cuda", "cpu"):
            raise ValueError(f"learner rank device must be cuda or cpu, got {device!r}")
        self._spec = spec
        self._mixed_precision = mixed_precision
        self._seed = seed
        self._device = device
        self._collective_timeout_seconds = collective_timeout_seconds
        self._delegate: LearnerWorker | None = None
        self._modules: FSDPModuleManager | None = None
        self._step = 0

    def initialize(self) -> int:
        if self._delegate is not None:
            raise RuntimeError("FSDP learner worker is already initialized")
        on_cuda = self._device == "cuda"
        if on_cuda:
            torch.cuda.set_device(0)
        if not dist.is_initialized():
            timeout = self._collective_timeout_seconds
            dist.init_process_group(
                backend="nccl" if on_cuda else "gloo",
                timeout=None if timeout is None else timedelta(seconds=timeout),
            )
        torch.manual_seed(self._seed)
        if on_cuda:
            torch.cuda.manual_seed_all(self._seed)
        device = "cuda:0" if on_cuda else "cpu"
        modules = FSDPModuleManager(self._mixed_precision, torch.device(device))
        delegate = self._spec(
            LearnerContext(
                learner_rank=int(self._rank),
                world_size=int(self._world_size),
                device=device,
            ),
            modules,
        )
        if not isinstance(delegate, LearnerWorker):
            raise TypeError(
                "learner spec must return an object implementing "
                "train_step, checkpoint_state, load_checkpoint_state, and close"
            )
        modules.validate()
        self._delegate = delegate
        self._modules = modules
        return int(self._rank)

    def train_step(
        self,
        batch: LearnerBatch,
        expected_step: int,
    ) -> LearnerStepResult:
        if self._delegate is None or self._modules is None:
            raise RuntimeError(_NOT_INITIALIZED)
        if expected_step != self._step + 1:
            raise RuntimeError(
                f"learner rank {self._rank} expected step {self._step + 1}, "
                f"got {expected_step}"
            )
        with self._modules.training_context():
            metrics = self._delegate.train_step(batch)
        normalized_metrics = {name: float(value) for name, value in metrics.items()}
        result = LearnerStepResult(
            learner_rank=int(self._rank),
            step=expected_step,
            samples=batch.samples,
            metrics=normalized_metrics,
        )
        self._step = expected_step
        return result

    def run_hook(self, name: str, *args: Any) -> Any:
        """Run one of the algorithm's :class:`~isaac_rl_posttraining.runtimes.learner.LearnerHooks` on this rank."""
        if self._delegate is None:
            raise RuntimeError(_NOT_INITIALIZED)
        return run_hook(self._delegate, name, *args)

    def policy_state(self, version: int) -> PolicyState | None:
        if self._modules is None:
            raise RuntimeError(_NOT_INITIALIZED)
        return self._modules.policy_state(version)

    def save_checkpoint(self, directory: str) -> int:
        if self._delegate is None or self._modules is None:
            raise RuntimeError(_NOT_INITIALIZED)
        self._modules.save_checkpoint(
            directory,
            self._delegate.checkpoint_state(),
            self._step,
        )
        return int(self._rank)

    def load_checkpoint(self, directory: str) -> tuple[int, int]:
        if self._delegate is None or self._modules is None:
            raise RuntimeError(_NOT_INITIALIZED)
        state, learner_step = self._modules.load_checkpoint(directory)
        self._delegate.load_checkpoint_state(state)
        self._step = learner_step
        return int(self._rank), learner_step

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None
        if dist.is_initialized():
            dist.destroy_process_group()
