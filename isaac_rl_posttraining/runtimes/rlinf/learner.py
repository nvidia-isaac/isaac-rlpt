# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf-managed PyTorch FSDP learner groups."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef
from isaac_rl_posttraining.runtimes.fsdp.checkpoint import (
    LEARNER_CHECKPOINT_SCHEMA_VERSION,
    validate_learner_checkpoint_manifest,
)
from isaac_rl_posttraining.runtimes.learner import (
    GroupLearnerHooks,
    LearnerBatch,
    LearnerStepResult,
)
from isaac_rl_posttraining.runtimes.rlinf.cluster import (
    rlinf_worker_class,
    terminate_worker_group,
)


def make_learner_worker(worker_base: type):
    """Build a generic FSDP learner on RLInf-managed accelerator workers.

    RLInf supplies placement, process identity, and the distributed rendezvous.
    PyTorch FSDP performs parameter, gradient, and optimizer-state sharding.
    RLInf's ``FSDPModelManager`` is intentionally not used because it constructs
    specific Hugging Face models and optimizers from RLInf recipe schemas.
    """
    from isaac_rl_posttraining.runtimes.fsdp.worker import FSDPLearnerWorker

    return rlinf_worker_class(
        "PosttrainFSDPLearnerWorker", FSDPLearnerWorker, worker_base
    )


class RLInfLearnerGroup:
    """Controller proxy that keeps every FSDP collective in one total order."""

    def __init__(
        self,
        group: Any,
        num_learner_ranks: int,
        *,
        release_accelerators: Callable[[], None] | None = None,
    ) -> None:
        self._group = group
        self.num_learner_ranks = num_learner_ranks
        self._step = 0
        self._policy_version = -1
        self._closed = False
        self._collective_failure: str | None = None
        self._collective_lock = threading.Lock()
        self._release_accelerators = release_accelerators or (lambda: None)

    def _ensure_available(self) -> None:
        if self._closed:
            raise RuntimeError("RLInf learner group is closed")
        if self._collective_failure is not None:
            raise RuntimeError(
                "RLInf learner group is unusable after a failed collective: "
                + self._collective_failure
            )

    def _record_collective_failure(self, exc: BaseException) -> None:
        self._collective_failure = f"{type(exc).__name__}: {exc}"

    def train_step(
        self,
        batches: Sequence[LearnerBatch],
    ) -> tuple[LearnerStepResult, ...]:
        if len(batches) != self.num_learner_ranks:
            raise ValueError(
                "expected one learner batch per learner rank "
                f"({self.num_learner_ranks}), "
                f"got {len(batches)}"
            )
        if not all(isinstance(batch, LearnerBatch) for batch in batches):
            raise TypeError("every learner batch must be a LearnerBatch")
        with self._collective_lock:
            self._ensure_available()
            try:
                expected_step = self._step + 1
                handles = tuple(
                    self._group.execute_on(rank).train_step(batch, expected_step)
                    for rank, batch in enumerate(batches)
                )
                results = tuple(handle.wait()[0] for handle in handles)
                ranks = tuple(result.learner_rank for result in results)
                if ranks != tuple(range(self.num_learner_ranks)):
                    raise RuntimeError(f"RLInf learner ranks are inconsistent: {ranks}")
                if any(result.step != expected_step for result in results):
                    raise RuntimeError(
                        f"RLInf learner ranks did not complete step {expected_step}"
                    )
                self._step = expected_step
                return results
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise

    def policy_state(self, *, version: int) -> PolicyState:
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("policy-state version must be an integer")
        with self._collective_lock:
            self._ensure_available()
            if version <= self._policy_version:
                raise ValueError(
                    f"policy-state version must advance beyond {self._policy_version}, "
                    f"got {version}"
                )
            try:
                rank_states = tuple(self._group.policy_state(version).wait())
                states = tuple(state for state in rank_states if state is not None)
                if len(states) != 1:
                    raise RuntimeError(
                        "FSDP policy export must return state on exactly rank zero, "
                        f"got {len(states)} states"
                    )
                state = states[0]
                if state.version != version:
                    raise RuntimeError(
                        f"FSDP policy export returned version {state.version}, "
                        f"expected {version}"
                    )
                self._policy_version = version
                return state
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise

    def publish(self, version: int) -> WeightsRef:
        """Export the published module on rank 0 and return it inline."""
        return WeightsRef.of(self.policy_state(version=version))

    def hooks(self) -> GroupLearnerHooks:
        """Algorithm hooks: queries answered by rank 0, notifications sent to every rank."""
        return GroupLearnerHooks(_RLInfHookDispatcher(self))

    def _query(self, name: str, *args: Any) -> Any:
        with self._collective_lock:
            self._ensure_available()
            try:
                return self._group.execute_on(0).run_hook(name, *args).wait()[0]
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise

    def _notify(self, name: str, *args: Any) -> Any:
        with self._collective_lock:
            self._ensure_available()
            try:
                results = tuple(self._group.run_hook(name, *args).wait())
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise
        if len(results) != self.num_learner_ranks:
            raise RuntimeError(
                f"learner hook {name!r} reached {len(results)} of "
                f"{self.num_learner_ranks} ranks"
            )
        return results[0]

    def save_checkpoint(self, directory: str) -> None:
        with self._collective_lock:
            self._ensure_available()
            try:
                ranks = tuple(
                    int(rank) for rank in self._group.save_checkpoint(directory).wait()
                )
                if ranks != tuple(range(self.num_learner_ranks)):
                    raise RuntimeError(
                        f"RLInf learner checkpoint ranks are inconsistent: {ranks}"
                    )
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise

    def load_checkpoint(self, directory: str) -> None:
        with self._collective_lock:
            self._ensure_available()
            try:
                restored = tuple(self._group.load_checkpoint(directory).wait())
                ranks = tuple(int(rank) for rank, _ in restored)
                if ranks != tuple(range(self.num_learner_ranks)):
                    raise RuntimeError(
                        f"RLInf learner checkpoint ranks are inconsistent: {ranks}"
                    )
                steps = tuple(int(step) for _, step in restored)
                if len(set(steps)) != 1:
                    raise RuntimeError(
                        f"RLInf learner checkpoint step is inconsistent: {steps}"
                    )
                self._step = steps[0]
            except BaseException as exc:
                self._record_collective_failure(exc)
                raise

    def close(self) -> None:
        with self._collective_lock:
            if self._closed:
                return
            self._closed = True
            try:
                error = terminate_worker_group(self._group)
            finally:
                self._release_accelerators()
        if error is not None:
            raise RuntimeError("an RLInf learner worker failed to close") from error


class _RLInfHookDispatcher:
    def __init__(self, group: RLInfLearnerGroup) -> None:
        self._group = group

    def query(self, name: str, *args: Any) -> Any:
        return self._group._query(name, *args)

    def notify(self, name: str, *args: Any) -> Any:
        return self._group._notify(name, *args)


__all__ = [
    "LEARNER_CHECKPOINT_SCHEMA_VERSION",
    "RLInfLearnerGroup",
    "make_learner_worker",
    "validate_learner_checkpoint_manifest",
]
