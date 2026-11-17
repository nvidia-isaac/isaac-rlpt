# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learner ranks as Ray actors forming one torch.distributed process group (NCCL on GPUs, gloo on CPUs).

Every rank runs :class:`~isaac_rl_posttraining.runtimes.fsdp.worker.FSDPLearnerWorker`:
modules the algorithm shards are wrapped in FSDP, modules it keeps whole
(off-policy SAC) are replicated with their gradients averaged. The controller
proxy keeps every collective in one order across ranks and publishes weights
once, from rank 0, as a Ray object reference.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaac_rl_posttraining.runtimes._serialization import serialize_specs_by_value
from isaac_rl_posttraining.runtimes.base import (
    ActorFailedError,
    PolicyState,
    WeightsRef,
)
from isaac_rl_posttraining.runtimes.fsdp.worker import FSDPLearnerWorker
from isaac_rl_posttraining.runtimes.learner import (
    GroupLearnerHooks,
    LearnerBatch,
    LearnerStepResult,
    LearnerWorkerSpec,
)
from isaac_rl_posttraining.runtimes.native_ray.cluster import (
    ActorPlacement,
    RayCluster,
    close_actors,
    free_port,
    import_ray,
    node_ip,
    remote_class,
    seed_process,
    wait_all,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.native import NativeRuntimeConfig

__all__ = ["LearnerRankActor", "RayLearnerGroup"]


class LearnerRankActor(FSDPLearnerWorker):
    """One learner rank in its own process."""

    def __init__(
        self,
        spec: LearnerWorkerSpec,
        *,
        rank: int,
        world_size: int,
        device: str,
        mixed_precision: str,
        seed: int,
        collective_timeout_seconds: float | None,
    ) -> None:
        import_ray()  # registers the resolver of weights published by reference
        seed_process(
            seed
        )  # torch is seeded again when the rank joins; NumPy and random only here
        self._rank = rank
        self._world_size = world_size
        super().__init__(
            spec,
            mixed_precision,
            seed,
            device=device,
            collective_timeout_seconds=collective_timeout_seconds,
        )

    def rendezvous(self) -> tuple[str, int]:
        """The address the other ranks join (asked of rank 0)."""
        return node_ip(), free_port()

    def join(self, address: str, port: int) -> int:
        """Join the process group at rank 0's address and build the algorithm's learner."""
        os.environ.update(
            MASTER_ADDR=address,
            MASTER_PORT=str(port),
            RANK=str(self._rank),
            WORLD_SIZE=str(self._world_size),
            LOCAL_RANK="0",
        )
        return self.initialize()

    def train_step(self, batch: LearnerBatch, expected_step: int) -> LearnerStepResult:
        # Arrays from the object store are read-only; the algorithm gets its own copies.
        batch = LearnerBatch(
            tensors=batch.tensors, samples=batch.samples, env_steps=batch.env_steps
        )
        return super().train_step(batch, expected_step)

    def publish(self, version: int) -> list[Any] | None:
        """Gather the published module (a collective); rank 0 puts it in the object store once."""
        state = self.policy_state(version)
        if state is None:
            return None
        # A list keeps the reference itself from being resolved on the way back.
        return [import_ray().put(state)]


class RayLearnerGroup:
    """The controller's proxy for learner-rank actors; implements :class:`~isaac_rl_posttraining.runtimes.learner.LearnerGroup`."""

    def __init__(
        self,
        cluster: RayCluster,
        spec: LearnerWorkerSpec,
        *,
        num_learner_ranks: int,
        name: str,
        config: NativeRuntimeConfig,
    ) -> None:
        self.num_learner_ranks = num_learner_ranks
        self.name = name
        self.config = config
        self._lock = threading.Lock()
        self._step = 0
        self._policy_version = -1
        self._failure: str | None = None
        self._closed = False
        self._actors: list[Any] = []
        self._placements: list[ActorPlacement] = []
        actor_class = remote_class(LearnerRankActor)
        token = uuid.uuid4().hex[:6]
        try:
            with serialize_specs_by_value(
                spec, enabled=config.serialize_specs_by_value
            ):
                spec_ref = import_ray().put(spec)
            for rank in range(num_learner_ranks):
                placement = cluster.place(
                    "learner", rank, name=f"{name}_rank{rank}_{token}"
                )
                self._placements.append(placement)
                self._actors.append(
                    actor_class.options(**placement.options).remote(
                        spec_ref,
                        rank=rank,
                        world_size=num_learner_ranks,
                        device="cuda" if placement.gpus else "cpu",
                        mixed_precision=config.learner_mixed_precision,
                        seed=config.learner_seed,
                        collective_timeout_seconds=config.collective_timeout_seconds,
                    )
                )
            address, port = self._wait(
                [self._actors[0].rendezvous.remote()],
                "learner rendezvous",
                config.startup_timeout_seconds,
                ranks=(0,),
            )[0]
            ranks = self._wait(
                [actor.join.remote(address, port) for actor in self._actors],
                "starting the learner ranks",
                config.startup_timeout_seconds,
            )
            if tuple(ranks) != tuple(range(num_learner_ranks)):
                raise RuntimeError(f"learner ranks joined as {ranks}")
        except BaseException:
            self._kill()
            raise

    def describe_rank(self, rank: int) -> str:
        return self._placements[rank].describe("learner rank")

    def _wait(
        self,
        calls: Sequence[Any],
        what: str,
        timeout_seconds: float | None,
        *,
        ranks: Sequence[int] | None = None,
    ) -> list[Any]:
        order = tuple(range(len(calls))) if ranks is None else tuple(ranks)
        return wait_all(
            calls,
            describe=lambda index: self.describe_rank(order[index]),
            timeout_seconds=timeout_seconds,
            what=f"learner group {self.name!r}: {what}",
        )

    def _collective(
        self, method: str, *args: Any, ranks: Sequence[int] | None = None
    ) -> list[Any]:
        """Submit ``method`` to the ranks (all by default) under the lock, so every rank sees one order."""
        with self._lock:
            self._require_usable()
            targets = range(self.num_learner_ranks) if ranks is None else ranks
            return [
                getattr(self._actors[rank], method).remote(*args) for rank in targets
            ]

    def _require_usable(self) -> None:
        if self._closed:
            raise RuntimeError(f"learner group {self.name!r} is closed")
        if self._failure is not None:
            raise ActorFailedError(
                f"learner group {self.name!r} is unusable after a failed call: {self._failure}"
            )

    def _finish(
        self,
        calls: Sequence[Any],
        what: str,
        timeout_seconds: float | None,
        *,
        ranks: Sequence[int] | None = None,
    ) -> list[Any]:
        try:
            return self._wait(calls, what, timeout_seconds, ranks=ranks)
        except BaseException as exc:
            # A rank that failed mid-collective leaves the others blocked in it.
            self._failure = f"{type(exc).__name__}: {exc}"
            raise

    def train_step_async(self, batches: Sequence[LearnerBatch]) -> _RayLearnerStep:
        """Start one update on every rank (rank ``r`` trains on ``batches[r]``)."""
        if len(batches) != self.num_learner_ranks:
            raise ValueError(
                f"expected one learner batch per rank ({self.num_learner_ranks}), got {len(batches)}"
            )
        with self._lock:
            self._require_usable()
            self._step += 1
            expected = self._step
            calls = [
                actor.train_step.remote(batch, expected)
                for actor, batch in zip(self._actors, batches, strict=True)
            ]
        return _RayLearnerStep(self, calls, expected)

    def train_step(
        self, batches: Sequence[LearnerBatch]
    ) -> tuple[LearnerStepResult, ...]:
        return self.train_step_async(batches).result()

    def policy_state(self, *, version: int) -> PolicyState:
        """The published module's weights at ``version``, inline."""
        self._advance_version(version)
        states = self._finish(
            self._collective("policy_state", version),
            f"exporting weights version {version}",
            self.config.control_timeout_seconds,
        )
        state = states[0]
        if not isinstance(state, PolicyState) or state.version != version:
            raise RuntimeError(f"rank 0 exported {state!r}, expected version {version}")
        return state

    def publish(self, version: int) -> WeightsRef:
        """Export the published module once on rank 0; receivers fetch it by Ray object reference."""
        self._advance_version(version)
        answers = self._finish(
            self._collective("publish", version),
            f"publishing weights version {version}",
            self.config.control_timeout_seconds,
        )
        if answers[0] is None or any(answer is not None for answer in answers[1:]):
            raise RuntimeError("exactly rank 0 must publish the weights")
        return WeightsRef(version=version, handle=answers[0][0])

    def _advance_version(self, version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("policy-state version must be an integer")
        if version <= self._policy_version:
            raise ValueError(
                f"policy-state version must advance beyond {self._policy_version}, got {version}"
            )
        self._policy_version = version

    def hooks(self) -> GroupLearnerHooks:
        """Algorithm hooks: queries answered by rank 0, notifications sent to every rank."""
        return GroupLearnerHooks(_Dispatcher(self))

    def run_hook(self, name: str, *args: Any, every_rank: bool) -> Any:
        ranks = None if every_rank else (0,)
        answers = self._finish(
            self._collective("run_hook", name, *args, ranks=ranks),
            f"learner hook {name!r}",
            self.config.control_timeout_seconds,
            ranks=ranks,
        )
        return answers[0]

    def save_checkpoint(self, directory: str) -> None:
        """Every rank writes its part; the directory must be visible to every learner node."""
        ranks = self._finish(
            self._collective("save_checkpoint", directory),
            f"saving a checkpoint to {directory}",
            self.config.call_timeout_seconds,
        )
        if tuple(int(rank) for rank in ranks) != tuple(range(self.num_learner_ranks)):
            raise RuntimeError(f"learner checkpoint ranks are inconsistent: {ranks}")

    def load_checkpoint(self, directory: str) -> None:
        restored = self._finish(
            self._collective("load_checkpoint", directory),
            f"loading the checkpoint {directory}",
            self.config.call_timeout_seconds,
        )
        steps = {int(step) for _, step in restored}
        if len(steps) != 1:
            raise RuntimeError(
                f"learner checkpoint step is inconsistent: {sorted(steps)}"
            )
        with self._lock:
            self._step = steps.pop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._failure is not None:
            # The ranks left alive may be blocked in the failed collective: stop them.
            self._kill()
            return
        errors = close_actors(
            self._actors,
            describe=self.describe_rank,
            timeout_seconds=self.config.control_timeout_seconds,
        )
        self._actors.clear()
        if errors:
            raise errors[0]

    def _kill(self) -> None:
        ray = import_ray()
        for actor in self._actors:
            ray.kill(actor)
        self._actors.clear()


class _RayLearnerStep:
    """One update in flight on every rank (an ``AsyncCall``)."""

    def __init__(self, group: RayLearnerGroup, calls: list[Any], step: int) -> None:
        self._group = group
        self._calls = calls
        self._step = step
        self._results: tuple[LearnerStepResult, ...] | None = None
        self._finished_at: float | None = None

    def result(self) -> tuple[LearnerStepResult, ...]:
        if self._results is None:
            results = tuple(
                self._group._finish(
                    self._calls,
                    f"update {self._step}",
                    self._group.config.call_timeout_seconds,
                )
            )
            if tuple(result.learner_rank for result in results) != tuple(
                range(self._group.num_learner_ranks)
            ) or any(result.step != self._step for result in results):
                raise RuntimeError(
                    f"learner ranks did not complete update {self._step}"
                )
            self._finished_at = time.time()
            self._results = results
        return self._results

    def finished_at(self) -> float | None:
        return self._finished_at


class _Dispatcher:
    def __init__(self, group: RayLearnerGroup) -> None:
        self._group = group

    def query(self, name: str, *args: Any) -> Any:
        return self._group.run_hook(name, *args, every_rank=False)

    def notify(self, name: str, *args: Any) -> Any:
        return self._group.run_hook(name, *args, every_rank=True)
