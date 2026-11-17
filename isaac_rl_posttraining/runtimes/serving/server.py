# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime-agnostic policy server: a serving engine behind a microbatcher and a session table.

A host owns the queue and the transport (a thread queue in-process, an
asyncio queue inside a Ray actor) and hands each drained batch to
:meth:`PolicyServer.serve`; everything that decides what is answered
together, and how, lives here so every runtime serves identically.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

from isaac_rl_posttraining.runtimes.base import WeightsRef
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.policy_adapters import first_value
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher, QueuedRequest
from isaac_rl_posttraining.runtimes.serving.messages import (
    BASE_MODEL_KINDS,
    DECISION_KINDS,
    PolicyResponse,
    RequestKind,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.runtimes.serving import ServingEngine

__all__ = ["PolicyServer", "SessionTable", "replica_stats"]

#: Names the server's timing counters carry per request kind.
_OPERATION_NAMES: dict[RequestKind, str] = {
    "act": "action",
    "encode": "prepare",
    "value": "successor",
    "act_and_value": "action_and_value",
    "decode": "decode",
    "reference": "reference",
    "features": "features",
}
_ACTING_KINDS = DECISION_KINDS


class SessionTable:
    """Per-session facts the server keeps: which sessions are evaluating."""

    def __init__(self) -> None:
        self._evaluating: set[Hashable] = set()

    def observe(
        self, kind: RequestKind, session_id: Hashable, mode: str | None
    ) -> None:
        """A session's latest decision mode decides whether it is evaluating."""
        if kind not in _ACTING_KINDS:
            return
        if mode == "eval":
            self._evaluating.add(session_id)
        else:
            self._evaluating.discard(session_id)

    def evaluating(self, session_id: Hashable) -> bool:
        return session_id in self._evaluating


class PolicyServer:
    """Answer batches of typed requests with one serving engine."""

    def __init__(
        self,
        engine: ServingEngine,
        batcher: Batcher,
        *,
        collect_metrics: bool = False,
    ) -> None:
        self.engine = engine
        self.batcher = batcher
        self.sessions = SessionTable()
        self._collect_metrics = bool(collect_metrics)
        self._batch_sizes: list[int] = []
        self._inference_seconds = 0.0
        self._request_latencies_seconds: list[float] = []
        # The fused kind's counters appear once a fused request was served.
        self._operation_seconds: dict[str, float] = dict.fromkeys(
            ("action", "prepare", "successor"), 0.0
        )
        self._operation_batch_sizes: dict[str, list[int]] = {
            name: [] for name in self._operation_seconds
        }

    @property
    def features(self) -> PolicyFeatures:
        return self.engine.features

    @property
    def version(self) -> int:
        return self.engine.version

    def serve(self, batch: Sequence[QueuedRequest]) -> list[PolicyResponse | Exception]:
        """Answer one drained batch; a failed group answers each of its requests with the error."""
        if self.batcher.serve_eval_unbatched:
            for item in batch:
                request = item.request
                self.sessions.observe(request.kind, request.session_id, request.mode)
        answers: dict[int, PolicyResponse | Exception] = {}
        for group in self.batcher.plan(
            batch,
            evaluating=self.sessions.evaluating,
            mixed_modes=self.features.mixed_mode_batches,
        ):
            self._serve_group(group, answers)
        return [answers[id(item)] for item in batch]

    def _serve_group(
        self,
        group: list[QueuedRequest],
        answers: dict[int, PolicyResponse | Exception],
    ) -> None:
        kind = group[0].request.kind
        started = time.perf_counter()
        try:
            outputs = self.engine.run(
                kind,
                [item.request.observation for item in group],
                [item.request.session_id for item in group],
                [item.request.mode for item in group],
                [item.request.base for item in group],
            )
            if len(outputs) != len(group):
                raise RuntimeError(
                    f"policy server answered {len(outputs)} of {len(group)} {kind!r} requests"
                )
            responses = [
                self._response(item, output)
                for item, output in zip(group, outputs, strict=True)
            ]
        except Exception as exc:  # noqa: BLE001 - every waiting request receives the error
            for item in group:
                answers[id(item)] = exc
        else:
            completed = time.perf_counter()
            for item, response in zip(group, responses, strict=True):
                answers[id(item)] = response
                if self._collect_metrics and kind in _ACTING_KINDS:
                    self._request_latencies_seconds.append(completed - item.enqueued_at)
        finally:
            if self._collect_metrics:
                elapsed = time.perf_counter() - started
                name = _OPERATION_NAMES[kind]
                self._operation_seconds[name] = (
                    self._operation_seconds.get(name, 0.0) + elapsed
                )
                self._operation_batch_sizes.setdefault(name, []).append(len(group))
                if kind in _ACTING_KINDS:
                    self._batch_sizes.append(len(group))
                    self._inference_seconds += elapsed

    def _response(self, item: QueuedRequest, output: Any) -> PolicyResponse:
        request = item.request
        version = self.engine.version
        if request.kind in BASE_MODEL_KINDS:
            result, random_state = output
            return PolicyResponse(
                request.request_id,
                request.kind,
                version,
                result=result,
                random_state=random_state,
            )
        if request.kind == "act":
            return PolicyResponse(
                request.request_id, "act", version, action_chunk=output
            )
        if request.kind == "encode":
            return PolicyResponse(
                request.request_id, "encode", version, observation=output
            )
        if request.kind == "value":
            observation, value = output
            return PolicyResponse(
                request.request_id,
                "value",
                version,
                observation=observation,
                value=first_value(value),
            )
        if request.kind == "act_and_value":
            chunk, observation, value = output
            return PolicyResponse(
                request.request_id,
                "act_and_value",
                version,
                action_chunk=chunk,
                observation=observation,
                value=first_value(value),
            )
        raise ValueError(f"unknown request kind {request.kind!r}")

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        self.engine.reset_sessions(session_ids, seeds)

    def set_weights(self, ref: WeightsRef) -> int:
        return self.engine.load_weights(ref)

    def stats(self) -> dict[str, Any]:
        """Batch sizes, latencies and per-kind compute time collected since the last reset."""
        return {
            "batch_sizes": tuple(self._batch_sizes),
            "inference_seconds": self._inference_seconds,
            "request_latencies_seconds": tuple(self._request_latencies_seconds),
            "operation_seconds": dict(self._operation_seconds),
            "operation_batch_sizes": {
                name: tuple(sizes)
                for name, sizes in self._operation_batch_sizes.items()
            },
            "policy_metadata": dict(self.engine.metadata()),
        }

    def reset_stats(self) -> None:
        self._batch_sizes.clear()
        self._inference_seconds = 0.0
        self._request_latencies_seconds.clear()
        for name in self._operation_seconds:
            self._operation_seconds[name] = 0.0
            self._operation_batch_sizes[name].clear()

    def close(self) -> None:
        self.engine.close()


def replica_stats(
    server: PolicyServer, *, index: int, initialization_seconds: float, device: str
) -> dict[str, Any]:
    """One replica's statistics, the same on every runtime: its index, load time, counters and peak memory."""
    allocated = reserved = None
    if device.startswith("cuda"):
        import torch

        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    return {
        "policy_server_index": index,
        "initialization_seconds": initialization_seconds,
        **server.stats(),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }
