# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The microbatcher: which queued requests a policy server answers together."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Hashable, Sequence
from dataclasses import dataclass

from isaac_rl_posttraining.runtimes.serving.messages import (
    DECISION_KINDS,
    PolicyRequest,
    RequestKind,
)

__all__ = ["DEFAULT_MAX_WAIT_MS", "Batcher", "QueuedRequest"]

#: How long a policy server waits after the first request for others to join
#: its batch, unless configured. RLInf's value: about 1% of a GR00T decode
#: (~400 ms on an L40), long enough for the requests of rollout workers that
#: reach a decision together to coalesce, which a zero wait never does.
DEFAULT_MAX_WAIT_MS = 5.0

#: The order a drained batch is answered in: decisions first, then the
#: encodings and values of next observations.
_KIND_ORDER: tuple[RequestKind, ...] = (
    "act",
    "encode",
    "value",
    "act_and_value",
    "reference",
    "features",
    "decode",
)
_ACTING_KINDS = DECISION_KINDS


@dataclass(frozen=True)
class QueuedRequest:
    """A request waiting in a server's queue.

    ``group`` numbers one client call: requests submitted together (a batched
    rollout worker's whole batch) share it. ``client`` is the connected client
    that sent it (``None``: a caller that never connected).
    """

    request: PolicyRequest
    group: int
    enqueued_at: float
    client: int | None = None


@dataclass(frozen=True)
class Batcher:
    """How a policy server groups requests.

    The server waits up to ``max_wait_ms`` after the first request for others
    and answers at most ``max_batch`` at once (``None``: no limit). It answers
    at once when every connected client already has a request waiting, since
    no other request can join the batch. With
    ``serve_eval_unbatched``, evaluation requests are only batched with
    requests of the same client call, so how requests happen to arrive
    cannot change a checkpoint's score; a session stays an evaluation session
    until it acts in another mode.
    """

    max_batch: int | None = None
    max_wait_ms: float = 0.0
    serve_eval_unbatched: bool = False

    def __post_init__(self) -> None:
        if self.max_batch is not None and (
            isinstance(self.max_batch, bool)
            or not isinstance(self.max_batch, int)
            or self.max_batch < 1
        ):
            raise ValueError("max_batch must be a positive integer or None")
        if (
            isinstance(self.max_wait_ms, bool)
            or not isinstance(self.max_wait_ms, (int, float))
            or not math.isfinite(self.max_wait_ms)
            or self.max_wait_ms < 0
        ):
            raise ValueError("max_wait_ms must be a finite non-negative number")
        if not isinstance(self.serve_eval_unbatched, bool):
            raise TypeError("serve_eval_unbatched must be a boolean")

    @property
    def max_wait_seconds(self) -> float:
        return float(self.max_wait_ms) / 1000.0

    def full(self, pending: int) -> bool:
        """Whether ``pending`` requests already fill one batch."""
        return self.max_batch is not None and pending >= self.max_batch

    def ready(self, pending: Sequence[QueuedRequest], clients: Collection[int]) -> bool:
        """Whether to answer ``pending`` now: a full batch, or a request waiting from every connected client."""
        if self.full(len(pending)):
            return True
        waiting = {item.client for item in pending}
        return bool(clients) and all(client in waiting for client in clients)

    def take(self, pending: list[QueuedRequest]) -> list[QueuedRequest]:
        """Remove and return the next batch, oldest first."""
        size = (
            len(pending)
            if self.max_batch is None
            else min(self.max_batch, len(pending))
        )
        batch = pending[:size]
        del pending[:size]
        return batch

    def plan(
        self,
        batch: Sequence[QueuedRequest],
        *,
        evaluating: Callable[[Hashable], bool],
        mixed_modes: bool,
    ) -> list[list[QueuedRequest]]:
        """Split a drained batch into the same-kind groups the engine answers, in order.

        ``evaluating`` tells whether a session is an evaluation session;
        ``mixed_modes=False`` also splits each group by decision mode.
        """
        groups: list[list[QueuedRequest]] = []
        for kind in _KIND_ORDER:
            items = [item for item in batch if item.request.kind == kind]
            if not items:
                continue
            if self.serve_eval_unbatched:
                groups.extend(self._isolate_evaluation(items, kind, evaluating))
            else:
                groups.append(items)
        if mixed_modes:
            return groups
        split: list[list[QueuedRequest]] = []
        for group in groups:
            by_mode: dict[object, list[QueuedRequest]] = {}
            for item in group:
                by_mode.setdefault(item.request.mode, []).append(item)
            split.extend(by_mode.values())
        return split

    @staticmethod
    def _isolate_evaluation(
        items: list[QueuedRequest],
        kind: RequestKind,
        evaluating: Callable[[Hashable], bool],
    ) -> list[list[QueuedRequest]]:
        """Evaluation requests first, one group per client call; the rest together after."""
        by_call: dict[int, list[QueuedRequest]] = {}
        rest: list[QueuedRequest] = []
        for item in items:
            request = item.request
            if kind in _ACTING_KINDS:
                isolated = request.mode == "eval"
            else:
                isolated = evaluating(request.session_id)
            if isolated:
                by_call.setdefault(item.group, []).append(item)
            else:
                rest.append(item)
        groups = list(by_call.values())
        if rest:
            groups.append(rest)
        return groups
