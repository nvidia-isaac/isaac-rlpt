# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request queues that feed a :class:`PolicyServer`: one for threads, one for asyncio.

Both wait up to the batcher's ``max_wait_ms`` after the first request, or
until every connected client has a request waiting, take at most
``max_batch`` requests, and hand the batch to the server; requests one client
submits together arrive together.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

from isaac_rl_posttraining.runtimes.serving.batcher import QueuedRequest
from isaac_rl_posttraining.runtimes.serving.messages import (
    PolicyRequest,
    PolicyResponse,
)
from isaac_rl_posttraining.runtimes.serving.server import PolicyServer

__all__ = ["AsyncRequestQueue", "ThreadRequestQueue"]

_T = TypeVar("_T")


def _responses(
    answers: Sequence[PolicyResponse | BaseException],
) -> list[PolicyResponse]:
    for answer in answers:
        if isinstance(answer, BaseException):
            raise answer
    return [answer for answer in answers if isinstance(answer, PolicyResponse)]


class _Clients:
    """The clients connected to one queue, numbered as they connect."""

    def __init__(self) -> None:
        self.connected: set[int] = set()
        self._numbers = itertools.count()

    def connect(self) -> int:
        client = next(self._numbers)
        self.connected.add(client)
        return client

    def disconnect(self, client: int) -> None:
        self.connected.discard(client)


class ThreadRequestQueue:
    """Serve requests from any number of threads in one process.

    No serving thread: the first caller to find the server idle becomes the
    batch's leader, waits for others up to ``max_wait_ms`` (or until every
    connected client has a request waiting), answers the batch on its own
    thread and wakes the callers it answered.
    """

    def __init__(self, server: PolicyServer) -> None:
        self.server = server
        self._condition = threading.Condition(threading.Lock())
        self._pending: list[QueuedRequest] = []
        self._answers: dict[int, PolicyResponse | BaseException] = {}
        self._busy = False
        # Callers waiting to run an exclusive call go before the next batch,
        # so a busy stream of requests cannot starve an episode reset.
        self._exclusive_waiting = 0
        self._groups = itertools.count()
        self._clients = _Clients()
        self._closed = False

    def connect(self) -> int:
        """Register a client; pass the number with its requests."""
        with self._condition:
            return self._clients.connect()

    def disconnect(self, client: int) -> None:
        with self._condition:
            self._clients.disconnect(client)
            self._condition.notify_all()

    def submit(
        self, requests: Sequence[PolicyRequest], client: int | None = None
    ) -> list[PolicyResponse]:
        """Queue requests as one call of ``client`` and return their responses in order."""
        with self._condition:
            if self._closed:
                raise RuntimeError("policy server is closed")
            group = next(self._groups)
            enqueued_at = time.perf_counter()
            items = [
                QueuedRequest(request, group, enqueued_at, client)
                for request in requests
            ]
            self._pending.extend(items)
            self._condition.notify_all()
            while any(id(item) not in self._answers for item in items):
                if self._busy or self._exclusive_waiting:
                    self._condition.wait()
                    continue
                self._lead_one_batch()
            answers = [self._answers.pop(id(item)) for item in items]
        return _responses(answers)

    def _lead_one_batch(self) -> None:
        """Gather a batch, answer it with the lock released, and file the answers."""
        self._busy = True
        try:
            batcher = self.server.batcher
            deadline = time.perf_counter() + batcher.max_wait_seconds
            while not batcher.ready(self._pending, self._clients.connected):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            batch = batcher.take(self._pending)
            self._condition.release()
            try:
                answers: Sequence[PolicyResponse | BaseException] = self.server.serve(
                    batch
                )
            except BaseException as exc:  # noqa: BLE001 - the callers of the batch receive it
                answers = [exc] * len(batch)
            finally:
                self._condition.acquire()
            for item, answer in zip(batch, answers, strict=True):
                self._answers[id(item)] = answer
        finally:
            self._busy = False
            self._condition.notify_all()

    def exclusive(self, call: Callable[[], _T]) -> _T:
        """Run ``call`` while no batch is being answered (weights, resets, statistics)."""
        with self._condition:
            self._exclusive_waiting += 1
            while self._busy:
                self._condition.wait()
            self._exclusive_waiting -= 1
            self._busy = True
        try:
            return call()
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()

    def reset_sessions(
        self, session_ids: Sequence[Hashable], seeds: Sequence[int | None]
    ) -> None:
        self.exclusive(lambda: self.server.reset_sessions(session_ids, seeds))

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self.exclusive(self.server.close)


class AsyncRequestQueue:
    """Serve requests arriving as coroutines on one event loop (a Ray async actor).

    One long-lived consumer takes the first request, gathers more for up to
    ``max_wait_ms`` (until the batch is ready) and answers them; it never
    exits while the queue is open, so a producer cannot find a consumer that
    is about to stop.
    """

    def __init__(self, server: PolicyServer) -> None:
        self.server = server
        self._queue: asyncio.Queue[
            tuple[QueuedRequest, asyncio.Future[PolicyResponse]]
        ] = asyncio.Queue()
        self._consumer: asyncio.Task[None] | None = None
        self._groups = itertools.count()
        self._clients = _Clients()
        self._closed = False

    def connect(self) -> int:
        """Register a client; pass the number with its requests."""
        return self._clients.connect()

    def disconnect(self, client: int) -> None:
        self._clients.disconnect(client)

    async def submit(
        self, requests: Sequence[PolicyRequest], client: int | None = None
    ) -> list[PolicyResponse]:
        if self._closed:
            raise RuntimeError("policy server is closed")
        loop = asyncio.get_running_loop()
        group = next(self._groups)
        enqueued_at = time.perf_counter()
        futures: list[asyncio.Future[PolicyResponse]] = []
        for request in requests:
            future: asyncio.Future[PolicyResponse] = loop.create_future()
            futures.append(future)
            await self._queue.put(
                (QueuedRequest(request, group, enqueued_at, client), future)
            )
        if self._consumer is None or self._consumer.done():
            self._consumer = loop.create_task(self._consume())
        return [await future for future in futures]

    async def _consume(self) -> None:
        while not self._closed:
            batch = await self._gather()
            try:
                answers = self.server.serve([item for item, _ in batch])
                for (_, future), answer in zip(batch, answers, strict=True):
                    if future.done():
                        continue
                    if isinstance(answer, BaseException):
                        future.set_exception(answer)
                    else:
                        future.set_result(answer)
            except Exception as exc:  # noqa: BLE001 - the callers of the batch receive it
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _gather(
        self,
    ) -> list[tuple[QueuedRequest, asyncio.Future[PolicyResponse]]]:
        """The next batch: the first request, then others until it is ready or ``max_wait_ms`` passed."""
        batcher = self.server.batcher
        loop = asyncio.get_running_loop()
        batch = [await self._queue.get()]
        deadline = loop.time() + batcher.max_wait_seconds
        while not batcher.full(len(batch)):
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0 or batcher.ready(
                [item for item, _ in batch], self._clients.connected
            ):
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                # Swallow the consumer's cancellation, not one of close() itself.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
        self.server.close()
