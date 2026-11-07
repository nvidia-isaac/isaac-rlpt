# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Average replicated gradients across learner ranks in a few bucketed all-reduces per backward.

Every parameter registered with :func:`average_gradients_across_ranks` gets a
post-accumulate-grad hook that only records it. At the end of the backward pass
the gradients it produced are flattened, per device and dtype, into buckets of at
most :data:`BUCKET_BYTES`, each bucket is summed with one ``all_reduce`` and
divided by the world size, and the result is copied back. Averaging stays
before gradient clipping and ``optimizer.step()``, as with one all-reduce per
tensor. The reduction is elementwise, so on gloo with two ranks the averaged
gradients are bit-identical to per-tensor all-reduces; with more ranks, or on
NCCL, the backend may sum a bucket in a different order than a lone tensor, so
the last bits can differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Largest flattened bucket per all-reduce.
BUCKET_BYTES = 64 * 1024 * 1024

#: The bucket position of a registered parameter (the same on every rank).
_ORDER = "_rlpt_gradient_bucket_order"


class _BackwardGradientAverager:
    """Collects the registered gradients of one backward and averages them when it ends."""

    def __init__(self) -> None:
        self._registered = 0
        self._pending: dict[int, Any] = {}
        self._graph_task: int | None = None

    def register(self, parameter: Any) -> None:
        if not parameter.requires_grad or hasattr(parameter, _ORDER):
            return
        setattr(parameter, _ORDER, self._registered)
        self._registered += 1
        parameter.register_post_accumulate_grad_hook(self._record)

    def _record(self, parameter: Any) -> None:
        import torch

        graph_task = torch._C._current_graph_task_id()
        if graph_task != self._graph_task:
            # A new backward; anything left over belongs to one that raised.
            self._pending.clear()
            self._graph_task = graph_task
            torch.autograd.Variable._execution_engine.queue_callback(self._flush)
        self._pending[getattr(parameter, _ORDER)] = parameter

    def _flush(self) -> None:
        parameters = [self._pending[order] for order in sorted(self._pending)]
        self._pending.clear()
        self._graph_task = None
        for grads in _buckets([p.grad for p in parameters if p.grad is not None]):
            _average(grads)


def _buckets(grads: Sequence[Any]) -> list[list[Any]]:
    """``grads`` in order, split by device and dtype and capped at :data:`BUCKET_BYTES`."""
    open_buckets: dict[tuple[Any, Any], tuple[list[Any], int]] = {}
    buckets: list[list[Any]] = []
    for grad in grads:
        key = (grad.device, grad.dtype)
        size = grad.numel() * grad.element_size()
        bucket, used = open_buckets.get(key, ([], 0))
        if bucket and used + size > BUCKET_BYTES:
            buckets.append(bucket)
            bucket, used = [], 0
        bucket.append(grad)
        open_buckets[key] = (bucket, used + size)
    buckets.extend(bucket for bucket, _ in open_buckets.values())
    return buckets


def _average(grads: Sequence[Any]) -> None:
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    flat = _flatten_dense_tensors(list(grads))
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    for grad, averaged in zip(
        grads, _unflatten_dense_tensors(flat, list(grads)), strict=True
    ):
        grad.copy_(averaged)


_AVERAGER = _BackwardGradientAverager()


def average_gradients_across_ranks(parameters: Sequence[Any]) -> None:
    """Average the gradients of ``parameters`` over the default process group after each backward.

    All registered parameters share one set of buckets, so a backward that reaches
    several modules (or a learned temperature) still costs one all-reduce per
    bucket. A parameter is averaged once however many callers ask. Every rank must
    register the same parameters in the same order. A no-op without an initialised
    ``torch.distributed`` group of more than one rank.
    """
    import torch.distributed as dist

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        return
    for parameter in parameters:
        _AVERAGER.register(parameter)
