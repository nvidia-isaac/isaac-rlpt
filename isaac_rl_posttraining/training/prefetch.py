# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw the next learner batch on a helper thread while the learner trains on the current one."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from isaac_rl_posttraining.runtimes.learner import LearnerBatch, LearnerGroup
from isaac_rl_posttraining.training.on_policy_metrics import rank_mean_metrics

__all__ = ["learner_updates", "prefetched"]

T = TypeVar("T")


def prefetched(draw: Callable[[], T], count: int) -> Iterator[T]:
    """Yield ``count`` results of ``draw()``, computing each next one while the caller uses the current one.

    The draws run one at a time and in order on a single helper thread, so a
    sampler with its own random generator consumes it exactly as a plain loop
    would. No draw is started beyond ``count``: the caller may change what
    ``draw`` reads (for example, add to the replay buffer) once the loop ends.
    The caller must not change it between draws inside the loop.
    """
    if count <= 0:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="learner-batch") as pool:
        pending = pool.submit(draw)
        for index in range(count):
            current = pending.result()
            if index + 1 < count:
                pending = pool.submit(draw)
            yield current


def learner_updates(
    learner_group: LearnerGroup, draw_batch: Callable[[], LearnerBatch], env_steps: int, count: int
) -> Iterator[Mapping[str, float]]:
    """Run ``count`` learner updates on replay draws, yielding each update's metrics averaged over ranks."""
    num_ranks = int(getattr(learner_group, "num_learner_ranks", 1))
    # The buffer is fixed during a burst, so the next draw overlaps this update.
    for parts in prefetched(lambda: draw_batch().at_env_steps(env_steps).split(num_ranks), count):
        yield rank_mean_metrics(learner_group.train_step(parts))
