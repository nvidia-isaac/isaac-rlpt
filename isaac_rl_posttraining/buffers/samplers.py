# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Samplers: how a learner draws batches from replay buffers.

A learner receives a :class:`Sampler`, never a buffer layout. The library
starts small: :class:`UniformSampler` draws over every stored transition (or
over the transitions a filter selects), :class:`SuccessFractionSampler` draws
a fixed share of every batch from successful episodes,
:class:`MixedSampler` draws fixed shares from several buffers or samplers
(demonstrations next to the robot's own experience), and
:class:`SequenceSampler` draws consecutive decisions of one episode for n-step
or recurrent critics. Shares are allocated the same way everywhere: each
member takes the floor of its share and the last one takes the rest, so a
batch is always full; members that cannot supply anything hand their share to
the others in proportion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from isaac_rl_posttraining.buffers.replay import Batch, ReplayBuffer

__all__ = [
    "MixedSampler",
    "Sampler",
    "SequenceSampler",
    "SuccessFractionSampler",
    "UniformSampler",
    "allocate_shares",
    "renormalized",
]


@runtime_checkable
class Sampler(Protocol):
    """Draw learner batches; ``can_sample`` says whether a draw would return anything."""

    def sample(self, batch_size: int) -> Batch: ...

    def can_sample(self, batch_size: int) -> bool: ...


def allocate_shares(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Split ``total`` samples over members: floor of each share, the last member takes the rest."""
    names = list(fractions)
    if not names:
        return {}
    counts: dict[str, int] = {}
    for name in names[:-1]:
        # A hair of slack keeps 100 * 0.29 from flooring to 28.
        counts[name] = math.floor(total * fractions[name] + 1e-9)
    counts[names[-1]] = total - sum(counts.values())
    return counts


def renormalized(fractions: Mapping[str, float], available: Iterable[str]) -> dict[str, float]:
    """The fractions of the available members, rescaled to sum to one (equal when they were all zero)."""
    keep = set(available)
    chosen = [name for name in fractions if name in keep]
    if not chosen:
        return {}
    total = sum(fractions[name] for name in chosen)
    if total <= 0.0:
        return {name: 1.0 / len(chosen) for name in chosen}
    return {name: fractions[name] / total for name in chosen}


def _validated_fractions(names: list[str], fractions: Mapping[str, float] | None) -> dict[str, float]:
    if fractions is None:
        return {name: 1.0 / len(names) for name in names}
    if set(fractions) != set(names):
        raise ValueError(f"fractions must name exactly {names}, got {sorted(fractions)}")
    values = {name: float(fractions[name]) for name in names}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("fractions must be non-negative")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must sum to 1, got {sum(values.values()):.6f}")
    return values


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    # Seeded from NumPy's global stream, which seed_everything controls, so an
    # unseeded sampler is still reproducible run to run.
    return rng if rng is not None else np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))  # NOSONAR - draws from the globally seeded NumPy stream


def _check_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")


def _draw(rng: np.random.Generator, pool: np.ndarray, count: int) -> np.ndarray:
    """``count`` indices from ``pool``: without replacement while the pool is large enough."""
    return rng.choice(pool, size=count, replace=len(pool) < count)


def _concatenate(batches: list[Batch]) -> Batch:
    if len(batches) == 1:
        return batches[0]
    if any("origin" in batch for batch in batches):
        # Members without imagined data omit the column; their rows are real.
        batches = [
            batch if "origin" in batch else {**batch, "origin": ["real"] * len(batch["reward"])} for batch in batches
        ]
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches[1:]):
        raise ValueError("samplers returned batches with different keys")
    joined: Batch = {}
    for key in batches[0]:
        first = batches[0][key]
        if isinstance(first, Mapping):
            joined[key] = {name: np.concatenate([batch[key][name] for batch in batches]) for name in first}
        elif isinstance(first, np.ndarray):
            joined[key] = np.concatenate([batch[key] for batch in batches])
        else:
            joined[key] = [entry for batch in batches for entry in batch[key]]
    return joined


class UniformSampler:
    """Uniform draws over the buffer's transitions, or over the ones a filter selects.

    ``select`` are the keyword filters of :meth:`ReplayBuffer.indices`, for
    example ``outcome="success"`` for the successful complete episodes only.
    """

    def __init__(self, buffer: ReplayBuffer, *, rng: np.random.Generator | None = None, **select: Any) -> None:
        self.buffer = buffer
        self.rng = _generator(rng)
        self.select = dict(select)

    def pool(self) -> np.ndarray:
        """The flat indices this sampler draws over (materialized; for inspection and tests)."""
        return self.buffer.indices(**self.select)

    def can_sample(self, batch_size: int) -> bool:
        return self.buffer.count(**self.select) > 0

    def sample(self, batch_size: int) -> Batch:
        _check_batch_size(batch_size)
        picked = self.buffer.draw(self.rng, batch_size, **self.select)
        if len(picked) == 0:
            return {}
        return self.buffer.collate(picked)


class MixedSampler:
    """Fixed shares of every batch from several members, renormalized over the ones that can supply.

    Members are samplers or buffers (a buffer becomes a uniform sampler). With
    ``fractions=None`` every member takes the same share.
    """

    def __init__(
        self,
        members: Mapping[str, Sampler | ReplayBuffer],
        fractions: Mapping[str, float] | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not members:
            raise ValueError("a mixed sampler needs at least one member")
        self.rng = _generator(rng)
        self.members: dict[str, Sampler] = {}
        for name, member in members.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"member names must be non-empty strings, got {name!r}")
            if isinstance(member, ReplayBuffer):
                member = UniformSampler(member, rng=self.rng)
            elif not isinstance(member, Sampler):
                raise TypeError(f"member {name!r} must be a sampler or a replay buffer, got {type(member).__name__}")
            self.members[name] = member
        self.fractions = _validated_fractions(list(self.members), fractions)

    def shares(self, available: Iterable[str] | None = None) -> dict[str, float]:
        """The fractions of the ``available`` members, rescaled to sum to one (in declaration order)."""
        return renormalized(self.fractions, self.members if available is None else available)

    def allocate(self, batch_size: int) -> dict[str, int]:
        """Samples per member for one batch, over the members that can supply any."""
        available = [name for name, member in self.members.items() if member.can_sample(1)]
        counts = allocate_shares(batch_size, self.shares(available))
        return {name: count for name, count in counts.items() if count > 0}

    def can_sample(self, batch_size: int) -> bool:
        return any(member.can_sample(1) for member in self.members.values())

    def sample(self, batch_size: int) -> Batch:
        _check_batch_size(batch_size)
        counts = self.allocate(batch_size)
        if not counts:
            return {}
        batches: list[Batch] = []
        for name, count in counts.items():
            batch = self.members[name].sample(count)
            if not batch:
                raise RuntimeError(f"sampler {name!r} returned no batch for {count} samples")
            batches.append(batch)
        return _concatenate(batches)


class SuccessFractionSampler(MixedSampler):
    """A fixed share of every batch from successful episodes, the rest from the other finished ones.

    Only complete episodes take part, so an episode is scored once its outcome
    is known; when one side has nothing yet the other supplies the whole batch.
    """

    def __init__(self, buffer: ReplayBuffer, fraction: float, *, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")
        rng = _generator(rng)
        super().__init__(
            {
                "success": UniformSampler(buffer, rng=rng, outcome="success"),
                "other": UniformSampler(buffer, rng=rng, complete=True, exclude_outcome="success"),
            },
            {"success": float(fraction), "other": 1.0 - float(fraction)},
            rng=rng,
        )
        self.buffer = buffer
        self.fraction = float(fraction)


class SequenceSampler:
    """Windows of ``length`` consecutive decisions of one episode, stacked along a time axis.

    Every array of the batch gains a second axis of size ``length``; windows
    are drawn uniformly over all valid starts of all stored episodes.
    """

    def __init__(self, buffer: ReplayBuffer, length: int, *, rng: np.random.Generator | None = None) -> None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"length must be a positive integer, got {length!r}")
        self.buffer = buffer
        self.length = length
        self.rng = _generator(rng)

    def starts(self) -> np.ndarray:
        """Flat indices at which a full window fits inside one episode."""
        offsets = self.buffer.episode_offsets()
        fits = self.buffer.episode_lengths() - self.length + 1
        positions = np.flatnonzero(fits > 0)
        if len(positions) == 0:
            return np.zeros(0, dtype=np.int64)
        counts = fits[positions]
        before = np.concatenate(([0], np.cumsum(counts)[:-1]))
        return np.repeat(offsets[positions] - before, counts) + np.arange(int(counts.sum()), dtype=np.int64)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.starts()) > 0

    def sample(self, batch_size: int) -> Batch:
        _check_batch_size(batch_size)
        starts = self.starts()
        if len(starts) == 0:
            return {}
        windows = [
            self.buffer.collate(range(int(start), int(start) + self.length))
            for start in _draw(self.rng, starts, batch_size)
        ]
        stacked: Batch = {}
        for key, first in windows[0].items():
            if isinstance(first, Mapping):
                stacked[key] = {name: np.stack([window[key][name] for window in windows]) for name in first}
            elif isinstance(first, np.ndarray):
                stacked[key] = np.stack([window[key] for window in windows])
            else:
                stacked[key] = [window[key] for window in windows]
        return stacked
