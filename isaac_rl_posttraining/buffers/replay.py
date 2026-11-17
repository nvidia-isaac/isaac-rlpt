# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One replay buffer organized by episode, and the batches learners draw from it.

A :class:`ReplayBuffer` stores :class:`~isaac_rl_posttraining.data.Transition`
records episode by episode: the decisions of an episode still running sit in
an open slot and are already sampleable, a finished episode is filed whole,
and eviction removes the oldest complete episode in one piece so an episode is
never left half present. Each episode's outcome is fixed when it completes and
the buffer keeps a per-episode index (lengths, offsets, outcomes) that filtered
draws work on, so a draw costs the batch, not the buffer. There is one buffer
type; a demonstration dataset is the same buffer loaded from disk. Learners never read the buffer directly:
they draw through a :class:`~isaac_rl_posttraining.buffers.samplers.Sampler`,
which asks the buffer for the indices it selects and for :func:`collate` of
the transitions behind them. The batch keeps the keys the algorithms read
(``obs``, ``action``, ``reward``, ``next_obs``, ``done``, ``truncated``,
``bootstrap_steps``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import torch

from isaac_rl_posttraining.data import (
    DynamicObservation,
    Episode,
    EpisodeMetadata,
    Transition,
)
from isaac_rl_posttraining.data.transitions import Origin

__all__ = [
    "BATCH_FIELDS",
    "Batch",
    "EpisodeOutcome",
    "ReplayBuffer",
    "RewardAggregation",
    "collate",
    "to_tensors",
]

#: A collated batch: stacked arrays under fixed keys, dict observations one level deep.
Batch = dict[str, Any]
#: How the per-step rewards of one decision become the one number a critic trains on.
RewardAggregation = Literal["sum", "last"]
EpisodeOutcome = Literal["ongoing", "success", "failure", "timeout"]
#: The batch keys every off-policy learner reads.
BATCH_FIELDS: tuple[str, ...] = ("obs", "action", "reward", "next_obs", "done", "truncated", "bootstrap_steps")


#: Outcome names as the small integers the episode index stores.
_OUTCOME_CODES: dict[str, int] = {"ongoing": 0, "success": 1, "failure": 2, "timeout": 3}


def _outcome_code(name: str) -> int:
    try:
        return _OUTCOME_CODES[name]
    except KeyError:
        raise ValueError(f"unknown episode outcome {name!r}; use one of {sorted(_OUTCOME_CODES)}") from None


class _StoredEpisode:
    """One episode's decisions in arrival order, with the bookkeeping the buffer indexes on."""

    __slots__ = ("complete", "id", "metadata", "outcome", "reward_stats", "succeeded", "transitions")

    def __init__(self, transition: Transition) -> None:
        self.id = transition.episode_id
        self.transitions: list[Transition] = []
        self.metadata = EpisodeMetadata(
            behavior_source=transition.behavior_source,
            seed=transition.metadata.reset_seed,
            policy_version_at_start=transition.metadata.policy_version,
        )
        self.complete = False
        self.succeeded = False
        #: ``ongoing`` until the ending decision arrives, then fixed for good.
        self.outcome: EpisodeOutcome = "ongoing"
        #: Per reward channel: running ``[count, mean, m2, min, max]`` of the decision rewards (Welford).
        self.reward_stats: dict[str, list[float]] = {}

    @property
    def next_step(self) -> int:
        last = self.transitions[-1]
        return last.metadata.step_in_episode + last.steps

    def append(self, transition: Transition, *, aggregation: RewardAggregation) -> None:
        self.transitions.append(transition)
        self.succeeded = self.succeeded or transition.succeeded
        if transition.ended:
            self.complete = True
            self.outcome = "success" if self.succeeded else transition.outcome
        for channel in transition.rewards:
            _welford_add(self.reward_stats, channel, transition.decision_reward(channel, aggregation=aggregation))

    def recount_rewards(self, channel: str, *, aggregation: RewardAggregation) -> None:
        """Rebuild one channel's running statistics after its rewards were rewritten."""
        self.reward_stats.pop(channel, None)
        for transition in self.transitions:
            if channel in transition.rewards:
                _welford_add(self.reward_stats, channel, transition.decision_reward(channel, aggregation=aggregation))

    def as_episode(self) -> Episode:
        return Episode(id=self.id, transitions=tuple(self.transitions), metadata=self.metadata)


def _welford_add(stats: dict[str, list[float]], channel: str, value: float) -> None:
    entry = stats.get(channel)
    if entry is None:
        stats[channel] = [1.0, value, 0.0, value, value]
        return
    entry[0] += 1.0
    delta = value - entry[1]
    entry[1] += delta / entry[0]
    entry[2] += delta * (value - entry[1])
    entry[3] = min(entry[3], value)
    entry[4] = max(entry[4], value)


def _welford_merge(into: list[float], other: list[float]) -> None:
    """Chan's combination of two running ``[count, mean, m2, min, max]`` records into ``into``."""
    count = into[0] + other[0]
    delta = other[1] - into[1]
    into[1] += delta * other[0] / count
    into[2] += other[2] + delta * delta * into[0] * other[0] / count
    into[0] = count
    into[3] = min(into[3], other[3])
    into[4] = max(into[4], other[4])


class _EpisodeIndex:
    """Per-episode arrays (length, flat offset, outcome, completeness) the filters and draws work on.

    Rebuilt lazily after the buffer changes and reused by every draw in
    between, so a training step of many updates pays for it once.
    """

    __slots__ = ("codes", "complete", "lengths", "offsets")

    def __init__(self, episodes: Sequence[_StoredEpisode]) -> None:
        count = len(episodes)
        self.lengths = np.fromiter((len(episode.transitions) for episode in episodes), dtype=np.int64, count=count)
        self.codes = np.fromiter((_OUTCOME_CODES[episode.outcome] for episode in episodes), dtype=np.int8, count=count)
        self.complete = np.fromiter((episode.complete for episode in episodes), dtype=bool, count=count)
        self.offsets = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(self.lengths, out=self.offsets[1:])

    def positions(
        self, *, outcome: str | None, exclude_outcome: str | None, complete: bool | None
    ) -> np.ndarray:
        """Positions (in storage order) of the episodes the filters keep."""
        mask = np.ones(len(self.lengths), dtype=bool)
        if complete is not None:
            mask &= self.complete == bool(complete)
        if outcome is not None:
            mask &= self.codes == _outcome_code(outcome)
        if exclude_outcome is not None:
            mask &= self.codes != _outcome_code(exclude_outcome)
        return np.flatnonzero(mask)

    def flat(self, positions: np.ndarray) -> np.ndarray:
        """Flat indices of every transition of the episodes at ``positions``, in storage order."""
        lengths = self.lengths[positions]
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        before = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        return np.repeat(self.offsets[positions] - before, lengths) + np.arange(total, dtype=np.int64)


class ReplayBuffer:
    """Episode-organized storage of typed transitions with whole-episode eviction.

    Args:
        capacity: Transitions kept; once exceeded, the oldest complete
            episodes are evicted whole. Episodes still running are never
            evicted, so the buffer may briefly exceed ``capacity`` by the
            length of the open episodes.
        drop_components: Observation components not stored (raw camera frames
            the learner never reads); the rest of the observation is kept as a
            :class:`~isaac_rl_posttraining.data.DynamicObservation`.
        reward_channel: The reward channel batches carry as ``reward``.
        reward_aggregation: How a decision's per-step rewards of that channel
            become its one ``reward``: their sum, or the last step's value.
    """

    def __init__(
        self,
        capacity: int,
        *,
        drop_components: Iterable[str] = (),
        reward_channel: str = "env",
        reward_aggregation: RewardAggregation = "last",
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if reward_aggregation not in ("sum", "last"):
            raise ValueError(f"reward_aggregation must be 'sum' or 'last', got {reward_aggregation!r}")
        if not isinstance(reward_channel, str) or not reward_channel:
            raise ValueError("reward_channel must be a non-empty string")
        self.capacity = capacity
        self.drop_components = tuple(drop_components)
        self.reward_channel = reward_channel
        self.reward_aggregation = reward_aggregation
        self._episodes: list[_StoredEpisode] = []
        self._open: dict[int | str, _StoredEpisode] = {}
        self._complete_ids: set[int | str] = set()
        self._size = 0
        self._index: _EpisodeIndex | None = None
        self._expert_transitions = 0

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """File one decision under its episode; the episode closes when the decision ends it."""
        if not isinstance(transition, Transition):
            raise TypeError(f"a replay buffer stores Transition records, got {type(transition).__name__}")
        transition = self._stored(transition)
        episode = self._open.get(transition.episode_id)
        if episode is None:
            if transition.episode_id in self._complete_ids:
                raise ValueError(
                    f"episode {transition.episode_id!r} is already complete "
                    "(data sources feeding one buffer need distinct source names)"
                )
            if transition.metadata.step_in_episode != 0:
                raise ValueError(
                    f"episode {transition.episode_id!r} starts at step {transition.metadata.step_in_episode}, expected 0"
                )
            episode = _StoredEpisode(transition)
            self._episodes.append(episode)
            self._open[episode.id] = episode
        elif transition.metadata.step_in_episode != episode.next_step:
            raise ValueError(
                f"episode {episode.id!r} continues at step {transition.metadata.step_in_episode}, "
                f"expected {episode.next_step}"
            )
        episode.append(transition, aggregation=self.reward_aggregation)
        self._size += 1
        if transition.behavior_source == "expert":
            self._expert_transitions += 1
        self._index = None
        if episode.complete:
            del self._open[episode.id]
            self._complete_ids.add(episode.id)
        self._evict()

    def _stored(self, transition: Transition) -> Transition:
        if not self.drop_components:
            return transition
        return replace(
            transition,
            observation=self._without_dropped(transition.observation.to_mapping()),
            next_observation=self._without_dropped(transition.next_observation.to_mapping()),
        )

    def _without_dropped(self, mapping: Mapping[str, Any]) -> DynamicObservation:
        return DynamicObservation.from_mapping(
            {name: value for name, value in mapping.items() if name not in self.drop_components}
        )

    def discard_open(self, episode_id: int | str) -> None:
        """Drop a running episode and its stored decisions (an episode the data source cut short)."""
        episode = self._open.pop(episode_id, None)
        if episode is not None:
            self._remove(self._episodes.index(episode))

    def _evict(self) -> None:
        while self._size > self.capacity:
            oldest = next((index for index, episode in enumerate(self._episodes) if episode.complete), None)
            if oldest is None:
                return
            self._complete_ids.discard(self._remove(oldest).id)

    def _remove(self, position: int) -> _StoredEpisode:
        episode = self._episodes.pop(position)
        self._size -= len(episode.transitions)
        self._expert_transitions -= sum(1 for transition in episode.transitions if transition.behavior_source == "expert")
        self._index = None
        return episode

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def clear(self) -> None:
        self._episodes.clear()
        self._open.clear()
        self._complete_ids.clear()
        self._size = 0
        self._index = None
        self._expert_transitions = 0

    def episodes(self, *, complete: bool | None = None) -> tuple[Episode, ...]:
        """The stored episodes in storage order; ``complete`` keeps only finished or only running ones."""
        return tuple(
            episode.as_episode()
            for episode in self._episodes
            if complete is None or episode.complete == complete
        )

    def transitions(self) -> list[Transition]:
        """Every stored transition, episode by episode in storage order."""
        return [transition for episode in self._episodes for transition in episode.transitions]

    def _episode_index(self) -> _EpisodeIndex:
        if self._index is None:
            self._index = _EpisodeIndex(self._episodes)
        return self._index

    def episode_offsets(self) -> np.ndarray:
        """Flat index at which each stored episode starts, plus the total as a final entry."""
        return self._episode_index().offsets

    def episode_lengths(self) -> np.ndarray:
        """Number of transitions of each stored episode, in storage order."""
        return self._episode_index().lengths

    def transition(self, index: int) -> Transition:
        """The transition at flat ``index`` of :meth:`transitions`."""
        if index < 0 or index >= self._size:
            raise IndexError(f"transition index {index} out of range for {self._size} transitions")
        offsets = self.episode_offsets()
        episode = int(np.searchsorted(offsets, index, side="right")) - 1
        return self._episodes[episode].transitions[index - int(offsets[episode])]

    def transitions_at(self, indices: Iterable[int]) -> list[Transition]:
        """The transitions at the flat ``indices``, in that order."""
        flat = np.asarray(list(indices), dtype=np.int64)
        if flat.size == 0:
            return []
        if flat.min() < 0 or flat.max() >= self._size:
            raise IndexError(f"transition index out of range for {self._size} transitions")
        offsets = self.episode_offsets()
        episodes = np.searchsorted(offsets, flat, side="right") - 1
        return [
            self._episodes[int(episode)].transitions[int(index - offsets[episode])]
            for episode, index in zip(episodes, flat, strict=True)
        ]

    def indices(
        self,
        *,
        outcome: EpisodeOutcome | None = None,
        exclude_outcome: EpisodeOutcome | None = None,
        complete: bool | None = None,
        source: str | None = None,
        origin: Origin | None = None,
    ) -> np.ndarray:
        """Flat indices of the transitions matching the filters, in storage order.

        ``outcome``, ``exclude_outcome`` and ``complete`` select episodes;
        ``source`` selects transitions by who produced them (the behavior
        source) and ``origin`` by where their dynamics came from.
        """
        index = self._episode_index()
        positions = index.positions(outcome=outcome, exclude_outcome=exclude_outcome, complete=complete)
        if source is None and origin is None:
            return index.flat(positions)
        matching: list[int] = []
        for position in positions.tolist():
            start = int(index.offsets[position])
            matching.extend(
                start + offset
                for offset, transition in enumerate(self._episodes[position].transitions)
                if (source is None or transition.behavior_source == source)
                and (origin is None or transition.origin == origin)
            )
        return np.asarray(matching, dtype=np.int64)

    def count(
        self,
        *,
        outcome: EpisodeOutcome | None = None,
        exclude_outcome: EpisodeOutcome | None = None,
        complete: bool | None = None,
        source: str | None = None,
        origin: Origin | None = None,
    ) -> int:
        """How many transitions :meth:`indices` would return for the same filters."""
        if source is not None or origin is not None:
            return len(
                self.indices(
                    outcome=outcome, exclude_outcome=exclude_outcome, complete=complete, source=source, origin=origin
                )
            )
        index = self._episode_index()
        positions = index.positions(outcome=outcome, exclude_outcome=exclude_outcome, complete=complete)
        return int(index.lengths[positions].sum())

    def draw(
        self,
        rng: np.random.Generator,
        count: int,
        *,
        outcome: EpisodeOutcome | None = None,
        exclude_outcome: EpisodeOutcome | None = None,
        complete: bool | None = None,
        source: str | None = None,
        origin: Origin | None = None,
    ) -> np.ndarray:
        """``count`` flat indices drawn uniformly over the transitions the filters keep.

        Without replacement while the selection is large enough, with
        replacement otherwise; identical in law and in random-stream use to
        ``rng.choice(self.indices(...), count, ...)`` without building that
        array, so the cost is the batch plus the number of stored episodes.
        Returns an empty array when nothing matches.
        """
        if source is not None or origin is not None:
            pool = self.indices(
                outcome=outcome, exclude_outcome=exclude_outcome, complete=complete, source=source, origin=origin
            )
            if len(pool) == 0:
                return pool
            return rng.choice(pool, size=count, replace=len(pool) < count)
        index = self._episode_index()
        positions = index.positions(outcome=outcome, exclude_outcome=exclude_outcome, complete=complete)
        lengths = index.lengths[positions]
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        picks = rng.choice(total, size=count, replace=total < count)
        ends = np.cumsum(lengths)
        which = np.searchsorted(ends, picks, side="right")
        within = picks - (ends[which] - lengths[which])
        return index.offsets[positions[which]] + within

    def collate(self, indices: Iterable[int]) -> Batch:
        """The learner batch of the transitions at ``indices``, in that order."""
        return collate(
            self.transitions_at(indices),
            reward_channel=self.reward_channel,
            reward_aggregation=self.reward_aggregation,
        )

    def get_statistics(self) -> dict[str, int]:
        """Episode and transition counts by outcome and source, for the training log."""
        codes = self._episode_index().codes
        counts = {"transitions": self._size, "episodes": len(self._episodes), "open_episodes": len(self._open)}
        for name in ("success", "failure", "timeout"):
            counts[f"{name}_episodes"] = int(np.count_nonzero(codes == _OUTCOME_CODES[name]))
        counts["expert_transitions"] = self._expert_transitions
        return counts

    def reward_statistics(self) -> dict[str, dict[str, float]]:
        """Per reward channel, ``mean``/``min``/``max``/``std`` of the decision rewards (under the buffer's aggregation).

        Combined from per-episode running records, so it costs the number of
        stored episodes, not of transitions; ``std`` is the population value
        :func:`numpy.std` would give.
        """
        merged: dict[str, list[float]] = {}
        for episode in self._episodes:
            for channel, record in episode.reward_stats.items():
                if channel in merged:
                    _welford_merge(merged[channel], record)
                else:
                    merged[channel] = list(record)
        return {
            channel: {"mean": mean, "min": low, "max": high, "std": math.sqrt(max(m2 / count, 0.0))}
            for channel, (count, mean, m2, low, high) in merged.items()
        }

    # ------------------------------------------------------------------
    # Relabeling
    # ------------------------------------------------------------------

    def relabel(self, channel: str, score: Callable[[Episode], Sequence[np.ndarray]]) -> int:
        """Score every complete episode and store the per-step rewards as ``channel``.

        ``score`` returns one array per transition of the episode, each of that
        transition's step count. The ``env`` channel and every other field stay
        as they are; the transitions are replaced by copies carrying the new
        channel. Returns the number of transitions relabeled.
        """
        if channel == "env":
            raise ValueError("the 'env' channel is the environment's own reward and is never rewritten")
        relabeled = 0
        for episode in self._episodes:
            if not episode.complete:
                continue
            rewards = list(score(episode.as_episode()))
            if len(rewards) != len(episode.transitions):
                raise ValueError(
                    f"score returned {len(rewards)} reward arrays for the {len(episode.transitions)} "
                    f"transitions of episode {episode.id!r}"
                )
            for position, (transition, values) in enumerate(zip(episode.transitions, rewards, strict=True)):
                relabeled_transition = replace(
                    transition, rewards={**transition.rewards, channel: np.asarray(values, dtype=np.float32)}
                )
                episode.transitions[position] = relabeled_transition
                relabeled += 1
            episode.recount_rewards(channel, aggregation=self.reward_aggregation)
        return relabeled


# ----------------------------------------------------------------------
# Batches
# ----------------------------------------------------------------------


def _learner_action(transition: Transition) -> np.ndarray:
    if transition.learner_action is not None:
        return transition.learner_action.value
    if transition.env_action is None:
        raise ValueError("a transition without a learner action needs its env_action to be batched")
    return transition.env_action.commands


def _stack(name: str, values: Sequence[np.ndarray]) -> np.ndarray:
    shapes = {tuple(np.shape(value)) for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{name} has mixed shapes across the batch: {sorted(shapes)[:10]}")
    return np.stack([np.asarray(value) for value in values])


def _stack_components(name: str, mappings: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(mappings[0])
    for mapping in mappings[1:]:
        if set(mapping) != keys:
            raise ValueError(f"{name} components differ across the batch: {sorted(keys)} vs {sorted(mapping)}")
    return {
        key: _as_batch_dtype(_stack(f"{name}[{key!r}]", [mapping[key] for mapping in mappings]))
        for key in mappings[0]
    }


def _as_batch_dtype(array: np.ndarray) -> np.ndarray:
    """Narrow a float component to float32, and leave any other dtype alone.

    Learners want one float width, so a float64 state is narrowed. A camera
    frame is not a float: an observation declares it ``Array[np.uint8, H, W,
    3]``, the simulator emits bytes, the launch plan checks that spec, and the
    policies that read frames reject anything else. Casting here would
    quadruple the size of every sampled batch and hand an image processor a
    float where it requires a byte -- with the batch no longer matching the
    spec the run was checked against.
    """
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    return array


def _legacy_info(transition: Transition) -> dict[str, Any]:
    info = dict(transition.metadata.extra)
    if transition.ended and "is_success" not in info and "success" not in info:
        info["is_success"] = transition.succeeded
    return info


def collate(
    transitions: Sequence[Transition],
    *,
    reward_channel: str = "env",
    reward_aggregation: RewardAggregation = "last",
) -> Batch:
    """Stack transitions into the batch the off-policy learners read.

    ``done`` is the terminal flag: a decision that both terminated and hit the
    time limit is terminal, as in the on-policy loop. An
    ``origin`` list is added only when some transition is not ``"real"``. Arrays
    are float32 (``bootstrap_steps`` int64), as the learners expect, except
    that an observation component keeps a non-float dtype of its own -- a
    uint8 camera frame stays uint8.
    """
    if not transitions:
        return {}
    terminated = np.asarray([transition.terminated[-1] for transition in transitions], dtype=bool)
    truncated = np.asarray([transition.truncated[-1] for transition in transitions], dtype=bool)
    rewards = [
        transition.decision_reward(reward_channel, aggregation=reward_aggregation) for transition in transitions
    ]
    batch: Batch = {
        "obs": _stack_components("obs", [transition.observation.arrays() for transition in transitions]),
        "action": _stack("action", [_learner_action(transition) for transition in transitions]).astype(
            np.float32, copy=False
        ),
        "reward": np.asarray(rewards, dtype=np.float32),
        "next_obs": _stack_components(
            "next_obs", [transition.next_observation.arrays() for transition in transitions]
        ),
        "done": terminated.astype(np.float32),
        "truncated": truncated.astype(np.float32),
        "bootstrap_steps": np.asarray([transition.steps for transition in transitions], dtype=np.int64),
        "reward_sequence": [transition.reward(reward_channel) for transition in transitions],
        "behavior_source": [transition.behavior_source for transition in transitions],
        "info": [_legacy_info(transition) for transition in transitions],
    }
    if any(transition.origin != "real" for transition in transitions):
        batch["origin"] = [transition.origin for transition in transitions]
    return batch


def to_tensors(batch: Batch, device: str | torch.device | None = None) -> Batch:
    """The batch's arrays as torch tensors, optionally on ``device``; lists stay lists."""
    converted: Batch = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            converted[key] = {name: _tensor(entry, device) for name, entry in value.items()}
        elif isinstance(value, np.ndarray) or torch.is_tensor(value):
            converted[key] = _tensor(value, device)
        else:
            converted[key] = value
    return converted


def _tensor(value: np.ndarray | torch.Tensor, device: str | torch.device | None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value))
    return tensor if device is None else tensor.to(device)
