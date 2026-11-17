# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save and load replay buffers as a directory of arrays and JSON.

A saved buffer is a directory rather than one pickle, so it can be read
without the code that wrote it::

    <buffer>/
      schema.json      # schema version, what the arrays are, the buffer's read settings
      metadata.json    # one entry per episode and per transition: ids, sources, metadata, text
      episodes.npz     # every array field over all transitions, episode by episode

Arrays that only some transitions carry (a learner action, a relabeled reward
channel, a policy output) are stored with a presence mask, so a buffer with
an episode still running or a channel added later saves like any other.
A buffer streamed while it was collected is a directory of such buffers,
``part-00000/``, ``part-00001/``, ...; :func:`load_buffer` reads either.
``schema_version`` lets a later format read an older one deliberately:
optional fields a newer writer adds are read with defaults when a file lacks
them.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from isaac_rl_posttraining.buffers.replay import ReplayBuffer, RewardAggregation
from isaac_rl_posttraining.data import (
    DynamicObservation,
    EnvAction,
    Intervention,
    LearnerAction,
    PolicyOutputs,
    Transition,
    TransitionMetadata,
)

__all__ = [
    "SCHEMA_FILE",
    "SCHEMA_VERSION",
    "buffer_parts",
    "load_buffer",
    "read_schema",
    "save_buffer",
    "save_buffer_part",
]

SCHEMA_VERSION = 1
SCHEMA_FILE = "schema.json"
METADATA_FILE = "metadata.json"
ARRAYS_FILE = "episodes.npz"
FORMAT = "isaac_rl_posttraining.replay"
#: Directory name prefix of the parts a streaming writer appends under one root.
PART_PREFIX = "part-"

_INTERVENTION_MASKS = (
    "policy_tick_mask",
    "expert_tick_mask",
    "pre_intervention_tick_mask",
    "post_intervention_tick_mask",
)
_METADATA_FIELDS = (
    "step_in_episode",
    "decision_index",
    "policy_version",
    "environment_index",
    "reset_seed",
    "reset_seed_scope",
    "wall_time",
    "max_steps_in_episode",
    "source",
    "worker_index",
    "policy_lag",
)


class _Arrays:
    """Collects the per-transition arrays of a buffer, stacked or concatenated, with presence masks."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.stored: dict[str, np.ndarray] = {}
        self.stringified_extra = 0

    def stack(self, name: str, values: Sequence[np.ndarray | None]) -> None:
        """Stack equally shaped per-transition arrays; ``None`` entries are recorded in ``<name>/mask``."""
        present = [value for value in values if value is not None]
        if not present:
            return
        shapes = {tuple(np.shape(value)) for value in present}
        if len(shapes) != 1:
            raise ValueError(f"{name} must have one shape across the buffer to be saved, got {sorted(shapes)[:6]}")
        self.stored[f"{name}/values"] = np.stack([np.asarray(value) for value in present])
        self.stored[f"{name}/mask"] = np.asarray([value is not None for value in values], dtype=bool)

    def concatenate(self, name: str, values: Sequence[np.ndarray | None]) -> None:
        """Concatenate per-step arrays along their first axis; ``None`` entries are recorded in ``<name>/mask``."""
        present = [np.asarray(value) for value in values if value is not None]
        if not present:
            return
        self.stored[f"{name}/values"] = np.concatenate(present, axis=0)
        self.stored[f"{name}/mask"] = np.asarray([value is not None for value in values], dtype=bool)


def _json_safe(value: Any, counter: list[int]) -> Any:
    """A JSON-serializable copy of an ``info`` value; arrays keep their dtype, the rest its repr."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": value.dtype.name}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(entry, counter) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry, counter) for entry in value]
    counter[0] += 1
    return repr(value)


def _from_json(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__array__", "dtype"}:
            return np.asarray(value["__array__"], dtype=value["dtype"])
        return {key: _from_json(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_from_json(entry) for entry in value]
    return value


def _collect_arrays(transitions: Sequence[Transition]) -> _Arrays:
    """Every per-transition array field of ``transitions``, with presence masks."""
    arrays = _Arrays(len(transitions))
    _collect_observation_arrays(arrays, transitions)
    _collect_action_and_reward_arrays(arrays, transitions)
    _collect_policy_output_arrays(arrays, transitions)
    _collect_intervention_arrays(arrays, transitions)
    return arrays


def _collect_observation_arrays(arrays: _Arrays, transitions: Sequence[Transition]) -> None:
    for side in ("observation", "next_observation"):
        components = sorted({name for t in transitions for name in getattr(t, side).arrays()})
        for name in components:
            arrays.stack(f"{side}/{name}", [getattr(t, side).arrays().get(name) for t in transitions])


def _collect_action_and_reward_arrays(arrays: _Arrays, transitions: Sequence[Transition]) -> None:
    arrays.stack(
        "learner_action/value", [None if t.learner_action is None else t.learner_action.value for t in transitions]
    )
    arrays.concatenate(
        "env_action/commands", [None if t.env_action is None else t.env_action.commands for t in transitions]
    )
    for channel in sorted({name for t in transitions for name in t.rewards}):
        arrays.concatenate(f"rewards/{channel}", [t.rewards.get(channel) for t in transitions])
    for flag in ("terminated", "truncated", "success"):
        arrays.stored[flag] = np.concatenate([getattr(t, flag) for t in transitions])
    arrays.stored["steps"] = np.asarray([t.steps for t in transitions], dtype=np.int64)


def _collect_intervention_arrays(arrays: _Arrays, transitions: Sequence[Transition]) -> None:
    for name in _INTERVENTION_MASKS:
        arrays.concatenate(
            f"intervention/{name}",
            [None if t.metadata.intervention is None else getattr(t.metadata.intervention, name) for t in transitions],
        )


def _collect_policy_output_arrays(arrays: _Arrays, transitions: Sequence[Transition]) -> None:
    for name in ("log_prob", "value", "next_value", "entropy", "action_mean", "action_std", "denoising_log_prob"):
        arrays.stack(f"policy_outputs/{name}", [getattr(t.policy_outputs, name) for t in transitions])
    feature_keys = sorted(
        {key for t in transitions if t.policy_outputs.features is not None for key in t.policy_outputs.features}
    )
    for key in feature_keys:
        arrays.stack(
            f"policy_outputs/features/{key}",
            [None if t.policy_outputs.features is None else t.policy_outputs.features.get(key) for t in transitions],
        )


def _transition_record(transition: Transition, stringified: list[int]) -> dict[str, Any]:
    """The JSON metadata entry of one transition."""
    metadata = transition.metadata
    return {
        "episode_id": transition.episode_id,
        "behavior_source": transition.behavior_source,
        "origin": transition.origin,
        "learner_action_kind": None if transition.learner_action is None else transition.learner_action.kind,
        "execution_length": None if transition.env_action is None else transition.env_action.execution_length,
        "metadata": {name: getattr(metadata, name) for name in _METADATA_FIELDS},
        "intervention_phase": None if metadata.intervention is None else metadata.intervention.phase,
        "text": transition.observation.text(),
        "next_text": transition.next_observation.text(),
        "extra": _json_safe(metadata.extra, stringified),
    }


def save_buffer(buffer: ReplayBuffer, path: str | os.PathLike[str]) -> None:
    """Write ``buffer`` as directory ``path``, episode by episode in storage order.

    The buffer is written to a hidden sibling first and renamed into place, so
    a crash never leaves a half-written buffer and an earlier save at ``path``
    survives until the new one is complete.
    """
    transitions = buffer.transitions()
    if not transitions:
        raise ValueError("cannot save an empty replay buffer")
    arrays = _collect_arrays(transitions)
    stringified = [0]

    episodes = [
        {
            "id": episode.id,
            "complete": episode.complete,
            "transitions": len(episode.transitions),
            "metadata": asdict(episode.metadata),
        }
        for episode in buffer.episodes()
    ]
    records = [_transition_record(transition, stringified) for transition in transitions]
    schema = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "capacity": buffer.capacity,
        "drop_components": list(buffer.drop_components),
        "reward_channel": buffer.reward_channel,
        "reward_aggregation": buffer.reward_aggregation,
        "reward_channels": sorted({name for t in transitions for name in t.rewards}),
        "observation": transitions[0].observation.spec().to_json(),
        "episodes": len(episodes),
        "transitions": len(transitions),
        "extra_values_stringified": stringified[0],
    }

    parent, name = os.path.split(os.path.abspath(path))
    staging = os.path.join(parent, f".{name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    with open(os.path.join(staging, SCHEMA_FILE), "w", encoding="utf-8") as stream:
        json.dump(schema, stream, indent=2)
    with open(os.path.join(staging, METADATA_FILE), "w", encoding="utf-8") as stream:
        json.dump({"episodes": episodes, "transitions": records}, stream)
    np.savez(os.path.join(staging, ARRAYS_FILE), **arrays.stored)
    previous = os.path.join(parent, f".{name}.old")
    if os.path.isfile(os.path.join(path, SCHEMA_FILE)):  # only ever replace an earlier saved buffer
        shutil.rmtree(previous, ignore_errors=True)
        os.replace(path, previous)
    try:
        os.replace(staging, path)
    except OSError:
        if os.path.isdir(previous):
            os.replace(previous, path)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def read_schema(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The ``schema.json`` of a saved buffer, checked to be one this reader understands."""
    if os.path.isfile(path):
        raise ValueError(
            f"{path} is a file, but a saved replay buffer is a directory holding {SCHEMA_FILE}, "
            f"{METADATA_FILE} and {ARRAYS_FILE}; legacy single-file .npz replay files are not supported "
            "(re-save the buffer with save_buffer from a run on this version, or rebuild its transitions "
            "with Transition.from_legacy_fields and save those)"
        )
    if not os.path.isfile(os.path.join(path, SCHEMA_FILE)):
        raise FileNotFoundError(
            f"{path} holds no {SCHEMA_FILE} and no {PART_PREFIX}NNNNN buffers; it is not a saved replay buffer directory"
        )
    with open(os.path.join(path, SCHEMA_FILE), encoding="utf-8") as stream:
        schema = json.load(stream)
    if schema.get("format") != FORMAT:
        raise ValueError(f"{path} is not a saved replay buffer (format {schema.get('format')!r})")
    version = schema.get("schema_version")
    if not isinstance(version, int) or version < 1 or version > SCHEMA_VERSION:
        raise ValueError(f"{path} has schema version {version!r}; this reader understands 1..{SCHEMA_VERSION}")
    return schema


class _Reader:
    """Reads the per-transition arrays back, honoring the presence masks."""

    def __init__(self, stored: Mapping[str, np.ndarray], steps: np.ndarray) -> None:
        self.stored = stored
        self.steps = steps
        self.offsets = np.cumsum(np.concatenate([[0], steps]))
        self._positions: dict[str, np.ndarray] = {}

    def names(self, prefix: str) -> list[str]:
        """The ``<prefix>/<name>`` entries stored, without the prefix and the ``/values`` suffix."""
        found = []
        for key in self.stored:
            if key.startswith(prefix + "/") and key.endswith("/values"):
                found.append(key[len(prefix) + 1 : -len("/values")])
        return sorted(found)

    def _position(self, name: str, index: int) -> int | None:
        """The row of transition ``index`` among the transitions carrying ``name``, or None."""
        mask_key = f"{name}/mask"
        if mask_key not in self.stored:
            return None
        if name not in self._positions:
            self._positions[name] = np.cumsum(self.stored[mask_key]) - 1
        if not self.stored[mask_key][index]:
            return None
        return int(self._positions[name][index])

    def stacked(self, name: str, index: int) -> np.ndarray | None:
        position = self._position(name, index)
        return None if position is None else self.stored[f"{name}/values"][position]

    def concatenated(self, name: str, index: int) -> np.ndarray | None:
        """The per-step slice of transition ``index`` from a concatenated array."""
        position = self._position(name, index)
        if position is None:
            return None
        mask = self.stored[f"{name}/mask"]
        start = int(self.steps[:index][mask[:index]].sum())
        return self.stored[f"{name}/values"][start : start + int(self.steps[index])]

    def per_step(self, name: str, index: int) -> np.ndarray:
        return self.stored[name][int(self.offsets[index]) : int(self.offsets[index + 1])]


def buffer_parts(path: str | os.PathLike[str]) -> list[str]:
    """The saved buffers under ``path`` in order: ``path`` itself, or its ``part-*`` directories."""
    if os.path.isfile(os.path.join(path, SCHEMA_FILE)) or not os.path.isdir(path):
        return [os.fspath(path)]
    parts = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.startswith(PART_PREFIX) and os.path.isfile(os.path.join(path, name, SCHEMA_FILE))
    )
    return parts or [os.fspath(path)]


def save_buffer_part(buffer: ReplayBuffer, root: str | os.PathLike[str]) -> str:
    """Write ``buffer`` as the next ``part-NNNNN`` directory under ``root``; returns its path.

    A reader (or a crash) never sees a half-written part (see :func:`save_buffer`).
    """
    os.makedirs(root, exist_ok=True)
    index = sum(1 for name in os.listdir(root) if name.startswith(PART_PREFIX))
    final = os.path.join(root, f"{PART_PREFIX}{index:05d}")
    save_buffer(buffer, final)
    return final


def _read_part(path: str) -> tuple[dict[str, Any], list[Transition]]:
    schema = read_schema(path)
    with open(os.path.join(path, METADATA_FILE), encoding="utf-8") as stream:
        metadata = json.load(stream)
    with np.load(os.path.join(path, ARRAYS_FILE)) as archive:
        stored = {key: archive[key] for key in archive.files}
    records = metadata["transitions"]
    reader = _Reader(stored, stored["steps"])
    if len(records) != len(reader.steps):
        raise ValueError(f"{path}: {len(records)} transition records but {len(reader.steps)} step counts")
    return schema, [_transition(reader, index, record) for index, record in enumerate(records)]


def load_buffer(
    path: str | os.PathLike[str],
    *,
    capacity: int | None = None,
    reward_channel: str | None = None,
    reward_aggregation: RewardAggregation | None = None,
) -> ReplayBuffer:
    """Rebuild a saved buffer, validating every transition on the way in.

    ``path`` is one saved buffer or a directory of ``part-*`` buffers (what a
    data source's persistence streams); parts are read in order into one
    buffer whose default capacity holds all of them. The read settings come
    from the (first) file unless given; a demonstration buffer is usually
    loaded with the learner's own ``reward_channel`` and
    ``reward_aggregation`` so its batches read like the online ones.
    """
    parts = buffer_parts(path)
    schema = read_schema(parts[0])
    available: list[str] | None
    if len(parts) > 1:
        available = sorted({name for part in parts for name in read_schema(part).get("reward_channels") or ()})
    else:
        available = schema.get("reward_channels")
    if reward_channel is not None and available is not None and reward_channel not in available:
        raise ValueError(
            f"{path} carries no reward channel {reward_channel!r} (its channels: {sorted(available)}); "
            "a demonstration buffer must have been recorded with the learner's reward channel"
        )
    loaded = [_read_part(part)[1] for part in parts]
    default_capacity = schema["capacity"] if len(parts) == 1 else max(1, sum(len(part) for part in loaded))
    buffer = ReplayBuffer(
        default_capacity if capacity is None else capacity,
        drop_components=schema.get("drop_components", ()),
        reward_channel=schema["reward_channel"] if reward_channel is None else reward_channel,
        reward_aggregation=schema["reward_aggregation"] if reward_aggregation is None else reward_aggregation,
    )
    for transitions in loaded:
        for transition in transitions:
            buffer.add(transition)
    return buffer


def _observation(reader: _Reader, side: str, index: int, text: Mapping[str, str]) -> DynamicObservation:
    mapping: dict[str, Any] = dict(text)
    for name in reader.names(side):
        value = reader.stacked(f"{side}/{name}", index)
        if value is not None:
            mapping[name] = value
    return DynamicObservation.from_mapping(mapping)


def _transition(reader: _Reader, index: int, record: Mapping[str, Any]) -> Transition:
    learner_value = reader.stacked("learner_action/value", index)
    learner_action = None if learner_value is None else LearnerAction(record["learner_action_kind"], learner_value)
    commands = reader.concatenated("env_action/commands", index)
    env_action = None if commands is None else EnvAction(commands=commands, execution_length=record["execution_length"])
    rewards = {}
    for channel in reader.names("rewards"):
        values = reader.concatenated(f"rewards/{channel}", index)
        if values is not None:
            rewards[channel] = values
    outputs: dict[str, Any] = {}
    for name in ("log_prob", "value", "next_value", "entropy", "action_mean", "action_std", "denoising_log_prob"):
        value = reader.stacked(f"policy_outputs/{name}", index)
        if value is not None:
            outputs[name] = value
    features = {}
    for key in reader.names("policy_outputs/features"):
        value = reader.stacked(f"policy_outputs/features/{key}", index)
        if value is not None:
            features[key] = value
    if features:
        outputs["features"] = features
    intervention = None
    if record.get("intervention_phase") is not None:
        intervention = Intervention(
            phase=record["intervention_phase"],
            **{name: reader.concatenated(f"intervention/{name}", index) for name in _INTERVENTION_MASKS},
        )
    fields = dict(record["metadata"])
    # Fields a later schema added read as their defaults from older files.
    metadata = TransitionMetadata(
        step_in_episode=int(fields["step_in_episode"]),
        decision_index=fields.get("decision_index"),
        policy_version=fields.get("policy_version"),
        environment_index=fields.get("environment_index"),
        reset_seed=fields.get("reset_seed"),
        reset_seed_scope=fields.get("reset_seed_scope"),
        wall_time=fields.get("wall_time"),
        max_steps_in_episode=fields.get("max_steps_in_episode"),
        intervention=intervention,
        extra=_from_json(record.get("extra", {})),
        source=fields.get("source"),
        worker_index=fields.get("worker_index"),
        policy_lag=fields.get("policy_lag"),
    )
    return Transition(
        episode_id=record["episode_id"],
        observation=_observation(reader, "observation", index, record.get("text", {})),
        next_observation=_observation(reader, "next_observation", index, record.get("next_text", {})),
        learner_action=learner_action,
        env_action=env_action,
        rewards=rewards,
        terminated=reader.per_step("terminated", index),
        truncated=reader.per_step("truncated", index),
        success=reader.per_step("success", index),
        behavior_source=record["behavior_source"],
        metadata=metadata,
        policy_outputs=PolicyOutputs(**outputs),
        origin=record.get("origin", "real"),
    )
