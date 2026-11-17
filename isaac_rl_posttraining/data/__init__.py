# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The records every layer exchanges, and the schema each record carries.

A simulator emits an :class:`Observation`, a policy turns it into a
:class:`LearnerAction` and an :class:`EnvAction`, a rollout worker files the
executed steps as a :class:`Transition`, and complete :class:`Episode` objects
fill the replay buffer. Every record is a frozen dataclass that validates
itself on construction; an observation type is declared by annotating its
array fields (:class:`Array`, :class:`Text`) and inheriting from
:class:`Observation`, which makes its :class:`~isaac_rl_posttraining.data.spec.ObservationSpec`
the contract producers and consumers are checked against. This package depends
on NumPy only, so the env, policy, buffer and runtime layers can all import it.
"""

from isaac_rl_posttraining.data.records import (
    Array,
    ArrayMap,
    DynamicObservation,
    Observation,
    Record,
    Text,
    dim,
)
from isaac_rl_posttraining.data.spec import (
    ActionSpec,
    ArraySpec,
    DimSize,
    ObservationSpec,
    SpecComparison,
    SpecMismatch,
    compare_action_specs,
    compare_observation_specs,
)
from isaac_rl_posttraining.data.transitions import (
    ACTION_KINDS,
    BEHAVIOR_SOURCES,
    RESET_SEED_SCOPES,
    ActionKind,
    BehaviorSource,
    EnvAction,
    Episode,
    EpisodeMetadata,
    Intervention,
    LearnerAction,
    PolicyOutputs,
    ResetSeedScope,
    Transition,
    TransitionMetadata,
    episode_success,
)

__all__ = [
    "ACTION_KINDS",
    "BEHAVIOR_SOURCES",
    "RESET_SEED_SCOPES",
    "ActionKind",
    "ActionSpec",
    "Array",
    "ArrayMap",
    "ArraySpec",
    "BehaviorSource",
    "DimSize",
    "DynamicObservation",
    "EnvAction",
    "Episode",
    "EpisodeMetadata",
    "Intervention",
    "LearnerAction",
    "Observation",
    "ObservationSpec",
    "PolicyOutputs",
    "Record",
    "ResetSeedScope",
    "SpecComparison",
    "SpecMismatch",
    "Text",
    "Transition",
    "TransitionMetadata",
    "compare_action_specs",
    "compare_observation_specs",
    "dim",
    "episode_success",
]
