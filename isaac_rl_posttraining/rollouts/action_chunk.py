# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-neutral action chunks produced by acting policies.

An RL post-training policy has two distinct action surfaces.
The learner chooses an action in its optimization space (for example a flow
latent, a joint flow/residual action, or a direct reference edit), while the
simulator executes a decoded sequence in the embodiment's action space.
Keeping both values in one record prevents a runtime backend from confusing
the action that belongs in replay with the actions sent to the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np

from isaac_rl_posttraining.data import ACTION_KINDS, ActionKind

ActionSource = Literal["policy", "warmup", "native", "expert"]

__all__ = ["ActionChunk", "ActionSource"]


@dataclass(frozen=True)
class ActionChunk:
    """One policy decision and the environment actions it decoded into.

    ``environment_actions`` always has shape ``[batch, horizon, action_dim]``.
    ``learner_action`` retains the policy's native geometry and only requires
    its leading dimension to match the environment-action batch;
    ``learner_action_kind`` says what it is (a steering latent, an edit, the
    chunk itself or a flow policy's denoising chain).
    It is ``None`` for behavior that has no trainable action, such as a native
    frozen-policy evaluation.

    ``execution_length`` is the receding-horizon prefix requested by the
    collector.
    A simulator may terminate before every requested row is executed; that
    realized count belongs on the resulting transition as ``bootstrap_steps``.

    ``learner_observation`` is the policy-side numeric representation used by
    the actor and critic, with the same leading batch dimension.
    This keeps simulator-only strings and raw policy images out of replay when
    serving computes embeddings outside the simulator worker.

    ``cached_features`` are the activations of the policy's frozen sub-modules
    at this decision (see ``policies/feature_cache.py``), with the same leading
    batch dimension; a worker stores them on the transition so the learner can
    skip the frozen forward pass.
    """

    learner_action: np.ndarray | None
    environment_actions: np.ndarray
    execution_length: int
    learner_action_kind: ActionKind | None = None
    source: ActionSource = "policy"
    policy_version: int = 0
    behavior_log_prob: np.ndarray | None = None
    behavior_value: np.ndarray | None = None
    learner_observation: Mapping[str, np.ndarray] | None = None
    cached_features: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        environment_actions = np.asarray(self.environment_actions)
        if environment_actions.ndim != 3:
            raise ValueError(
                "environment_actions must have shape [batch, horizon, action_dim], "
                f"got {environment_actions.shape}"
            )
        if any(width < 1 for width in environment_actions.shape):
            raise ValueError(
                "environment_actions dimensions must all be positive, "
                f"got {environment_actions.shape}"
            )
        if isinstance(self.execution_length, bool) or not isinstance(
            self.execution_length, int
        ):
            raise TypeError("execution_length must be an integer")
        if not 1 <= self.execution_length <= environment_actions.shape[1]:
            raise ValueError(
                f"execution_length must be in [1, {environment_actions.shape[1]}], "
                f"got {self.execution_length}"
            )
        if isinstance(self.policy_version, bool) or not isinstance(
            self.policy_version, int
        ):
            raise TypeError("policy_version must be an integer")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.source not in ("policy", "warmup", "native", "expert"):
            raise ValueError(f"unknown action source {self.source!r}")

        learner_action = self._checked_learner_action(environment_actions.shape[0])
        behavior_log_prob = self._batched_vector(
            self.behavior_log_prob,
            "behavior_log_prob",
            environment_actions.shape[0],
        )
        behavior_value = self._batched_vector(
            self.behavior_value, "behavior_value", environment_actions.shape[0]
        )

        learner_observation = self._batched_mapping(
            self.learner_observation,
            "learner_observation",
            environment_actions.shape[0],
        )
        cached_features = self._batched_mapping(
            self.cached_features, "cached_features", environment_actions.shape[0]
        )

        object.__setattr__(self, "environment_actions", environment_actions)
        object.__setattr__(self, "learner_action", learner_action)
        object.__setattr__(self, "behavior_log_prob", behavior_log_prob)
        object.__setattr__(self, "behavior_value", behavior_value)
        object.__setattr__(self, "learner_observation", learner_observation)
        object.__setattr__(self, "cached_features", cached_features)

    def _checked_learner_action(self, batch_size: int) -> np.ndarray | None:
        """Coerce the learner action and check its batch size and declared kind."""
        learner_action = self.learner_action
        if learner_action is not None:
            learner_action = np.asarray(learner_action)
            if learner_action.ndim < 1:
                raise ValueError("learner_action must carry a leading batch dimension")
            if learner_action.shape[0] != batch_size:
                raise ValueError(
                    "learner_action and environment_actions batch sizes differ: "
                    f"{learner_action.shape[0]} != {batch_size}"
                )
            if self.learner_action_kind not in ACTION_KINDS:
                raise ValueError(
                    f"a learner action needs its kind, one of {ACTION_KINDS}; got {self.learner_action_kind!r}"
                )
        elif self.learner_action_kind is not None:
            raise ValueError("learner_action_kind is given without a learner action")
        return learner_action

    @staticmethod
    def _batched_vector(value: Any, name: str, batch_size: int) -> np.ndarray | None:
        """Coerce a per-environment value (a scalar becomes ``[1]``) and check its batch size."""
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.shape[0] != batch_size:
            raise ValueError(
                f"{name} and environment_actions batch sizes differ: "
                f"{array.shape[0]} != {batch_size}"
            )
        return array

    @staticmethod
    def _batched_mapping(
        mapping: Mapping[str, Any] | None, name: str, batch_size: int
    ) -> dict[str, np.ndarray] | None:
        """Coerce a per-key array mapping and check every entry leads with the batch axis."""
        if mapping is None:
            return None
        arrays = {key: np.asarray(value) for key, value in mapping.items()}
        for key, value in arrays.items():
            if value.ndim < 1:
                raise ValueError(f"{name}[{key!r}] must carry a batch dimension")
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"{name}[{key!r}] and environment_actions batch sizes differ: "
                    f"{value.shape[0]} != {batch_size}"
                )
        return arrays

    @classmethod
    def from_arrays(
        cls,
        *,
        learner_action: np.ndarray | None,
        environment_actions: np.ndarray,
        execution_length: int,
        learner_action_kind: ActionKind | None = None,
        source: ActionSource = "policy",
        policy_version: int = 0,
        behavior_log_prob: np.ndarray | None = None,
        behavior_value: np.ndarray | None = None,
        learner_observation: Mapping[str, Any] | None = None,
        cached_features: Mapping[str, Any] | None = None,
    ) -> ActionChunk:
        """Build a chunk while accepting an unbatched ``[horizon, dim]`` decode."""
        decoded = np.asarray(environment_actions)
        if decoded.ndim == 2:
            decoded = decoded[None, ...]
        learned = None if learner_action is None else np.asarray(learner_action)
        if learned is not None and learned.ndim == 1:
            learned = learned[None, ...]
        return cls(
            learner_action=learned,
            environment_actions=decoded,
            execution_length=execution_length,
            learner_action_kind=learner_action_kind,
            source=source,
            policy_version=policy_version,
            behavior_log_prob=behavior_log_prob,
            behavior_value=behavior_value,
            learner_observation=learner_observation,
            cached_features=cached_features,
        )

    @property
    def batch_size(self) -> int:
        return int(self.environment_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.environment_actions.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.environment_actions.shape[2])

    @property
    def execution_actions(self) -> np.ndarray:
        """The requested receding-horizon prefix, shaped ``[B, K, A]``."""
        return self.environment_actions[:, : self.execution_length]

    def for_environment(self, index: int) -> ActionChunk:
        """Select one batch element while retaining a batch dimension."""
        if not 0 <= index < self.batch_size:
            raise IndexError(
                f"environment index {index} outside batch of {self.batch_size}"
            )
        learned = (
            None
            if self.learner_action is None
            else self.learner_action[index : index + 1]
        )
        log_prob = (
            None
            if self.behavior_log_prob is None
            else self.behavior_log_prob[index : index + 1]
        )
        value = (
            None
            if self.behavior_value is None
            else self.behavior_value[index : index + 1]
        )
        learner_observation = (
            None
            if self.learner_observation is None
            else {
                key: value[index : index + 1]
                for key, value in self.learner_observation.items()
            }
        )
        cached_features = (
            None
            if self.cached_features is None
            else {
                key: value[index : index + 1]
                for key, value in self.cached_features.items()
            }
        )
        return cast(
            ActionChunk,
            replace(
                self,
                learner_action=learned,
                environment_actions=self.environment_actions[index : index + 1],
                behavior_log_prob=log_prob,
                behavior_value=value,
                learner_observation=learner_observation,
                cached_features=cached_features,
            ),
        )

    @staticmethod
    def concatenate(chunks: list[ActionChunk] | tuple[ActionChunk, ...]) -> ActionChunk:
        """Concatenate compatible chunks along their batch axis."""
        if not chunks:
            raise ValueError("cannot concatenate an empty chunk sequence")
        first = chunks[0]
        for chunk in chunks[1:]:
            if (
                chunk.execution_length != first.execution_length
                or chunk.source != first.source
                or chunk.policy_version != first.policy_version
                or chunk.learner_action_kind != first.learner_action_kind
                or chunk.environment_actions.shape[1:]
                != first.environment_actions.shape[1:]
            ):
                raise ValueError(
                    "all chunks must have matching geometry and provenance"
                )
        learner_action = ActionChunk._concatenate_arrays(
            [chunk.learner_action for chunk in chunks], "learner actions"
        )
        behavior_log_prob = ActionChunk._concatenate_arrays(
            [chunk.behavior_log_prob for chunk in chunks],
            "behavior log probabilities",
        )
        behavior_value = ActionChunk._concatenate_arrays(
            [chunk.behavior_value for chunk in chunks], "behavior values"
        )
        return ActionChunk(
            learner_action=learner_action,
            environment_actions=np.concatenate(
                [chunk.environment_actions for chunk in chunks], axis=0
            ),
            execution_length=first.execution_length,
            learner_action_kind=first.learner_action_kind,
            source=first.source,
            policy_version=first.policy_version,
            behavior_log_prob=behavior_log_prob,
            behavior_value=behavior_value,
            learner_observation=ActionChunk._concatenate_mappings(
                [chunk.learner_observation for chunk in chunks], "learner observations"
            ),
            cached_features=ActionChunk._concatenate_mappings(
                [chunk.cached_features for chunk in chunks], "cached features"
            ),
        )

    @staticmethod
    def _concatenate_arrays(
        arrays: list[np.ndarray | None], name: str
    ) -> np.ndarray | None:
        """Concatenate along the batch axis; all chunks carry the array or none does."""
        if any(array is None for array in arrays):
            if not all(array is None for array in arrays):
                raise ValueError(f"cannot mix chunks with and without {name}")
            return None
        return np.concatenate([array for array in arrays if array is not None], axis=0)

    @staticmethod
    def _concatenate_mappings(
        mappings: list[Mapping[str, np.ndarray] | None], name: str
    ) -> dict[str, np.ndarray] | None:
        """Concatenate per-key arrays along the batch axis; all chunks carry them or none does."""
        if any(mapping is None for mapping in mappings):
            if not all(mapping is None for mapping in mappings):
                raise ValueError(f"cannot mix chunks with and without {name}")
            return None
        present = [mapping for mapping in mappings if mapping is not None]
        keys = set(present[0])
        if any(set(mapping) != keys for mapping in present[1:]):
            raise ValueError(f"{name} keys must match")
        return {
            key: np.concatenate([mapping[key] for mapping in present], axis=0)
            for key in keys
        }
