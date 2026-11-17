# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt the reference-edit actor and its frozen base model to the portable ActionChunk policy."""

from __future__ import annotations

import copy
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.envs.reference_action import (
    REFERENCE_KEY,
    environment_actions_from_learner,
    reference_observation_entries,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk, ActionSource
from isaac_rl_posttraining.rollouts.steering_step import SteeringStep
from isaac_rl_posttraining.runtimes.base import (
    DecisionMode,
    PolicyContext,
    PolicyState,
)
from isaac_rl_posttraining.runtimes.base_part import ServedBasePartSpec
from isaac_rl_posttraining.runtimes.frozen_base_model import (
    FrozenBaseModelSpec,
    build_dino_wrapper,
)
from isaac_rl_posttraining.runtimes.policy import (
    Observation,
    PolicyFeatures,
    RolloutPolicy,
)
from isaac_rl_posttraining.runtimes.policy_batching import (
    numpy_observation,
    validate_session_requests,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import apply_policy_state

__all__ = ["EditActionChunkPolicy", "EditPolicySpec"]


@dataclass(frozen=True)
class EditPolicySpec:
    """Everything a worker needs to build its reference-edit policy; picklable, provider-free."""

    actor: torch.nn.Module
    base_model: FrozenBaseModelSpec | ServedBasePartSpec
    action_indices: tuple[int, ...]
    environment_action_defaults: np.ndarray
    action_horizon: int
    reference_action_transform: str
    relative_action_scale: float
    reference_image_transform_mode: str
    state_key: str
    image_keys: tuple[str, ...]
    dino_model_name: str | None = None
    dino_wrapper: Any = None
    dino_image_keys: tuple[str, ...] = ()
    deterministic_evaluation: bool = False
    native: bool = False

    def with_base_part(self, base_model: ServedBasePartSpec) -> EditPolicySpec:
        """The same policy with its base model served (``placement.parts.base``)."""
        return replace(self, base_model=base_model)

    def __call__(self, context: PolicyContext) -> EditActionChunkPolicy:
        device = torch.device(context.device)
        return EditActionChunkPolicy(
            actor=copy.deepcopy(self.actor).to(device).eval(),
            base_model=self.base_model(context),
            device=device,
            action_indices=self.action_indices,
            environment_action_defaults=self.environment_action_defaults,
            action_horizon=self.action_horizon,
            reference_action_transform=self.reference_action_transform,
            relative_action_scale=self.relative_action_scale,
            reference_image_transform_mode=self.reference_image_transform_mode,
            state_key=self.state_key,
            image_keys=self.image_keys,
            dino_wrapper=build_dino_wrapper(
                device,
                dino_model_name=self.dino_model_name,
                dino_wrapper=self.dino_wrapper,
                image_keys=self.dino_image_keys,
            ),
            deterministic_evaluation=self.deterministic_evaluation,
            native=self.native,
        )


class EditActionChunkPolicy(RolloutPolicy):
    """Frozen reference decode plus a learned bounded edit, executed in the raw action space.

    Every decision rests on one frozen decode of its observation: the reference
    chunk the actor edits, the pooled backbone features it conditions on, and an
    action mask. ``encode_observations`` runs that decode for a next
    observation and caches it, so the replay ``next_obs`` and the following
    decision share one decode, exactly as the serial loop's reference-action
    environment decodes once per observation. The edit actor has no value
    head, so the policy encodes next observations but does not value them.

    ``warmup`` executes the reference itself (``source="native"``); ``train``
    samples the actor and ``eval`` samples it too (its mean under
    ``deterministic_evaluation``). A ``native`` policy evaluates the frozen
    reference only and accepts only ``eval`` decisions. Evaluation reset seeds
    reach the environment only; the actor and the frozen decode draw from the
    process RNG.
    """

    features = PolicyFeatures(encodes_observations=True)

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        base_model: Any,
        device: torch.device,
        action_indices: Sequence[int],
        environment_action_defaults: np.ndarray,
        action_horizon: int,
        reference_action_transform: str,
        relative_action_scale: float,
        reference_image_transform_mode: str,
        state_key: str,
        image_keys: Sequence[str],
        dino_wrapper: Any = None,
        deterministic_evaluation: bool = False,
        native: bool = False,
    ) -> None:
        if not callable(getattr(actor, "act", None)):
            raise TypeError("actor must expose act(...)")
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon < 1
        ):
            raise ValueError("action_horizon must be a positive integer")
        if not action_indices:
            raise ValueError(
                "action_indices must name at least one controlled dimension"
            )
        defaults = np.asarray(environment_action_defaults, dtype=np.float32)
        if defaults.ndim != 1 or defaults.shape[0] <= max(action_indices):
            raise ValueError(
                "environment_action_defaults must be one environment command covering "
                f"action_indices {tuple(action_indices)}, got shape {defaults.shape}"
            )
        if reference_action_transform not in ("identity", "absolute_to_relative"):
            raise ValueError(
                "reference_action_transform must be 'identity' or 'absolute_to_relative'"
            )
        if reference_image_transform_mode not in ("train", "eval"):
            raise ValueError("reference_image_transform_mode must be train or eval")
        self.actor = actor
        self.base_model = base_model
        self.device = device
        self.action_indices = tuple(int(index) for index in action_indices)
        self.environment_action_defaults = defaults
        self.action_horizon = action_horizon
        self.reference_action_transform = reference_action_transform
        self.relative_action_scale = float(relative_action_scale)
        self.reference_image_transform_mode = reference_image_transform_mode
        self.state_key = state_key
        self.image_keys = frozenset(image_keys)
        self.dino_wrapper = dino_wrapper
        self.deterministic_evaluation = bool(deterministic_evaluation)
        self.native = bool(native)
        self._policy_version = 0
        self._prepared_observations: dict[Hashable, dict[str, torch.Tensor]] = {}

    def _decode_reference(self, observation: Observation) -> dict[str, torch.Tensor]:
        """One frozen decode: the batch-one learner observation carrying the reference entries."""
        if not isinstance(observation, Mapping):
            raise TypeError("edit policies require a mapping observation")
        batched = SteeringStep.vectorize_obs(observation)
        if "dino_embedding" not in batched and self.dino_wrapper is not None:
            batched = self.dino_wrapper.observation(batched)
        with torch.inference_mode():
            features, reference = self.base_model.infer_reference(
                batched, image_transform_mode=self.reference_image_transform_mode
            )
        entries = reference_observation_entries(
            features,
            reference,
            batched[self.state_key],
            action_indices=self.action_indices,
            action_horizon=self.action_horizon,
            reference_action_transform=self.reference_action_transform,
            relative_action_scale=self.relative_action_scale,
        )
        learner: dict[str, torch.Tensor] = {}
        for key, value in batched.items():
            # Camera frames feed the frozen decode and DINO only; the learner
            # keeps their embedding and every other array of the observation.
            if key in self.image_keys or not isinstance(value, np.ndarray):
                continue
            learner[key] = torch.as_tensor(value, device=self.device)
        for key, value in entries.items():
            learner[key] = torch.as_tensor(value, device=self.device)
        return learner

    def encode_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> tuple[dict[str, np.ndarray], ...]:
        """Decode next observations and cache them for each session's next decision."""
        if len(observations) != len(session_ids):
            raise ValueError("observations and session_ids must have equal lengths")
        for session_id in session_ids:
            if not isinstance(session_id, Hashable):
                raise TypeError("policy-service session ids must be hashable")
        results = []
        for observation, session_id in zip(observations, session_ids, strict=True):
            learner_observation = self._decode_reference(observation)
            self._prepared_observations[session_id] = learner_observation
            results.append(numpy_observation(learner_observation, unbatch=True))
        return tuple(results)

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> tuple[ActionChunk, ...]:
        """Decide for independent sessions; each decision is one frozen decode plus one actor call."""
        validate_session_requests(observations, mode, session_ids)
        if self.native and any(decision_mode != "eval" for decision_mode in mode):
            raise ValueError("a native edit policy evaluates the frozen reference only")
        return tuple(
            self._edit_chunk(observation, decision_mode, session_id)
            for observation, decision_mode, session_id in zip(
                observations, mode, session_ids, strict=True
            )
        )

    def _edit_chunk(
        self,
        observation: Observation,
        decision_mode: DecisionMode,
        session_id: Hashable,
    ) -> ActionChunk:
        learner_observation = self._prepared_observations.pop(session_id, None)
        if learner_observation is None:
            learner_observation = self._decode_reference(observation)
        source: ActionSource
        if self.native or decision_mode == "warmup":
            selected = learner_observation[REFERENCE_KEY]
            source = "native"
        elif decision_mode in ("train", "eval"):
            selected, _ = self.actor.act(
                learner_observation,
                deterministic=decision_mode == "eval" and self.deterministic_evaluation,
            )
            source = "policy"
        else:
            raise ValueError(f"unhandled decision mode {decision_mode!r}")
        selected = selected.detach()
        expected_shape = (1, self.action_horizon, len(self.action_indices))
        if tuple(selected.shape) != expected_shape:
            raise ValueError(
                f"edit decisions must be one {expected_shape[1:]} chunk, got {tuple(selected.shape)}"
            )
        executed = selected.cpu().numpy().astype(np.float32)
        if self.actor.normalize_actions:
            learner_action = self.actor.normalize_action(selected).cpu().numpy()
        else:
            learner_action = executed
        return ActionChunk.from_arrays(
            learner_action=learner_action,
            environment_actions=environment_actions_from_learner(
                executed,
                action_indices=self.action_indices,
                environment_action_defaults=self.environment_action_defaults,
            ),
            execution_length=self.action_horizon,
            learner_action_kind="edit",
            source=source,
            policy_version=self._policy_version,
            learner_observation=numpy_observation(learner_observation, unbatch=False),
        )

    def reset_episode(self, session_id: Hashable = 0, seed: int | None = None) -> None:
        # Evaluation reset seeds reach the environment only (see the class docstring).
        del seed
        self._prepared_observations.pop(session_id, None)

    def set_policy_state(self, state: PolicyState) -> int:
        self._policy_version = apply_policy_state(
            self.actor, state, current_version=self._policy_version
        )
        return self._policy_version
