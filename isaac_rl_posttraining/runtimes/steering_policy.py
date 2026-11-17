# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt posttrain flow-steering actors to the portable ActionChunk policy."""

from __future__ import annotations

import copy
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import torch

from isaac_rl_posttraining.policies.base import DEFAULT_LATENT_CLIP
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk, ActionSource
from isaac_rl_posttraining.rollouts.steering_step import SteeringStep
from isaac_rl_posttraining.runtimes.base import (
    DecisionMode,
    PolicyContext,
    PolicyState,
)
from isaac_rl_posttraining.runtimes.base_part import (
    DecodeOptions,
    ServedBasePartSpec,
    base_part,
)
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
    batch_observation,
    combine_observations,
    numpy_observation,
    split_observation,
    validate_session_requests,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import apply_policy_state
from isaac_rl_posttraining.runtimes.session_randomness import SessionRandomness

__all__ = [
    "FrozenBaseModelSpec",
    "SteeringActionChunkPolicy",
    "SteeringPolicySpec",
    "build_dino_wrapper",
]

#: A per-session decision: the latent the decoder executes, its replayed form,
#: PPO's behaviour log-probability and value, and where the latent came from.
_Selection = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    ActionSource,
]


class _LatentSampler(Protocol):
    def sample_action(self) -> np.ndarray: ...


@dataclass(frozen=True)
class SteeringPolicySpec:
    """Everything a worker needs to build its steering policy; picklable, provider-free."""

    actor: torch.nn.Module
    base_model: FrozenBaseModelSpec | ServedBasePartSpec
    latent_sampler: _LatentSampler
    steering_key_mapping: dict[str, Any]
    action_exec_len: int
    action_modulator: Any = None
    dino_model_name: str | None = None
    dino_wrapper: Any = None
    dino_image_keys: tuple[str, ...] = ()
    use_vlm_features: bool = False
    reference_action_key: str | None = None
    reference_image_transform_mode: str = "train"
    deterministic_inference: bool = False
    sample_evaluation_actions: bool = False
    deterministic_evaluation: bool = False
    native: bool = False
    max_decode_batch_size: int | None = None
    batched_actor: bool = False

    def with_base_part(self, base_model: ServedBasePartSpec) -> SteeringPolicySpec:
        """The same policy with its base model served (``placement.parts.base``)."""
        return replace(self, base_model=base_model)

    def __call__(self, context: PolicyContext) -> SteeringActionChunkPolicy:
        if self.deterministic_inference:
            from isaac_rl_posttraining.utils.training_utils import (
                enable_deterministic_torch,
            )

            enable_deterministic_torch()
        device = torch.device(context.device)
        actor = copy.deepcopy(self.actor).to(device).eval()
        base_model = self.base_model(context)
        dino_wrapper = build_dino_wrapper(
            device,
            dino_model_name=self.dino_model_name,
            dino_wrapper=self.dino_wrapper,
            image_keys=self.dino_image_keys,
        )
        return SteeringActionChunkPolicy(
            actor=actor,
            base_model=base_model,
            latent_sampler=copy.deepcopy(self.latent_sampler),
            device=device,
            steering_key_mapping=copy.deepcopy(self.steering_key_mapping),
            action_exec_len=self.action_exec_len,
            action_modulator=copy.deepcopy(self.action_modulator),
            dino_wrapper=dino_wrapper,
            use_vlm_features=self.use_vlm_features,
            reference_action_key=self.reference_action_key,
            reference_image_transform_mode=self.reference_image_transform_mode,
            sample_evaluation_actions=self.sample_evaluation_actions,
            deterministic_evaluation=self.deterministic_evaluation,
            native=self.native,
            max_decode_batch_size=self.max_decode_batch_size,
            batched_actor=self.batched_actor,
        )


class SteeringActionChunkPolicy(RolloutPolicy):
    """Composite actor and frozen decoder used identically by every runtime.

    Algorithm-specific behavior remains in the objects it composes:

    - DSRL supplies a latent actor and no action modulator.
    - RFS supplies the joint flow/residual actor and its action modulator.
    - A smaller flow actor simply emits ``[batch, latent_dim]`` instead of a
      per-flow-step tensor; the frozen base model retains responsibility for
      broadcasting or tiling it.

    The worker always receives an unbatched simulator observation.
    This adapter adds one leading dimension because posttrain actor and frozen
    base models use a batch-one inference contract.

    Each decision carries a mode: ``warmup`` draws a random latent from the
    sampler, ``train`` samples the actor, and ``eval`` samples the actor too
    (its mean under ``deterministic_evaluation``). A ``native`` policy has no
    steering at all: it evaluates the frozen base model with its own noise and
    accepts only ``eval`` decisions.

    With ``batched_actor`` the actor runs once for all the sessions of a call
    that act from it, and each session then draws its sampling noise as an
    unbatched call would (a seeded evaluation session from its own stream).

    ``base_model`` is the base-model part: a frozen base model decoded in this process,
    or a :class:`~isaac_rl_posttraining.runtimes.base_part.ServedBasePart`
    when ``placement.parts.base`` serves it; the actor step stays here either
    way and the decodes are the same.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        base_model: Any,
        latent_sampler: _LatentSampler,
        device: torch.device,
        steering_key_mapping: dict[str, Any],
        action_exec_len: int,
        action_modulator: Any = None,
        dino_wrapper: Any = None,
        use_vlm_features: bool = False,
        reference_action_key: str | None = None,
        reference_image_transform_mode: str = "train",
        sample_evaluation_actions: bool = False,
        deterministic_evaluation: bool = False,
        native: bool = False,
        max_decode_batch_size: int | None = None,
        batched_actor: bool = False,
    ) -> None:
        if isinstance(action_exec_len, bool) or not isinstance(action_exec_len, int):
            raise TypeError("action_exec_len must be a positive integer")
        if action_exec_len < 1:
            raise ValueError("action_exec_len must be a positive integer")
        if max_decode_batch_size is not None and (
            isinstance(max_decode_batch_size, bool)
            or not isinstance(max_decode_batch_size, int)
            or max_decode_batch_size < 1
        ):
            raise ValueError("max_decode_batch_size must be a positive integer")
        if not callable(getattr(actor, "act", None)) and not callable(
            getattr(actor, "sample_action_chunk", None)
        ):
            raise TypeError("actor must expose act(...) or sample_action_chunk(...)")
        self.actor = actor
        self.latent_sampler = latent_sampler
        self.action_exec_len = action_exec_len
        self.use_vlm_features = use_vlm_features
        self.sample_evaluation_actions = bool(sample_evaluation_actions)
        self.deterministic_evaluation = bool(deterministic_evaluation)
        self.native = bool(native)
        self.max_decode_batch_size = max_decode_batch_size
        self._step = SteeringStep(
            base_model=base_model,
            device=device,
            steering_key_mapping=steering_key_mapping,
            dino_wrapper=dino_wrapper,
            action_modulator=action_modulator,
            reference_action_key=reference_action_key,
            reference_image_transform_mode=reference_image_transform_mode,
        )
        self._policy_version = 0
        self._actor_states: dict[Hashable, list[Any] | None] = {
            0: self._initial_actor_state()
        }
        self._prepared_observations: dict[Hashable, dict[str, torch.Tensor]] = {}
        self._randomness = SessionRandomness(device)
        self._base = base_part(base_model, device)
        self._decode_options = DecodeOptions(
            action_modulator=action_modulator,
            # A reference-conditioned actor decodes session by session, as the serial collector did.
            batched=reference_action_key is None,
            max_batch=max_decode_batch_size,
        )
        self.batched_actor = bool(batched_actor)
        if self.batched_actor:
            self._check_batched_actor()
        self.features = PolicyFeatures(
            encodes_observations=True,
            values_observations=True,
            behavior_values=callable(getattr(actor, "sample_action_chunk", None)),
        )

    def _initial_actor_state(self) -> list[Any] | None:
        initializer = getattr(self.actor, "get_initial_state", None)
        return [initializer()] if callable(initializer) else None

    def _check_batched_actor(self) -> None:
        """One actor call can serve several sessions only for a stateless, sampling (SAC-style) actor."""
        actor = self.actor
        if callable(getattr(actor, "sample_action_chunk", None)):
            raise TypeError(
                "batched_actor applies to latent-sampling actors, not PPO actors"
            )
        if self._initial_actor_state() not in ([None], [{}]):
            raise ValueError(
                "batched_actor needs a stateless actor; recurrent actors keep per-session state"
            )
        if (
            not callable(getattr(actor, "get_action_dist_params", None))
            or getattr(actor, "action_dist", None) is None
        ):
            raise TypeError(
                "batched_actor needs an actor exposing get_action_dist_params and action_dist"
            )
        if getattr(actor, "preprocess_obs_transform", None) is not None:
            raise ValueError(
                "batched_actor does not support actors with observation transforms"
            )

    def _encode_actor_observations(
        self, observations: Sequence[Observation]
    ) -> tuple[dict[str, torch.Tensor], ...]:
        """Batch expensive visual encoding, then split learner observations."""
        if not observations:
            return ()
        batched = [batch_observation(observation) for observation in observations]
        if self.native:
            # A native policy has no actor to feed; keeping only the state in
            # the replay payload keeps evaluation results small.
            state_key = self._step.steering_key_mapping["state"]
            return tuple(
                {
                    state_key: torch.as_tensor(
                        observation[state_key], device=self._step.device
                    ).float()
                }
                for observation in batched
            )
        combined = combine_observations(batched)
        if "dino_embedding" not in combined and self._step.dino_wrapper is not None:
            combined = self._step.dino_wrapper.observation(combined)
            batched = [
                split_observation(combined, index, len(observations))
                for index in range(len(observations))
            ]
        return tuple(
            self._step.process_obs(
                observation,
                squeeze=False,
                vlm_features="enabled" if self.use_vlm_features else "off",
            )
            for observation in batched
        )

    def encode_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> tuple[dict[str, np.ndarray], ...]:
        """Encode next observations and cache them for each session's next decision."""
        if len(observations) != len(session_ids):
            raise ValueError("observations and session_ids must have equal lengths")
        for session_id in session_ids:
            if not isinstance(session_id, Hashable):
                raise TypeError("policy-service session ids must be hashable")
        encoded = self._encode_actor_observations(observations)
        results = []
        for session_id, learner_observation in zip(session_ids, encoded, strict=True):
            self._prepared_observations[session_id] = learner_observation
            results.append(numpy_observation(learner_observation, unbatch=True))
        return tuple(results)

    def encode_and_value_next_observations(
        self,
        observations: Sequence[Observation],
        *,
        session_ids: Sequence[Hashable],
    ) -> tuple[tuple[dict[str, np.ndarray], np.ndarray], ...]:
        """Encode next observations and return one bootstrap value per session.

        Actors without a value head (SAC) return NaN values. Their decisions
        carry no behavior value, so the worker uses only the observations.
        """
        prepared = self.encode_observations(
            observations,
            session_ids=session_ids,
        )
        value_fn = getattr(self.actor, "values", None)
        if not callable(value_fn):
            return tuple(
                (observation, np.full((1,), np.nan, dtype=np.float32))
                for observation in prepared
            )
        ready = [self._prepared_observations[session_id] for session_id in session_ids]
        combined = combine_observations(ready)
        values = value_fn(combined).detach().cpu().numpy().reshape(-1)
        if len(values) != len(prepared):
            raise RuntimeError(
                f"actor returned {len(values)} bootstrap values for "
                f"{len(prepared)} observations"
            )
        return tuple(
            (observation, np.asarray([value], dtype=np.float32))
            for observation, value in zip(prepared, values, strict=True)
        )

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> tuple[ActionChunk, ...]:
        """Batch independent environment sessions through one frozen decoder."""
        validate_session_requests(observations, mode, session_ids)
        if not observations:
            return ()
        if self.native and any(decision_mode != "eval" for decision_mode in mode):
            raise ValueError(
                "a native steering policy evaluates the frozen base model only"
            )
        ppo_sampler = getattr(self.actor, "sample_action_chunk", None)
        if callable(ppo_sampler) and "warmup" in mode:
            raise ValueError(
                "PPO actors act from their policy and have no warm-up mode"
            )

        ready_observations = self._actor_observations(observations, session_ids)
        selected_actions: list[np.ndarray] = []
        learner_actions: list[np.ndarray] = []
        behavior_log_probs: list[np.ndarray | None] = []
        behavior_values: list[np.ndarray | None] = []
        sources: list[ActionSource] = []
        for selection in self._select_actions(
            ready_observations, mode, session_ids, ppo_sampler
        ):
            selected_action, replay_action, log_prob, value, source = selection
            behavior_log_probs.append(log_prob)
            behavior_values.append(value)
            sources.append(source)
            if selected_action is not None and replay_action is not None:
                selected_actions.append(selected_action)
                learner_actions.append(replay_action)

        combined_actions = (
            None if self.native else np.concatenate(selected_actions, axis=0)
        )
        decoded = self._base.decode(
            observations,
            None if self.native else selected_actions,
            session_ids=session_ids,
            modes=mode,
            randomness=self._randomness,
            options=self._decode_options,
        )
        replay_actions = self._replay_actions(
            learner_actions, combined_actions, sources, ppo_sampler
        )
        return tuple(
            ActionChunk.from_arrays(
                learner_action=replay_actions[index],
                environment_actions=decoded[index : index + 1],
                execution_length=self.action_exec_len,
                learner_action_kind=None if replay_actions[index] is None else "latent",
                source=sources[index],
                policy_version=self._policy_version,
                behavior_log_prob=behavior_log_probs[index],
                behavior_value=behavior_values[index],
                learner_observation=numpy_observation(
                    ready_observations[index], unbatch=False
                ),
            )
            for index in range(len(observations))
        )

    def _actor_observations(
        self,
        observations: Sequence[Observation],
        session_ids: Sequence[Hashable],
    ) -> list[dict[str, torch.Tensor]]:
        """Each session's learner observation: prepared by a successor call, else encoded now."""
        actor_observations: list[dict[str, torch.Tensor] | None] = [None] * len(
            observations
        )
        missing_indices = []
        for index, session_id in enumerate(session_ids):
            cached = self._prepared_observations.pop(session_id, None)
            if cached is None:
                missing_indices.append(index)
            else:
                actor_observations[index] = cached
        if missing_indices:
            encoded = self._encode_actor_observations(
                tuple(observations[index] for index in missing_indices)
            )
            for index, learner_observation in zip(
                missing_indices, encoded, strict=True
            ):
                actor_observations[index] = learner_observation
        ready_observations = []
        for actor_observation in actor_observations:
            if actor_observation is None:
                raise RuntimeError("failed to prepare a policy observation")
            ready_observations.append(actor_observation)
        return ready_observations

    def _select_actions(
        self,
        ready_observations: Sequence[dict[str, torch.Tensor]],
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
        ppo_sampler: Any,
    ) -> list[_Selection]:
        """Every session's selection; one actor call for the actor-driven ones under ``batched_actor``."""
        batched = [
            index
            for index, decision_mode in enumerate(mode)
            if self.batched_actor
            and not self.native
            and decision_mode in ("train", "eval")
        ]
        selections: list[_Selection | None] = [None] * len(mode)
        if len(batched) > 1:
            latents = self._batched_latents(
                [ready_observations[index] for index in batched],
                [mode[index] for index in batched],
                [session_ids[index] for index in batched],
            )
            for index, latent in zip(batched, latents, strict=True):
                selections[index] = (latent, latent, None, None, "policy")
        for index, (observation, decision_mode, session_id) in enumerate(
            zip(ready_observations, mode, session_ids, strict=True)
        ):
            if selections[index] is None:
                selections[index] = self._select_action(
                    observation, decision_mode, session_id, ppo_sampler
                )
        return [selection for selection in selections if selection is not None]

    def _batched_latents(
        self,
        observations: Sequence[dict[str, torch.Tensor]],
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> list[np.ndarray]:
        """One actor forward for all sessions; each samples its latent as an unbatched call would."""
        actor = self.actor
        combined = combine_observations(list(observations))
        removed = getattr(actor, "remove_obs_keys", None) or ()
        actor_input = {
            key: value for key, value in combined.items() if key not in removed
        }
        with torch.no_grad():
            mean, log_std, extra = actor.get_action_dist_params(actor_input)
            latents = []
            for index, (decision_mode, session_id) in enumerate(
                zip(mode, session_ids, strict=True)
            ):
                evaluation = decision_mode == "eval"
                with self._randomness.stream(session_id, enabled=evaluation):
                    latent = actor.action_dist.actions_from_params(
                        mean[index : index + 1],
                        log_std[index : index + 1],
                        deterministic=evaluation and self.deterministic_evaluation,
                        **extra,
                    )
                if actor.normalize_actions:
                    latent = actor.unnormalize_action(latent)
                latents.append(latent.detach().cpu().numpy())
        return latents

    def _select_action(
        self,
        actor_observation: dict[str, torch.Tensor],
        decision_mode: DecisionMode,
        session_id: Hashable,
        ppo_sampler: Any,
    ) -> _Selection:
        """One session's latent, its replayed form, PPO metadata, and source; no latent when native."""
        evaluation = decision_mode == "eval"
        behavior_log_prob: np.ndarray | None = None
        behavior_value: np.ndarray | None = None
        with self._randomness.stream(session_id, enabled=evaluation):
            source: ActionSource
            if self.native:
                return None, None, None, None, "native"
            if callable(ppo_sampler):
                (
                    decoder_action,
                    learner_action,
                    log_prob,
                    value,
                ) = ppo_sampler(
                    actor_observation,
                    deterministic=(evaluation and not self.sample_evaluation_actions),
                )
                selected_action = decoder_action.detach().cpu().numpy()
                replay_action = learner_action.detach().cpu().numpy()
                if not evaluation:
                    # Evaluation only consumes actions and outcomes.
                    # Omitting PPO metadata also avoids a bootstrap-value request.
                    behavior_log_prob = log_prob.detach().cpu().numpy().reshape(1)
                    behavior_value = value.detach().cpu().numpy().reshape(1)
                source = "policy"
            elif decision_mode == "warmup":
                # Warm-up latents bypass the actor. The base model executes them
                # after its injection clip; they are replayed in the actor's
                # normalized coordinates below, like policy latents.
                selected_action = np.clip(
                    np.asarray(self.latent_sampler.sample_action(), dtype=np.float32),
                    -DEFAULT_LATENT_CLIP,
                    DEFAULT_LATENT_CLIP,
                )
                replay_action = selected_action
                source = "warmup"
            elif decision_mode in ("train", "eval"):
                if session_id not in self._actor_states:
                    self._actor_states[session_id] = self._initial_actor_state()
                selected_action, actor_state = self._step.select_latent(
                    self.actor,
                    actor_observation,
                    deterministic=evaluation and self.deterministic_evaluation,
                    actor_state=self._actor_states[session_id],
                )
                self._actor_states[session_id] = actor_state
                source = "policy"
                replay_action = np.asarray(selected_action)
            else:
                raise ValueError(f"unhandled decision mode {decision_mode!r}")
        selected_action = np.asarray(selected_action)
        if selected_action.shape[0] != 1:
            raise ValueError(
                "each policy-service request must produce one learner action; "
                f"got {selected_action.shape}"
            )
        return (
            selected_action,
            np.asarray(replay_action),
            behavior_log_prob,
            behavior_value,
            source,
        )

    def _replay_actions(
        self,
        learner_actions: Sequence[np.ndarray],
        combined_actions: np.ndarray | None,
        sources: Sequence[ActionSource],
        ppo_sampler: Any,
    ) -> list[np.ndarray | None]:
        """The latents replay stores: in the actor's normalized coordinates, none when native."""
        if self.native:
            return [None] * len(sources)
        replay_actions: list[np.ndarray | None] = list(learner_actions)
        # The critic and the actor objective read every replayed latent in the
        # actor's normalized coordinates, warm-up ones included; otherwise a
        # noise bound other than 1 leaves the warm-up transitions in another
        # scale. Warm-up exploration may exceed the actor's reachable range,
        # so normalization must preserve rather than clamp that executed value.
        if not callable(ppo_sampler) and bool(
            getattr(self.actor, "normalize_actions", False)
        ):
            assert combined_actions is not None
            normalized = (
                self.actor.normalize_action(
                    torch.as_tensor(
                        combined_actions,
                        device=self._step.device,
                        dtype=torch.float32,
                    ),
                    clamp=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
            for index, source in enumerate(sources):
                if source in ("policy", "warmup"):
                    replay_actions[index] = normalized[index : index + 1]
        return replay_actions

    def reset_episode(self, session_id: Hashable = 0, seed: int | None = None) -> None:
        self._actor_states[session_id] = self._initial_actor_state()
        self._prepared_observations.pop(session_id, None)
        self._randomness.seed(session_id, seed)
        self._base.start_episode(session_id)

    def set_policy_state(self, state: PolicyState) -> int:
        self._policy_version = apply_policy_state(
            self.actor, state, current_version=self._policy_version
        )
        return self._policy_version
