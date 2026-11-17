# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A trainable flow policy as the portable ActionChunk policy of a batched worker.

Base-model-neutral: the worker drives any
:class:`~isaac_rl_posttraining.policies.trainable.TrainablePolicy` (GR00T N1.5,
GR00T N1.7, pi0, ...) that a :class:`~isaac_rl_posttraining.policies.trainable.TrainablePolicySpec`
loads on the worker's device.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from isaac_rl_posttraining.policies.feature_cache import pack_features
from isaac_rl_posttraining.policies.trainable import (
    STOCHASTIC_STEP_KEY,
    TrainablePolicy,
    TrainablePolicySpec,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import (
    DECISION_MODES,
    DecisionMode,
    PolicyContext,
    PolicyState,
)
from isaac_rl_posttraining.runtimes.policy import (
    Observation,
    PolicyFeatures,
    RolloutPolicy,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_tensors

__all__ = ["FlowSDEActionChunkPolicy", "FlowSDEPolicySpec"]

#: Offsets the exploration RNG seed from the simulation seed of the same worker.
_EXPLORATION_SEED_OFFSET = 7919


@dataclass(frozen=True)
class FlowSDEPolicySpec:
    """Load one trainable flow policy per rollout worker.

    Attributes:
        spec: How the worker loads the policy (base model, checkpoint, flow-SDE
            settings); the learner ranks build theirs from the same spec.
        action_window: Executed steps per decision, which the loaded model
            must agree on (the worker sizes its episode horizon by it).
        instruction: The task instruction every decision is conditioned on.
        base_seed: Exploration seed of policy 0; policy ``p`` uses ``base_seed + p``.
        cache_features: Ship the frozen part's outputs with each decision
            instead of the inputs they came from, so the learner skips it.
    """

    spec: TrainablePolicySpec
    action_window: int
    instruction: str
    base_seed: int = 0
    cache_features: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.action_window, bool) or not isinstance(self.action_window, int):
            raise TypeError("action_window must be a positive integer")
        if self.action_window < 1:
            raise ValueError("action_window must be a positive integer")
        if not self.instruction:
            raise ValueError("instruction must be non-empty")
        policy_class = self.spec.policy_class
        if self.cache_features and not policy_class.cached_feature_names:
            raise ValueError(f"{self.spec.base_model} has no cacheable features; set cache_features=false")
        if policy_class.requires_cached_features and not self.cache_features:
            raise ValueError(
                f"{self.spec.base_model} decisions can only travel as cached features; set cache_features=true"
            )

    def __call__(self, context: PolicyContext) -> FlowSDEActionChunkPolicy:
        actor_critic = self.spec.build(context.device)
        if actor_critic.action_window != self.action_window:
            raise ValueError(
                f"the checkpoint executes {actor_critic.action_window}-step action chunks, "
                f"not the configured action_window={self.action_window}"
            )
        return FlowSDEActionChunkPolicy(
            actor_critic,
            instruction=self.instruction,
            rng=np.random.default_rng(self.base_seed + context.policy_index + _EXPLORATION_SEED_OFFSET),
            cache_features=self.cache_features,
        )


class FlowSDEActionChunkPolicy(RolloutPolicy):
    """Decide for a batch of scenes with one flow-SDE decode.

    Training decisions noise one denoising step, drawn per decision batch from
    the policy's own RNG, and carry the denoising chain as the learner action
    together with the behavior log-probability and value. Evaluation decisions
    decode deterministically. The learner observation is the packed model input
    (or the inputs the learner still needs plus the frozen part's cached
    outputs) and the stochastic-step marker the learner re-evaluates the chain at.

    Published policy states carry every parameter outside the policy's
    ``frozen_submodules``; ``frozen_submodules`` overrides that set when given.

    Decisions carry behaviour values but the policy has no separate value
    pass, so a batched worker takes bootstrap values from the next decision
    (``bootstrap_values="reuse_next_decision"``). A batch shares one mode.

    A policy server runs this policy for many rollout workers: each batch it
    answers draws one stochastic step, and every decode draws its noise from
    the server process's torch generator. Served decisions therefore have the
    local distribution but match a local copy's numbers only when the server
    runs in the worker's process and serves that worker alone.
    """

    features = PolicyFeatures(behavior_values=True, mixed_mode_batches=False)

    def __init__(
        self,
        actor_critic: TrainablePolicy,
        *,
        instruction: str,
        rng: np.random.Generator,
        cache_features: bool,
        frozen_submodules: Sequence[str] | None = None,
    ) -> None:
        self.actor_critic = actor_critic.eval()
        self.instruction = instruction
        self.cache_features = bool(cache_features)
        self._rng = rng
        frozen = actor_critic.frozen_submodules if frozen_submodules is None else tuple(frozen_submodules)
        frozen_prefixes = tuple(f"{path}." for path in frozen)
        self._published_keys = {
            name for name in actor_critic.state_dict() if not name.startswith(frozen_prefixes)
        }
        self._policy_version = 0

    @property
    def action_window(self) -> int:
        return int(self.actor_critic.action_window)

    def action_chunks(
        self,
        observations: Sequence[Observation],
        *,
        mode: Sequence[DecisionMode],
        session_ids: Sequence[Hashable],
    ) -> tuple[ActionChunk, ...]:
        if not (len(observations) == len(mode) == len(session_ids)):
            raise ValueError("observations, mode, and session_ids must have equal lengths")
        if not observations:
            return ()
        modes = set(mode)
        if len(modes) != 1:
            raise ValueError("a flow-SDE batch shares one decision mode")
        (decision_mode,) = modes
        if decision_mode not in DECISION_MODES:
            raise ValueError(f"decision mode must be one of {DECISION_MODES}, got {decision_mode!r}")
        if decision_mode == "warmup":
            raise ValueError("the flow-SDE policy acts from the model and has no warm-up mode")
        batch = len(observations)
        frames = {
            key: np.stack([np.asarray(observation[key]) for observation in observations])
            for key in self.actor_critic.video_keys
        }
        states = {
            key: np.stack([np.asarray(observation[key]) for observation in observations])
            for key in self.actor_critic.state_keys
        }
        inputs = self.actor_critic.build_inputs(frames=frames, states=states, instruction=self.instruction)
        stochastic_step = (
            int(self._rng.integers(self.actor_critic.denoising_steps))
            if decision_mode == "train"
            else None
        )
        sample = self.actor_critic.act(inputs, stochastic_step=stochastic_step)
        raw_actions = self.actor_critic.decode(sample.actions, inputs=inputs, states=states)
        if self.cache_features:
            records = self.actor_critic.pack_inputs(self.actor_critic.learner_inputs(inputs), batch)
            cached_records = pack_features(sample.cached_features, batch)
        else:
            records = self.actor_critic.pack_inputs(inputs, batch)
            cached_records = None
        chains = sample.chains.float().cpu().numpy()
        log_probs = sample.log_prob.cpu().numpy()
        values = sample.value.cpu().numpy()
        step_marker = np.asarray([-1 if stochastic_step is None else stochastic_step], dtype=np.int64)
        chunks = []
        for index in range(batch):
            learner_observation: dict[str, np.ndarray] = {
                key: value[None] for key, value in records[index].items()
            }
            learner_observation[STOCHASTIC_STEP_KEY] = step_marker
            chunks.append(
                ActionChunk.from_arrays(
                    learner_action=chains[index][None],
                    environment_actions=raw_actions[index][None],
                    execution_length=self.action_window,
                    learner_action_kind="chain",
                    source="policy",
                    policy_version=self._policy_version,
                    behavior_log_prob=log_probs[index : index + 1],
                    behavior_value=values[index : index + 1],
                    learner_observation=learner_observation,
                    cached_features=(
                        None
                        if cached_records is None
                        else {key: value[None] for key, value in cached_records[index].items()}
                    ),
                )
            )
        return tuple(chunks)

    def set_policy_state(self, state: PolicyState) -> int:
        if state.version <= self._policy_version:
            raise ValueError(
                f"policy state must advance beyond {self._policy_version}, got {state.version}"
            )
        received = set(state.tensors)
        if received != self._published_keys:
            missing = sorted(self._published_keys - received)[:5]
            unexpected = sorted(received - self._published_keys)[:5]
            raise ValueError(
                "policy state keys differ from the worker's trainable state; "
                f"missing={missing} unexpected={unexpected}"
            )
        current: Mapping[str, torch.Tensor] = self.actor_critic.state_dict()
        tensors = {
            name: tensor.to(device=current[name].device, dtype=current[name].dtype)
            for name, tensor in policy_state_tensors(state).items()
        }
        result = self.actor_critic.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"policy state carried unexpected keys {result.unexpected_keys[:5]}")
        self._policy_version = state.version
        return self._policy_version
