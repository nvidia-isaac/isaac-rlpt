# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny flow-SDE actor-critic and the batched scenes it acts in, for served-PPO tests.

:class:`TinyFlowSDE` is a real :class:`~isaac_rl_posttraining.policies.flow_sde.FlowSDEActorCritic`
(the shared sampling, log-probability and value math) over a few linear
layers, with a frozen "backbone" whose output can travel as a cached feature.
Its evaluation decodes start from zero noise, so they are deterministic in
any process; training decisions draw noise as a real model does.
:class:`TinyTrainableSpec` stands in for a ``TrainablePolicySpec``, so the
production :class:`~isaac_rl_posttraining.policies.gr00t.flow_sde_policy.FlowSDEPolicySpec`
builds the rollout policy from it on a worker or a policy server.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from entry_doubles import ScriptedSimulator, scripted_environment
from torch import nn

from isaac_rl_posttraining.policies.feature_cache import unpack_features
from isaac_rl_posttraining.policies.flow_sde import FlowConditioning, FlowSDEActorCritic
from isaac_rl_posttraining.policies.gr00t.flow_sde_policy import FlowSDEPolicySpec
from isaac_rl_posttraining.policies.trainable import FlowSDEConfig
from isaac_rl_posttraining.runtimes import BatchedWorkerSpec, SparseOutcomeReward

__all__ = [
    "INSTRUCTION",
    "TinyFlowSDE",
    "TinyScenes",
    "TinyTrainableSpec",
    "tiny_flow_sde_spec",
    "tiny_worker_spec",
]

STATE, FEATURES, HORIZON, WIDTH, WINDOW, STEPS = 3, 4, 4, 2, 2, 3
INSTRUCTION = "stack the cubes"


class TinyFlowSDE(FlowSDEActorCritic):
    """Flow-SDE over linear layers; every copy starts from the same weights."""

    frozen_submodules = ("backbone",)
    cached_feature_names = ("features",)

    def __init__(self, noise_level: float = 0.5) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(3)
            super().__init__(
                video_keys=["video.cam"],
                state_keys=["state.q"],
                action_window=WINDOW,
                valid_action_dim=WIDTH,
                chunk_shape=(HORIZON, WIDTH),
                value_input_dim=FEATURES,
                config=FlowSDEConfig(
                    noise_level=noise_level,
                    denoising_steps=STEPS,
                    value_hidden_sizes=(8,),
                ),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.backbone = nn.Linear(STATE, FEATURES)
            self.backbone.requires_grad_(False)
            self.head = nn.Linear(FEATURES + WIDTH + 1, WIDTH)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def build_inputs(self, frames, states, instruction):
        assert instruction == INSTRUCTION and frames["video.cam"].ndim == 4
        return {"state": torch.as_tensor(np.asarray(states["state.q"], np.float32))}

    def act(self, inputs, *, stochastic_step, noise=None):
        # Real models start every decode from random noise, drawn from the acting process's
        # torch generator; evaluation here starts from zeros, so it is deterministic anywhere.
        if noise is None and stochastic_step is None:
            noise = torch.zeros(inputs["state"].shape[0], HORIZON, WIDTH)
        return super().act(inputs, stochastic_step=stochastic_step, noise=noise)

    def decode(self, normalized, *, inputs, states):
        return normalized[:, :WINDOW, :].float().numpy()

    def condition(
        self,
        inputs: Mapping[str, torch.Tensor],
        cached_features: Mapping[str, Any] | None = None,
    ):
        state = inputs["state"].float().reshape(-1, STATE)
        if cached_features is None:
            features = self.backbone(state)
        else:
            features = (
                unpack_features(cached_features, self.device)["features"]
                .float()
                .reshape(-1, FEATURES)
            )
        return FlowConditioning(
            context=features, cached={"features": features}, batch_size=state.shape[0]
        )

    def velocity(self, conditioning, x_t, step_index):
        batch = x_t.shape[0]
        time = (step_index.float() / STEPS)[:, None, None].expand(batch, HORIZON, 1)
        context = conditioning.context[:, None, :].expand(batch, HORIZON, FEATURES)
        return torch.tanh(self.head(torch.cat([context, x_t.float(), time], dim=-1)))

    def value_features(self, conditioning):
        return conditioning.context

    @classmethod
    def spec_options(cls, base_model):
        return {}

    @classmethod
    def load(cls, checkpoint, *, device, flow_sde, value_head_seed=0, **options):
        return cls(noise_level=flow_sde.noise_level)


@dataclass(frozen=True)
class TinyTrainableSpec:
    """The parts of a ``TrainablePolicySpec`` that ``FlowSDEPolicySpec`` reads."""

    base_model: str = "tiny_flow_sde"
    noise_level: float = 0.5

    @property
    def policy_class(self) -> type[TinyFlowSDE]:
        return TinyFlowSDE

    def build(self, device: str | None = None) -> TinyFlowSDE:
        return TinyFlowSDE(noise_level=self.noise_level)


def tiny_flow_sde_spec(
    *, cache_features: bool = False, base_seed: int = 0
) -> FlowSDEPolicySpec:
    """The production rollout-policy spec over the tiny model."""
    return FlowSDEPolicySpec(
        spec=TinyTrainableSpec(),  # type: ignore[arg-type]
        action_window=WINDOW,
        instruction=INSTRUCTION,
        base_seed=base_seed,
        cache_features=cache_features,
    )


class TinyScenes:
    """One worker's scenes in lockstep; scene ``i`` succeeds once its first action coordinate summed past 0.

    Episodes run ``chunks_per_episode`` chunks unless a scene succeeds, so
    rewards depend on the actions and PPO has a signal.
    """

    def __init__(self, indices: tuple[int, ...]) -> None:
        self.indices = np.asarray(indices, np.float32)
        self.num_envs = len(indices)
        self.t = 0
        self.progress = np.zeros(self.num_envs, np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        state = np.stack(
            [
                0.1 * self.indices,
                self.progress,
                np.full(self.num_envs, 0.05 * self.t, np.float32),
            ],
            axis=1,
        )
        return {
            "video.cam": np.zeros((self.num_envs, 2, 2, 3), np.uint8),
            "state.q": state.astype(np.float32),
        }

    def reset(self, *, seed=None):
        self.t = 0
        self.progress = np.zeros(self.num_envs, np.float32)
        return self._observation(), {}

    def step(self, actions):
        self.t += 1
        self.progress = self.progress + np.asarray(actions, np.float32)[:, 0]
        terminated = self.progress > 0.5
        return (
            self._observation(),
            np.zeros(self.num_envs),
            terminated,
            np.zeros(self.num_envs, dtype=bool),
            [{"success": bool(done)} for done in terminated],
        )

    def close(self) -> None:
        pass


def _make_scenes(context) -> TinyScenes:
    return TinyScenes(context.environment_indices)


def tiny_worker_spec(
    num_envs: int, *, chunks_per_episode: int = 3, cache_features: bool = False
) -> BatchedWorkerSpec:
    """The flow-SDE PPO rollout workers of ``flow_sde_ppo_launch`` over :class:`TinyScenes`."""
    return BatchedWorkerSpec(
        entry=ScriptedSimulator(_make_scenes),
        environment=scripted_environment(num_envs),
        phase="train",
        policy=tiny_flow_sde_spec(cache_features=cache_features),
        chunks_per_episode=chunks_per_episode,
        reward_transform=SparseOutcomeReward(
            success_reward=1.0, failure_reward=0.0, step_reward=0.0
        ),
        bootstrap_values="reuse_next_decision",
        supports_served_policy=True,
    )
