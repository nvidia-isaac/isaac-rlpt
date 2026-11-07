# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Doubles for policies over a frozen base model: a base model that draws from torch's generator in every call,
latent and edit actors, scripted environments and the worker specs over them."""

from __future__ import annotations

import numpy as np
import torch
from entry_doubles import ScriptedSimulator, scripted_environment, scripted_worker_spec

from isaac_rl_posttraining.algorithms.rfs.modulation import RFSActionModulator
from isaac_rl_posttraining.envs.reference_action import REFERENCE_KEY
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    EditPolicySpec,
    FrozenBaseModelSpec,
    NativeRuntime,
    SteeringPolicySpec,
)

HORIZON, WIDTH = 3, 2


class DrawingBaseModel:
    """A frozen base model whose every call draws from torch's generator."""

    supports_batched_infer = True

    def __init__(self) -> None:
        self.started: list[object] = []

    @staticmethod
    def _offset(observations) -> np.ndarray:
        state = np.asarray(observations["state"], np.float32)
        return state.reshape(state.shape[0], -1).sum(axis=1)[:, None, None]

    @staticmethod
    def _steer(noise, batch: int) -> np.ndarray | float:
        if noise is None:
            return 0.0
        return np.asarray(noise, np.float32).reshape(batch, 1, -1)[..., :WIDTH]

    def infer(self, observations, noise):
        draw = torch.randn(1, HORIZON, WIDTH).numpy()
        actions = 0.1 * draw + self._steer(noise, 1) + self._offset(observations)
        return {"actions": actions.astype(np.float32)}

    def infer_batch(self, observations, noise):
        offset = self._offset(observations)
        draw = torch.randn(offset.shape[0], HORIZON, WIDTH).numpy()
        actions = 0.1 * draw + self._steer(noise, offset.shape[0]) + offset
        return {"actions": actions.astype(np.float32)}

    def prepare_seeded_batch_noise(self, noise):
        return self._steer(noise, 1) + torch.randn(1, HORIZON, WIDTH).numpy()

    def prepare_seeded_native_noise(self):
        return torch.randn(1, HORIZON, WIDTH).numpy()

    def infer_prepared_batch(self, observations, initial_noise):
        actions = np.asarray(initial_noise) + self._offset(observations)
        return {"actions": actions.astype(np.float32)}

    def infer_reference(self, observations, *, image_transform_mode="eval"):
        features = torch.randn(1, 4) + (1.0 if image_transform_mode == "eval" else 0.0)
        reference = 0.1 * torch.randn(1, HORIZON, WIDTH).numpy() + self._offset(
            observations
        )
        return features, reference.astype(np.float32)

    def start_episode(self, session_id=None):
        self.started.append(session_id)

    def __call__(self, observations, noise):
        """Like real base models, callable as ``infer`` (Ray workers then receive the class by value)."""
        return self.infer(observations, noise)


class ReferenceBaseModel(DrawingBaseModel):
    """A base model whose reference chunk draws nothing (SAC-edit's actor then decides alike anywhere)."""

    def infer_reference(self, observations, *, image_transform_mode="eval"):
        offset = self._offset(observations)
        reference = (
            np.broadcast_to(offset, (1, HORIZON, WIDTH))
            + np.arange(HORIZON)[None, :, None]
        )
        return torch.ones(1, 4), reference.astype(np.float32)


class NoisyActor(torch.nn.Module):
    """A latent actor that samples around a mean read from the state."""

    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(width))

    def act(self, observation, actor_state=None, deterministic=False):
        mean = torch.tanh(observation["state"][:, :1]).expand(-1, self.bias.shape[0])
        mean = mean + self.bias
        if deterministic:
            return mean, actor_state
        return mean + 0.5 * torch.randn_like(mean), actor_state


class EditActor(torch.nn.Module):
    """An edit actor: half the reference chunk, plus noise when sampling."""

    normalize_actions = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def act(self, observation, deterministic=False):
        edit = self.scale * observation[REFERENCE_KEY]
        if deterministic:
            return edit, None
        return edit + 0.1 * torch.randn_like(edit), None


class Sampler:
    def __init__(self, width: int = WIDTH) -> None:
        self.width = width

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(1, self.width)).astype(np.float32)


class Env:
    """One environment; its state counts steps from an offset and from the reset seed."""

    def __init__(self, offset: float, horizon: int = 4) -> None:
        self.offset, self.horizon, self.t, self.seed = offset, horizon, 0, 0

    def _observation(self):
        return {
            "state": np.asarray([self.offset + self.t, 0.01 * self.seed], np.float32)
        }

    def reset(self, *, seed=None):
        self.t, self.seed = 0, 0 if seed is None else seed % 97
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        return (
            self._observation(),
            float(np.sum(action)),
            done,
            False,
            {"success": done},
        )

    def close(self):
        pass


class BatchedEnv:
    """Environments of one worker in lockstep; environment ``i`` ends after ``3 + i % 2`` steps."""

    def __init__(self, indices) -> None:
        self.indices = np.asarray(indices, np.float32)
        self.num_envs = len(indices)
        self.t, self.seed = 0, 0

    def _observation(self):
        return {
            "state": np.stack(
                [
                    10.0 * self.indices + self.t,
                    np.full(self.num_envs, 0.01 * self.seed, np.float32),
                ],
                axis=1,
            ).astype(np.float32)
        }

    def reset(self, *, seed=None):
        self.t, self.seed = 0, 0 if seed is None else seed % 97
        return self._observation(), {}

    def step(self, actions):
        self.t += 1
        terminated = self.t >= 3 + (self.indices % 2)
        return (
            self._observation(),
            np.asarray(actions, np.float64).sum(axis=1),
            terminated,
            np.zeros(self.num_envs, dtype=bool),
            [{"success": bool(done)} for done in terminated],
        )

    def close(self):
        pass


def steering_spec(
    base_model, family: str, *, deterministic: bool = False, native: bool = False
):
    width = 2 * WIDTH if family == "rfs" else WIDTH
    return SteeringPolicySpec(
        actor=NoisyActor(width),
        base_model=FrozenBaseModelSpec(prebuilt=base_model),
        latent_sampler=Sampler(width),
        steering_key_mapping={"state": "state", "image": []},
        action_exec_len=2,
        action_modulator=RFSActionModulator(WIDTH, WIDTH, 0.1)
        if family == "rfs"
        else None,
        reference_action_key=REFERENCE_KEY if family == "rfs_reference" else None,
        deterministic_evaluation=deterministic,
        native=native,
    )


def edit_spec(base_model, *, deterministic: bool = False):
    return EditPolicySpec(
        actor=EditActor(),
        base_model=FrozenBaseModelSpec(prebuilt=base_model),
        action_indices=(0, 1),
        environment_action_defaults=np.zeros(WIDTH, np.float32),
        action_horizon=HORIZON,
        reference_action_transform="identity",
        relative_action_scale=1.0,
        reference_image_transform_mode="train",
        state_key="state",
        image_keys=(),
        deterministic_evaluation=deterministic,
    )


def policy_spec(family: str, base_model, **options):
    return (
        edit_spec(base_model, **options)
        if family == "edit"
        else steering_spec(base_model, family, **options)
    )


def gym_spec(policy, num_envs: int):
    return scripted_worker_spec(
        lambda context: Env(offset=10.0 * context.environment_indices[0]),
        policy,
        num_envs=num_envs,
        supports_batched_policy=True,
    )


def batched_spec(policy, num_envs: int):
    return BatchedWorkerSpec(
        entry=ScriptedSimulator(
            lambda context: BatchedEnv(context.environment_indices)
        ),
        environment=scripted_environment(num_envs),
        phase="train",
        policy=policy,
        chunks_per_episode=3,
        supports_served_policy=True,
        bootstrap_values="reuse_next_decision",
    )


def fingerprint(results) -> list[tuple]:
    rows = []
    for result in results:
        for decision in result.decisions:
            learner_action = decision.transition.learner_action
            rows.append(
                (
                    result.worker_index,
                    decision.environment_index,
                    decision.action_chunk.source,
                    np.asarray(decision.action_chunk.environment_actions)
                    .round(6)
                    .tolist(),
                    None
                    if learner_action is None
                    else np.asarray(learner_action.value).round(6).tolist(),
                    decision.transition.rewards["env"].round(6).tolist(),
                )
            )
    return rows


def collect(runtime: NativeRuntime, spec, *, workers: int, num_envs: int, requests):
    """Seed this process's generators, run ``requests`` (one per worker per round) in order, close the runtime."""
    torch.manual_seed(5)
    np.random.seed(5)
    try:
        group = runtime.open_rollout_group(
            spec, num_rollout_workers=workers, num_envs=num_envs, name="parity"
        )
        rows = []
        for request in requests:
            rows.extend(fingerprint(group.collect([request] * workers)))
        return rows
    finally:
        runtime.close()
