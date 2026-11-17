# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose a simulator entry and a steering policy into the rollout worker spec of one phase."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
    requested_layout,
)
from isaac_rl_posttraining.rollouts.steering_step import SteeringStep
from isaac_rl_posttraining.runtimes.base import PolicyContext, RolloutWorkerSpec
from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec
from isaac_rl_posttraining.runtimes.gym_worker import (
    GymWorkerSpec,
    RewardTransform,
)
from isaac_rl_posttraining.runtimes.policy import RolloutPolicy
from isaac_rl_posttraining.runtimes.steering_policy import (
    FrozenBaseModelSpec,
    SteeringPolicySpec,
)

if TYPE_CHECKING:
    import gymnasium as gym

    from isaac_rl_posttraining.algorithms.rfs.modulation import RFSActionModulator
    from isaac_rl_posttraining.envs.steering_wrappers import LatentSpaceEnv
    from isaac_rl_posttraining.policies.base import BaseModelPolicy

__all__ = [
    "ImageKeys",
    "frozen_base_model_spec",
    "policy_decode_batch_size",
    "resolve_image_keys",
    "steering_policy_spec",
    "steering_worker_spec",
]


def policy_decode_batch_size(cfg: DictConfig, phase: Phase) -> int | None:
    """Frozen-policy decode cap for one phase.

    Evaluation deliberately stays uncapped so every active evaluation
    environment is decoded in one call. Training has an independent optional
    cap because its environment count and memory envelope can be much larger.
    """
    if phase == "eval":
        return None
    if phase != "train":
        raise ValueError(f"unknown rollout phase {phase!r}")
    configured = OmegaConf.select(
        cfg, "training.policy_decode_batch_size", default=None
    )
    if configured is None:
        return None
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise TypeError("training.policy_decode_batch_size must be a positive integer")
    if configured < 1:
        raise ValueError("training.policy_decode_batch_size must be a positive integer")
    return configured


@dataclass(frozen=True)
class ImageKeys:
    """The camera entries a run's image encoder embeds, and the setting that named them."""

    keys: tuple[str, ...]
    source: str

    def mismatch(self, cameras: Iterable[str]) -> str | None:
        """Why an observation with these camera entries cannot serve the keys; ``None`` when it can."""
        available = list(cameras)
        missing = [key for key in self.keys if key not in available]
        if not missing:
            return None
        return (
            f"image keys {missing} from {self.source} are not camera entries of the observation "
            f"(cameras: {available}); set env.image_keys to camera entries of the observation"
        )


def resolve_image_keys(
    cfg: DictConfig, entry: SimulatorEntry, spec: EnvironmentSpec
) -> ImageKeys:
    """The cameras the learner embeds: ``env.image_keys`` when set, else the entry's mapped image keys.

    A run that sets ``env.image_keys`` chooses its cameras explicitly; a null
    setting follows the simulator entry, so an entry brought from outside the
    repository needs no config change for its own camera names.
    """
    configured = OmegaConf.select(cfg, "env.image_keys", default=None)
    if configured is not None:
        return ImageKeys(tuple(str(key) for key in configured), "env.image_keys")
    mapped = entry.steering_key_mapping(spec)["image"]
    return ImageKeys(
        tuple(str(key) for key in mapped), f"{entry.name}.steering_key_mapping"
    )


def frozen_base_model_spec(
    cfg: DictConfig,
    *,
    base_model: BaseModelPolicy,
    controller_env: gym.Env | gym.vector.VectorEnv,
) -> tuple[FrozenBaseModelSpec, Any, str | None]:
    """How a rollout policy obtains its frozen base model and DINO encoder under the selected runtime.

    The native runtime in one process reuses the controller's already-loaded
    frozen policy and DINO wrapper (found on ``controller_env``); runtimes with
    other processes rebuild both inside them from the resolved config. Returns
    ``(base_model, dino_wrapper, dino_model_name)``.
    """
    # Lowering the placement section is the runners' job; imported here to keep it in one place.
    from isaac_rl_posttraining.runners.run_placement import native_runs_processes

    if str(cfg.runtime.name) == "native" and not native_runs_processes(cfg):
        return (
            FrozenBaseModelSpec(prebuilt=base_model),
            SteeringStep.find_dino_wrapper(controller_env),
            None,
        )
    return (
        FrozenBaseModelSpec(config=OmegaConf.to_container(cfg, resolve=True)),
        None,
        OmegaConf.select(cfg, "model.dinov2_model", default=None),
    )


def steering_policy_spec(
    cfg: DictConfig,
    *,
    actor: torch.nn.Module,
    base_model: BaseModelPolicy,
    controller_env: gym.Env | gym.vector.VectorEnv,
    latent_env: LatentSpaceEnv,
    action_modulator: RFSActionModulator | None,
    steering_key_mapping: dict[str, Any],
    dino_image_keys: Sequence[str],
    decode_batch_size: int | None,
    phase: Phase,
    deterministic_inference: bool = False,
    sample_evaluation_actions: bool = False,
    deterministic_evaluation: bool = False,
    native: bool = False,
) -> SteeringPolicySpec:
    """Build the steering policy spec the rollout workers of one phase share.

    ``dsrl.batched_actor`` makes a served policy run the actor once for all
    the sessions of a batch (off by default; per-session evaluation noise is kept).
    """
    base_model_spec, dino_wrapper, dino_model_name = frozen_base_model_spec(
        cfg, base_model=base_model, controller_env=controller_env
    )
    latent_sampler = copy.deepcopy(latent_env)
    reference_action_key = (
        str(latent_sampler.reference_action_key)
        if latent_sampler.include_reference_action
        else None
    )
    return SteeringPolicySpec(
        actor=copy.deepcopy(actor).cpu(),
        base_model=base_model_spec,
        latent_sampler=latent_sampler,
        steering_key_mapping=dict(steering_key_mapping),
        action_exec_len=int(cfg.dsrl.action_exec_len),
        action_modulator=copy.deepcopy(action_modulator),
        dino_model_name=dino_model_name,
        dino_wrapper=dino_wrapper,
        dino_image_keys=tuple(dino_image_keys),
        use_vlm_features=bool(cfg.dsrl.use_vlm_features),
        reference_action_key=reference_action_key,
        # The frozen policy's reference decode sees evaluation-time image
        # preprocessing only in the evaluation phase, as the serial collector
        # and evaluator did.
        reference_image_transform_mode="eval" if phase == "eval" else "train",
        deterministic_inference=deterministic_inference,
        sample_evaluation_actions=sample_evaluation_actions,
        deterministic_evaluation=deterministic_evaluation,
        native=native,
        max_decode_batch_size=decode_batch_size,
        batched_actor=bool(OmegaConf.select(cfg, "dsrl.batched_actor", default=False)),
    )


def steering_worker_spec(
    entry: SimulatorEntry,
    spec: EnvironmentSpec,
    phase: Phase,
    *,
    policy: Callable[[PolicyContext], RolloutPolicy],
    action_exec_len: int,
    reward_transform: RewardTransform | None,
    configured_workers: int | None,
) -> RolloutWorkerSpec:
    """Attach the entry's environments and a steering-family policy for one phase.

    Unbatched simulators get one persistent environment per worker behind a
    shared policy; natively batched simulators get one fixed-horizon worker per
    simulator server, each driving its share of the environments in lockstep
    with its own policy, or a policy server's when the runtime serves it.
    The workers build the environments themselves, through
    ``entry.make_environment``.
    """
    layout = requested_layout(
        entry, spec.num_envs, num_servers=spec.num_servers, workers=configured_workers
    )
    if not layout.batched:
        return GymWorkerSpec(
            entry=entry,
            environment=spec,
            phase=phase,
            policy=policy,
            reward_transform=reward_transform,
            supports_batched_policy=True,
            chunked_step=entry.supports_chunked_step,
        )
    if configured_workers not in (None, layout.workers):
        raise ValueError(
            f"{entry.name} hosts all environments in {layout.workers} batched "
            f"worker(s); set placement.envs.workers={layout.workers}"
        )
    return BatchedWorkerSpec(
        entry=entry,
        environment=spec,
        phase=phase,
        policy=policy,
        chunks_per_episode=math.ceil(spec.max_episode_steps / action_exec_len),
        reward_transform=reward_transform,
        supports_served_policy=True,
    )
