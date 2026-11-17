# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route an algorithm family to the rollout policy its runtime workers run."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.envs.reference_action import ReferenceActionEnv
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
)
from isaac_rl_posttraining.runtimes.base import RolloutWorkerSpec
from isaac_rl_posttraining.runtimes.edit_policy import EditPolicySpec
from isaac_rl_posttraining.runtimes.gym_worker import (
    RewardTransform,
    SparseOutcomeReward,
)
from isaac_rl_posttraining.runtimes.steering_rollouts import (
    frozen_base_model_spec,
    policy_decode_batch_size,
    resolve_image_keys,
    steering_policy_spec,
    steering_worker_spec,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.algorithms.rfs.modulation import RFSActionModulator

__all__ = [
    "PolicyFamily",
    "edit_policy_spec",
    "reward_transform_for",
    "rollout_worker_spec",
]

#: The off-policy families: ``steering`` acts in the frozen policy's flow-noise
#: space (DSRL, RFS); ``edit`` executes a bounded edit of the frozen policy's
#: decoded chunk (SAC-edit).
PolicyFamily = Literal["steering", "edit"]


def reward_transform_for(
    family: PolicyFamily, cfg: DictConfig, *, phase: Phase
) -> RewardTransform | None:
    """The reward shaping the family's workers record in ``phase``, next to the simulator's own reward.

    Training workers store the sparse outcome reward when ``env.sparse_success_reward``
    is set. Steering evaluation scores the raw environment reward; edit
    evaluation scores the shaped reward.
    """
    if family not in ("steering", "edit"):
        raise ValueError(f"unknown policy family {family!r}")
    if not bool(cfg.env.sparse_success_reward):
        return None
    if family == "steering" and phase == "eval":
        return None
    return SparseOutcomeReward(
        success_reward=float(cfg.env.success_reward),
        failure_reward=float(cfg.env.failure_reward),
        step_reward=float(cfg.env.step_reward),
    )


def edit_policy_spec(
    cfg: DictConfig,
    *,
    actor: torch.nn.Module,
    base_model: Any,
    controller_env: Any,
    reference_env: ReferenceActionEnv,
    image_keys: Sequence[str],
    dino_image_keys: Sequence[str],
    reference_image_transform_mode: str,
    deterministic_evaluation: bool = False,
    native: bool = False,
) -> EditPolicySpec:
    """Build the edit policy spec the rollout workers of one phase share.

    The chunk geometry comes from the reference-action environment the
    controller built, so the workers and the learner cannot disagree on it.
    """
    base_model_spec, dino_wrapper, dino_model_name = frozen_base_model_spec(
        cfg, base_model=base_model, controller_env=controller_env
    )
    return EditPolicySpec(
        actor=copy.deepcopy(actor).cpu(),
        base_model=base_model_spec,
        action_indices=tuple(reference_env.action_indices),
        environment_action_defaults=np.array(
            reference_env.environment_action_defaults, copy=True
        ),
        action_horizon=int(reference_env.action_horizon),
        reference_action_transform=str(reference_env.reference_action_transform),
        relative_action_scale=float(reference_env.relative_action_scale),
        reference_image_transform_mode=reference_image_transform_mode,
        state_key=str(reference_env.state_key),
        image_keys=tuple(image_keys),
        dino_model_name=dino_model_name,
        dino_wrapper=dino_wrapper,
        dino_image_keys=tuple(dino_image_keys),
        deterministic_evaluation=deterministic_evaluation,
        native=native,
    )


def decode_batch_size_for(
    cfg: DictConfig, entry: SimulatorEntry, *, phase: Phase
) -> int | None:
    """Resolve the training decode cap; evaluation decodes all active environments.

    An explicit simulator override takes precedence over the training cap.
    Without either setting, the simulator entry supplies its default.
    """
    if phase == "eval":
        return None
    training_cap = policy_decode_batch_size(cfg, phase)
    configured = OmegaConf.select(cfg, "simulator.decode_batch_size")
    if configured is None:
        if training_cap is not None:
            return training_cap
        return entry.decode_batch_size
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        raise ValueError(
            f"simulator.decode_batch_size must be a positive integer, got {configured!r}"
        )
    return configured


def rollout_worker_spec(
    family: PolicyFamily,
    cfg: DictConfig,
    entry: SimulatorEntry,
    spec: EnvironmentSpec,
    *,
    actor: torch.nn.Module,
    base_model: Any,
    controller_env: Any,
    latent_env: Any,
    action_modulator: RFSActionModulator | None,
    phase: Phase,
    configured_workers: int | None,
) -> RolloutWorkerSpec:
    """Compose the entry's environments with the family's policy for one phase.

    ``controller_env`` is the controller's sizing environment (the native
    runtime finds the frozen policy's DINO wrapper on it); ``latent_env`` is
    the learner's environment stand-in the policy samples from: the latent
    sampler for the steering family, the reference-action environment for the
    edit family. The reward the workers record is :func:`reward_transform_for`.
    """
    evaluation = phase == "eval"
    steering_key_mapping = dict(entry.steering_key_mapping(spec))
    image_keys = resolve_image_keys(cfg, entry, spec).keys
    deterministic_evaluation = evaluation and str(cfg.eval.action_selection) == "mean"
    native = evaluation and bool(cfg.eval.eval_base)
    reward_transform = reward_transform_for(family, cfg, phase=phase)
    if family == "steering":
        policy: Any = steering_policy_spec(
            cfg,
            actor=actor,
            base_model=base_model,
            controller_env=controller_env,
            latent_env=latent_env,
            action_modulator=action_modulator,
            steering_key_mapping=steering_key_mapping,
            dino_image_keys=image_keys,
            decode_batch_size=decode_batch_size_for(cfg, entry, phase=phase),
            phase=phase,
            deterministic_evaluation=deterministic_evaluation,
            native=native,
        )
        action_exec_len = int(cfg.dsrl.action_exec_len)
    elif family == "edit":
        policy = edit_policy_spec(
            cfg,
            actor=actor,
            base_model=base_model,
            controller_env=controller_env,
            reference_env=latent_env,
            image_keys=steering_key_mapping["image"],
            dino_image_keys=image_keys,
            reference_image_transform_mode=(
                "eval"
                if evaluation
                else str(cfg.direct_action.reference_image_transform_mode)
            ),
            deterministic_evaluation=deterministic_evaluation,
            native=native,
        )
        action_exec_len = int(latent_env.action_horizon)
    else:
        raise ValueError(f"unknown policy family {family!r}")
    return steering_worker_spec(
        entry,
        spec,
        phase,
        policy=policy,
        action_exec_len=action_exec_len,
        reward_transform=reward_transform,
        configured_workers=configured_workers,
    )
