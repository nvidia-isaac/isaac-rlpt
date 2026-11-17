# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat physical reset draws independently for each simulator environment."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JointNoiseGroup(TypedDict):
    joint_names: list[str]
    range: tuple[float, float]


def repeat_reset_draws(
    reset: Callable[[torch.Tensor], None],
    env_ids: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> None:
    """Keep each environment's draws stable under reordered or partial resets."""
    if device.type == "cpu":
        cuda_devices = []
    elif device.type == "cuda":
        cuda_devices = [device]
    else:
        raise ValueError("Fixed V2D resets require a CPU or CUDA device")
    # V2D's trajectory reset samples physical perturbations using torch only.
    # Restore the surrounding streams, including when the reset raises.
    with torch.random.fork_rng(devices=cuda_devices):
        for env_id in env_ids:
            env_seed = (seed + int(env_id)) % (2**63)
            cpu_generator = torch.Generator(device="cpu").manual_seed(env_seed)
            torch.set_rng_state(cpu_generator.get_state())
            if device.type == "cuda":
                generator = torch.Generator(device=device).manual_seed(env_seed)
                torch.cuda.set_rng_state(generator.get_state(), device=device)
            reset(env_id.reshape(1))


def reset_to_fixed_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    seed: int,
    command_name: str,
    trajectory_time_index: tuple[int, int],
    joint_position_noise_groups: dict[str, JointNoiseGroup],
    object_xy_noise_range: tuple[float, float],
    object_yaw_noise_range: tuple[float, float],
) -> None:
    """Apply the released reset with a fixed per-environment perturbation seed."""
    from robotic_grounding.tasks.v2d_whole_body.mdp.events import (
        reset_robot_to_trajectory_start,
    )

    reset = partial(
        reset_robot_to_trajectory_start,
        env,
        command_name=command_name,
        trajectory_time_index=trajectory_time_index,
        joint_position_noise_groups=joint_position_noise_groups,
        object_xy_noise_range=object_xy_noise_range,
        object_yaw_noise_range=object_yaw_noise_range,
    )
    repeat_reset_draws(reset, env_ids, seed=seed, device=torch.device(env.device))
