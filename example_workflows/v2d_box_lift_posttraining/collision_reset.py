# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision validation in the native Isaac Lab reset event lifecycle."""

from functools import partial

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase
from robotic_grounding.tasks.v2d_whole_body.mdp.events import (
    reset_robot_to_trajectory_start,
)

from example_workflows.v2d_box_lift_posttraining.fixed_reset import JointNoiseGroup
from example_workflows.v2d_box_lift_posttraining.reset_collision import (
    ResetCollisionChecker,
)
from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)
from example_workflows.v2d_box_lift_posttraining.reset_validation import ResetSampler


class CollisionValidatedReset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        config: ResetValidationConfig = cfg.params["reset_validation"]
        self.sampler = ResetSampler(
            env.num_envs,
            seed=cfg.params["seed"],
            fixed=cfg.params["fixed_reset"],
            device=torch.device(env.device),
            config=config,
        )
        self.collision_checker = ResetCollisionChecker(env, config)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        *,
        reset_validation: ResetValidationConfig,
        seed: int,
        fixed_reset: bool,
        command_name: str,
        trajectory_time_index: tuple[int, int],
        joint_position_noise_groups: dict[str, JointNoiseGroup],
        object_xy_noise_range: tuple[float, float],
        object_yaw_noise_range: tuple[float, float],
    ) -> None:
        sample = partial(
            reset_robot_to_trajectory_start,
            env,
            command_name=command_name,
            trajectory_time_index=trajectory_time_index,
            joint_position_noise_groups=joint_position_noise_groups,
            object_xy_noise_range=object_xy_noise_range,
            object_yaw_noise_range=object_yaw_noise_range,
        )
        self.sampler.reset(env_ids, sample=sample, check=self.collision_checker.check)

    def reseed(self, env_ids: torch.Tensor | None, seed: int) -> None:
        self.sampler.reseed(env_ids, seed)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return self.sampler.diagnostics()
