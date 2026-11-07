# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulator entry the rollout workers build ``MyRobotEnv`` from."""

from __future__ import annotations

from isaac_rl_posttraining.data import ActionSpec
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

from .env import IMAGE_SIZE, NUM_JOINTS, MyRobotEnv, MyRobotObservation


class MyRobot(SimulatorEntry):
    name = "my_robot"

    def make_environment(self, spec, phase, context):
        return MyRobotEnv(
            task=spec.env_name.split("/", 1)[1],
            max_episode_steps=spec.max_episode_steps,
            seed=None
            if spec.seed is None
            else spec.seed + context.environment_indices[0],
        )

    def observation_spec(self, spec):
        return MyRobotObservation.declared_spec(
            {"H": IMAGE_SIZE, "W": IMAGE_SIZE, "state_dim": NUM_JOINTS}
        )

    def action_spec(self, spec):
        return ActionSpec(width=NUM_JOINTS)
