# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V2D's lightweight attachment to the simulator-neutral runtime layer."""

from __future__ import annotations

from isaac_rl_posttraining.data import ActionSpec, ObservationSpec
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
    WorkerLayout,
)
from isaac_rl_posttraining.runtimes import RolloutWorkerContext
from simulators.v2d.contract import (
    IMAGE_KEYS,
    JOINT_DIM,
    v2d_action_spec,
    v2d_observation_spec,
)
from simulators.v2d.env import V2DBatchedEnv, V2DEnv

__all__ = ["V2D"]


def task_id_from_env_name(env_name: str) -> str:
    prefix = "v2d/"
    task_id = env_name[len(prefix) :]
    if not env_name.startswith(prefix) or not task_id:
        raise ValueError(f"the V2D entry expects env.env_name={prefix}<task>, got {env_name!r}")
    return task_id


class V2D(SimulatorEntry):
    """A dedicated remote V2D server per phase, scalar or natively batched."""

    name = "v2d"
    evaluation = "dedicated"
    server_url_variables = ("V2D_SERVER_URL", "V2D_EVAL_SERVER_URL")
    base_models = ("gr00t",)
    supports_chunked_step = True

    def make_environment(
        self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext
    ) -> V2DEnv | V2DBatchedEnv:
        if spec.server_url is None:
            raise ValueError(
                f"V2D {phase} environments need a server address; export "
                "V2D_SERVER_URL and V2D_EVAL_SERVER_URL"
            )
        if spec.num_servers != 1:
            raise ValueError(f"V2D connects one client to one server; got {spec.num_servers} addresses")
        if spec.action_dim != JOINT_DIM:
            raise ValueError(f"V2D requires simulator.action_dim={JOINT_DIM}")
        if spec.image_resolution != 0:
            raise ValueError("V2D requires native-resolution cameras (simulator.image_resolution=0)")
        task_id = task_id_from_env_name(spec.env_name)
        if spec.num_envs == 1:
            if len(context.environment_indices) != 1:
                raise ValueError("scalar V2D workers own exactly one environment")
            return V2DEnv(
                spec.server_url,
                max_episode_steps=spec.max_episode_steps,
                task_id=task_id,
            )
        return V2DBatchedEnv(
            spec.server_url,
            num_envs=len(context.environment_indices),
            max_episode_steps=spec.max_episode_steps,
            task_id=task_id,
        )

    def worker_layout(self, num_envs: int, num_servers: int = 1, *, workers: int | None = None) -> WorkerLayout:
        if num_servers != 1:
            raise ValueError(f"V2D hosts a phase's environments on one server; {num_servers} addresses were exported")
        if workers is not None and workers != 1:
            raise ValueError(
                f"V2D runs one batched rollout worker on its one simulator server; placement.envs.workers={workers} "
                "asks for more; leave placement.envs.workers unset"
            )
        return WorkerLayout(workers=1, batched=num_envs > 1)

    def steering_key_mapping(self, spec: EnvironmentSpec) -> dict[str, object]:
        del spec
        return {"state": "state", "language": None, "image": list(IMAGE_KEYS)}

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        del spec
        return v2d_observation_spec()

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        if spec.action_dim != JOINT_DIM:
            raise ValueError(f"V2D requires simulator.action_dim={JOINT_DIM}")
        return v2d_action_spec()
