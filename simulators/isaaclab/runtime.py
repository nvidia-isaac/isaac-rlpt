# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab's entry into the runtime layer.

Isaac Sim hosts one simulation per process, so a rollout worker cannot build
an Isaac Lab environment in-process the way the LIBERO entry does. Each
worker instead owns a simulator child process
(:mod:`simulators.isaaclab.subprocess_env`): a single scene behind the
unbatched surface, or all of its scenes cloned in one simulation behind the
batched surface (the steering families with several scenes, and the flow
policy family, GR00T flow-SDE PPO).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from isaac_rl_posttraining.data.spec import ActionSpec, ArraySpec, ObservationSpec
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
    WorkerLayout,
)
from isaac_rl_posttraining.runtimes import RolloutWorkerContext
from simulators.isaaclab.env import ENV_NAME_PREFIX, mapping_for_task
from simulators.isaaclab.subprocess_env import (
    IsaacLabEnvSpec,
    SubprocessIsaacLabEnv,
    startup_delay_seconds,
)

__all__ = [
    "IsaacLab",
    "IsaacLabBatchedScenesEnv",
    "IsaacLabSceneEnv",
    "task_id_from_env_name",
]

#: Spreads the reset seeds of workers apart so no two scenes replay the same draw.
_WORKER_SEED_STRIDE = 1_000_003


class IsaacLabSceneEnv(gym.Env):
    """One scene of a simulator child process behind the unbatched ``GymEnvironment`` surface.

    :class:`SubprocessIsaacLabEnv` speaks the natively batched protocol
    (``[N, ...]`` observations, ``[N, A]`` actions); a steering rollout worker
    owns exactly one scene, so this view strips the batch axis in both directions.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, child: SubprocessIsaacLabEnv) -> None:
        if child.num_envs != 1:
            raise ValueError(f"IsaacLabSceneEnv wraps a one-scene simulation, got {child.num_envs} scenes")
        self._child = child
        # The launcher sizes the learner from these (an evaluation-only run has no other environment).
        self.observation_space = child.single_observation_space
        self.action_space = child.single_action_space
        # The networks see the flat state and image embeddings; the per-group states are the policy's.
        self.steering_remove_obs_keys = [*child.video_keys, *child.state_keys]

    @staticmethod
    def _scene(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.array(value[0], copy=True) for key, value in observation.items()}

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, info = self._child.reset(seed=seed)
        return self._scene(observation), dict(info)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observation, rewards, terminated, truncated, infos = self._child.step(
            np.asarray(action, dtype=np.float32)[None, :]
        )
        return self._scene(observation), float(rewards[0]), bool(terminated[0]), bool(truncated[0]), dict(infos[0])

    def close(self) -> None:
        self._child.close()


class IsaacLabBatchedScenesEnv:
    """All scenes of one worker's simulation behind the batched environment surface.

    A reset seed is offset by the worker's stride, so workers that receive the
    same request seed still draw distinct scenes; a dead simulation child can
    be replaced with :meth:`restart`.
    """

    def __init__(self, child: SubprocessIsaacLabEnv, *, seed_offset: int) -> None:
        self._child = child
        self._seed_offset = int(seed_offset)
        self.num_envs = child.num_envs
        self.single_observation_space = child.single_observation_space
        self.single_action_space = child.single_action_space

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self._child.reset(seed=None if seed is None else seed + self._seed_offset)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        return self._child.step(actions)

    def restart(self) -> None:
        self._child.restart()

    def close(self) -> None:
        self._child.close()


def task_id_from_env_name(env_name: str) -> str:
    """The Isaac Lab task id of ``isaaclab/<Task-Id>``."""
    task_id = env_name[len(ENV_NAME_PREFIX) :]
    if not env_name.startswith(ENV_NAME_PREFIX) or not task_id:
        raise ValueError(f"the Isaac Lab entry expects env.env_name={ENV_NAME_PREFIX}<Task-Id>, got {env_name!r}")
    return task_id


def _scene_task(spec: EnvironmentSpec) -> str:
    """The task id of ``spec``, after checking the values every Isaac Lab simulation needs."""
    if spec.seed is None:
        raise ValueError("Isaac Lab scenes need training.seed to seed each worker's simulation")
    if spec.action_dim is None:
        raise ValueError("Isaac Lab scenes need the client-side action_dim (simulator.action_dim)")
    if spec.image_resolution is None:
        raise ValueError(
            "Isaac Lab scenes need the camera resolution (simulator.image_resolution)"
        )
    return task_id_from_env_name(spec.env_name)


def _task_mapping(spec: EnvironmentSpec) -> dict[str, Any]:
    """The registered GR00T mapping of ``spec``'s task, as a configuration error when there is none."""
    try:
        return mapping_for_task(task_id_from_env_name(spec.env_name))
    except KeyError as exc:
        raise ValueError(f"env.env_name={spec.env_name!r}: {exc.args[0]}") from exc


def _kit_log_path(name: str) -> str:
    """Kit's log file for one worker's simulation, e.g. ``/tmp/kit_isaaclab_ppo_worker0.log``."""
    return os.path.join(tempfile.gettempdir(), f"kit_isaaclab_{name}.log")


class IsaacLab(SimulatorEntry):
    """Isaac Lab 3.0 tasks, one simulator child process per rollout worker.

    Each worker runs one simulation per process, so evaluation shares the
    training workers at episode boundaries when it asks for as many scenes
    (``eval.eval_num_envs``); a different count evaluates on a simulation of
    its own. The steering families run all of
    their scenes as clones in one simulation, stepped as one batch by a single
    worker: one Isaac Sim per scene would not fit a single GPU beyond a few
    scenes.
    """

    name = "isaaclab"
    base_models = ("gr00t", "gr00t_n15", "gr00t_n17")
    simulates_on_gpu = True
    # The learner is sized from the task mapping, so no Isaac Sim starts outside the workers.
    sized_from_declaration = True

    def worker_layout(
        self, num_envs: int, num_servers: int = 1, *, workers: int | None = None
    ) -> WorkerLayout:
        """One worker per phase by default: a single scene unbatched, several scenes as one batched simulation.

        ``workers`` (``placement.envs.workers``) splits the scenes evenly over
        that many workers, one Isaac Sim each (on its own GPU when the
        placement gives the rollout workers GPUs).
        """
        if num_servers != 1:
            raise ValueError(
                f"{self.name} builds its environments on one simulator; {num_servers} server addresses were exported"
            )
        if workers is None or workers == 1:
            return WorkerLayout(workers=1, batched=num_envs > 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"placement.envs.workers must be a positive integer, got {workers!r}")
        if num_envs % workers:
            raise ValueError(
                f"{self.name}: {num_envs} scenes (training.num_envs) do not split evenly over {workers} rollout "
                "workers (placement.envs.workers); make training.num_envs a multiple of placement.envs.workers"
            )
        return WorkerLayout(workers=workers, batched=num_envs > workers)

    def make_environment(
        self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext
    ) -> IsaacLabSceneEnv | IsaacLabBatchedScenesEnv:
        """The worker's scenes in one child process, seeded from the run seed.

        A single scene seeds by its environment index; several scenes share one
        simulation whose resets are offset per worker, as in :meth:`make_scenes`.
        Unlike the flow family, the steering families cap episodes client-side
        at the spec's ``max_episode_steps``.
        """
        task_id = _scene_task(spec)
        if len(context.environment_indices) > 1:
            return IsaacLabBatchedScenesEnv(
                SubprocessIsaacLabEnv(
                    IsaacLabEnvSpec(
                        task_id=task_id,
                        num_envs=len(context.environment_indices),
                        image_size=spec.image_resolution,
                        action_dim=spec.action_dim,
                        seed=spec.seed + context.worker_index,
                        max_episode_steps=spec.max_episode_steps,
                        kit_log_path=_kit_log_path(f"steering_{phase}_worker{context.worker_index}"),
                        startup_delay_seconds=startup_delay_seconds(context.worker_index),
                    )
                ),
                seed_offset=context.worker_index * _WORKER_SEED_STRIDE,
            )
        return IsaacLabSceneEnv(
            SubprocessIsaacLabEnv(
                IsaacLabEnvSpec(
                    task_id=task_id,
                    num_envs=1,
                    image_size=spec.image_resolution,
                    action_dim=spec.action_dim,
                    seed=spec.seed + context.environment_indices[0],
                    max_episode_steps=spec.max_episode_steps,
                    kit_log_path=_kit_log_path(f"steering_worker{context.worker_index}"),
                    startup_delay_seconds=startup_delay_seconds(context.worker_index),
                )
            )
        )

    def make_scenes(self, spec: EnvironmentSpec, context: RolloutWorkerContext) -> IsaacLabBatchedScenesEnv:
        """Every scene assigned to the worker, in one simulation.

        The flow policy family plays whole episodes bounded by its chunk
        horizon, so the spec's episode cap is not enforced client-side here.
        """
        task_id = _scene_task(spec)
        return IsaacLabBatchedScenesEnv(
            SubprocessIsaacLabEnv(
                IsaacLabEnvSpec(
                    task_id=task_id,
                    num_envs=len(context.environment_indices),
                    image_size=spec.image_resolution,
                    action_dim=spec.action_dim,
                    seed=spec.seed + context.worker_index,
                    kit_log_path=_kit_log_path(f"ppo_worker{context.worker_index}"),
                    startup_delay_seconds=startup_delay_seconds(context.worker_index),
                )
            ),
            seed_offset=context.worker_index * _WORKER_SEED_STRIDE,
        )

    def instruction(self, spec: EnvironmentSpec) -> str:
        return str(_task_mapping(spec)["task_description"])

    def steering_key_mapping(self, spec: EnvironmentSpec) -> dict[str, Any]:
        """The checkpoint's own observation keys for the task (cameras are per task)."""
        return {"state": "state", "language": None, "image": list(_task_mapping(spec)["cameras"])}

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        """The task mapping's cameras at ``simulator.image_resolution``, its state groups and their concatenation."""
        if spec.image_resolution is None:
            raise ValueError("Isaac Lab scenes need the camera resolution (simulator.image_resolution)")
        mapping = _task_mapping(spec)
        size = spec.image_resolution
        arrays = {camera: ArraySpec((size, size, 3), "uint8") for camera in mapping["cameras"]}
        widths = {key: int(group["slice"][1]) - int(group["slice"][0]) for key, group in mapping["states"].items()}
        arrays.update({key: ArraySpec((width,), "float32") for key, width in widths.items()})
        arrays["state"] = ArraySpec((sum(widths.values()),), "float32")
        return ObservationSpec(arrays)

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        """One model action of ``simulator.action_dim`` (the scene pads the uncontrolled joints itself)."""
        if spec.action_dim is None:
            raise ValueError("Isaac Lab scenes need the client-side action_dim (simulator.action_dim)")
        return ActionSpec(width=spec.action_dim)
