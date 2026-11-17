# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Factory's entry into the runtime layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from isaac_rl_posttraining.data.spec import ActionSpec, ObservationSpec
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
    WorkerLayout,
)
from isaac_rl_posttraining.runtimes import RolloutWorkerContext
from simulators.g1_arena.embodiment import G1_WBC_MODALITY, G1_WBC_STATE_GROUP_ORDER
from simulators.g1_arena.env import (
    ArenaBatchedEnv,
    ArenaEnv,
    arena_action_spec,
    arena_observation_spec,
)

__all__ = ["G1Arena", "G1Scenes"]

#: Width of each group of the 31-dim policy state, in the order the simulator packs it.
_STATE_GROUP_WIDTHS = tuple(
    (group, G1_WBC_MODALITY["state"][group]["end"] - G1_WBC_MODALITY["state"][group]["start"])
    for group in G1_WBC_STATE_GROUP_ORDER
)


def _groot_observation(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add the GR00T modality keys (``video.ego_view``, ``state.<group>``) to a batched Arena observation."""
    result = dict(observation)
    result["video.ego_view"] = observation["ego_view_image"]
    state = np.asarray(observation["state"])
    expected = sum(width for _, width in _STATE_GROUP_WIDTHS)
    if state.shape[-1] != expected:
        raise ValueError(f"G1 policy state must be {expected}-dim, got {state.shape}")
    start = 0
    for group, width in _STATE_GROUP_WIDTHS:
        result[f"state.{group}"] = state[..., start : start + width]
        start += width
    return result


class G1Scenes:
    """One batched Arena client seen through the GR00T modality keys the flow-SDE policy reads."""

    def __init__(self, env: ArenaBatchedEnv) -> None:
        self._env = env
        self.num_envs = env.num_envs

    @property
    def need_obs(self) -> bool:
        return self._env.need_obs

    @need_obs.setter
    def need_obs(self, value: bool) -> None:
        self._env.need_obs = value

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
        observation, info = self._env.reset(seed=seed)
        return _groot_observation(observation), info

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observation, reward, terminated, truncated, infos = self._env.step(actions)
        return _groot_observation(observation), reward, terminated, truncated, infos

    def close(self) -> None:
        self._env.close()


class G1Arena(SimulatorEntry):
    """Unitree G1 whole-body control in IsaacLab-Arena, served over the transport.

    One Arena server serves one client, so the layout is one worker per
    server: a scalar client for one environment, or one batched client for a
    server hosting many. A phase spread over several servers (the launcher
    exports their addresses as one comma-separated list) gets one batched
    worker per server, each stepping an equal share of the environments, so
    one server can step while another's frames decode. Training and
    evaluation run on separate servers, so evaluation uses a dedicated
    persistent group; the scalar client executes a whole action chunk per
    server round trip.
    """

    name = "g1_arena"
    evaluation = "dedicated"
    server_url_variables = ("ARENA_SERVER_URL", "ARENA_EVAL_SERVER_URL")
    supports_chunked_step = True
    #: GR00T decodes the whole batch in one forward; this bounds its memory when
    #: one shared policy serves many cloned environments.
    decode_batch_size = 16

    def make_environment(
        self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext
    ) -> ArenaEnv | ArenaBatchedEnv:
        if spec.server_url is None:
            raise ValueError(
                f"G1 Factory {phase} environments need a server address; the launcher exports "
                "ARENA_SERVER_URL and ARENA_EVAL_SERVER_URL"
            )
        if spec.action_dim is None:
            raise ValueError("G1 Factory environments need the client-side action_dim (simulator.action_dim)")
        if spec.image_resolution is None:
            raise ValueError(
                "G1 Factory environments need the client-side image_resolution (simulator.image_resolution)"
            )
        addresses = spec.server_addresses()
        if spec.num_envs == 1:
            if len(context.environment_indices) != 1:
                raise ValueError("scalar Arena workers own exactly one environment")
            if len(addresses) != 1:
                raise ValueError(f"one Arena environment connects to one server; got {len(addresses)} addresses")
            return ArenaEnv(
                server_url=addresses[0],
                image_resolution=spec.image_resolution,
                action_dim=spec.action_dim,
                max_episode_steps=spec.max_episode_steps,
            )
        if len(addresses) != context.num_rollout_workers:
            raise ValueError(
                f"G1 Factory connects one batched worker to one server; {len(addresses)} addresses were exported for "
                f"{context.num_rollout_workers} workers"
            )
        return ArenaBatchedEnv(
            server_url=addresses[context.worker_index],
            num_envs=len(context.environment_indices),
            image_resolution=spec.image_resolution,
            action_dim=spec.action_dim,
            max_episode_steps=spec.max_episode_steps,
        )

    def worker_layout(self, num_envs: int, num_servers: int = 1, *, workers: int | None = None) -> WorkerLayout:
        if isinstance(num_servers, bool) or not isinstance(num_servers, int) or num_servers < 1:
            raise ValueError("num_servers must be a positive integer")
        if workers is not None and workers != num_servers:
            raise ValueError(
                f"G1 Factory runs one batched worker per simulator server; placement.envs.workers={workers} does not "
                f"match the {num_servers} server address(es) exported (NUM_SERVERS in the launchers); leave "
                "placement.envs.workers unset"
            )
        if num_envs % num_servers:
            raise ValueError(
                f"G1 Factory splits the environments equally over its servers; {num_envs} environments do not divide "
                f"by {num_servers} servers"
            )
        if num_servers > 1 and num_envs // num_servers < 2:
            raise ValueError(
                f"each of several Arena servers hosts a batch of at least two environments; {num_envs} environments "
                f"over {num_servers} servers leave {num_envs // num_servers} per server"
            )
        return WorkerLayout(workers=num_servers, batched=num_envs > 1)

    def make_scenes(self, spec: EnvironmentSpec, context: RolloutWorkerContext) -> G1Scenes:
        """The worker's batched client on its training server, for the flow-SDE fine-tuning family."""
        if spec.num_envs < 2:
            raise ValueError("G1 Factory scenes need a batched server (training.num_envs >= 2)")
        environment = self.make_environment(spec, "train", context)
        if not isinstance(environment, ArenaBatchedEnv):
            raise TypeError(f"G1 Factory scenes need a batched client, got {type(environment).__name__}")
        return G1Scenes(environment)

    def instruction(self, spec: EnvironmentSpec) -> str:
        raise ValueError(
            "the G1 Factory task is chosen by the server, so its instruction cannot be derived from "
            f"env.env_name={spec.env_name!r}; set base_model.task_instruction"
        )

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        """What one Arena client emits, from the same declaration its Gymnasium spaces come from."""
        if spec.image_resolution is None:
            raise ValueError(
                "G1 Factory environments need the client-side image_resolution (simulator.image_resolution)"
            )
        return arena_observation_spec(image_resolution=spec.image_resolution)

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        if spec.action_dim is None:
            raise ValueError("G1 Factory environments need the client-side action_dim (simulator.action_dim)")
        return arena_action_spec(spec.action_dim)
