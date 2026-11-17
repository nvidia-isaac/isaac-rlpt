# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium client for a remote Isaac Lab Arena environment.

The Arena simulator runs under its own Python environment (see
``simulators/g1_arena/env_server.py``); the learner drives it from this process
through an :class:`~isaac_rl_posttraining.envs.transport.EnvTransport`. The
client is deliberately thin: it turns transport payloads into the Gymnasium
``reset``/``step`` API, enforces a client-side episode-length cap, and
survives server restarts by reconnecting between episodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger

from isaac_rl_posttraining.data import (
    ActionSpec,
    Array,
    ArraySpec,
    Observation,
    ObservationSpec,
    dim,
)
from isaac_rl_posttraining.envs.transport import (
    EnvTransport,
    TransportError,
    handshake,
    transport_from_uri,
)
from simulators.g1_arena.image_ops import shrunk_hw
from simulators.g1_arena.timers import RoundTripTimer

NATIVE_IMAGE_HW: tuple[int, int] = (480, 640)
"""Native height/width of the G1 ego (head) camera."""


def spec_image_hw(spec: dict[str, Any], native_hw: tuple[int, int]) -> tuple[int, int]:
    """Frame size the server will send, from its spec.

    A server started with ``--image-shortest-edge`` advertises the edge under
    ``spaces.observation.ego_view.shortest_edge`` and sends frames shrunk to it
    (aspect preserved); native-frame servers and older servers advertise no
    edge (or ``None``) and send ``native_hw`` frames.
    """
    # Compatibility boundary: servers predating the spaces block, and the minimal
    # transports the runtime tests stand in with, advertise no image spaces at all.
    spaces = spec["spaces"]["observation"] if "spaces" in spec else {}
    ego = spaces["ego_view"] if "ego_view" in spaces else {}
    edge = ego["shortest_edge"] if "shortest_edge" in ego else None
    if edge is None:
        return native_hw
    return shrunk_hw(native_hw, int(edge))

DEFAULT_ACTION_DIM = 35
"""Whole-body-control joint action: arms (14) + hands (14) + waist (3) +
base height (1) + navigation (3)."""

DEFAULT_STATE_DIM = 31
"""Proprioceptive state in the policy's modality order: left_arm, right_arm,
left_hand, right_hand, waist (7+7+7+7+3). Legs are whole-body-controlled and
excluded, matching the checkpoint's 31-dim state modality."""

ARENA_KEY_MAPPING: dict[str, Any] = {
    "state": "state",
    "language": None,  # single-task Arena environments carry no language embedding
    "image": ["ego_view_image"],
}
"""Observation-key mapping consumed by the latent-steering rollout workers."""


@dataclass(frozen=True)
class ArenaObservation(Observation):
    """What one Arena client emits per step: the ego camera frame and the proprioceptive state.

    The annotations are the schema the launch plan checks the policy against;
    ``H`` and ``W`` are the camera size the client is configured for and
    ``state_dim`` the policy's arm/hand/waist proprioception width.
    """

    ego_view_image: Array[np.uint8, dim.H, dim.W, 3]
    state: Array[np.float32, dim.state_dim]


def arena_image_hw(image_resolution: int) -> tuple[int, int]:
    """The camera frame size a client with ``image_resolution`` declares (native when 0)."""
    if image_resolution > 0:
        return (int(image_resolution), int(image_resolution))
    return NATIVE_IMAGE_HW


def arena_observation_spec(*, image_resolution: int, state_dim: int = DEFAULT_STATE_DIM) -> ObservationSpec:
    """The observation spec of one Arena client, bound to its configured geometry."""
    height, width = arena_image_hw(image_resolution)
    return ArenaObservation.declared_spec({"H": height, "W": width, "state_dim": int(state_dim)})


def arena_action_spec(action_dim: int) -> ActionSpec:
    """The per-step whole-body-control command one Arena client accepts."""
    return ActionSpec(width=int(action_dim), dtype="float32")


def arena_wire_observation_spec(*, image_hw: tuple[int, int], state_dim: int) -> ObservationSpec:
    """What a client reads from the server's payload, under the wire's own names."""
    return ObservationSpec(
        {"ego_view": ArraySpec((*image_hw, 3), "uint8"), "state": ArraySpec((int(state_dim),), "float32")}
    )


def _check_server_specs(transport: EnvTransport, *, image_hw: tuple[int, int], state_dim: int, action_dim: int) -> None:
    """Refuse a server whose observation or action disagrees with this client, before the first episode."""
    comparison = handshake(
        transport,
        reads=arena_wire_observation_spec(image_hw=image_hw, state_dim=state_dim),
        emits=arena_action_spec(action_dim),
        context="Arena server and client disagree",
    )
    logger.info(f"Arena spec handshake: {comparison.summary()}")


class ArenaEnv(gym.Env):
    """Isaac Lab Arena environment behind a transport.

    Observations are ``{"ego_view_image": uint8 [H, W, 3], "state": float32
    [state_dim]}``; actions are a flat ``float32 [action_dim]`` box (the
    whole-body-control joint targets for the G1 tasks).

    The ego image is carried at the camera's native resolution by default
    (``image_resolution <= 0``) so the policy's own image processor performs
    the only resize; pre-shrinking the frame here is out-of-distribution for
    the policy and demonstrably hurts manipulation. Set ``image_resolution``
    to a positive value only if the server is resizing to the same value.

    Without a transport or server URL the environment runs in *stub* mode:
    ``reset`` returns a zero observation and ``step`` raises. This lets
    offline pipelines and unit tests construct the full model stack (spaces,
    key mapping) with no simulator attached.

    Args:
        server_url: Environment-server URI, e.g. ``tcp://127.0.0.1:6000``.
            Ignored when ``transport`` is given; ``None`` selects stub mode.
        transport: Pre-built transport to use instead of constructing one
            from ``server_url``.
        image_resolution: Square observation size when positive; any value
            ``<= 0`` keeps the native camera frame (``image_hw``).
        image_hw: Native ``(height, width)`` of the ego camera, used when
            ``image_resolution <= 0``. Defaults to :data:`NATIVE_IMAGE_HW`.
        action_dim: Length of the flat action vector.
        state_dim: Length of the proprioceptive state vector.
        max_episode_steps: Client-side episode cap in simulator steps; the
            server enforces no horizon, so without a cap a failing episode
            never ends. ``None`` disables the cap.
        socket_timeout: Transport timeout in seconds for ``step`` requests.
        connect_timeout: Total budget in seconds for the initial connection.
        retry_interval: Seconds between reconnection attempts.
        stub: Force stub mode on or off; by default it is inferred from
            whether a server address is available.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url: str | None = None,
        *,
        transport: EnvTransport | None = None,
        image_resolution: int = 224,
        image_hw: tuple[int, int] | None = None,
        action_dim: int = DEFAULT_ACTION_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        max_episode_steps: int | None = None,
        socket_timeout: float = 120.0,
        connect_timeout: float = float("inf"),
        retry_interval: float = 5.0,
        stub: bool | None = None,
    ) -> None:
        super().__init__()

        self.server_url = server_url
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.retry_interval = float(retry_interval)
        self._episode_steps = 0

        # Rollout workers flip this off to skip server-side observation
        # capture on intermediate steps of an action chunk.
        self.need_obs: bool = True
        # Reported by the server at reset: whether one step may carry a
        # whole ``[chunk, action_dim]`` chunk.
        self.supports_action_chunking: bool = False

        # Mapping the steering-side actor/critic use to locate state / image /
        # language entries in the observation dict.
        self.steering_key_mapping: dict[str, Any] = dict(ARENA_KEY_MAPPING)
        # Raw camera keys the SAC networks drop (they use the DINO embedding);
        # the frozen policy still consumes the ego image via infer.
        self.steering_remove_obs_keys: list[str] = ["ego_view_image"]

        self.image_resolution = int(image_resolution)
        if self.image_resolution > 0:
            self.image_hw: tuple[int, int] = (self.image_resolution, self.image_resolution)
        else:
            self.image_hw = tuple(image_hw) if image_hw is not None else NATIVE_IMAGE_HW

        self._transport: EnvTransport | None = transport
        if transport is None and server_url:
            self._transport = transport_from_uri(
                server_url,
                request_timeout=socket_timeout,
                connect_timeout=connect_timeout,
                retry_interval=retry_interval,
            )
        self.stub = (self._transport is None) if stub is None else bool(stub)

        self.observation_space = gym.spaces.Dict(
            {
                "ego_view_image": gym.spaces.Box(
                    low=0, high=255, shape=(*self.image_hw, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
        )

        if not self.stub and self._transport is not None:
            self._transport.connect()
            if self.image_resolution <= 0:
                self.image_hw = spec_image_hw(self._transport.spec(), self.image_hw)
                self.observation_space.spaces["ego_view_image"] = gym.spaces.Box(
                    low=0, high=255, shape=(*self.image_hw, 3), dtype=np.uint8
                )
            _check_server_specs(
                self._transport, image_hw=self.image_hw, state_dim=self.state_dim, action_dim=self.action_dim
            )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the remote episode.

        Retries indefinitely across connection failures so training survives
        simulator restarts.

        Args:
            seed: Forwarded to the server (and to Gymnasium's RNG seeding).
            options: Unused; present for Gymnasium API compatibility.

        Returns:
            The ``(observation, info)`` pair.
        """
        super().reset(seed=seed)
        self._episode_steps = 0
        if self.stub or self._transport is None:
            return self._zero_observation(), {}

        while True:
            try:
                self._transport.connect()
                payload = self._transport.reset(seed=seed)
                info = dict(payload.get("info", {}))
                self.supports_action_chunking = bool(info.get("supports_action_chunking", False))
                return self._format_observation(payload), info
            except TransportError as exc:
                logger.warning(
                    f"Arena reset failed ({exc}); retrying in {self.retry_interval:.0f}s"
                )
                self._transport.close()
                time.sleep(self.retry_interval)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute an action (or an ``[chunk, action_dim]`` action chunk).

        A lost connection raises instead of becoming a task truncation.
        Runtime adapters can retry infrastructure failures without recording
        them as valid environment outcomes.

        Args:
            action: ``float32 [action_dim]`` action, or ``[chunk, action_dim]``
                chunk the server executes sequentially.

        Returns:
            The Gymnasium ``(obs, reward, terminated, truncated, info)`` tuple.
        """
        if self.stub or self._transport is None:
            raise RuntimeError(
                "ArenaEnv is in stub mode (no server address). Start the Arena "
                "env server and pass its URL to step the environment."
            )
        action = np.asarray(action, dtype=np.float32)
        try:
            self._transport.connect()
            payload = self._transport.step(action.tolist(), need_obs=self.need_obs)
        except TransportError as exc:
            self._transport.close()
            raise RuntimeError(
                f"Arena server at {self.server_url} disconnected during step; "
                "the transition was not recorded"
            ) from exc

        reward = float(payload.get("reward", 0.0))
        terminated = bool(payload.get("done", False))
        truncated = bool(payload.get("truncated", False))
        success = bool(payload.get("success", False))
        info = dict(payload.get("info", {}))
        info["success"] = success
        info["is_success"] = success

        # Enforce the client-side episode cap; a chunked action consumes
        # several simulator steps, which the server reports via "num_steps".
        default_steps = action.shape[0] if action.ndim == 2 else 1
        executed_steps = int(payload.get("num_steps", default_steps))
        info["num_steps"] = executed_steps
        self._episode_steps += executed_steps
        if (
            self.max_episode_steps is not None
            and not terminated
            and self._episode_steps >= self.max_episode_steps
        ):
            truncated = True

        return self._format_observation(payload), reward, terminated, truncated, info

    def step_chunk(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], tuple[float, ...], bool, bool, dict[str, Any]]:
        """Execute a ``[chunk, action_dim]`` chunk, stopping early at an episode end.

        A server that supports chunking runs the whole chunk in one round
        trip and reports the last step's reward; the returned reward sequence
        places that reward on the last executed step, matching the sparse
        outcome convention of per-action stepping. Other servers are stepped
        one action at a time with camera capture only on the last step.

        Returns:
            ``(obs, rewards, terminated, truncated, info)`` with one reward
            per executed simulator step.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"step_chunk expects a [chunk, action_dim] array, got {actions.shape}")
        if self.supports_action_chunking:
            observation, reward, terminated, truncated, info = self.step(actions)
            executed = int(info["num_steps"])
            rewards = (0.0,) * (executed - 1) + (float(reward),)
            return observation, rewards, terminated, truncated, info
        rewards_list: list[float] = []
        for index, action in enumerate(actions):
            self.need_obs = index == len(actions) - 1
            try:
                observation, reward, terminated, truncated, info = self.step(action)
            finally:
                self.need_obs = True
            rewards_list.append(float(reward))
            if terminated or truncated:
                break
        return observation, tuple(rewards_list), terminated, truncated, info

    def close(self) -> None:
        """Close the transport connection (safe to call repeatedly)."""
        if self._transport is not None:
            self._transport.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def spec_info(self) -> dict[str, Any]:
        """Return the server's ``SPEC`` handshake payload (``{}`` in stub mode)."""
        if self.stub or self._transport is None:
            return {}
        self._transport.connect()
        return self._transport.spec()

    def _zero_observation(self) -> dict[str, Any]:
        """An all-zero observation matching the observation space."""
        return {
            "ego_view_image": np.zeros((*self.image_hw, 3), dtype=np.uint8),
            "state": np.zeros((self.state_dim,), dtype=np.float32),
        }

    def _format_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Convert a transport payload into the observation dict."""
        image = np.asarray(payload["ego_view"])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return {
            "ego_view_image": image,
            "state": np.asarray(payload["state"], dtype=np.float32),
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown: transport internals may be gone


# Name used by earlier revisions of this package.
ArenaRemoteEnv = ArenaEnv


RESET_RETRY_BUDGET_S = 600.0
"""Total seconds the batched client retries an unreachable server on reset.

The launchers never restart a dead simulator, so an unreachable server means
the run is over; fail the job loudly instead of retrying forever.
"""


class ArenaBatchedEnv:
    """N Arena envs inside ONE simulator instance, behind one transport.

    The natively-batched counterpart of :class:`ArenaEnv` for a server started
    with ``--num-envs N``: every wire payload carries a leading ``[N]`` axis
    and one ``step`` advances all envs by one simulator step. It implements
    the ``BatchedEnvAdapter`` contract (``reset`` honoring
    ``options={"reset_mask": mask}``, batched ``step``), so the ``batched``
    vector backend presents it through the standard ``gym.vector`` interface.

    Isaac Lab auto-resets a terminated env inside the simulator's own step,
    so the observation returned alongside ``terminated=True`` is already the
    fresh episode's first frame; the masked reset still round-trips to the
    server because client-side truncations (the episode cap below) are
    invisible to the simulator and need an explicit per-env reset.

    Args:
        server_url: Environment-server URI hosting the batched simulator.
            ``None`` selects stub mode (zero observations, for tests).
        num_envs: Env count; must match the server's ``--num-envs`` (checked
            against the SPEC handshake).
        image_resolution: Square observation size when positive; ``<= 0``
            keeps the native camera frame.
        image_hw: Native ``(height, width)`` of the ego camera.
        action_dim: Length of each env's flat action vector.
        state_dim: Length of each env's proprioceptive state vector.
        max_episode_steps: Client-side per-env episode cap in simulator
            steps; the server enforces no horizon.
        socket_timeout: Transport timeout in seconds for ``step`` requests.
        connect_timeout: Total budget in seconds for the initial connection.
        retry_interval: Seconds between reconnection attempts.
    """

    def __init__(
        self,
        server_url: str | None = None,
        *,
        num_envs: int,
        image_resolution: int = 0,
        image_hw: tuple[int, int] | None = None,
        action_dim: int = DEFAULT_ACTION_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        max_episode_steps: int | None = None,
        socket_timeout: float = 120.0,
        connect_timeout: float = float("inf"),
        retry_interval: float = 5.0,
    ) -> None:
        self.server_url = server_url
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.retry_interval = float(retry_interval)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._round_trip_timer = RoundTripTimer(self.num_envs)

        if image_resolution > 0:
            self.image_hw: tuple[int, int] = (int(image_resolution), int(image_resolution))
        else:
            self.image_hw = tuple(image_hw) if image_hw is not None else NATIVE_IMAGE_HW

        # Rollout workers flip this off to skip the image payload on
        # intermediate steps of an action chunk (the dominant wire cost at
        # large N); the server still ships images on any episode end.
        self.need_obs: bool = True

        # The steering contract lives on the client too (the builder also
        # stamps the outermost wrapper).
        self.steering_key_mapping: dict[str, Any] = dict(ARENA_KEY_MAPPING)
        self.steering_remove_obs_keys: list[str] = ["ego_view_image"]

        self._transport: EnvTransport | None = None
        if server_url:
            self._transport = transport_from_uri(
                server_url,
                request_timeout=socket_timeout,
                connect_timeout=connect_timeout,
                retry_interval=retry_interval,
            )
        self.stub = self._transport is None

        if not self.stub:
            self._transport.connect()
            spec = self._transport.spec()
            server_envs = int(spec["num_envs"])
            if server_envs != self.num_envs:
                raise ValueError(
                    f"server at {server_url} hosts {server_envs} envs but "
                    f"training.num_envs={self.num_envs}; restart the server "
                    f"with --num-envs {self.num_envs}."
                )
            if int(image_resolution) <= 0:
                self.image_hw = spec_image_hw(spec, self.image_hw)
            _check_server_specs(
                self._transport, image_hw=self.image_hw, state_dim=self.state_dim, action_dim=self.action_dim
            )

        self.single_observation_space = gym.spaces.Dict(
            {
                "ego_view_image": gym.spaces.Box(
                    low=0, high=255, shape=(*self.image_hw, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
                ),
            }
        )
        self.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset all envs, or only the ``reset_mask`` subset (rolling resets)."""
        mask = None if options is None else options.get("reset_mask")
        env_ids = None
        if mask is not None:
            env_ids = [int(i) for i in np.flatnonzero(np.asarray(mask))]
            self._episode_steps[env_ids] = 0
        else:
            self._episode_steps[:] = 0
        if self.stub:
            return self._zero_observation(), {}

        deadline = time.monotonic() + RESET_RETRY_BUDGET_S
        while True:
            try:
                self._transport.connect()
                payload = self._transport.reset(seed=seed, env_ids=env_ids)
                return self._format_observation(payload), dict(payload.get("info", {}))
            except TransportError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Arena batched server at {self.server_url} unreachable for "
                        f"{RESET_RETRY_BUDGET_S:.0f}s across reset retries ({exc}); "
                        "the simulator has died — check its sim log."
                    ) from exc
                logger.warning(
                    f"Arena batched reset failed ({exc}); retrying in {self.retry_interval:.0f}s"
                )
                self._transport.close()
                time.sleep(self.retry_interval)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Advance every env by one simulator step with its own action."""
        if self.stub:
            raise RuntimeError(
                "ArenaBatchedEnv is in stub mode (no server address). Start the Arena "
                "env server with --num-envs and pass its URL to step the environment."
            )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"actions must be [num_envs={self.num_envs}, {self.action_dim}] "
                f"(the batched protocol carries no chunks), got {actions.shape}"
            )
        t_start = time.perf_counter()
        try:
            self._transport.connect()
            payload = self._transport.step(actions.tolist(), need_obs=self.need_obs)
        except TransportError as exc:
            self._transport.close()
            raise RuntimeError(
                f"Arena batched server at {self.server_url} disconnected during "
                "step; no environment transitions were recorded"
            ) from exc

        reward = np.asarray(payload["reward"], dtype=np.float64)
        terminated = np.asarray(payload["done"], dtype=bool)
        truncated = np.asarray(payload["truncated"], dtype=bool)
        success = np.asarray(payload["success"], dtype=bool)
        num_steps = np.asarray(payload["num_steps"], dtype=np.int64)

        # Enforce the client-side episode cap per env; the simulator never
        # sees these truncations, so the masked reset restarts the env.
        self._episode_steps += num_steps
        if self.max_episode_steps is not None:
            over_cap = (self._episode_steps >= self.max_episode_steps) & ~terminated
            truncated = truncated | over_cap

        infos: list[dict[str, Any]] = [
            {
                "success": bool(success[i]),
                "is_success": bool(success[i]),
                "num_steps": int(num_steps[i]),
            }
            for i in range(self.num_envs)
        ]
        t_received = time.perf_counter()
        observation = self._format_observation(payload)
        # The transport reports where its part of the round trip went; a server
        # predating the echo reports no time of its own.
        timing = self._transport.last_timing
        summary = self._round_trip_timer.record(
            t_received - t_start,
            float(payload["server_s"]) if "server_s" in payload else 0.0,
            timing["recv_s"] if "recv_s" in timing else 0.0,
            timing["decode_s"] if "decode_s" in timing else 0.0,
            time.perf_counter() - t_received,
            with_images="ego_view" in payload,
        )
        if summary is not None:
            logger.info(summary)
        return observation, reward, terminated, truncated, infos

    def close(self) -> None:
        """Close the transport connection (safe to call repeatedly)."""
        if self._transport is not None:
            self._transport.close()

    def _zero_observation(self) -> dict[str, np.ndarray]:
        return {
            "ego_view_image": np.zeros((self.num_envs, *self.image_hw, 3), dtype=np.uint8),
            "state": np.zeros((self.num_envs, self.state_dim), dtype=np.float32),
        }

    def _format_observation(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        if "ego_view" in payload:
            images = np.asarray(payload["ego_view"])
            if images.dtype != np.uint8:
                images = np.clip(images, 0, 255).astype(np.uint8)
        else:
            # Intermediate chunk step (need_obs off): the collector discards
            # this observation, so a zero frame stands in for the image.
            images = np.zeros((self.num_envs, *self.image_hw, 3), dtype=np.uint8)
        return {
            "ego_view_image": images,
            "state": np.asarray(payload["state"], dtype=np.float32),
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown: transport internals may be gone


def build_dsrl_env(
    env_name,
    *,
    num_envs: int = 4,
    vectorized: bool = False,
    max_episode_steps: int = 400,
    chunk_size=None,
    dinov2_model=None,
    dinov2_processor=None,
    device=None,
    sentence_model=None,
    image_keys=None,
    seed=None,
    server_url=None,
    action_dim: int = DEFAULT_ACTION_DIM,
    image_resolution: int = 224,
    critic_image_size=None,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    vector_backend: str = "sync",
):
    """Build the wrapped (vectorized) Arena DSRL environment for ``env_name``.

    Registered for the ``arena``/``isaac`` env_name prefixes; the generic
    learner resolves and calls this through the provider registry, so core
    never imports this module. Without a ``server_url`` the client runs in stub
    mode (host-only model work with no live simulator).
    """
    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper
    from isaac_rl_posttraining.envs.language_wrappers import (
        LanguageInstructionVectorWrapper,
    )
    from isaac_rl_posttraining.envs.vector_backends import build_vector_env

    task = env_name.split("/", 1)[1] if "/" in env_name else "drill_lift"
    image_keys = image_keys or ["ego_view_image"]
    n = num_envs if vectorized else 1

    if vector_backend == "batched":
        # One natively-batched simulator hosts all N envs (--num-envs N on
        # the server); a single URL addresses it.
        if n == 1:
            raise ValueError(
                "vector_backend='batched' needs num_envs > 1 for g1_arena "
                "(the single-env server speaks the scalar protocol); use 'sync' for "
                "one env."
            )
        batched_client = ArenaBatchedEnv(
            server_url=server_url,
            num_envs=n,
            image_resolution=image_resolution,
            action_dim=action_dim,
            state_dim=DEFAULT_STATE_DIM,
            max_episode_steps=max_episode_steps,
        )
        env: gym.Env = build_vector_env(vector_backend, batched_env=batched_client, num_envs=n)
    else:
        if n > 1:
            raise ValueError(
                "g1_arena parallel collection runs one natively-batched simulator: "
                f"set vector_backend=batched (got {vector_backend!r} with "
                f"num_envs={n}) and start the env server with --num-envs."
            )

        def _make_arena_env() -> gym.Env:
            return ArenaEnv(
                server_url=server_url,
                image_resolution=image_resolution,
                action_dim=action_dim,
                state_dim=DEFAULT_STATE_DIM,
                max_episode_steps=max_episode_steps,
            )

        env = gym.vector.SyncVectorEnv([_make_arena_env])
    if sentence_model is not None:
        env = LanguageInstructionVectorWrapper(env, task, sentence_model)
    if dinov2_model is not None:
        env = VectorDinoEmbeddingWrapper(
            env,
            dinov2_model,
            dinov2_processor,
            device=device,
            image_keys=image_keys,
            critic_image_size=critic_image_size,
        )
    if chunk_size is not None:
        env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)
    # Expose the steering contract on the outermost wrapper: the batched
    # backend has no ``envs`` list for the collector to reach through, so it
    # resolves the mapping on the wrapper chain instead.
    env.steering_key_mapping = dict(ARENA_KEY_MAPPING)
    env.steering_remove_obs_keys = ["ego_view_image"]
    return env
