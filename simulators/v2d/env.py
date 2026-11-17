# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin scalar and native-batched clients for a served V2D simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from isaac_rl_posttraining.envs.transport import (
    PROTOCOL_VERSION,
    EnvTransport,
    handshake,
    transport_from_uri,
)
from simulators.v2d.contract import (
    IMAGE_KEYS,
    JOINT_DIM,
    JsonValue,
    v2d_action_spec,
    v2d_observation_spec,
)


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V2D {name} must be a mapping")
    return value


def _check_endpoint_contract(
    transport: EnvTransport,
    *,
    task_id: str,
    max_episode_steps: int,
    num_envs: int,
) -> dict[str, Any]:
    spec = transport.spec()
    if spec["protocol"] != PROTOCOL_VERSION or spec["task_id"] != task_id:
        raise ValueError("V2D protocol/task identity mismatch")
    expected = (max_episode_steps, JOINT_DIM, num_envs)
    actual = (spec["episode_horizon"], spec["action_dim"], spec["num_envs"])
    if actual != expected:
        raise ValueError(f"V2D horizon/action/environment contract mismatch: {actual} != {expected}")
    image_shapes = _require_mapping(spec["image_shapes"], name="image_shapes")
    if tuple(image_shapes) != IMAGE_KEYS:
        raise ValueError("V2D camera ordering mismatch")
    handshake(
        transport,
        reads=v2d_observation_spec(),
        emits=v2d_action_spec(),
        context="V2D server and client disagree",
    )
    return spec


def _spaces(spec: Mapping[str, Any]) -> gym.spaces.Dict:
    image_shapes = _require_mapping(spec["image_shapes"], name="image_shapes")
    spaces: dict[str, gym.Space] = {
        "state": gym.spaces.Box(-np.inf, np.inf, (JOINT_DIM,), np.float32)
    }
    for key in IMAGE_KEYS:
        shape = tuple(int(size) for size in image_shapes[key])
        if len(shape) != 3 or shape[-1] != 3 or any(size < 1 for size in shape):
            raise ValueError(f"Invalid V2D image shape for {key}: {shape}")
        spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return gym.spaces.Dict(spaces)


def _observation(
    payload: Mapping[str, object],
    spaces: gym.spaces.Dict,
    *,
    num_envs: int | None,
    allow_missing_images: bool,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, space in spaces.spaces.items():
        expected_shape = space.shape if num_envs is None else (num_envs, *space.shape)
        if key not in payload:
            if allow_missing_images and key in IMAGE_KEYS and num_envs is not None:
                result[key] = np.zeros(expected_shape, dtype=space.dtype)
                continue
            raise ValueError(f"Missing required V2D observation {key}")
        value = np.asarray(payload[key])
        if value.dtype != space.dtype or value.shape != expected_shape:
            raise ValueError(
                f"Invalid V2D observation {key}: got {value.shape}/{value.dtype}, "
                f"expected {expected_shape}/{space.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite V2D observation {key}")
        result[key] = value
    return result


def _terminal_info(info: Mapping[str, Any], success: bool) -> dict[str, Any]:
    result = dict(info)
    task_metrics = result.get("task_metrics")
    if task_metrics is not None:
        metrics = _require_mapping(task_metrics, name="task_metrics")
        if bool(metrics["success"]) is not success:
            raise ValueError("V2D reward and terminal task metrics disagree")
    result.update(success=success, is_success=success)
    return result


class V2DEnv(gym.Env):
    """One served V2D environment following the 58-D joint contract."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        server_url: str | None = None,
        *,
        max_episode_steps: int,
        task_id: str,
        transport: EnvTransport | None = None,
    ) -> None:
        super().__init__()
        if transport is None:
            if server_url is None:
                raise ValueError("V2D requires an explicit server_url")
            transport = transport_from_uri(
                server_url, connect_timeout=120, request_timeout=120
            )
        self.transport = transport
        self.transport.connect()
        self.runtime_contract = _check_endpoint_contract(
            transport,
            task_id=task_id,
            max_episode_steps=max_episode_steps,
            num_envs=1,
        )
        self.max_episode_steps = max_episode_steps
        self.observation_space = _spaces(self.runtime_contract)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (JOINT_DIM,), np.float32)
        self.language_instruction = str(self.runtime_contract["prompt"])
        self.steering_key_mapping = {
            "state": "state",
            "language": None,
            "image": list(IMAGE_KEYS),
        }
        self.steering_remove_obs_keys = list(IMAGE_KEYS)
        self.need_obs = True
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, JsonValue] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None:
            raise ValueError("Scalar V2D reset has no options")
        payload = self.transport.reset(seed=seed)
        self.steps = 0
        info = dict(_require_mapping(payload["info"], name="reset info"))
        info["runtime_contract"] = self.runtime_contract
        return _observation(payload, self.observation_space, num_envs=None, allow_missing_images=False), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (JOINT_DIM,):
            raise ValueError(f"V2D action must have shape ({JOINT_DIM},), got {values.shape}")
        observation, rewards, terminated, truncated, info = self._step_actions(
            values, requested=1
        )
        return observation, rewards[-1], terminated, truncated, info

    def step_chunk(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], list[float], bool, bool, dict[str, Any]]:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != JOINT_DIM or len(values) == 0:
            raise ValueError(
                f"V2D action chunk must have shape (chunk, {JOINT_DIM}), got {values.shape}"
            )
        return self._step_actions(values, requested=len(values))

    def _step_actions(
        self,
        values: np.ndarray,
        *,
        requested: int,
    ) -> tuple[dict[str, np.ndarray], list[float], bool, bool, dict[str, Any]]:
        payload = self.transport.step(values.tolist(), need_obs=True)
        count = int(payload["num_steps"])
        if not 1 <= count <= requested:
            raise ValueError("Invalid executed V2D action count")
        self.steps += count
        terminated = bool(payload["done"])
        truncated = bool(payload["truncated"])
        success = bool(payload["success"])
        if success and not (terminated or truncated):
            raise ValueError("V2D success must be terminal")
        if self.steps > self.max_episode_steps or (
            self.steps == self.max_episode_steps and not (terminated or truncated)
        ):
            raise RuntimeError("V2D simulator violated the task horizon")
        info = _terminal_info(
            _require_mapping(payload["info"], name="step info"), success
        )
        if int(info["episode_steps"]) != self.steps:
            raise ValueError("V2D client/server simulator-step counts differ")
        info["num_steps"] = count
        terminal_reward = float(payload["reward"])
        rewards = [0.0] * (count - 1) + [terminal_reward]
        return (
            _observation(payload, self.observation_space, num_envs=None, allow_missing_images=False),
            rewards,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self.transport.close()


class V2DBatchedEnv:
    """One transport connected to multiple V2D environments in one Isaac process."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        num_envs: int,
        max_episode_steps: int,
        task_id: str,
        transport: EnvTransport | None = None,
    ) -> None:
        if num_envs < 2:
            raise ValueError("V2DBatchedEnv requires num_envs > 1")
        if transport is None and server_url is None:
            raise ValueError("Batched V2D requires an explicit server_url")
        self.server_url = server_url
        self._external_transport = transport is not None
        self.transport = transport if transport is not None else self._make_transport()
        self.transport.connect()
        self.runtime_contract = _check_endpoint_contract(
            self.transport,
            task_id=task_id,
            max_episode_steps=max_episode_steps,
            num_envs=num_envs,
        )
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.single_observation_space = _spaces(self.runtime_contract)
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (JOINT_DIM,), np.float32)
        self.language_instruction = str(self.runtime_contract["prompt"])
        self.steering_key_mapping = {
            "state": "state",
            "language": None,
            "image": list(IMAGE_KEYS),
        }
        self.steering_remove_obs_keys = list(IMAGE_KEYS)
        self.need_obs = True
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)
        self._active = np.zeros(num_envs, dtype=np.bool_)

    def _make_transport(self) -> EnvTransport:
        if self.server_url is None:
            raise RuntimeError("Cannot recreate an externally supplied V2D transport")
        return transport_from_uri(
            self.server_url, connect_timeout=120, request_timeout=120
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, JsonValue] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        mask_value = None if options is None else options["reset_mask"]
        if mask_value is None:
            env_ids = None
            selected = np.ones(self.num_envs, dtype=np.bool_)
        else:
            selected = np.asarray(mask_value, dtype=np.bool_)
            if selected.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must have shape ({self.num_envs},)")
            env_ids = np.flatnonzero(selected).astype(int).tolist()
            if not env_ids:
                raise ValueError("reset_mask must select at least one environment")
        payload = self.transport.reset(seed=seed, env_ids=env_ids)
        self._episode_steps[selected] = 0
        self._active[selected] = True
        return (
            _observation(
                payload,
                self.single_observation_space,
                num_envs=self.num_envs,
                allow_missing_images=False,
            ),
            dict(_require_mapping(payload["info"], name="reset info")),
        )

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Sequence[Mapping[str, Any]],
    ]:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.num_envs, JOINT_DIM):
            raise ValueError(f"actions must have shape ({self.num_envs}, {JOINT_DIM})")
        payload = self.transport.step(values.tolist(), need_obs=self.need_obs)
        reward = np.asarray(payload["reward"], dtype=np.float64)
        terminated = np.asarray(payload["done"], dtype=np.bool_)
        truncated = np.asarray(payload["truncated"], dtype=np.bool_)
        success = np.asarray(payload["success"], dtype=np.bool_)
        counts = np.asarray(payload["num_steps"], dtype=np.int64)
        expected = (self.num_envs,)
        for name, value in (
            ("reward", reward),
            ("done", terminated),
            ("truncated", truncated),
            ("success", success),
            ("num_steps", counts),
        ):
            if value.shape != expected:
                raise ValueError(f"Batched V2D {name} must have shape {expected}")
        if np.any((counts != 0) & (counts != 1)) or np.any(self._active & (counts != 1)):
            raise ValueError("Active V2D environments advance exactly one step per request")
        terminal = terminated | truncated
        if np.any(success & ~terminal):
            raise ValueError("V2D success must be terminal")
        self._episode_steps += counts
        invalid_horizon = self._active & (
            (self._episode_steps > self.max_episode_steps)
            | ((self._episode_steps == self.max_episode_steps) & ~terminal)
        )
        if invalid_horizon.any():
            raise RuntimeError("V2D simulator violated a batched task horizon")
        raw_infos = payload["info"]
        if not isinstance(raw_infos, list) or len(raw_infos) != self.num_envs:
            raise ValueError("Batched V2D info must contain one mapping per environment")
        infos = []
        for env_index, raw_info in enumerate(raw_infos):
            info = _terminal_info(
                _require_mapping(raw_info, name=f"info[{env_index}]"),
                bool(success[env_index]),
            )
            if self._active[env_index] and int(info["episode_steps"]) != int(
                self._episode_steps[env_index]
            ):
                raise ValueError("V2D client/server simulator-step counts differ")
            info["num_steps"] = int(counts[env_index])
            infos.append(info)
        self._active[terminal] = False
        return (
            _observation(
                payload,
                self.single_observation_space,
                num_envs=self.num_envs,
                allow_missing_images=not self.need_obs,
            ),
            reward,
            terminated,
            truncated,
            infos,
        )

    def restart(self) -> None:
        """Reconnect after a failed simulator episode."""
        if self._external_transport:
            raise RuntimeError("Cannot restart a V2D client with an externally supplied transport")
        self.transport.close()
        self.transport = self._make_transport()
        self.transport.connect()
        self.runtime_contract = _check_endpoint_contract(
            self.transport,
            task_id=str(self.runtime_contract["task_id"]),
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
        )
        self._active[:] = False

    def close(self) -> None:
        self.transport.close()
