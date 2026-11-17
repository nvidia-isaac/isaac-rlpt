# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from robosuite (MIT) and modified; see THIRD_PARTY_NOTICES.md.
#
# SPDX-FileCopyrightText: Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""LIBERO in the pi0 checkpoint conventions.

Presents :class:`~simulators.libero.env.LiberoEnv` the way the
``pi0_libero`` checkpoints were trained (the openpi LIBERO example):

- observations ``{"observation/image", "observation/wrist_image",
  "observation/state", "prompt"}``; both cameras are the raw MuJoCo buffer
  rotated 180 degrees, pad-resized to 224x224 uint8;
- ``observation/state`` is the 8-dim end-effector state
  ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``;
- actions are pi0's decoded 7-dim OSC commands, passed to the simulator
  unchanged (the checkpoint's output transform already emits env-space
  actions, gripper included);
- reward is shifted to ``-1`` per step and ``0`` on the success step (the
  DSRL LIBERO convention: finish fast), with ``info["success"]`` set.

The ``openpi_client`` package (image resizing) is imported lazily so this
module never makes it a base dependency.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from simulators.libero.env import EEF_STATE_KEYS, LiberoEnv

__all__ = [
    "PI0_LIBERO_CAMERAS",
    "PI0_LIBERO_CAMERA_KEYS",
    "PI0_LIBERO_IMAGE_SIZE",
    "PI0_LIBERO_STATE_KEY",
    "PI0_LIBERO_DSRL_KEY_MAPPING",
    "preprocess_libero_observation",
    "LiberoPi0Adapter",
    "make_libero_pi0_env",
]

#: Cameras the pi0_libero checkpoints consume (scene, wrist).
PI0_LIBERO_CAMERAS: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")

#: pi0's input resolution (pad-resize target).
PI0_LIBERO_IMAGE_SIZE: int = 224

#: Render resolution before the pad-resize (the openpi example renders 256).
_RENDER_SIZE: int = 256

#: Observation keys of the openpi LIBERO example.
_IMAGE_KEY = "observation/image"
_WRIST_IMAGE_KEY = "observation/wrist_image"
_STATE_KEY = "observation/state"
#: The camera entries and the state entry of a pi0-convention observation.
PI0_LIBERO_CAMERA_KEYS: tuple[str, ...] = (_IMAGE_KEY, _WRIST_IMAGE_KEY)
PI0_LIBERO_STATE_KEY = _STATE_KEY

#: Observation entries the latent-steering workers consume: the SAC critic
#: embeds the scene camera with DINO; single-task envs carry no language
#: embedding.
PI0_LIBERO_DSRL_KEY_MAPPING: dict[str, Any] = {
    "state": _STATE_KEY,
    "language": None,
    "image": [_IMAGE_KEY],
}


def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Pad-resize one HWC uint8 frame via openpi's own image tools."""
    from openpi_client import image_tools

    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


# Adapted from robosuite (MIT) ``transform_utils.quat2axisangle``; see THIRD_PARTY_NOTICES.md.
def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert an ``[x, y, z, w]`` quaternion to an axis-angle vector."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def preprocess_libero_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert a LIBERO environment observation into pi0 format.

    Flips both camera images (the pi0 training convention for LIBERO),
    resizes with padding to 224x224 uint8, and assembles the 8-dim state
    ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``.

    Args:
        raw_obs: LIBERO observation with ``agentview_image``,
            ``robot0_eye_in_hand_image``, end-effector pose keys, and an
            optional ``prompt`` / ``language_instruction``.

    Returns:
        pi0-format observation dict.

    Raises:
        ValueError: If no end-effector orientation key is present.
    """
    from openpi_client import image_tools as _image_tools

    image = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = _image_tools.convert_to_uint8(_image_tools.resize_with_pad(image, 224, 224))
    wrist_image = _image_tools.convert_to_uint8(_image_tools.resize_with_pad(wrist_image, 224, 224))

    if "robot0_eef_quat" in raw_obs:
        state = np.concatenate(
            [
                raw_obs["robot0_eef_pos"],
                _quat_to_axis_angle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            ]
        )
    elif "robot0_eef_ori" in raw_obs:
        state = np.concatenate(
            [raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_ori"], raw_obs["robot0_gripper_qpos"]]
        )
    else:
        raise ValueError(
            f"No end-effector orientation key found. Available keys: {list(raw_obs)}"
        )

    return {
        _IMAGE_KEY: image,
        _WRIST_IMAGE_KEY: wrist_image,
        _STATE_KEY: state,
        "prompt": raw_obs.get("prompt", raw_obs.get("language_instruction", "unknown task")),
    }


class LiberoPi0Adapter(gym.Wrapper):
    """Present a :class:`LiberoEnv` in the pi0_libero conventions.

    Args:
        env: Inner environment; must render both pi0 cameras and expose the
            8-dim end-effector state -- :func:`make_libero_pi0_env`
            constructs a conforming one.

    Raises:
        ValueError: If the inner environment lacks a required camera view.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        self._camera_keys = tuple(f"{camera}_image" for camera in PI0_LIBERO_CAMERAS)
        missing = [key for key in self._camera_keys if key not in spaces]
        if missing:
            raise ValueError(
                f"inner environment lacks camera views {missing}; construct it with "
                f"cameras={PI0_LIBERO_CAMERAS} (see make_libero_pi0_env)"
            )

        image_space = gym.spaces.Box(
            low=0, high=255, shape=(PI0_LIBERO_IMAGE_SIZE, PI0_LIBERO_IMAGE_SIZE, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                _IMAGE_KEY: image_space,
                _WRIST_IMAGE_KEY: image_space,
                _STATE_KEY: spaces["state"],
            }
        )
        # pi0's decoded action space: 7-dim env-space OSC commands.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.steering_key_mapping: dict[str, Any] = dict(PI0_LIBERO_DSRL_KEY_MAPPING)
        # The SAC networks drop both raw camera views (they use the scene
        # camera's DINO embedding); the frozen policy consumes them via infer.
        self.steering_remove_obs_keys: list[str] = [
            _IMAGE_KEY,
            _WRIST_IMAGE_KEY,
        ]

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._observation(observation), info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        info.setdefault("success", bool(info.get("is_success", terminated)))
        # Finish-fast reward: -1 per step, 0 on the success step.
        return self._observation(observation), float(reward) - 1.0, terminated, truncated, info

    def _observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Rename cameras, restore the training orientation, pad-resize."""
        adapted: dict[str, Any] = {
            key: value for key, value in observation.items() if key not in self._camera_keys
        }
        scene, wrist = (observation[key] for key in self._camera_keys)
        # Training frames are the raw buffer rotated 180 degrees; the inner
        # env emits the upright frame (vertical flip of the buffer), so one
        # horizontal flip completes the rotation.
        adapted[_IMAGE_KEY] = _resize_with_pad(
            np.ascontiguousarray(scene[:, ::-1]), PI0_LIBERO_IMAGE_SIZE
        )
        adapted[_WRIST_IMAGE_KEY] = _resize_with_pad(
            np.ascontiguousarray(wrist[:, ::-1]), PI0_LIBERO_IMAGE_SIZE
        )
        adapted[_STATE_KEY] = adapted.pop("state")
        return adapted


def make_libero_pi0_env(
    suite: str = "libero_10",
    task_id: int = 0,
    *,
    task_name: str | None = None,
    max_episode_steps: int | None = 400,
    sequential_init_states: bool = False,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    seed: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> LiberoPi0Adapter:
    """Construct a LIBERO task in the pi0_libero checkpoint conventions.

    Args:
        suite: LIBERO benchmark suite name.
        task_id: Task index within the suite; ignored when ``task_name`` is
            given.
        task_name: Task selected by name instead of index.
        max_episode_steps: Client-side episode cap in simulator steps.
        sequential_init_states: Walk the benchmark init states in episode
            order (fixed-initial-state evaluation).
        use_init_states: Draw initial scenes from the benchmark init-state
            files.
        seed: Initial seed for the init-state sampler.
        env_kwargs: Extra keyword arguments for LIBERO's
            ``OffScreenRenderEnv`` (e.g. ``render_gpu_device_id``).

    Returns:
        The adapted environment.
    """
    # Renderer device: training pins one GPU via CUDA_VISIBLE_DEVICES, so
    # the visible device is always index 0. LIBERO's default (-1) defers to
    # environment-variable lookups that resolve to the PHYSICAL id and
    # crash EGL on any pinned GPU other than 0.
    merged_kwargs = {"render_gpu_device_id": 0, **(env_kwargs or {})}
    return LiberoPi0Adapter(
        LiberoEnv(
            suite=suite,
            task_id=task_id,
            task_name=task_name,
            image_size=_RENDER_SIZE,
            cameras=PI0_LIBERO_CAMERAS,
            state_keys=EEF_STATE_KEYS,
            max_episode_steps=max_episode_steps,
            use_init_states=use_init_states,
            fixed_init_state_id=fixed_init_state_id,
            sequential_init_states=sequential_init_states,
            seed=seed,
            env_kwargs=merged_kwargs,
        )
    )


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
    action_dim: int = 35,
    image_resolution: int = 224,
    critic_image_size=None,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    vector_backend: str = "sync",
):
    """Build the wrapped (vectorized) LIBERO-pi0 DSRL environment for ``env_name``.

    Registered for the ``libero_pi0`` env_name prefix (pi0_libero checkpoint
    conventions: 224x224 pad-resized cameras, eef state, env-space actions).
    """
    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper
    from isaac_rl_posttraining.envs.vector_backends import build_vector_env

    parts = env_name.split("/")
    suite = parts[1] if len(parts) > 1 else "libero_10"
    task = parts[2] if len(parts) > 2 else "0"
    image_keys = image_keys or [_IMAGE_KEY]
    n = num_envs if vectorized else 1

    def _libero_pi0_env_fn(idx: int):
        def _make_env():
            task_kwargs = {"task_id": int(task)} if task.isdigit() else {"task_name": task}
            worker_seed = None if seed is None else seed + idx
            return make_libero_pi0_env(
                suite=suite,
                max_episode_steps=max_episode_steps,
                seed=worker_seed,
                use_init_states=use_init_states,
                fixed_init_state_id=fixed_init_state_id,
                **task_kwargs,
            )

        return _make_env

    env = build_vector_env(
        vector_backend, env_fns=[_libero_pi0_env_fn(i) for i in range(n)], num_envs=n
    )
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
    # Expose the steering contract on the outermost wrapper. The async
    # (SubprocVectorEnv) backend keeps each adapter in a worker process, so the
    # collector, evaluator, and learner cannot reach it through ``env.envs`` --
    # they read the mapping / removed-image keys from here instead.
    env.steering_key_mapping = dict(PI0_LIBERO_DSRL_KEY_MAPPING)
    env.steering_remove_obs_keys = [_IMAGE_KEY, _WRIST_IMAGE_KEY]
    return env
