# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO in the DreamZero checkpoint conventions.

Presents :class:`~simulators.libero.env.LiberoEnv` the way the DreamZero
LIBERO SFT checkpoints were trained (the ``libero_sim`` embodiment):

- observations ``{"main_image", "wrist_image", "state", "prompt"}``; both
  cameras are the raw MuJoCo buffer rotated 180 degrees at the native 256x256
  render (the base model's transform chain owns crop/resize/normalize);
- ``state`` is the 8-dim end-effector state
  ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``;
- actions are DreamZero's decoded 7-dim OSC commands with the gripper
  binarized to ±1 (LIBERO's native convention), passed to the simulator
  unchanged;
- reward is shifted to ``-1`` per step and ``0`` on the success step (the
  DSRL LIBERO convention: finish fast), with ``info["success"]`` set.

The model-side preprocessing (view concat, prompt template, q99
normalization) lives with the base model
(:mod:`isaac_rl_posttraining.policies.dreamzero`); this adapter only shapes
raw simulator output into the base model's canonical observation keys.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from simulators.libero.env import EEF_STATE_KEYS, LiberoEnv

__all__ = [
    "DREAMZERO_LIBERO_CAMERAS",
    "DREAMZERO_LIBERO_IMAGE_SIZE",
    "DREAMZERO_LIBERO_DSRL_KEY_MAPPING",
    "LiberoDreamZeroAdapter",
    "make_libero_dreamzero_env",
    "build_dsrl_env",
]

#: Cameras the DreamZero libero_sim checkpoints consume (scene, wrist).
DREAMZERO_LIBERO_CAMERAS: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")

#: Native render resolution; the base model's transform chain crops and resizes.
DREAMZERO_LIBERO_IMAGE_SIZE: int = 256

#: Observation entries the latent-steering workers consume: the SAC critic
#: embeds the scene camera with DINO; single-task envs carry no language
#: embedding.
DREAMZERO_LIBERO_DSRL_KEY_MAPPING: dict[str, Any] = {
    "state": "state",
    "language": None,
    "image": ["main_image"],
}


class LiberoDreamZeroAdapter(gym.Wrapper):
    """Present a :class:`LiberoEnv` in the DreamZero libero_sim conventions.

    Args:
        env: Inner environment; must render both DreamZero cameras and expose
            the 8-dim end-effector state --
            :func:`make_libero_dreamzero_env` constructs a conforming one.

    Raises:
        ValueError: If the inner environment lacks a required camera view.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        self._camera_keys = tuple(f"{camera}_image" for camera in DREAMZERO_LIBERO_CAMERAS)
        missing = [key for key in self._camera_keys if key not in spaces]
        if missing:
            raise ValueError(
                f"inner environment lacks camera views {missing}; construct it with "
                f"cameras={DREAMZERO_LIBERO_CAMERAS} (see make_libero_dreamzero_env)"
            )

        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(DREAMZERO_LIBERO_IMAGE_SIZE, DREAMZERO_LIBERO_IMAGE_SIZE, 3),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "main_image": image_space,
                "wrist_image": image_space,
                "state": spaces["state"],
            }
        )
        # DreamZero's decoded action space: 7-dim env-space OSC commands.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.steering_key_mapping: dict[str, Any] = dict(DREAMZERO_LIBERO_DSRL_KEY_MAPPING)
        # The SAC networks drop both raw camera views (they use the scene
        # camera's DINO embedding); the frozen policy consumes them via infer.
        self.steering_remove_obs_keys: list[str] = ["main_image", "wrist_image"]

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._observation(observation), info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        info.setdefault("success", bool(info["is_success"]))
        # Finish-fast reward: -1 per step, 0 on the success step.
        return self._observation(observation), float(reward) - 1.0, terminated, truncated, info

    def _observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Rename cameras and restore the checkpoint's training orientation."""
        adapted: dict[str, Any] = {
            key: value for key, value in observation.items() if key not in self._camera_keys
        }
        scene, wrist = (observation[key] for key in self._camera_keys)
        # Training frames are the raw buffer rotated 180 degrees; the inner
        # env emits the upright frame (vertical flip of the buffer), so one
        # horizontal flip completes the rotation.
        adapted["main_image"] = np.ascontiguousarray(scene[:, ::-1])
        adapted["wrist_image"] = np.ascontiguousarray(wrist[:, ::-1])
        return adapted


def make_libero_dreamzero_env(
    suite: str = "libero_spatial",
    task_id: int = 0,
    *,
    task_name: str | None = None,
    max_episode_steps: int | None = 240,
    sequential_init_states: bool = False,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    seed: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> LiberoDreamZeroAdapter:
    """Construct a LIBERO task in the DreamZero libero_sim conventions.

    Args:
        suite: LIBERO benchmark suite name.
        task_id: Task index within the suite; ignored when ``task_name`` is
            given.
        task_name: Task selected by name instead of index.
        max_episode_steps: Client-side episode cap in simulator steps (the
            validated recipe runs 240 = 15 chunks of 16).
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
    return LiberoDreamZeroAdapter(
        LiberoEnv(
            suite=suite,
            task_id=task_id,
            task_name=task_name,
            image_size=DREAMZERO_LIBERO_IMAGE_SIZE,
            cameras=DREAMZERO_LIBERO_CAMERAS,
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
    max_episode_steps: int = 240,
    chunk_size=None,
    dinov2_model=None,
    dinov2_processor=None,
    device=None,
    sentence_model=None,
    image_keys=None,
    seed=None,
    server_url=None,
    action_dim: int = 7,
    image_resolution: int = 256,
    critic_image_size=None,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    vector_backend: str = "sync",
):
    """Build the wrapped (vectorized) LIBERO-DreamZero DSRL environment.

    Registered for the ``libero_dreamzero`` env_name prefix (DreamZero
    libero_sim checkpoint conventions: 256x256 rotated cameras, eef state,
    env-space actions).
    """
    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper
    from isaac_rl_posttraining.envs.vector_backends import build_vector_env

    parts = env_name.split("/")
    suite = parts[1] if len(parts) > 1 else "libero_spatial"
    task = parts[2] if len(parts) > 2 else "0"
    image_keys = image_keys or ["main_image"]
    n = num_envs if vectorized else 1

    def _libero_dreamzero_env_fn(idx: int):
        def _make_env():
            task_kwargs = {"task_id": int(task)} if task.isdigit() else {"task_name": task}
            worker_seed = None if seed is None else seed + idx
            return make_libero_dreamzero_env(
                suite=suite,
                max_episode_steps=max_episode_steps,
                seed=worker_seed,
                use_init_states=use_init_states,
                fixed_init_state_id=fixed_init_state_id,
                **task_kwargs,
            )

        return _make_env

    env = build_vector_env(
        vector_backend,
        env_fns=[_libero_dreamzero_env_fn(i) for i in range(n)],
        num_envs=n,
    )
    if dinov2_model is not None or critic_image_size:
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
    env.steering_key_mapping = dict(DREAMZERO_LIBERO_DSRL_KEY_MAPPING)
    env.steering_remove_obs_keys = ["main_image", "wrist_image"]
    return env
