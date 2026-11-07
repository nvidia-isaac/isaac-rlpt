# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO environment adapter for the GR00T-N1.7 LIBERO checkpoints.

The released GR00T-N1.7-LIBERO checkpoints (embodiment tag ``libero_sim``)
were trained on a specific rendering of the LIBERO simulator, established by
the reference evaluation in the Isaac-GR00T repository
(``gr00t/eval/sim/LIBERO``). :class:`LiberoGr00tAdapter` reproduces those
conventions exactly on top of the generic :class:`~.libero.LiberoEnv`
provider, so the policy sees precisely its training distribution:

* **Cameras** -- two 256x256 views: the scene camera (``agentview``) as
  modality view ``image`` and the wrist camera (``robot0_eye_in_hand``) as
  ``wrist_image``. The training frames are the raw MuJoCo buffer rotated
  180 degrees; ``LiberoEnv`` emits the upright frame (a vertical flip of
  the buffer), so the adapter applies the remaining horizontal flip.
* **State** -- the 8-dim end-effector layout ``(x, y, z, roll, pitch, yaw,
  gripper[2])``: position, orientation as axis-angle, and the two gripper
  joint positions (``EEF_STATE_KEYS`` on the inner environment).
* **Actions** -- 7-dim ``(dx, dy, dz, droll, dpitch, dyaw, gripper)``. The
  first six dimensions pass through (OSC deltas in the simulator's own
  units). The model emits the gripper in the dataset convention ``[0, 1]``
  (1 = open); the simulator expects ``-1`` = open / ``+1`` = close, so the
  adapter re-normalizes to ``[-1, 1]``, binarizes, and flips the sign --
  the reference evaluation's ``normalize_gripper_action`` +
  ``invert_gripper_action`` composition.

Observation keys follow the base model's per-view contract (``video.image``,
``video.wrist_image``); ``state`` and ``prompt`` ride through unchanged, so
the adapted environment plugs directly into
:class:`~isaac_rl_posttraining.policies.gr00t.Gr00tBaseModelPolicy` and the
steering rollout/eval workers.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from simulators.libero.env import (
    DEFAULT_ACTION_DIM,
    EEF_STATE_KEYS,
    LiberoEnv,
)

__all__ = [
    "GR00T_LIBERO_CAMERAS",
    "GR00T_LIBERO_IMAGE_SIZE",
    "GR00T_LIBERO_MAX_EPISODE_STEPS",
    "GR00T_LIBERO_VIDEO_KEYS",
    "LiberoGr00tAdapter",
    "gripper_to_env",
    "make_libero_gr00t_env",
]

GR00T_LIBERO_CAMERAS: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
"""Simulator cameras rendered for the checkpoint, in modality-view order."""

GR00T_LIBERO_VIDEO_KEYS: tuple[str, ...] = ("image", "wrist_image")
"""The checkpoint's video modality views, aligned with
:data:`GR00T_LIBERO_CAMERAS`."""

GR00T_LIBERO_IMAGE_SIZE = 256
"""Camera resolution of the reference evaluation (and the training data)."""

GR00T_LIBERO_MAX_EPISODE_STEPS = 720
"""Per-episode simulator-step cap of the reference evaluation
(``gr00t/eval/rollout_policy.py`` in the Isaac-GR00T repository)."""


def gripper_to_env(gripper: np.ndarray) -> np.ndarray:
    """Convert model-space gripper commands to simulator gripper commands.

    The model decodes the gripper in the training data's ``[0, 1]``
    convention (1 = open). The simulator's Panda gripper takes ``-1`` =
    open / ``+1`` = close. Following the reference evaluation exactly:
    re-normalize to ``[-1, 1]``, binarize with ``sign``, and flip the sign.

    Args:
        gripper: Model-space gripper values (any shape).

    Returns:
        Simulator gripper commands of the same shape (values in
        ``{-1, 0, +1}``; 0 only at exactly 0.5, where ``sign`` is 0).
    """
    return -np.sign(2.0 * np.asarray(gripper, dtype=np.float32) - 1.0)


class LiberoGr00tAdapter(gym.Wrapper):
    """Present a :class:`LiberoEnv` in the GR00T-N1.7 LIBERO conventions.

    Observations become ``{"video.image", "video.wrist_image", "state",
    "prompt"}`` (see the module docstring for the exact conventions);
    actions are accepted in the model's raw decoded space -- a ``[7]``
    action or ``[chunk, 7]`` chunk -- and converted to simulator commands
    before stepping.

    Args:
        env: Inner environment; must render the two checkpoint cameras
            (:data:`GR00T_LIBERO_CAMERAS`) and expose the 8-dim
            end-effector state (:data:`~.libero.EEF_STATE_KEYS`) --
            :func:`make_libero_gr00t_env` constructs a conforming one.

    Raises:
        ValueError: If the inner environment lacks a required camera view.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        self._camera_keys = tuple(f"{camera}_image" for camera in GR00T_LIBERO_CAMERAS)
        missing = [key for key in self._camera_keys if key not in spaces]
        if missing:
            raise ValueError(
                f"inner environment lacks camera views {missing}; construct it with "
                f"cameras={GR00T_LIBERO_CAMERAS} (see make_libero_gr00t_env)"
            )

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"video.{video_key}": spaces[camera_key]
                    for video_key, camera_key in zip(GR00T_LIBERO_VIDEO_KEYS, self._camera_keys)
                },
                "state": spaces["state"],
            }
        )
        # The model's raw decoded action space: OSC deltas in [-1, 1] and the
        # gripper in [0, 1] -- both within the declared bounds.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(DEFAULT_ACTION_DIM,), dtype=np.float32
        )
        self.steering_key_mapping: dict[str, Any] = {
            "state": "state",
            "language": None,
            "image": [f"video.{key}" for key in GR00T_LIBERO_VIDEO_KEYS],
        }
        # The SAC networks drop both raw camera views (they use the scene
        # camera's DINO embedding); the frozen policy consumes them via infer.
        self.steering_remove_obs_keys: list[str] = [
            f"video.{key}" for key in GR00T_LIBERO_VIDEO_KEYS
        ]

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._observation(observation), info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(self._action(action))
        return self._observation(observation), reward, terminated, truncated, info

    def _observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Rename the camera views and restore the training orientation."""
        adapted: dict[str, Any] = {
            key: value
            for key, value in observation.items()
            if key not in self._camera_keys
        }
        for video_key, camera_key in zip(GR00T_LIBERO_VIDEO_KEYS, self._camera_keys):
            # Training frames are the raw buffer rotated 180 degrees; the
            # inner env emits the upright frame (vertical flip of the
            # buffer), so one horizontal flip completes the rotation.
            adapted[f"video.{video_key}"] = np.ascontiguousarray(
                observation[camera_key][:, ::-1]
            )
        return adapted

    @staticmethod
    def _action(action: np.ndarray) -> np.ndarray:
        """Convert a model-space action (or chunk) to simulator commands."""
        env_action = np.array(action, dtype=np.float32, copy=True)
        env_action[..., -1] = gripper_to_env(env_action[..., -1])
        return env_action


def make_libero_gr00t_env(
    suite: str = "libero_10",
    task_id: int = 0,
    *,
    task_name: str | None = None,
    max_episode_steps: int | None = GR00T_LIBERO_MAX_EPISODE_STEPS,
    sequential_init_states: bool = False,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    seed: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> LiberoGr00tAdapter:
    """Construct a LIBERO task in the GR00T-N1.7 checkpoint conventions.

    Args:
        suite: LIBERO benchmark suite name.
        task_id: Task index within the suite; ignored when ``task_name``
            is given.
        task_name: Task selected by name instead of index.
        max_episode_steps: Client-side episode cap in simulator steps;
            defaults to the reference evaluation's 720 (``None`` disables
            the cap).
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
    return LiberoGr00tAdapter(
        LiberoEnv(
            suite=suite,
            task_id=task_id,
            task_name=task_name,
            image_size=GR00T_LIBERO_IMAGE_SIZE,
            cameras=GR00T_LIBERO_CAMERAS,
            state_keys=EEF_STATE_KEYS,
            max_episode_steps=max_episode_steps,
            use_init_states=use_init_states,
            fixed_init_state_id=fixed_init_state_id,
            sequential_init_states=sequential_init_states,
            seed=seed,
            env_kwargs=env_kwargs,
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
    """Build the wrapped (vectorized) LIBERO-GR00T DSRL environment for ``env_name``.

    Registered for the ``libero`` env_name prefix (GR00T-N1.7 checkpoint
    conventions: two cameras, eef state, model-space actions). Requires the
    LIBERO stack (see simulators/libero/README.md).
    """
    import gymnasium as gym

    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper

    parts = env_name.split("/")
    suite = parts[1] if len(parts) > 1 else "libero_10"
    task = parts[2] if len(parts) > 2 else "0"
    image_keys = image_keys or ["video.image"]
    n = num_envs if vectorized else 1

    def _make_libero_env() -> gym.Env:
        task_kwargs = {"task_id": int(task)} if task.isdigit() else {"task_name": task}
        return make_libero_gr00t_env(
            suite=suite,
            max_episode_steps=max_episode_steps,
            seed=seed,
            use_init_states=use_init_states,
            fixed_init_state_id=fixed_init_state_id,
            **task_kwargs,
        )

    env = gym.vector.SyncVectorEnv([_make_libero_env for _ in range(n)])
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
    return env
