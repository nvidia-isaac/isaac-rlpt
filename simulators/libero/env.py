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

"""In-process Gymnasium provider for the LIBERO benchmark suites.

`LIBERO <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ is a
MuJoCo/robosuite tabletop-manipulation benchmark (a Franka Panda arm, four
task suites of ten language-conditioned tasks each). Unlike the G1 Factory
integration, which drives a simulator process over a transport, LIBERO is
light enough to step inside the learner process, so :class:`LiberoEnv` wraps
LIBERO's environment API directly and adapts it to the same observation
contract the steering wrappers and rollout workers consume:

* observations are ``{"<camera>_image": uint8 [H, W, 3], "state": float32
  [state_dim], "prompt": str}`` -- the frame is upright (LIBERO's dataset
  convention; the raw MuJoCo offscreen buffer is vertically flipped),
* actions are a flat ``float32 [7]`` box (OSC delta pose + gripper), with
  ``[chunk, 7]`` chunks executed sequentially like the Arena env server,
* ``info["success"]`` / ``info["is_success"]`` carry the task's BDDL goal
  predicate, and episodes terminate on success (LIBERO environments never
  terminate otherwise, so the client-side ``max_episode_steps`` cap ends
  failed episodes as truncations).

The LIBERO stack (``libero``, ``robosuite``, ``mujoco``) is an optional
dependency; see ``simulators/libero/README.md`` for the tested install and
the headless-rendering (``MUJOCO_GL=egl``) notes. Importing this module does
not require it -- only constructing an environment does.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

DEFAULT_ACTION_DIM = 7
"""OSC end-effector action: delta position (3) + delta axis-angle rotation
(3) + gripper open/close (1), all in ``[-1, 1]``."""

DEFAULT_STATE_KEYS: tuple[str, ...] = ("robot0_joint_pos", "robot0_gripper_qpos")
"""Proprioceptive entries concatenated into the ``state`` vector (7 + 2 =
9-dim), matching LIBERO's own low-dim policy inputs (``joint_states`` +
``gripper_states``)."""

EEF_STATE_KEYS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_axis_angle",
    "robot0_gripper_qpos",
)
"""End-effector proprioception (3 + 3 + 2 = 8-dim): position, orientation as
axis-angle, gripper joint positions -- the state layout of the released
GR00T-N1.7 LIBERO checkpoints (``x, y, z, roll, pitch, yaw, gripper``)."""


# Adapted from robosuite (MIT) ``transform_utils.quat2axisangle``; see THIRD_PARTY_NOTICES.md.
def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to axis-angle coordinates.

    robosuite's convention (``robosuite.utils.transform_utils.quat2axisangle``):
    a unit direction vector scaled by the rotation angle in radians, with the
    zero rotation mapping to the zero vector.

    Args:
        quat: Quaternion as ``(x, y, z, w)``.

    Returns:
        ``float32 [3]`` exponential coordinates.
    """
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * np.arccos(w)) / den).astype(np.float32)


#: Derived proprioceptive entries: state keys computed from raw robosuite
#: observations rather than read directly. Usable anywhere a raw key is
#: (``state_keys``), keeping the flat-state contract uniform.
DERIVED_STATE_KEYS: dict[str, Any] = {
    "robot0_eef_axis_angle": lambda raw: quat_to_axis_angle(raw["robot0_eef_quat"]),
}

DEFAULT_IMAGE_SIZE = 256
"""Square camera resolution. 256 matches the public LeRobot conversions of
the LIBERO demonstrations; set 128 to match the original HDF5 datasets."""

DEFAULT_MAX_EPISODE_STEPS = 600
"""LIBERO's own evaluation cap (``libero/configs/eval/default.yaml``),
applied to every suite."""

DEFAULT_SETTLE_STEPS = 5
"""Zero-action steps executed after an initial state is injected, letting
physics settle (LIBERO's evaluation protocol)."""

LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
"""Benchmark suite names accepted by :class:`LiberoEnv`."""

LIBERO_KEY_MAPPING: dict[str, Any] = {
    "state": "state",
    "language": None,  # one LiberoEnv instance is one task; the prompt is constant
    "image": ["agentview_image"],
}
"""Observation-key mapping consumed by the latent-steering rollout workers
(for the default ``agentview`` camera)."""


def _import_libero() -> tuple[Any, Any, Any]:
    """Import the LIBERO API, failing fast with actionable messages.

    Returns:
        The ``(get_libero_path, get_benchmark, OffScreenRenderEnv)`` triple.

    Raises:
        ImportError: If the LIBERO stack is not installed.
        RuntimeError: If LIBERO has never been configured on this machine.
            LIBERO's first import blocks on an interactive path prompt when
            its ``config.yaml`` is missing, which would wedge a headless
            learner; the pre-check turns that into an error instead.
    """
    # LIBERO's own config-location variable (external package, process
    # boundary): read only to fail fast instead of blocking on LIBERO's
    # interactive first-run prompt.
    config_dir = os.environ.get("LIBERO_CONFIG_PATH") or os.path.expanduser("~/.libero")
    if not os.path.isfile(os.path.join(config_dir, "config.yaml")):
        raise RuntimeError(
            f"LIBERO is not configured: no config.yaml under {config_dir!r}. "
            "Importing libero would block on an interactive prompt; pre-seed "
            "the config as shown in simulators/libero/README.md (Part 1)."
        )
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "The LIBERO stack is not installed (needs the libero package plus "
            "robosuite/mujoco/gym); follow simulators/libero/README.md"
        ) from exc
    return get_libero_path, get_benchmark, OffScreenRenderEnv


class LiberoEnv(gym.Env):
    """One LIBERO task as an in-process Gymnasium environment.

    Observations are ``{"<camera_name>_image": uint8 [image_size,
    image_size, 3], "state": float32 [state_dim], "prompt": str}``. The
    declared ``observation_space`` covers the array entries; the ``prompt``
    string (the task's language instruction) rides along in the observation
    dict for the language wrappers, mirroring how the Arena stack carries
    it.

    Episode semantics match the remote Arena environment: LIBERO terminates
    only on success (the BDDL goal predicate), so ``max_episode_steps`` is
    the client-side cap that ends failed episodes as truncations. Chunked
    actions ``[chunk, 7]`` are executed sequentially, stopping early on
    termination, and consume one cap step per row.

    Args:
        suite: Benchmark suite name (one of :data:`LIBERO_SUITES`).
        task_id: Task index within the suite's canonical ordering
            (``task_order_index=0``); ignored when ``task_name`` is given.
        task_name: Task selected by name instead of index.
        image_size: Square camera resolution (see
            :data:`DEFAULT_IMAGE_SIZE` for the dataset conventions).
        camera_name: Camera rendered into the image observation
            (``"agentview"`` is the benchmark's scene camera;
            ``"robot0_eye_in_hand"`` is the wrist view).
        cameras: Multiple cameras rendered per step, each emitted as its
            own ``"<camera>_image"`` observation entry; overrides
            ``camera_name`` (whose camera stays the convention for
            ``image_key`` when ``cameras`` starts with it).
        state_keys: Proprioceptive observation entries concatenated (in
            order) into the ``state`` vector. Raw robosuite entries and the
            derived entries in :data:`DERIVED_STATE_KEYS` may be mixed.
        max_episode_steps: Client-side episode cap in simulator steps;
            ``None`` disables the cap (an episode then only ends on
            success).
        use_init_states: Draw each episode's initial scene from the task's
            benchmark init-state file (LIBERO's evaluation protocol). A
            specific state can be forced per episode via
            ``reset(options={"init_state_id": k})``.
        fixed_init_state_id: Pin every reset to this benchmark init-state
            index (single-scenario training); requires ``use_init_states``.
        sequential_init_states: Walk the benchmark init states in order
            (episode ``k`` gets init state ``k`` modulo the file size)
            instead of sampling them -- the fixed-initial-state evaluation
            protocol, reproducible without seed bookkeeping. Requires
            ``use_init_states``.
        settle_steps: Zero-action steps after init-state injection.
        seed: Initial seed for the init-state sampler (``reset(seed=...)``
            reseeds).
        env_kwargs: Extra keyword arguments forwarded to LIBERO's
            ``OffScreenRenderEnv`` (e.g. ``render_gpu_device_id``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        suite: str = "libero_spatial",
        task_id: int = 0,
        *,
        task_name: str | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        camera_name: str = "agentview",
        cameras: tuple[str, ...] | None = None,
        state_keys: tuple[str, ...] = DEFAULT_STATE_KEYS,
        max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
        use_init_states: bool = True,
        sequential_init_states: bool = False,
        fixed_init_state_id: int | None = None,
        settle_steps: int = DEFAULT_SETTLE_STEPS,
        seed: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        get_libero_path, get_benchmark, offscreen_env_cls = _import_libero()

        if suite not in LIBERO_SUITES:
            raise ValueError(f"unknown LIBERO suite {suite!r}; expected one of {LIBERO_SUITES}")
        self.suite = suite
        self._benchmark = get_benchmark(suite)()

        self.task_id = self._resolve_task_id(task_id, task_name)
        self._task = self._benchmark.get_task(self.task_id)
        self.task_name: str = self._task.name
        self.language_instruction: str = str(self._task.language)

        self.image_size = int(image_size)
        self.camera_names: tuple[str, ...] = tuple(cameras) if cameras else (str(camera_name),)
        self.camera_name = self.camera_names[0]
        self.image_keys: tuple[str, ...] = tuple(f"{cam}_image" for cam in self.camera_names)
        self.image_key = self.image_keys[0]
        self.state_keys = tuple(state_keys)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.settle_steps = int(settle_steps)
        self.use_init_states = bool(use_init_states)
        self.sequential_init_states = bool(sequential_init_states)
        if self.sequential_init_states and not self.use_init_states:
            raise ValueError("sequential_init_states requires use_init_states=True")
        self.fixed_init_state_id = None if fixed_init_state_id is None else int(fixed_init_state_id)
        if self.fixed_init_state_id is not None:
            if not self.use_init_states:
                raise ValueError("fixed_init_state_id requires use_init_states=True")
            if self.sequential_init_states:
                raise ValueError(
                    "fixed_init_state_id and sequential_init_states are mutually exclusive"
                )
        self._episode_index = 0
        self._episode_steps = 0
        self._closed = False

        bddl_file = os.path.join(
            get_libero_path("bddl_files"), self._task.problem_folder, self._task.bddl_file
        )
        self._env = offscreen_env_cls(
            bddl_file_name=bddl_file,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            camera_names=list(self.camera_names),
            **(env_kwargs or {}),
        )

        self._init_states = (
            self._load_init_states(get_libero_path) if self.use_init_states else None
        )
        state_dim = self._probe_state_dim()

        image_space = gym.spaces.Box(
            low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                **dict.fromkeys(self.image_keys, image_space),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(DEFAULT_ACTION_DIM,), dtype=np.float32
        )

        # Mapping the steering-side actor/critic use to locate state / image /
        # language entries in the observation dict.
        self.steering_key_mapping: dict[str, Any] = {
            "state": "state",
            "language": None,
            "image": list(self.image_keys),
        }

        if seed is not None:
            super().reset(seed=seed)  # seeds self.np_random for init-state draws

    def _resolve_task_id(self, task_id: int, task_name: str | None) -> int:
        """The suite index of ``task_name`` if given, else the range-checked ``task_id``."""
        task_names = list(self._benchmark.get_task_names())
        if task_name is not None:
            if task_name not in task_names:
                raise ValueError(
                    f"task {task_name!r} not in suite {self.suite!r}; tasks: {task_names}"
                )
            task_id = task_names.index(task_name)
        if not 0 <= task_id < len(task_names):
            raise ValueError(
                f"task_id {task_id} out of range for suite {self.suite!r} ({len(task_names)} tasks)"
            )
        return int(task_id)

    def _load_init_states(self, get_libero_path: Any) -> Any:
        """The task's benchmark initial states."""
        try:
            return self._benchmark.get_task_init_states(self.task_id)
        except Exception:
            # LIBERO's init-state files are numpy-pickled; torch >= 2.6
            # rejects them under its weights_only default. They are data
            # files from the user's own LIBERO checkout (the same source
            # the simulator code runs from), so loading them fully is no
            # new trust.
            import torch

            init_path = os.path.join(
                get_libero_path("init_states"),
                self._task.problem_folder,
                self._task.init_states_file,
            )
            return torch.load(  # NOSONAR - numpy-pickled LIBERO data file
                init_path, weights_only=False
            )

    def _probe_state_dim(self) -> int:
        """Width of the state vector, measured on a real observation.

        Robosuite composes proprioception from configurable observables, so
        the widths are not knowable without probing.
        """
        probe = self._env.env._get_observations(force_update=True)
        missing = [
            k for k in self.state_keys if k not in probe and k not in DERIVED_STATE_KEYS
        ]
        if missing:
            raise ValueError(
                f"state_keys {missing} not in the LIBERO observation; "
                f"available: {sorted(probe)} plus derived {sorted(DERIVED_STATE_KEYS)}"
            )
        return int(sum(self._state_entry(probe, key).size for key in self.state_keys))

    def _next_init_state_id(self, options: dict[str, Any] | None) -> int | None:
        """The init state the next episode starts from, or ``None`` for LIBERO's own reset."""
        if options and "init_state_id" in options:
            return int(options["init_state_id"])
        if self.fixed_init_state_id is not None:
            return self.fixed_init_state_id
        if self._init_states is not None and len(self._init_states) > 0:
            if self.sequential_init_states:
                return self._episode_index % len(self._init_states)
            return int(self.np_random.integers(len(self._init_states)))
        return None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the task, optionally injecting a benchmark initial state.

        Args:
            seed: Seeds the init-state sampler and LIBERO's scene
                randomization (LIBERO seeds the global NumPy RNG; this is
                its API, not ours).
            options: ``{"init_state_id": k}`` forces initial state ``k``
                from the task's init-state file (LIBERO's fixed-init
                evaluation protocol indexes episodes this way).

        Returns:
            The ``(observation, info)`` pair; ``info`` carries the task
            identity and the injected ``init_state_id`` (``None`` when the
            scene came from LIBERO's own randomized reset).
        """
        super().reset(seed=seed)
        self._assert_open()
        self._episode_steps = 0
        if seed is not None:
            self._env.seed(seed)
        raw = self._env.reset()

        init_state_id = self._next_init_state_id(options)
        self._episode_index += 1
        if init_state_id is not None:
            if self._init_states is None or not 0 <= init_state_id < len(self._init_states):
                raise ValueError(
                    f"init_state_id {init_state_id} out of range "
                    f"(task has {0 if self._init_states is None else len(self._init_states)} init states)"
                )
            raw = self._env.set_init_state(self._init_states[init_state_id])
            zero = np.zeros(DEFAULT_ACTION_DIM, dtype=np.float32)
            for _ in range(self.settle_steps):
                raw, _, _, _ = self._env.step(zero)

        info = {
            "suite": self.suite,
            "task_name": self.task_name,
            "prompt": self.language_instruction,
            "init_state_id": init_state_id,
            # step() executes [chunk, 7] chunks natively; rollout/eval
            # workers key their batched-step fast path on this flag.
            "supports_action_chunking": True,
        }
        return self._format_observation(raw), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute an action (or an ``[chunk, 7]`` action chunk).

        Chunk rows run sequentially and stop early on termination or on the
        episode cap, mirroring the Arena env server's chunk semantics.
        Rewards are summed across the executed rows (LIBERO's reward is a
        sparse 1.0 on success).

        Args:
            action: ``float32 [7]`` action, or ``[chunk, 7]`` chunk.

        Returns:
            The Gymnasium ``(obs, reward, terminated, truncated, info)``
            tuple. ``terminated`` is the task's success predicate (LIBERO
            episodes end no other way); ``info["num_steps"]`` reports the
            simulator steps consumed.
        """
        self._assert_open()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[-1] != DEFAULT_ACTION_DIM:
            raise ValueError(
                f"expected an action of shape [{DEFAULT_ACTION_DIM}] or "
                f"[chunk, {DEFAULT_ACTION_DIM}], got {action.shape}"
            )

        total_reward = 0.0
        terminated = truncated = False
        steps_consumed = 0
        raw: dict[str, Any] = {}
        step_info: dict[str, Any] = {}
        for row in action:
            raw, reward, done, step_info = self._env.step(row)
            steps_consumed += 1
            self._episode_steps += 1
            total_reward += float(reward)
            # LIBERO overrides `done` with the BDDL success predicate; the
            # extra check keeps that property explicit rather than assumed.
            terminated = bool(done) or bool(self._env.check_success())
            if terminated:
                break
            if self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps:
                truncated = True
                break

        success = terminated
        info = dict(step_info or {})
        info["success"] = success
        info["is_success"] = success
        info["num_steps"] = steps_consumed
        return self._format_observation(raw), total_reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying LIBERO environment (safe to call repeatedly)."""
        if self._closed:
            return
        self._closed = True
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("LiberoEnv is closed")

    @staticmethod
    def _state_entry(raw: dict[str, Any], key: str) -> np.ndarray:
        """One flat float32 state entry: a raw observable or a derived one."""
        if key in DERIVED_STATE_KEYS:
            value = DERIVED_STATE_KEYS[key](raw)
        else:
            value = raw[key]
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _format_observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map a raw robosuite observation onto the wrapper contract."""
        observation: dict[str, Any] = {}
        for image_key in self.image_keys:
            image = np.asarray(raw[image_key])
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            # MuJoCo offscreen buffers are bottom-up; flip to the upright
            # frame (the orientation of LIBERO's demonstration datasets).
            observation[image_key] = np.ascontiguousarray(image[::-1])
        observation["state"] = np.concatenate(
            [self._state_entry(raw, key) for key in self.state_keys]
        )
        observation["prompt"] = self.language_instruction
        return observation

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown: simulator internals may be gone
