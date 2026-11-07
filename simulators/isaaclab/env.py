# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium views of IsaacLab 3.0 manager-based tasks, inside the process that hosts Isaac Sim.

A rollout worker hosts its simulation in a child process
(``simulators/isaaclab/subprocess_env.py``), which serves
:class:`IsaacLabBatchedGr00tEnv`; :class:`IsaacLabGr00tEnv` is the one-scene
view. Both wrap one ``ManagerBasedRLEnv`` and adapt it to the observation
contract the steering wrappers and rollout workers consume, in the GR00T-native
keys of the checkpoint's data config:

* observations are ``{"video.<view>": uint8 [H, W, 3], "state.<part>":
  float32 [d], "state": float32 [sum d], "prompt": str}`` -- the camera views
  and the state parts the checkpoint was fine-tuned on, plus their
  concatenation as the critic's proprioceptive input,
* actions are a flat ``float32 [A]`` box in the model's action-key order; the
  wrapper zero-pads them to the simulator's action width (uncontrolled joints)
  and executes ``[chunk, A]`` chunks row by row, stopping early on termination,
* ``terminated`` is IsaacLab's success termination, so success == terminated;
  time-outs and object drops are IsaacLab truncations, and the client-side
  ``max_episode_steps`` cap also ends an episode as a truncation.

The simulator-to-model mapping (camera terms -> video keys, state terms,
slices and derived quantities, action padding/bounds, task description) is a
plain dict per task id (:data:`TASK_MAPPINGS`, mirrored by the simulator
configs under ``simulators/isaaclab/configs/simulator/``), so another IsaacLab
task plugs in with a mapping and no code. Two task families ship:

* ``Isaac-Assemble-Trocar-G129-Dex3-v0`` (Unitree G1 29-DoF + Dex3 hands; three
  224x224 cameras; 28 joint-position targets padded to 43) -- IsaacLab's
  ``IsaacLabDataConfig`` layout (:data:`TROCAR_GR00T_MAPPING`),
* ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`` (Franka Panda stacking three
  cubes; table and wrist cameras at 256x256; 7-dim relative end-effector pose
  deltas plus a binary gripper) -- the LIBERO-style layout of RLinf's
  ``RLinf-Gr00t-SFT-Stack-cube`` checkpoint (:data:`FRANKA_STACK_GR00T_MAPPING`).

Isaac Sim hosts one application and one simulation per process. The app is
started once (:func:`launch_app`), and the training and evaluation wrappers
share one underlying IsaacLab environment through
:func:`get_or_create_isaaclab_env`; collection and evaluation run
sequentially in the trainer, so sharing is safe. The IsaacLab stack is an
optional dependency imported lazily; see ``simulators/isaaclab/README.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from loguru import logger
import torch

from simulators.isaaclab.task_mappings import (
    FRANKA_STACK_GR00T_MAPPING,
    FRANKA_STACK_TASKS,
    TROCAR_GR00T_MAPPING,
)

__all__ = [
    "APP_LAUNCHED_FLAG",
    "ASSET_ROOT_NAMES",
    "DEFAULT_ACTION_DIM",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_JOINT_BOUND",
    "DEFAULT_MAX_EPISODE_STEPS",
    "DEFAULT_TASK_ID",
    "ENV_NAME_PREFIX",
    "FRANKA_STACK_GR00T_MAPPING",
    "FRANKA_STACK_TASKS",
    "TASK_MAPPINGS",
    "TROCAR_GR00T_MAPPING",
    "TROCAR_TASKS",
    "IsaacLabBatchedGr00tEnv",
    "IsaacLabGr00tEnv",
    "get_or_create_isaaclab_env",
    "launch_app",
    "mapping_for_task",
    "quat_wxyz_to_axis_angle",
    "register_task_mapping",
    "release_isaaclab_env",
]

ENV_NAME_PREFIX = "isaaclab/"
"""``env_name`` prefix routed to this provider: ``isaaclab/<Task-Id>``."""

DEFAULT_TASK_ID = "Isaac-Assemble-Trocar-G129-Dex3-v0"
"""Training variant of the trocar-assembly task (randomized tray yaw)."""

TROCAR_TASKS: tuple[str, ...] = (
    DEFAULT_TASK_ID,
    "Isaac-Assemble-Trocar-G129-Dex3-Eval-v0",
)
"""Registered task ids; the ``-Eval-v0`` variant fixes a per-env tray yaw."""

DEFAULT_MAX_EPISODE_STEPS = 1000
"""IsaacLab's own horizon: 20 s at 50 Hz control (sim dt 1/200, decimation 4)."""

DEFAULT_IMAGE_SIZE = 224
"""Square resolution of the task's ``TiledCamera`` sensors."""

DEFAULT_ACTION_DIM = 28
"""Controlled joints: left arm (7) + right arm (7) + left hand (7) + right hand (7)."""

DEFAULT_JOINT_BOUND = 3.2
"""Per-joint action bound (radians) when the environment exposes no
articulation to read soft joint-position limits from (injected doubles)."""

APP_LAUNCHED_FLAG = "ISAAC_RLPT_APP_LAUNCHED"
"""Process-state flag (``"1"``) set by an outer IsaacLab launcher after its own
``AppLauncher``; :func:`launch_app` then does not start a second app."""

TASK_MAPPINGS: dict[str, dict[str, Any]] = {
    **dict.fromkeys(TROCAR_TASKS, TROCAR_GR00T_MAPPING),
    **dict.fromkeys(FRANKA_STACK_TASKS, FRANKA_STACK_GR00T_MAPPING),
}
"""IsaacLab task id -> simulator-to-model mapping (:func:`mapping_for_task`)."""


def register_task_mapping(task_id: str, mapping: dict[str, Any]) -> None:
    """Make ``isaaclab/<task_id>`` buildable with ``mapping`` (validated eagerly)."""
    _validate_mapping(mapping)
    TASK_MAPPINGS[str(task_id)] = dict(mapping)


def mapping_for_task(task_id: str) -> dict[str, Any]:
    """The registered mapping of ``task_id``.

    Raises:
        KeyError: If no mapping is registered for the task.
    """
    try:
        return TASK_MAPPINGS[task_id]
    except KeyError as exc:
        raise KeyError(
            f"no GR00T mapping registered for IsaacLab task {task_id!r}; known: "
            f"{sorted(TASK_MAPPINGS)}. Register one with register_task_mapping()."
        ) from exc


def quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Unit quaternion(s) ``[..., 4]`` in (w, x, y, z) order to axis-angle ``[..., 3]``.

    The exponential-coordinates convention of robosuite/LIBERO (axis scaled by
    the rotation angle), which RLinf's stack-cube wrapper applies to IsaacLab's
    ``eef_quat`` before feeding the GR00T state; zero rotation maps to zeros.
    """
    q = np.asarray(quat, dtype=np.float64)
    w = np.clip(q[..., 0], -1.0, 1.0)
    den = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    scaled = q[..., 1:4] * (2.0 * np.arccos(w))[..., None]
    safe = np.where(den[..., None] > 1e-8, den[..., None], 1.0)
    out = np.where(den[..., None] > 1e-8, scaled / safe, 0.0)
    return out.astype(np.float32)


_STATE_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "quat_wxyz_to_axis_angle": quat_wxyz_to_axis_angle,
}
"""Named derived-state transforms a mapping's state entry may request."""


ASSET_ROOT_NAMES: tuple[str, ...] = (
    "NUCLEUS_ASSET_ROOT_DIR",
    "ISAAC_NUCLEUS_DIR",
    "ISAACLAB_NUCLEUS_DIR",
    "NVIDIA_NUCLEUS_DIR",
)
"""Isaac Lab asset-root constants an override value may reference as ``{NAME}``."""


def _asset_roots() -> dict[str, str]:
    """Isaac Lab's asset-root constants (``isaaclab.utils.assets``)."""
    from isaaclab.utils import assets

    return {name: getattr(assets, name) for name in ASSET_ROOT_NAMES}


def _expand_override_value(value: Any) -> Any:
    """Expand ``{ISAACLAB_NUCLEUS_DIR}``-style asset-root placeholders in a string value."""
    if not isinstance(value, str) or "{" not in value:
        return value
    return value.format(**_asset_roots())


def _apply_cfg_overrides(cfg: Any, overrides: dict[str, Any] | None) -> Any:
    """Assign ``overrides`` (dotted attribute paths -> values) onto an env config.

    String values may reference Isaac Lab's asset roots (:data:`ASSET_ROOT_NAMES`)
    as ``{NAME}`` placeholders.

    Raises:
        AttributeError: If a path does not exist on the config (typo guard).
    """
    for path, value in (overrides or {}).items():
        *parents, leaf = str(path).split(".")
        target = cfg
        for name in parents:
            target = getattr(target, name)
        if not hasattr(target, leaf):
            raise AttributeError(f"env config has no attribute {path!r} to override")
        setattr(target, leaf, _expand_override_value(value))
    return cfg

_APP_LAUNCHER: Any = None


def launch_app() -> None:
    """Start the Isaac Sim application once per process (headless, cameras on).

    Skipped when an outer launcher already started it (:data:`APP_LAUNCHED_FLAG`
    set to ``"1"``) or when this function already ran.

    Raises:
        ImportError: If the IsaacLab stack is not installed.
    """
    global _APP_LAUNCHER
    # Process-boundary flag, not a tuning value: an IsaacLab-side launcher
    # sets it after its own AppLauncher (Isaac Sim allows one app per process).
    if _APP_LAUNCHER is not None or os.environ.get(APP_LAUNCHED_FLAG) == "1":
        return
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ImportError(
            "The IsaacLab stack is not installed (needs isaaclab + isaaclab_tasks under "
            "Isaac Sim); follow simulators/isaaclab/README.md"
        ) from exc
    _APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=True)


def _registered_task_id(task_id: str) -> str:
    """``task_id`` as Gymnasium registers it: Isaac Lab 3.0.0 moved the Franka stack task to ``IsaacContrib-*``."""
    contrib = "IsaacContrib-" + task_id.removeprefix("Isaac-").removesuffix("-v0")
    return task_id if task_id in gym.registry or contrib not in gym.registry else contrib


def _make_isaaclab_env(
    task_id: str, cfg_overrides: dict[str, Any] | None = None, num_envs: int = 1
) -> Any:
    """Create the IsaacLab environment for ``task_id`` hosting ``num_envs`` scenes.

    Args:
        task_id: Registered IsaacLab task id.
        cfg_overrides: Dotted env-config paths to assign before construction
            (a mapping's ``env_cfg_overrides``, e.g. camera resolutions).
        num_envs: Scenes cloned in the one simulation (IsaacLab's native
            vectorization; each has its own cameras).
    """
    launch_app()
    # Importing isaaclab_tasks registers every task id with Gymnasium.
    from isaaclab_tasks.utils import parse_env_cfg

    task_id = _registered_task_id(task_id)
    env_cfg = parse_env_cfg(task_id)  # also resolves the presets of Isaac Lab 3.0.0 task configs
    env_cfg.scene.num_envs = int(num_envs)
    _apply_cfg_overrides(env_cfg, cfg_overrides)
    return gym.make(task_id, cfg=env_cfg, render_mode=None)


@dataclass
class _SharedEnv:
    """One process-wide IsaacLab simulation shared by the train and eval views.

    ``generation`` counts every reset/step any view performs, so the batched
    training view can tell when the evaluation view stepped the simulation
    in between and its running episodes are no longer valid.
    """

    env: Any
    handles: int = 0
    num_envs: int = 1
    generation: int = 0


_SHARED_ENVS: dict[str, _SharedEnv] = {}


def get_or_create_isaaclab_env(
    task_id: str,
    *,
    make_env: Callable[[str], Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    num_envs: int = 1,
) -> Any:
    """Return the process-wide IsaacLab environment for ``task_id``.

    The first call creates it, with ``num_envs`` scenes; later calls (the
    evaluation view, which drives scene 0) return the same object and add a
    handle, released by :func:`release_isaaclab_env`. Sharing is safe because
    the trainer collects and evaluates sequentially; the batched training view
    detects the evaluation view's steps through the shared generation counter.

    Args:
        task_id: Registered IsaacLab task id.
        make_env: Builds the environment from the task id; defaults to
            ``gym.make`` over the registry config (tests inject a fake here).
        cfg_overrides: Env-config overrides for the default builder (ignored
            for an injected ``make_env``).
        num_envs: Scenes the simulation must host (the training view's
            ``training.num_envs``); a view needing fewer attaches to a larger
            simulation.

    Raises:
        RuntimeError: If a different task's environment is already live
            (Isaac Sim hosts one simulation per process), or the live one
            hosts fewer scenes than requested.
    """
    live = [tid for tid in _SHARED_ENVS if tid != task_id]
    if live:
        raise RuntimeError(
            f"IsaacLab environment {live[0]!r} is already live in this process; "
            f"cannot also create {task_id!r} (one simulation per process)"
        )
    shared = _SHARED_ENVS.get(task_id)
    if shared is None:
        if make_env is not None:
            env = make_env(task_id)
        elif cfg_overrides or num_envs != 1:
            env = _make_isaaclab_env(task_id, cfg_overrides=cfg_overrides, num_envs=num_envs)
        else:
            env = _make_isaaclab_env(task_id)
        shared = _SharedEnv(env=env, num_envs=int(getattr(env.unwrapped, "num_envs", num_envs)))
        _SHARED_ENVS[task_id] = shared
    elif int(num_envs) > shared.num_envs:
        raise RuntimeError(
            f"the live IsaacLab simulation for {task_id!r} hosts {shared.num_envs} scene(s); "
            f"cannot serve a view of {num_envs}. Create the training view first."
        )
    shared.handles += 1
    return shared.env


def _shared_record(task_id: str) -> _SharedEnv | None:
    return _SHARED_ENVS.get(task_id)


def release_isaaclab_env(task_id: str) -> None:
    """Drop one handle on the shared environment; close it with the last one.

    Raises:
        KeyError: If no shared environment exists for ``task_id``.
    """
    shared = _SHARED_ENVS[task_id]
    shared.handles -= 1
    if shared.handles <= 0:
        del _SHARED_ENVS[task_id]
        shared.env.close()


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensors (any device) or array-likes to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _state_width(spec: dict[str, Any]) -> int:
    start, stop = spec["slice"]
    return int(stop) - int(start)


def _validate_mapping(mapping: dict[str, Any]) -> None:
    if not mapping["cameras"]:
        raise ValueError("gr00t mapping needs at least one camera (video key -> camera term)")
    if not mapping["states"]:
        raise ValueError("gr00t mapping needs at least one state entry")
    for key, spec in mapping["states"].items():
        start, stop = spec["slice"]
        if not (0 <= start < stop):
            raise ValueError(f"state entry {key!r} has an invalid slice {spec['slice']}")
        transform = spec.get("transform")
        if transform is not None and transform not in _STATE_TRANSFORMS:
            raise ValueError(
                f"state entry {key!r} requests unknown transform {transform!r}; "
                f"known: {sorted(_STATE_TRANSFORMS)}"
            )
    for pad in ("prefix_pad", "suffix_pad"):
        if int(mapping[pad]) < 0:
            raise ValueError(f"{pad} must be non-negative, got {mapping[pad]}")
    bounds = mapping.get("action_bounds")
    if bounds is not None and len(bounds["low"]) != len(bounds["high"]):
        raise ValueError("action_bounds low/high must have the same length")
    if bounds is None and not mapping.get("action_term"):
        raise ValueError("gr00t mapping needs action_bounds or an action_term to read joint limits from")


def _sim_num_envs(sim: Any) -> int:
    return int(getattr(sim, "num_envs", 1))


def _find_term(raw: dict[str, Any], term: str, groups: tuple[str, ...]) -> Any:
    """Look ``term`` up in the first observation group that carries it."""
    for group in groups:
        entries = raw.get(group)
        if isinstance(entries, dict) and term in entries:
            return entries[term]
    raise KeyError(f"observation term {term!r} not found in groups {list(groups)}")


def _observation_arrays(
    raw: dict[str, Any], mapping: dict[str, Any], image_size: int
) -> dict[str, np.ndarray]:
    """Map IsaacLab's observation groups (all scenes) onto GR00T-keyed ``[N, ...]`` arrays.

    Cameras become uint8 ``[N, H, W, 3]`` (looked up in ``camera_images``,
    then ``policy``); each state entry is its term (after the optional
    transform) sliced to ``[N, d]``; ``state`` concatenates them ``[N, D]``.
    """
    observation: dict[str, np.ndarray] = {}
    for video_key, camera in mapping["cameras"].items():
        image = _to_numpy(_find_term(raw, camera, ("camera_images", "policy")))
        if image.shape[1:] != (image_size, image_size, 3):
            raise ValueError(
                f"camera {camera!r} renders {image.shape[1:]}, but image_size="
                f"{image_size} was declared; match the task's camera resolution"
            )
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        observation[video_key] = np.ascontiguousarray(image)
    parts = []
    for state_key, spec in mapping["states"].items():
        start, stop = spec["slice"]
        term = _to_numpy(_find_term(raw, spec["term"], ("policy",)))
        transform = spec.get("transform")
        if transform is not None:
            term = _STATE_TRANSFORMS[transform](term)
        value = np.ascontiguousarray(term[:, start:stop], dtype=np.float32)
        observation[state_key] = value
        parts.append(value)
    observation["state"] = np.concatenate(parts, axis=-1)
    return observation


def _stage_terms_count(raw: dict[str, Any] | None, terms: list[str] | None) -> np.ndarray | None:
    """How many of the boolean ``terms`` hold per scene (``[N]``), or None."""
    if not terms or not raw:
        return None
    try:
        flags = [_to_numpy(_find_term(raw, t, ("subtask_terms", "policy"))).reshape(-1) for t in terms]
    except KeyError:
        return None
    return np.sum(np.stack([f.astype(bool) for f in flags], axis=0), axis=0).astype(int)


def _observation_space(mapping: dict[str, Any], image_size: int) -> gym.spaces.Dict:
    image_space = gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
    state_dim = int(sum(_state_width(spec) for spec in mapping["states"].values()))
    return gym.spaces.Dict(
        {
            **dict.fromkeys(mapping["cameras"], image_space),
            **{
                key: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_state_width(spec),), dtype=np.float32)
                for key, spec in mapping["states"].items()
            },
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
        }
    )


def _action_bounds(
    sim: Any, mapping: dict[str, Any], action_dim: int, prefix_pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension action bounds: the mapping's explicit ``action_bounds`` when
    given, else the soft joint-position limits of the controlled joints in
    action order, else :data:`DEFAULT_JOINT_BOUND` (an injected environment
    without a ``scene``). A misnamed action term or robot raises."""
    explicit = mapping.get("action_bounds")
    if explicit is not None:
        low = np.asarray(explicit["low"], dtype=np.float32)
        high = np.asarray(explicit["high"], dtype=np.float32)
        if low.shape != (action_dim,):
            raise ValueError(f"action_bounds have {low.shape[0]} entries but action_dim={action_dim}")
        return low, high
    if not hasattr(sim, "scene"):
        return (
            np.full(action_dim, -DEFAULT_JOINT_BOUND, dtype=np.float32),
            np.full(action_dim, DEFAULT_JOINT_BOUND, dtype=np.float32),
        )
    robot = sim.scene["robot"]
    action_term = getattr(sim.cfg.actions, mapping["action_term"])
    joint_names = list(action_term.joint_names)
    controlled = joint_names[prefix_pad : prefix_pad + action_dim]
    joint_ids, _ = robot.find_joints(controlled, preserve_order=True)
    # IsaacLab 3.0 data buffers are ProxyArrays; ``.torch`` is the tensor view.
    limits = _to_numpy(robot.data.soft_joint_pos_limits.torch).astype(np.float32)
    bounds = limits[0, joint_ids]
    return bounds[:, 0], bounds[:, 1]


def _env_actions(
    rows: np.ndarray, mapping: dict[str, Any], env_action_dim: int, prefix_pad: int, device: Any
) -> torch.Tensor:
    """Model actions ``[N, A]`` -> simulator actions ``[N, env_action_dim]``.

    Zero-pads the uncontrolled dimensions; dimensions listed in the mapping's
    ``sign_dims`` (binary commands such as a gripper) are sent as their sign.
    """
    rows = np.array(rows, dtype=np.float32, copy=True)
    for dim in mapping.get("sign_dims") or ():
        rows[:, dim] = np.sign(rows[:, dim])
    env_action = torch.zeros((rows.shape[0], env_action_dim), dtype=torch.float32, device=device)
    env_action[:, prefix_pad : prefix_pad + rows.shape[1]] = torch.from_numpy(np.ascontiguousarray(rows))
    return env_action


class IsaacLabGr00tEnv(gym.Env):
    """One IsaacLab task as an in-process Gymnasium environment (env 0 of the sim).

    Observations are the GR00T-native dict described in the module docstring;
    the declared ``observation_space`` covers the array entries, and the
    ``prompt`` string rides along for the language wrappers. Actions are the
    model's ``float32 [action_dim]`` joint targets or a ``[chunk, action_dim]``
    chunk executed sequentially, one episode-cap step per row.

    IsaacLab resets a finished env inside ``step`` (the observation returned
    on a terminal step is post-reset); :meth:`reset` then re-applies the
    task's reset events, so every episode starts from a fresh draw.

    Args:
        task_id: Registered IsaacLab task id.
        env: Pre-built IsaacLab environment (tests inject a fake here); by
            default the shared environment from
            :func:`get_or_create_isaaclab_env`.
        mapping: Simulator-to-model mapping (see :data:`TROCAR_GR00T_MAPPING`,
            the default).
        max_episode_steps: Client-side episode cap in control steps; ``None``
            leaves only IsaacLab's own time-out.
        image_size: Square camera resolution declared in the observation space.
        action_dim: Width of the model action (the controlled joints).
        seed: Seed applied on the first ``reset`` (IsaacLab seeds its RNGs
            from it) and to this env's ``np_random``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: str = DEFAULT_TASK_ID,
        *,
        env: Any | None = None,
        mapping: dict[str, Any] | None = None,
        max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.mapping = dict(mapping if mapping is not None else TROCAR_GR00T_MAPPING)
        _validate_mapping(self.mapping)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.prefix_pad = int(self.mapping["prefix_pad"])
        self.suffix_pad = int(self.mapping["suffix_pad"])
        self.env_action_dim = self.prefix_pad + self.action_dim + self.suffix_pad
        self.language_instruction: str = str(self.mapping["task_description"])

        self._owns_shared_handle = env is None
        self._env = (
            env
            if env is not None
            else get_or_create_isaaclab_env(
                self.task_id, cfg_overrides=self.mapping.get("env_cfg_overrides")
            )
        )
        self._shared = _shared_record(self.task_id) if env is None else None
        self._sim = self._env.unwrapped
        self.device = self._sim.device
        self._pending_seed = seed
        self._episode_steps = 0
        self._closed = False

        self.video_keys: list[str] = list(self.mapping["cameras"])
        self.state_keys: list[str] = list(self.mapping["states"])
        self.state_dim = int(sum(_state_width(spec) for spec in self.mapping["states"].values()))

        self.observation_space = _observation_space(self.mapping, self.image_size)
        low, high = _action_bounds(self._sim, self.mapping, self.action_dim, self.prefix_pad)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Mapping the steering-side actor/critic use to locate state / image /
        # language entries in the observation dict.
        self.steering_key_mapping: dict[str, Any] = {
            "state": "state",
            "language": None,
            "image": list(self.video_keys),
        }
        # The SAC networks see only the flat state plus the DINO embeddings;
        # the frozen policy consumes the views and per-group states via infer.
        self.steering_remove_obs_keys: list[str] = [*self.video_keys, *self.state_keys]

        if seed is not None:
            super().reset(seed=seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the task and return ``(observation, info)``.

        Args:
            seed: Forwarded to IsaacLab's reset (which seeds torch/NumPy);
                the constructor's ``seed`` is used on the first reset when
                none is given.
            options: Unused; accepted for the Gymnasium signature.
        """
        super().reset(seed=seed)
        self._assert_open()
        if seed is None:
            seed, self._pending_seed = self._pending_seed, None
        self._episode_steps = 0
        raw, _ = self._env.reset(seed=seed)
        self._touch()
        info = {
            "task_id": self.task_id,
            "prompt": self.language_instruction,
            # step() executes [chunk, action_dim] chunks natively; rollout/eval
            # workers key their batched-step fast path on this flag.
            "supports_action_chunking": True,
        }
        return self._observation(raw), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute a model action (or an ``[chunk, action_dim]`` chunk).

        Rows run sequentially and stop early on termination, on an IsaacLab
        truncation (time-out, object drop), or on the client-side cap; each
        row consumes one cap step. Rewards (IsaacLab's, env 0) are summed.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` with ``terminated``
            == task success and ``info["num_steps"]`` the rows executed.
        """
        self._assert_open()
        action = self._action_rows(action)

        total_reward = 0.0
        terminated = truncated = sim_reset = False
        steps_consumed = 0
        raw: dict[str, Any] = getattr(self, "_last_raw", None) or {}
        stage_before = None
        for row in action:
            if self._episode_steps == 0:
                self._log_first_step_diagnostics(row)
            # IsaacLab resets a finished env inside step() and returns the fresh
            # observation, so the progress diagnostic is read before every step.
            stage_before = self._task_stage(raw)
            raw, reward, done, timed_out, _ = self._env.step(self._env_action(row))
            self._touch()
            steps_consumed += 1
            self._episode_steps += 1
            total_reward += float(_to_numpy(reward).reshape(-1)[0])
            terminated = bool(_to_numpy(done).reshape(-1)[0])
            truncated = bool(_to_numpy(timed_out).reshape(-1)[0])
            if terminated or truncated:
                sim_reset = True
                break
            if self._step_cap_reached():
                truncated = True
                break

        info = {"success": terminated, "is_success": terminated, "num_steps": steps_consumed}
        if terminated or truncated:
            stage = stage_before if sim_reset else self._task_stage(raw)
            if stage is not None:
                info["task_stage"] = stage
            self._log_episode_end(stage, terminated, truncated)
        return self._observation(raw), total_reward, terminated, truncated, info

    def _action_rows(self, action: np.ndarray) -> np.ndarray:
        """``action`` as ``[rows, action_dim]`` float32, a single action becoming one row."""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected an action of shape [{self.action_dim}] or "
                f"[chunk, {self.action_dim}], got {action.shape}"
            )
        return action

    def _step_cap_reached(self) -> bool:
        """Whether the episode has used up the client-side step cap."""
        return self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps

    def _log_episode_end(self, stage: int | None, terminated: bool, truncated: bool) -> None:
        """Log how the episode ended and how far the robot and the scene moved (pipeline checks)."""
        moved = frame_change = None
        if getattr(self, "_episode_start_state", None) is not None and self._last_state is not None:
            moved = float(np.abs(self._last_state - self._episode_start_state).max())
        if getattr(self, "_episode_start_frame", None) is not None and getattr(self, "_last_frame", None) is not None:
            frame_change = float(np.abs(self._last_frame.astype(np.int16) - self._episode_start_frame).mean())
        logger.info(
            f"[isaaclab] episode end: steps={self._episode_steps} success={terminated} "
            f"truncated={truncated and not terminated} stage={stage} "
            f"max|state_end-state_start|={moved} mean|frame_end-frame_start|={frame_change}"
        )

    def _log_first_step_diagnostics(self, row: np.ndarray) -> None:
        """Log the first action of an episode next to the current state (pipeline checks)."""
        state = getattr(self, "_last_state", None)
        env_action = _to_numpy(self._env_action(row)).reshape(-1)
        low, high = self.action_space.low, self.action_space.high
        msg = (
            f"[isaaclab-diag] first action min/max={row.min():.3f}/{row.max():.3f} "
            f"env action width={env_action.shape[0]} (prefix pad {self.prefix_pad}) "
            f"bounds low min/max={low.min():.2f}/{low.max():.2f} high min/max={high.min():.2f}/{high.max():.2f}"
        )
        if state is not None and state.shape == row.shape:
            # Joint-position actions: how far the first target sits from the current state.
            diff = np.abs(row - state)
            msg += f" | state min/max={state.min():.3f}/{state.max():.3f} max|action-state|={diff.max():.3f} (joint {int(diff.argmax())})"
        logger.info(msg)

    def _task_stage(self, raw: dict[str, Any] | None = None) -> int | None:
        """Task stage of env 0: the mapping's ``stage_fn`` (read from the live
        simulation), else how many of its boolean ``stage_terms`` (group
        ``subtask_terms``) hold in ``raw``; None when unavailable."""
        terms = self.mapping.get("stage_terms")
        if terms and not self.mapping.get("stage_fn"):
            raw = raw if raw is not None else (getattr(self, "_last_raw", None) or {})
            try:
                flags = [bool(_to_numpy(self._term(raw, t, ("subtask_terms", "policy"))).reshape(-1)[0]) for t in terms]
            except KeyError:
                return None
            return int(sum(flags))
        spec = self.mapping.get("stage_fn")
        if not spec or getattr(self, "_stage_fn_failed", False):
            return None
        try:
            fn = self._stage_fn if hasattr(self, "_stage_fn") else None
            if fn is None:
                if callable(spec):
                    fn = spec
                else:
                    import importlib

                    module_name, func_name = str(spec).split(":", 1)
                    fn = getattr(importlib.import_module(module_name), func_name)
                self._stage_fn = fn
            return int(_to_numpy(fn(self._sim)).reshape(-1)[0])
        except Exception as exc:  # diagnostics must never break a step
            self._stage_fn_failed = True
            logger.warning(f"[isaaclab] task stage unavailable ({type(exc).__name__}: {exc}); disabling")
            return None

    def close(self) -> None:
        """Release the IsaacLab environment (the app stays up); idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_shared_handle:
            release_isaaclab_env(self.task_id)
        else:
            self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("IsaacLabGr00tEnv is closed")

    def _touch(self) -> None:
        """Record that this view advanced the shared simulation."""
        if self._shared is not None:
            self._shared.generation += 1

    def _env_action(self, row: np.ndarray) -> torch.Tensor:
        """One model action -> the simulator's ``[num_scenes, env_action_dim]`` command.

        Scene 0 receives the action (zero-padded to the simulator's width,
        ``sign_dims`` as their sign); any other scene of a larger shared
        simulation receives zeros.
        """
        rows = np.zeros((_sim_num_envs(self._sim), self.action_dim), dtype=np.float32)
        rows[0] = np.asarray(row, dtype=np.float32)
        return _env_actions(rows, self.mapping, self.env_action_dim, self.prefix_pad, self.device)

    @staticmethod
    def _term(raw: dict[str, Any], term: str, groups: tuple[str, ...]) -> Any:
        """Look ``term`` up in the first observation group that carries it."""
        return _find_term(raw, term, groups)

    def _observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map IsaacLab's observation groups (scene 0) onto the GR00T-native dict."""
        self._last_raw = raw
        arrays = _observation_arrays(raw, self.mapping, self.image_size)
        observation: dict[str, Any] = {key: value[0] for key, value in arrays.items()}
        observation["prompt"] = self.language_instruction
        self._last_state = observation["state"]
        if self._episode_steps == 0:
            self._episode_start_state = observation["state"].copy()
            self._episode_start_frame = observation[self.video_keys[0]].astype(np.int16)
            if getattr(self, "_frame_diag_count", 0) < 3:
                self._frame_diag_count = getattr(self, "_frame_diag_count", 0) + 1
                stats = {k: (float(observation[k].mean()), float(observation[k].std())) for k in self.mapping["cameras"]}
                logger.info(f"[isaaclab-diag] first frames mean/std per view={stats} state={np.round(observation['state'], 3).tolist()}")
        self._last_frame = observation[self.video_keys[0]]
        return observation

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown: simulator internals may be gone


class IsaacLabBatchedGr00tEnv:
    """N scenes of one IsaacLab simulation as a natively batched environment.

    The training-side counterpart of :class:`IsaacLabGr00tEnv` for
    ``training.num_envs > 1``: IsaacLab clones the scene ``num_envs`` times in
    one simulation (each clone with its own cameras) and steps them together,
    so ``step`` takes ``[N, A]`` model actions and returns ``[N, ...]``
    observations, rewards and flags. It implements the contract of
    :class:`~isaac_rl_posttraining.envs.vector_backends.BatchedEnvAdapter`
    (``reset`` honoring ``options={"reset_mask": mask}``, no action chunking:
    the collector's per-scene action queues feed one action per scene per
    step), which presents it through the standard ``gym.vector`` interface
    as the ``batched`` vector backend.

    IsaacLab resets a finished scene inside its own ``step`` and returns the
    new episode's first observation with ``terminated``; the adapter's masked
    reset the step after restarts that scene once more (one spent step per
    episode) and is what restarts scenes cut by the client-side cap, which
    the simulator never sees.

    The evaluation view (:class:`IsaacLabGr00tEnv`) drives scene 0 of the same
    simulation between collection phases. That invalidates the running
    training episodes, so the first ``step`` afterwards returns them as
    truncated ``disconnected`` transitions (the collector closes the episodes
    without storing the bogus step) and the adapter restarts every scene.

    Args:
        task_id: Registered IsaacLab task id.
        num_envs: Scenes hosted by the simulation.
        mapping: Simulator-to-model mapping (see :data:`TASK_MAPPINGS`).
        max_episode_steps: Client-side per-scene episode cap in control steps.
        image_size: Square camera resolution the mapping's cameras render at.
        action_dim: Width of one model action.
        seed: Seed of the first reset.
        env: Injected IsaacLab-shaped environment (tests); defaults to the
            shared simulation created with ``num_envs`` scenes.
    """

    def __init__(
        self,
        task_id: str,
        *,
        num_envs: int,
        mapping: dict[str, Any] | None = None,
        max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        seed: int | None = None,
        env: Any | None = None,
    ) -> None:
        self.task_id = str(task_id)
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.mapping = dict(mapping if mapping is not None else mapping_for_task(self.task_id))
        _validate_mapping(self.mapping)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.prefix_pad = int(self.mapping["prefix_pad"])
        self.suffix_pad = int(self.mapping["suffix_pad"])
        self.env_action_dim = self.prefix_pad + self.action_dim + self.suffix_pad
        self.language_instruction: str = str(self.mapping["task_description"])

        self._owns_shared_handle = env is None
        self._env = (
            env
            if env is not None
            else get_or_create_isaaclab_env(
                self.task_id,
                cfg_overrides=self.mapping.get("env_cfg_overrides"),
                num_envs=self.num_envs,
            )
        )
        self._shared = _shared_record(self.task_id) if env is None else None
        self._sim = self._env.unwrapped
        if _sim_num_envs(self._sim) != self.num_envs:
            raise RuntimeError(
                f"the IsaacLab simulation hosts {_sim_num_envs(self._sim)} scene(s) but the batched "
                f"view was built for {self.num_envs}; create the training view first"
            )
        self.device = self._sim.device
        self._pending_seed = seed
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._generation_seen: int | None = None
        self._last_raw: dict[str, Any] | None = None
        self._last_obs: dict[str, np.ndarray] | None = None
        self._closed = False
        self.need_obs = True  # accepted from the collector; observations are always produced

        self.video_keys: list[str] = list(self.mapping["cameras"])
        self.state_keys: list[str] = list(self.mapping["states"])
        self.single_observation_space = _observation_space(self.mapping, self.image_size)
        low, high = _action_bounds(self._sim, self.mapping, self.action_dim, self.prefix_pad)
        self.single_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.steering_key_mapping: dict[str, Any] = {"state": "state", "language": None, "image": list(self.video_keys)}
        self.steering_remove_obs_keys: list[str] = [*self.video_keys, *self.state_keys]

    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every scene, or only those flagged in ``options["reset_mask"]``."""
        self._assert_open()
        mask = None if options is None else options.get("reset_mask")
        if mask is None:
            if seed is None:
                seed, self._pending_seed = self._pending_seed, None
            raw, _ = self._env.reset(seed=seed)
            self._episode_steps[:] = 0
        else:
            env_ids = [int(i) for i in np.flatnonzero(np.asarray(mask))]
            if env_ids:
                # IsaacLab's managers index with the ids as a tensor on the sim device.
                ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self.device)
                raw, _ = self._sim.reset(env_ids=ids)
            else:
                raw = self._last_raw
            self._episode_steps[env_ids] = 0
        self._mark_generation()
        obs = _observation_arrays(raw, self.mapping, self.image_size)
        self._last_raw, self._last_obs = raw, obs
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Advance every scene by one control step with its own action."""
        self._assert_open()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"actions must be [num_envs={self.num_envs}, {self.action_dim}], got {actions.shape}"
            )
        if self._interrupted():
            # Another view stepped the simulation since our last step: the
            # running episodes are gone. Close them without storing this step;
            # the adapter restarts every scene next step.
            logger.info("[isaaclab] batched view: simulation was used by another view; ending all episodes")
            infos = [{"disconnected": True, "success": False, "is_success": False} for _ in range(self.num_envs)]
            self._episode_steps[:] = 0
            return (
                self._last_obs,
                np.zeros(self.num_envs, dtype=np.float64),
                np.zeros(self.num_envs, dtype=bool),
                np.ones(self.num_envs, dtype=bool),
                infos,
            )

        stage_before = _stage_terms_count(self._last_raw, self.mapping.get("stage_terms"))
        env_actions = _env_actions(actions, self.mapping, self.env_action_dim, self.prefix_pad, self.device)
        raw, reward, done, timed_out, _ = self._env.step(env_actions)
        self._mark_generation()
        self._episode_steps += 1
        reward = _to_numpy(reward).reshape(self.num_envs).astype(np.float64)
        terminated = _to_numpy(done).reshape(self.num_envs).astype(bool)
        sim_truncated = _to_numpy(timed_out).reshape(self.num_envs).astype(bool)
        truncated = sim_truncated.copy()
        if self.max_episode_steps is not None:
            truncated |= (self._episode_steps >= self.max_episode_steps) & ~terminated
        obs = _observation_arrays(raw, self.mapping, self.image_size)
        stage_after = _stage_terms_count(raw, self.mapping.get("stage_terms"))

        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            info = {"success": bool(terminated[i]), "is_success": bool(terminated[i]), "num_steps": 1}
            if terminated[i] or truncated[i]:
                # The simulator already reset a finished scene: read its stage from before the step.
                source = stage_before if (terminated[i] or sim_truncated[i]) else stage_after
                if source is not None:
                    info["task_stage"] = int(source[i])
                logger.info(
                    f"[isaaclab] scene {i} episode end: steps={int(self._episode_steps[i])} "
                    f"success={bool(terminated[i])} truncated={bool(truncated[i] and not terminated[i])} "
                    f"stage={info.get('task_stage')}"
                )
                self._episode_steps[i] = 0
            infos.append(info)
        self._last_raw, self._last_obs = raw, obs
        return obs, reward, terminated, truncated, infos

    def close(self) -> None:
        """Release the IsaacLab environment (the app stays up); idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_shared_handle:
            release_isaaclab_env(self.task_id)
        else:
            self._env.close()

    # ------------------------------------------------------------------
    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("IsaacLabBatchedGr00tEnv is closed")

    def _mark_generation(self) -> None:
        if self._shared is not None:
            self._shared.generation += 1
            self._generation_seen = self._shared.generation

    def _interrupted(self) -> bool:
        return (
            self._shared is not None
            and self._generation_seen is not None
            and self._shared.generation != self._generation_seen
        )
