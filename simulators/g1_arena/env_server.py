#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Arena environment server.

This script is the simulator side of the two-process split: it runs under an
environment with Isaac Sim and IsaacLab-Arena (the G1 workflow's own
``example_workflows/g1_arena_posttraining/.venv``, or ``SIM_PYTHON``), and the
learner connects to it as a client. The
script builds the requested Arena environment and serves it over the native
TCP transport (see ``isaac_rl_posttraining/envs/transport.py`` for the wire
schema) or, with ``--transport zmq``, over the ZeroMQ transport
(``isaac_rl_posttraining/envs/zmq_transport.py``; the client then connects
with a ``zmq+tcp://`` URL). The learner drives it through
``simulators.g1_arena.env.ArenaEnv``.

With ``--num-envs N`` (N > 1) one simulator instance hosts N cloned
environments (Isaac Lab's natively batched scene): wire payloads carry a
leading ``[N]`` axis, one MSG_STEP advances every env by one simulator step,
and MSG_RESET honors ``env_ids`` for rolling per-env restarts. The learner's
rollout worker then connects through ``simulators.g1_arena.env.ArenaBatchedEnv``.

Only ``isaac_rl_posttraining.envs.transport`` (standard library; plus, for
``--transport zmq`` only, the ``[zmq]`` extra's pyzmq/msgpack) is imported
from this repository, so the package does not need to be installed in the
Arena environment -- the script adds its own repository root to ``sys.path``.

Environment variables:
    ISAACLAB_ARENA_PATH: IsaacLab-Arena checkout; used to locate the Arena
        runtime and the G1 joint-space configuration files.
    ARENA_SIM_DEVICE: Simulation/physics device, ``cpu`` or ``gpu`` (also
        accepts an explicit ``cuda[:N]``). Forwarded to Isaac Lab's
        ``--device`` argument; unset keeps Isaac Lab's default (GPU). The
        resulting device is printed at boot.

Lifecycle contract (relied on by the launchers and the integration tests):
    * readiness: the line ``[arena_env_server] listening on <host>:<port>``
      is printed once the socket is bound and the simulator is ready;
    * shutdown: SIGTERM triggers a clean exit through the simulation
      context's normal teardown path.

Example (from the Arena checkout's venv):
    ISAACLAB_ARENA_PATH=<arena checkout> \\
        <arena venv python> simulators/g1_arena/env_server.py \\
        --task LMDrillLift --embodiment g1_wbc_joint \\
        --port 6000 --action-scale 1.0 --image-resolution 0
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

# The Arena venv does not have this package installed; import it from the
# repository checkout this script lives in.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaac_rl_posttraining.data.spec import ActionSpec, ArraySpec, ObservationSpec
from isaac_rl_posttraining.envs.transport import (  # noqa: E402
    MSG_CLOSE,
    MSG_RESET,
    MSG_SPEC,
    MSG_STEP,
    MSG_SUCCESS_CHECK,
    PROTOCOL_VERSION,
    NativeTcpServer,
    spec_payload,
)
from simulators.g1_arena.image_ops import letterbox_square, shrink_shortest_edge  # noqa: E402
from simulators.g1_arena.stall_dump import StallDump  # noqa: E402
from simulators.g1_arena.task_prompt import resolve_task_prompt  # noqa: E402
from simulators.g1_arena.timers import StepTimer  # noqa: E402
from simulators.g1_arena.embodiment import (  # noqa: E402
    G1_WBC_ACTION_DIM,
    G1_WBC_STATE_GROUP_ORDER,
    wbc_base_height_slice,
    wbc_joint_action_slices,
    wbc_navigate_slice,
)

LOG_PREFIX = "[arena_env_server]"

# The G1 whole-body-control state/action layout is owned by
# simulators.g1_arena.embodiment -- one source of truth shared
# with the offline buffer (the two used to be maintained separately and
# silently diverged). Joint-group order of the policy's 31-dim proprioceptive
# state (arms, hands, waist only; the legs are whole-body-controlled):
STATE_GROUP_ORDER = G1_WBC_STATE_GROUP_ORDER

# Layout of the 35-dim whole-body-control policy action. ACTION_SLICES keys the
# arm/hand/waist groups with the bare group names the Arena joint remap matches
# (a prefixed key silently zeros that group); base-height and navigate ride
# separately.
ACTION_SLICES = wbc_joint_action_slices()
BASE_HEIGHT_SLICE = wbc_base_height_slice()
NAVIGATE_SLICE = wbc_navigate_slice()
POLICY_ACTION_DIM = G1_WBC_ACTION_DIM


def parse_args() -> argparse.Namespace:
    """Parse this script's arguments, ignoring Isaac Lab / Arena arguments.

    ``add_help`` is disabled and unknown arguments are tolerated so flags
    meant for the Arena CLI (e.g. ``--viz kit``) pass through untouched.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task", default="LMDrillLift", help="Arena example-environment name"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="environments hosted by this one simulator instance (Isaac Lab's "
        "natively batched scene). With N > 1 the wire payloads carry a leading "
        "[N] axis and MSG_RESET honors env_ids for rolling per-env resets — "
        "the client is simulators.g1_arena.env.ArenaBatchedEnv. "
        "The default keeps the validated single-env "
        "protocol byte-identical.",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=None,
        help="distance in meters between cloned environments; forwarded to the "
        "Arena CLI's --env_spacing (unset: Arena's default for one env, 200 m for --num-envs > 1 -- "
        "closer clones overlap each other's factory floor, see resolve_env_spacing)",
    )
    parser.add_argument(
        "--task-module",
        default="g1_factory.environments",
        help=(
            "Python module holding the task class. The G1 factory LM* tasks live "
            "outside the Arena's built-in registry, so the server registers "
            "'<task-module>.<task>:<task>' via the Arena CLI's "
            "external_environment_class_path before selecting it."
        ),
    )
    parser.add_argument("--embodiment", default="g1_factory_wbc_joint")
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help="GR00T embodiment tag used for policy->sim joint remapping (matches the checkpoint's finetune metadata).",
    )
    parser.add_argument(
        "--object",
        default="locomanip_power_drill",
        help="task object; pass 'none' for tasks that define their own object",
    )
    parser.add_argument(
        "--render-once-prims",
        default=None,
        help="comma-separated prim paths relative to each env clone whose render is kept in env_0 only and "
        "hidden in every other clone (colliders stay). For a clone prim whose effect already covers every "
        "env -- the factory's ambient dome light -- N copies change every env's image while one copy draws "
        "the single-env image. Unset: the factory's dome light for --num-envs > 1, nothing for one env; "
        "'none' disables (see resolve_render_once_prims)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind; 0.0.0.0 serves other hosts, so only on a trusted network (frames are pickled)",
    )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument(
        "--transport",
        choices=("native", "zmq"),
        default="native",
        help="wire transport: 'native' (plain TCP, pickle frames; the default "
        "the launchers rely on) or 'zmq' (ZeroMQ REP socket, msgpack frames; "
        "requires pyzmq/msgpack in the Arena environment and a client using "
        "a zmq+tcp:// URL)",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
        help="square observation size; <= 0 keeps the native camera frame",
    )
    parser.add_argument(
        "--image-shortest-edge",
        type=int,
        default=0,
        help="shrink frames so the shortest side is this many pixels, aspect preserved, exactly as "
        "the frozen policy's image transform does first (GR00T-N1.7: 256); 0 sends native frames. "
        "Exclusive with a positive --image-resolution",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="real_action = action_scale * policy_action (1.0: raw joint targets)",
    )
    parser.add_argument("--camera-key", default="robot_head_cam")
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.16,
        help="success object-height threshold used as a last-resort fallback",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset-state-dir",
        default=None,
        help="optional directory that records each full seeded reset as an "
        "Isaac Lab scene-state snapshot and restores it on later resets. "
        "Reuse the directory for after-the-fact checkpoint evaluation.",
    )
    parser.add_argument(
        "--reset-rerenders",
        type=int,
        default=1,
        help="RTX sensor renders forced around a recorded or restored full reset "
        "(default 1; 0 preserves the simulator's configured behavior)",
    )
    parser.add_argument(
        "--reset-observation-dir",
        default=None,
        help="optional diagnostic directory for compressed camera and state "
        "arrays from each full seeded reset; disabled by default",
    )
    parser.add_argument(
        "--renderer-antialiasing",
        choices=("off", "taa", "fxaa", "dlss", "rtxaa"),
        default=None,
        help="optional Isaac Sim renderer anti-aliasing mode; unset keeps its "
        "default (currently temporal DLSS)",
    )
    parser.add_argument(
        "--strict-reproducibility",
        action="store_true",
        help="enable Isaac Lab's deterministic RTX mode, PhysX enhanced "
        "determinism, and deterministic torch kernels; opt-in because these "
        "settings may reduce simulator throughput",
    )
    parser.add_argument(
        "--disable-texture-streaming",
        action="store_true",
        default=bool(os.environ.get("ARENA_DISABLE_TEXTURE_STREAMING")),
        help="force textures resident; avoids an RTX texture-streaming stall "
        "that can wedge camera rendering on very long rollouts (also enabled "
        "by the ARENA_DISABLE_TEXTURE_STREAMING environment variable)",
    )
    parser.add_argument(
        "--no-wbc-assembly",
        action="store_true",
        help="disable the 35-to-50 dim whole-body-control action assembly "
        "(enabled by default for the g1_wbc_joint embodiment)",
    )
    return parser.parse_known_args()[0]


def make_server(transport: str, host: str, port: int) -> Any:
    """Build the requested transport's server binding.

    Args:
        transport: ``"native"`` or ``"zmq"``.
        host: Interface to bind.
        port: Port to bind.

    Returns:
        A server object exposing ``serve_forever(handler, ...)``.

    Raises:
        SystemExit: For an unknown transport, or when the ZeroMQ transport
            is requested but its optional dependencies are missing.
    """
    if transport == "native":
        return NativeTcpServer(host=host, port=port)
    if transport == "zmq":
        try:
            from isaac_rl_posttraining.envs.zmq_transport import ZmqServer
        except ImportError as exc:
            raise SystemExit(
                f"{LOG_PREFIX} --transport zmq unavailable: {exc}"
            ) from exc
        return ZmqServer(f"tcp://{host}:{port}")
    raise SystemExit(f"{LOG_PREFIX} unknown transport {transport!r}")


# Env spacing for batched scenes. Each env clones the whole factory background
# (gear_factory: a 200 m x 200 m floor, walls spanning 62 m x 36 m). Any
# spacing below the floor's 200 m lays the clones' floors over each other
# (coplanar copies under every camera: z-fighting and a render cost that grows
# with N) and below 62 m puts a neighbour's wall inside this env's hall; 200 m
# tiles the floors edge to edge. An explicit --env-spacing always wins.
DEFAULT_BATCHED_ENV_SPACING_M = 200.0


def resolve_env_spacing(num_envs: int, env_spacing: float | None) -> float | None:
    """``--env-spacing`` default: Arena's own for one env, 200 m for --num-envs > 1."""
    if env_spacing is not None:
        return env_spacing
    return DEFAULT_BATCHED_ENV_SPACING_M if num_envs > 1 else None


# The factory background carries an ambient dome light, and a dome light is
# infinite: N clones light every env N times over. Keeping only env_0's copy
# brings a batched env's frame close to the single-env frame (LMDrillLift, N=8,
# head-camera mean brightness: single env 53, env_0-only dome light 55-58, all
# 8 dome lights 39-42).
DEFAULT_BATCHED_RENDER_ONCE_PRIMS = ("factory_room/Lights/AmbientFill",)

# Isaac Sim's SimulationApp accepts integer anti-aliasing modes. Keep the
# user-facing option readable while preserving its documented values.
RENDERER_ANTIALIASING_MODES = {
    "off": 0,
    "taa": 1,
    "fxaa": 2,
    "dlss": 3,
    "rtxaa": 4,
}


def resolve_renderer_antialiasing(mode: str | None) -> int | None:
    """Map a public anti-aliasing name to Isaac Sim's launcher value."""
    if mode is None:
        return None
    try:
        return RENDERER_ANTIALIASING_MODES[mode.lower()]
    except KeyError as exc:
        choices = ", ".join(RENDERER_ANTIALIASING_MODES)
        raise ValueError(
            f"renderer antialiasing must be one of {choices}, got {mode!r}"
        ) from exc


def configure_renderer_antialiasing(
    launcher_args: argparse.Namespace, mode: str | None
) -> int | None:
    """Attach the renderer mode directly to Isaac Lab's launcher namespace."""
    value = resolve_renderer_antialiasing(mode)
    if value is not None:
        launcher_args.anti_aliasing = value
    return value


def resolve_render_once_prims(
    num_envs: int, render_once_prims: str | None
) -> list[str]:
    """``--render-once-prims`` default: the factory's dome light for --num-envs > 1, nothing for one env.

    ``"none"`` disables the default explicitly (a task without the factory
    room would otherwise fail at boot, since every path must exist).
    """
    if render_once_prims is None:
        return list(DEFAULT_BATCHED_RENDER_ONCE_PRIMS) if num_envs > 1 else []
    if render_once_prims.strip().lower() == "none":
        return []
    paths = [p.strip() for p in render_once_prims.split(",") if p.strip()]
    if not paths:
        raise SystemExit("--render-once-prims needs at least one prim path, or 'none'")
    return paths


class ResetStateStore:
    """Record and restore complete Isaac Lab scene state for seeded evaluation."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: str,
        *,
        num_envs: int,
        torch_module: Any,
        reset_rerenders: int = 1,
    ) -> None:
        if reset_rerenders < 0:
            raise ValueError("reset_rerenders must be non-negative")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.num_envs = int(num_envs)
        self.torch = torch_module
        self.reset_rerenders = int(reset_rerenders)
        self._states: dict[int, dict[str, Any]] = {}

    def _path(self, seed: int) -> Path:
        return self.directory / f"seed_{seed}_envs_{self.num_envs}.pt"

    def _to_cpu(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {name: self._to_cpu(item) for name, item in value.items()}
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().clone()
        raise TypeError(f"unsupported scene-state value {type(value).__name__}")

    def _load(self, seed: int) -> dict[str, Any] | None:
        if seed in self._states:
            return self._states[seed]
        path = self._path(seed)
        if not path.is_file():
            return None
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"reset-state snapshot {path} is not a mapping")
        expected = {
            "schema_version": self.SCHEMA_VERSION,
            "seed": seed,
            "num_envs": self.num_envs,
        }
        actual = {name: payload.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"reset-state snapshot {path} metadata {actual} does not match {expected}"
            )
        state = payload.get("scene_state")
        if not isinstance(state, dict):
            raise TypeError(f"reset-state snapshot {path} has no scene_state mapping")
        self._states[seed] = state
        print(f"{LOG_PREFIX} restored reset state from {path}", flush=True)
        return state

    def _record(self, env: Any, seed: int) -> None:
        state = self._to_cpu(env.unwrapped.scene.get_state(is_relative=True))
        path = self._path(seed)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "seed": seed,
            "num_envs": self.num_envs,
            "scene_state": state,
        }
        try:
            self.torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._states[seed] = state
        print(f"{LOG_PREFIX} recorded reset state at {path}", flush=True)

    def _reset_with_fresh_camera(
        self, env: Any, operation: Callable[[], tuple[Any, Any]]
    ) -> tuple[Any, Any]:
        """Refresh RTX sensors for this reset without slowing later auto-resets."""
        cfg = getattr(env.unwrapped, "cfg", None)
        if (
            self.reset_rerenders == 0
            or cfg is None
            or not hasattr(cfg, "num_rerenders_on_reset")
        ):
            return operation()
        previous = int(cfg.num_rerenders_on_reset)
        cfg.num_rerenders_on_reset = max(previous, self.reset_rerenders)
        try:
            return operation()
        finally:
            cfg.num_rerenders_on_reset = previous

    def reset(self, env: Any, seed: int) -> tuple[Any, Any]:
        state = self._load(seed)
        if state is None:
            # Generate the snapshot once, then evaluate through the same
            # restore path used by later and retained evaluations.
            env.reset(seed=seed)
            self._record(env, seed)
            state = self._states[seed]

        def restore() -> tuple[Any, Any]:
            # Mark Gymnasium's outer OrderEnforcing wrapper as reset before
            # restoring through the Isaac Lab API on the unwrapped env.
            env.reset(seed=seed)
            return env.unwrapped.reset_to(
                state,
                env_ids=None,
                seed=seed,
                is_relative=True,
            )

        return self._reset_with_fresh_camera(
            env,
            restore,
        )


class ResetObservationCapture:
    """Persist full-reset observations for pixel-level replay diagnostics."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: str,
        *,
        num_envs: int,
        numpy_module: Any,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.num_envs = int(num_envs)
        self.np = numpy_module
        self._reset_counts: dict[int, int] = {}

    def record(self, seed: int, observation: dict[str, Any]) -> Path:
        """Atomically write one reset batch without changing the wire payload."""
        reset_index = self._reset_counts.get(seed, 0)
        self._reset_counts[seed] = reset_index + 1
        path = self.directory / (
            f"seed_{seed}_envs_{self.num_envs}_reset_{reset_index:03d}.npz"
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        arrays = {
            str(name): self.np.ascontiguousarray(self.np.asarray(value))
            for name, value in observation.items()
            if name != "prompt"
        }
        metadata = {
            "schema_version": self.np.asarray(self.SCHEMA_VERSION, dtype=self.np.int64),
            "seed": self.np.asarray(seed, dtype=self.np.int64),
            "num_envs": self.np.asarray(self.num_envs, dtype=self.np.int64),
            "reset_index": self.np.asarray(reset_index, dtype=self.np.int64),
        }
        try:
            with temporary.open("wb") as handle:
                self.np.savez_compressed(handle, **metadata, **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{LOG_PREFIX} captured reset observation at {path}", flush=True)
        return path


def run_at_50hz(arena_env: Any) -> None:
    """Keep the 50 Hz control rate (sim dt 1/200, decimation 4) the G1 controller is built for: Arena now defaults to 15 Hz."""
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        set_control_rate_50hz,
    )

    previous = arena_env.env_cfg_callback
    arena_env.env_cfg_callback = lambda cfg: set_control_rate_50hz(
        previous(cfg) if previous is not None else cfg
    )


def reset_with_index_tensors(env: Any) -> None:
    """Hand ``_reset_idx`` index tensors: Isaac Lab 3.0 resets every env with ``slice(None)``, which Arena's reset terms reject."""
    import torch

    unwrapped = env.unwrapped
    all_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    reset_idx = unwrapped._reset_idx

    def _reset_idx(env_ids: Any) -> None:
        reset_idx(all_ids[env_ids] if isinstance(env_ids, slice) else env_ids)

    unwrapped._reset_idx = _reset_idx


def render_prims_once(env: Any, num_envs: int, relative_paths: list[str]) -> None:
    """Keep env_0's copy of each prim visible and hide the copies in env_1..env_{N-1} (render only).

    For prims whose effect is not local to their clone, such as the factory's
    infinite ambient dome light, N copies change every env's image; one copy
    reproduces the single-env image. Colliders are untouched.
    """
    from pxr import UsdGeom

    stage = env.unwrapped.sim.stage
    for relative in relative_paths:
        for env_idx in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/{relative}")
            if not prim.IsValid():
                raise RuntimeError(
                    f"--render-once-prims: /World/envs/env_{env_idx}/{relative} not found"
                )
            if env_idx > 0:
                UsdGeom.Imageable(prim).MakeInvisible()
        print(
            f"{LOG_PREFIX} rendering {relative} from env_0 only ({num_envs - 1} clone copies hidden)",
            flush=True,
        )


def resolve_sim_device() -> str | None:
    """Map ``ARENA_SIM_DEVICE`` to Isaac Lab's ``--device`` value.

    Returns:
        ``"cpu"``, ``"cuda"``/``"cuda:N"``, or ``None`` when the variable is
        unset (keep Isaac Lab's default).

    Raises:
        SystemExit: If the variable holds an unrecognized value.
    """
    raw = os.environ.get("ARENA_SIM_DEVICE", "").strip()
    if not raw:
        print(
            f"{LOG_PREFIX} sim device: default (ARENA_SIM_DEVICE unset; Isaac Lab picks GPU)",
            flush=True,
        )
        return None
    lowered = raw.lower()
    if lowered == "cpu":
        device = "cpu"
    elif lowered == "gpu":
        device = "cuda"
    elif lowered.startswith("cuda"):
        device = raw
    else:
        raise SystemExit(
            f"{LOG_PREFIX} invalid ARENA_SIM_DEVICE={raw!r} (expected cpu or gpu)"
        )
    print(f"{LOG_PREFIX} sim device: {device} (ARENA_SIM_DEVICE={raw})", flush=True)
    return device


def bootstrap_arena_runtime() -> None:
    """Wire up the IsaacLab-Arena runtime from ``ISAACLAB_ARENA_PATH``.

    Running under the Arena venv's interpreter is necessary but not always
    sufficient: Isaac Sim expects the environment variables its activation
    script exports, and its native kit libraries must be loaded with
    ``RTLD_GLOBAL`` before Python extensions that link against them. This
    replays both. No-op when ``ISAACLAB_ARENA_PATH`` is unset (e.g. running
    inside a container whose interpreter is already fully configured).
    """
    arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
    if not arena_path:
        return
    _replay_venv_activation(arena_path)
    _preload_native_libraries(arena_path)
    print(f"{LOG_PREFIX} bootstrapped Arena runtime at {arena_path}", flush=True)


def _replay_venv_activation(arena_path: str) -> None:
    """Recover the environment, import paths and site dirs the Arena venv's activate script sets up."""
    import glob
    import site
    import subprocess

    venv_activate = os.path.join(arena_path, ".venv", "bin", "activate")
    out = subprocess.run(
        ["bash", "-c", f"source {venv_activate} && env -0"],
        capture_output=True,
        text=True,
    ).stdout
    pairs = (line.split("=", 1) for line in out.split("\0") if "=" in line)
    activated = dict(pairs)
    for key in (
        "ISAACLAB_PATH",
        "RESOURCE_NAME",
        "CARB_APP_PATH",
        "EXP_PATH",
        "ISAAC_PATH",
        "LD_LIBRARY_PATH",
    ):
        if key in activated:
            os.environ[key] = activated[key]
    for path in activated.get("PYTHONPATH", "").split(":"):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    for site_packages in glob.glob(
        os.path.join(arena_path, ".venv", "lib", "python*", "site-packages")
    ):
        site.addsitedir(site_packages)


def _preload_native_libraries(arena_path: str) -> None:
    """Load libpython and the Isaac Sim kit libraries with ``RTLD_GLOBAL``.

    Native extensions then resolve their symbols regardless of import order.
    """
    import ctypes
    import glob
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    shared_objects = []
    if libdir:
        libpython = os.path.join(
            libdir, f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
        )
        if os.path.exists(libpython):
            shared_objects.append(libpython)
    kit_dir = os.path.join(arena_path, "submodules", "IsaacLab", "_isaac_sim", "kit")
    shared_objects.extend(glob.glob(os.path.join(kit_dir, "lib*.so")))
    for shared_object in shared_objects:
        try:
            ctypes.CDLL(shared_object, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _disable_texture_streaming() -> None:
    """Force textures resident via carb settings (best effort, never fatal)."""
    try:
        import carb

        settings = carb.settings.get_settings()
        for key, value in (
            ("/rtx-transient/resourcemanager/enableTextureStreaming", False),
            ("/rtx/resourcemanager/enableTextureStreaming", False),
            ("/rtx/materialDb/syncLoads", True),
            ("/omni.kit.plugin/syncUsdLoads", True),
        ):
            try:
                settings.set(key, value)
            except Exception:
                pass  # setting paths vary across kit versions
        print(f"{LOG_PREFIX} texture streaming disabled", flush=True)
    except Exception as exc:
        print(
            f"{LOG_PREFIX} WARN could not disable texture streaming: {exc}", flush=True
        )


def configure_strict_environment(
    env_cfg: Any,
    *,
    physics_cfg_factory: Callable[[], Any] | None = None,
) -> Any:
    """Enable the environment-side controls needed for strict replay."""
    physics_cfg = env_cfg.sim.physics
    if physics_cfg is None:
        if physics_cfg_factory is None:
            from isaaclab_physx.physics import PhysxCfg

            physics_cfg_factory = PhysxCfg
        physics_cfg = physics_cfg_factory()
        env_cfg.sim.physics = physics_cfg
    if not hasattr(physics_cfg, "enable_enhanced_determinism"):
        raise RuntimeError(
            "strict reproducibility requires a PhysX configuration with "
            "enable_enhanced_determinism"
        )
    physics_cfg.enable_enhanced_determinism = True
    return env_cfg


def install_strict_environment_callback(arena_environment: Any) -> None:
    """Compose strict replay settings with an Arena task callback."""
    previous_callback = arena_environment.env_cfg_callback

    def _strict_callback(env_cfg: Any) -> Any:
        if previous_callback is not None:
            env_cfg = previous_callback(env_cfg)
        return configure_strict_environment(env_cfg)

    arena_environment.env_cfg_callback = _strict_callback


def configure_strict_launcher(launcher_args: Any, enabled: bool) -> bool:
    """Enable Isaac Lab's deterministic RTX launch mode when requested."""
    if enabled:
        launcher_args.deterministic = True
    return bool(enabled)


def configure_strict_renderer(settings_manager: Any) -> None:
    """Disable frame-varying RTX effects for strict visual replay."""
    deterministic_settings = {
        "/rtx/directLighting/sampledLighting/stochasticLCRefresh": False,
        "/rtx/pathtracing/temporalAccumulation": False,
        "/rtx/post/tonemap/dither": False,
        "/rtx/raytracing/caustics/useFrameSeed": False,
        "/rtx/raytracing/subpixel/uniformSubpixelJittering": False,
        "/rtx/reflections/sampledLighting/stochasticLCRefresh": False,
        "/rtx/shadows/constantSeed": True,
    }
    for path, value in deterministic_settings.items():
        settings_manager.set_bool(path, value)


def configure_strict_torch(torch_module: Any) -> None:
    """Restore deterministic torch flags reset by Isaac Lab reseeding."""
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False


def _load_wbc_configs(args: argparse.Namespace) -> dict[str, Any]:
    """Load the joint-space configs used for the 35-to-50 action assembly.

    The ``g1_wbc_joint`` action term expects a 50-dim action: 43 joint targets
    (legs, waist, arms, hands, in simulation order) followed by the whole-body
    commands (navigate 3, base height 1, torso roll/pitch/yaw 3). The policy
    emits the 35-dim vector described by :data:`ACTION_SLICES`; the arms,
    hands, and waist are remapped by joint name into the 43-joint vector
    (legs stay zero -- the decoupled whole-body controller drives them) and
    torso roll/pitch/yaw is zero.

    Returns:
        ``{"policy_cfg", "sim_cfg", "remap"}`` -- the group-to-joint-names
        config, the joint-name-to-sim-index config, and the Arena remap
        utility -- or an empty dict when assembly is disabled.
    """
    use_wbc = (
        args.embodiment in ("g1_wbc_joint", "g1_factory_wbc_joint")
        and not args.no_wbc_assembly
    )
    if not use_wbc:
        return {}

    arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
    if not arena_path:
        raise SystemExit(
            f"{LOG_PREFIX} ISAACLAB_ARENA_PATH must be set to locate the G1 joint-space configs"
        )
    from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml
    from isaaclab_arena_gr00t.utils.joints_conversion import (
        remap_policy_joints_to_sim_joints,
    )

    g1_dir = Path(arena_path) / "isaaclab_arena_gr00t" / "embodiments" / "g1"
    wbc = {
        "policy_cfg": load_robot_joints_config_from_yaml(
            g1_dir / "gr00t_43dof_joint_space.yaml"
        ),
        "sim_cfg": load_robot_joints_config_from_yaml(
            g1_dir / "43dof_joint_space.yaml"
        ),
        "remap": partial(
            remap_policy_joints_to_sim_joints, embodiment_tag=args.embodiment_tag
        ),
    }
    print(
        f"{LOG_PREFIX} whole-body action assembly ON: {len(wbc['sim_cfg'])} sim joints, "
        f"groups={list(wbc['policy_cfg'].keys())}",
        flush=True,
    )
    return wbc


def _validate_batched_support(env: Any, wbc: dict[str, Any], num_envs: int) -> None:
    """Fail loud at boot on anything the batched (``--num-envs > 1``) mode needs.

    A silently wrong multi-env setup costs days of GPU time (e.g. success only
    ever observable for env 0), so every requirement is checked before the
    server starts listening.
    """
    import inspect

    if not wbc:
        raise SystemExit(
            f"{LOG_PREFIX} --num-envs {num_envs} requires the whole-body action assembly "
            "(a *_wbc_joint embodiment without --no-wbc-assembly): the batched state and "
            "action protocol is defined in the policy's grouped joint layout"
        )
    active_terms = env.unwrapped.termination_manager.active_terms
    if "success" not in active_terms:
        raise SystemExit(
            f"{LOG_PREFIX} --num-envs {num_envs} requires a 'success' termination term for "
            f"per-env success scoring; active terms: {list(active_terms)}"
        )
    if "env_ids" not in inspect.signature(env.unwrapped.reset).parameters:
        raise SystemExit(
            f"{LOG_PREFIX} --num-envs {num_envs} requires per-env partial resets "
            "(ManagerBasedEnv.reset(env_ids=...)), which this Isaac Lab version lacks"
        )
    if os.environ.get("ARENA_SKIP_WBC_PATCH_CHECK"):
        print(
            f"{LOG_PREFIX} WBC patch check skipped (ARENA_SKIP_WBC_PATCH_CHECK set)",
            flush=True,
        )
        return
    try:
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_homie_policy import (
            G1HomiePolicyV2,
        )
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} WARN cannot import the homie WBC policy to verify the "
            f"multi-env patch ({exc}); continuing unverified",
            flush=True,
        )
        return
    # The stock homie policy wipes EVERY env's recurrent state on any per-env
    # reset and gates stand-vs-walk on the whole command batch; the patch in
    # simulators/g1_arena/patches/ fixes both and stamps this marker. The
    # marker probe is a compatibility check on an external, optionally-patched
    # class, hence the defaulted getattr.
    if not getattr(G1HomiePolicyV2, "supports_env_ids_reset", False):
        raise SystemExit(
            f"{LOG_PREFIX} --num-envs {num_envs} needs the per-env homie WBC fixes: apply "
            "simulators/g1_arena/patches/arena-homie-per-env-wbc.patch to the "
            "IsaacLab-Arena checkout (see simulators/g1_arena/README.md), or set "
            "ARENA_SKIP_WBC_PATCH_CHECK=1 if your embodiment does not use the homie "
            "whole-body policy"
        )


def build_handler(
    env: Any,
    args: argparse.Namespace,
    device: Any,
    task_description: str,
    wbc: dict[str, Any],
    np: Any,
    torch: Any,
    cv2: Any,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any] | None], StepTimer]:
    """Build the transport request handler around a live Arena environment.

    Returns the handler and its step timer; the serve loop feeds the timer
    the per-reply send timing the handler itself cannot see.

    The numpy/torch/cv2 modules are passed in (rather than imported at module
    scope) because they must only be imported after the simulation app has
    initialized.

    Args:
        env: The built (wrapped) Arena environment.
        args: Parsed script arguments.
        device: The environment's compute device.
        task_description: Natural-language task prompt sent with observations.
        wbc: Whole-body-control configs from :func:`_load_wbc_configs`.
        np: The ``numpy`` module.
        torch: The ``torch`` module.
        cv2: The ``cv2`` module.

    Returns:
        A ``handler(message) -> response | None`` for
        :meth:`NativeTcpServer.serve_forever`.
    """
    handler = ArenaRequestHandler(env, args, device, task_description, wbc, np, torch, cv2)
    return handler, handler.step_timer


class ArenaRequestHandler:
    """Answers transport requests against a live Arena environment (see :func:`build_handler`).

    With ``--num-envs > 1`` payloads carry a leading [N] axis; one MSG_STEP
    advances every env by ONE simulator step (the learner's vectorized
    collector paces chunks itself, so the wire never carries chunks here).
    Isaac Lab auto-resets a terminated env inside step(), so the obs returned
    at done is already the fresh episode's first frame; MSG_RESET with
    env_ids performs an explicit partial reset (needed for client-side
    truncations, which the sim never sees). Requires the WBC action assembly
    (validated at boot).
    """

    def __init__(
        self,
        env: Any,
        args: argparse.Namespace,
        device: Any,
        task_description: str,
        wbc: dict[str, Any],
        np: Any,
        torch: Any,
        cv2: Any,
    ) -> None:
        self.env, self.args, self.device = env, args, device
        self.task_description, self.wbc = task_description, wbc
        self.np, self.torch, self.cv2 = np, torch, cv2
        self.num_envs = int(args.num_envs)
        self.batched = self.num_envs > 1
        self.step_timer = StepTimer(self.num_envs, log_prefix=LOG_PREFIX)
        self._logged_once: set[str] = set()
        reset_state_dir = getattr(args, "reset_state_dir", None)
        self.reset_state_store = (
            ResetStateStore(
                str(reset_state_dir),
                num_envs=self.num_envs,
                torch_module=torch,
                reset_rerenders=int(getattr(args, "reset_rerenders", 1)),
            )
            if reset_state_dir
            else None
        )
        reset_observation_dir = getattr(args, "reset_observation_dir", None)
        self.reset_observation_capture = (
            ResetObservationCapture(
                str(reset_observation_dir),
                num_envs=self.num_envs,
                numpy_module=np,
            )
            if reset_observation_dir
            else None
        )
        if self.reset_state_store is not None:
            print(
                f"{LOG_PREFIX} evaluation reset replay enabled; forcing "
                f"{self.reset_state_store.reset_rerenders} RTX sensor rerender(s) per full reset",
                flush=True,
            )
        if self.reset_observation_capture is not None:
            print(
                f"{LOG_PREFIX} reset observation capture enabled at "
                f"{self.reset_observation_capture.directory}",
                flush=True,
            )
        # Column gather that reorders the [N, 43] sim joint vector into the
        # policy's grouped 31-dim state, precomputed once.
        self.state_reindex: list[int] = []
        if wbc:
            self.state_reindex = [
                int(wbc["sim_cfg"][joint_name])
                for group in STATE_GROUP_ORDER
                for joint_name in wbc["policy_cfg"][group]
            ]

    def __call__(self, message: dict[str, Any]) -> dict[str, Any] | None:
        with StallDump.from_environment(LOG_PREFIX):
            return self._dispatch(message)

    def _dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        message_type = message.get("type")
        if message_type == MSG_RESET:
            return self._handle_reset_batched(message) if self.batched else self._handle_reset(message)
        if message_type == MSG_STEP:
            return self._handle_step_batched(message) if self.batched else self._handle_step(message)
        if message_type == MSG_SPEC:
            return self._handle_spec()
        if message_type == MSG_SUCCESS_CHECK:
            return {"done": True}  # single-stage tasks
        if message_type == MSG_CLOSE:
            return None
        return {"error": f"unknown message type {message_type}"}

    def _log_once(self, key: str, message: str) -> None:
        if key not in self._logged_once:
            self._logged_once.add(key)
            print(f"{LOG_PREFIX} {message}", flush=True)

    def _to_numpy(self, value: Any) -> Any:
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else self.np.asarray(value)
        )

    @staticmethod
    def _unbatch(value: Any) -> Any:
        has_batch = (
            hasattr(value, "ndim")
            and value.ndim >= 1
            and getattr(value, "shape", [1])[0] == 1
        )
        return value[0] if has_batch else value

    def _to_rgb8(self, rgb: Any) -> Any:
        """Drop an alpha channel and cast to uint8."""
        np = self.np
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _resize(self, rgb: Any) -> Any:
        """Apply the configured server-side letterbox or shortest-edge shrink."""
        args = self.args
        if args.image_resolution > 0:
            rgb = letterbox_square(rgb, args.image_resolution, self.cv2)
        if args.image_shortest_edge > 0:
            rgb = shrink_shortest_edge(rgb, args.image_shortest_edge, self.cv2)
        return rgb

    # ---- observation extraction -------------------------------------------------
    def _extract_image(self, obs: dict[str, Any]) -> Any:
        cameras = obs["camera_obs"]
        self._log_once("camera_keys", f"camera_obs keys: {list(cameras.keys())}")
        key = self.args.camera_key
        if f"{key}_rgb" in cameras:
            resolved = f"{key}_rgb"
        elif key in cameras:
            resolved = key
        else:
            resolved = "robot_head_cam_rgb"
        rgb = self._to_rgb8(self._to_numpy(self._unbatch(cameras[resolved])))
        self._log_once(
            "camera_shape",
            f"native camera shape={rgb.shape} (image resolution arg={self.args.image_resolution})",
        )
        return self._resize(rgb)

    def _extract_state(self, obs: dict[str, Any]) -> Any:
        """Build the proprioceptive state in the policy's group order.

        The simulator's ``robot_joint_pos`` is in interleaved simulation
        order; the policy expects the grouped order of
        :data:`STATE_GROUP_ORDER`, so joints are re-indexed by name using the
        joint-space configs. Falls back to per-group observation terms, then
        to the raw simulation vector.
        """
        np, wbc = self.np, self.wbc
        policy_group = obs.get("policy", {})
        sim_state = None
        for group in (policy_group, obs.get("wbc", {})):
            if "robot_joint_pos" in group:
                sim_state = (
                    self._to_numpy(self._unbatch(group["robot_joint_pos"]))
                    .reshape(-1)
                    .astype(np.float32)
                )
                break
        if sim_state is not None and wbc.get("sim_cfg") and wbc.get("policy_cfg"):
            sim_cfg, policy_cfg = wbc["sim_cfg"], wbc["policy_cfg"]
            ordered = [
                sim_state[int(sim_cfg[joint_name])]
                for group in STATE_GROUP_ORDER
                for joint_name in policy_cfg[group]
            ]
            return np.asarray(ordered, dtype=np.float32)
        if all(key in policy_group for key in STATE_GROUP_ORDER):
            return np.concatenate(
                [
                    self._to_numpy(self._unbatch(policy_group[key])).reshape(-1)
                    for key in STATE_GROUP_ORDER
                ]
            ).astype(np.float32)
        if sim_state is not None:
            return sim_state
        raise KeyError(
            f"cannot build the state vector; policy keys={list(policy_group.keys())}"
        )

    def _extract_success(self, obs: dict[str, Any]) -> bool:
        """Read the live "success" termination term, with a height fallback."""
        try:
            return bool(
                self.np.any(self._to_numpy(self.env.unwrapped.termination_manager.get_term("success")))
            )
        except Exception:
            pass
        try:
            object_height = float(self._to_numpy(self._unbatch(obs["task_obs"])).reshape(-1)[0])
            return bool(object_height >= self.args.lift_height)
        except Exception:
            return False

    def _observation_payload(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "ego_view": self._extract_image(obs),
            "state": self._extract_state(obs),
            "prompt": self.task_description,
        }

    # ---- action assembly --------------------------------------------------------
    def _to_wbc50(self, actions: Any) -> Any:
        """Assemble [B, 35] policy actions into the [B, 50] env action."""
        batch = actions.shape[0]
        values = actions.detach().cpu().numpy()
        # The Arena remap utility converts from numpy, so pass numpy [B, 1, n].
        named = {
            name: values[:, sl].reshape(batch, 1, sl.stop - sl.start)
            for name, sl in ACTION_SLICES.items()
        }
        wbc = self.wbc
        remapped = wbc["remap"](named, wbc["policy_cfg"], wbc["sim_cfg"], self.device)
        joints = remapped.get_joints_pos()  # [B, 1, 43]
        navigate = actions[:, NAVIGATE_SLICE].reshape(batch, 1, 3)
        base_height = actions[:, BASE_HEIGHT_SLICE].reshape(batch, 1, 1)
        torso_rpy = self.torch.zeros(batch, 1, 3, device=self.device)
        return self.torch.cat([joints, navigate, base_height, torso_rpy], dim=2).reshape(
            batch, -1
        )

    def _step_once(self, action_1d: Any) -> tuple[Any, ...]:
        action = action_1d.unsqueeze(0)
        return self.env.step(self._to_wbc50(action) if self.wbc else action)

    # ---- batched mode (--num-envs > 1) ------------------------------------------
    def _extract_images_batched(self, obs: dict[str, Any]) -> Any:
        cameras = obs["camera_obs"]
        key = self.args.camera_key
        resolved = f"{key}_rgb" if f"{key}_rgb" in cameras else key
        rgb = self._to_numpy(cameras[resolved])  # [N, H, W, C] (tiled camera term)
        return self._resize(self._to_rgb8(rgb))

    def _extract_state_batched(self, obs: dict[str, Any]) -> Any:
        for group in (obs["policy"], obs.get("wbc", {})):
            if "robot_joint_pos" in group:
                sim_state = self._to_numpy(group["robot_joint_pos"]).reshape(self.num_envs, -1)
                return sim_state[:, self.state_reindex].astype(self.np.float32)
        raise KeyError(
            f"batched state needs robot_joint_pos; policy keys={list(obs['policy'].keys())}"
        )

    def _observation_payload_batched(
        self, obs: dict[str, Any], include_images: bool
    ) -> dict[str, Any]:
        payload = {"state": self._extract_state_batched(obs), "prompt": self.task_description}
        if include_images:
            payload["ego_view"] = self._extract_images_batched(obs)
        return payload

    # ---- request handlers -------------------------------------------------------
    def _reset_full(self, seed: Any) -> Any:
        """Reset every env, replaying a stored reset state for seeded evaluation resets."""
        if seed is not None and self.reset_state_store is not None:
            obs, _ = self.reset_state_store.reset(self.env, int(seed))
        else:
            obs, _ = self.env.reset(seed=None if seed is None else int(seed))
        return obs

    def _handle_reset_batched(self, message: dict[str, Any]) -> dict[str, Any]:
        env_ids = message.get("env_ids")
        seed = message.get("seed")
        if env_ids:
            # Partial reset (client-side truncation, rolling per-env restarts);
            # naturally-terminated envs were already auto-reset inside step(),
            # re-resetting them just draws a fresh episode, which is fine.
            ids = self.torch.as_tensor(
                [int(i) for i in env_ids], dtype=self.torch.long, device=self.device
            )
            reset_kwargs = {"env_ids": ids}
            if seed is not None:
                reset_kwargs["seed"] = int(seed)
            obs, _ = self.env.unwrapped.reset(**reset_kwargs)
        else:
            obs = self._reset_full(seed)
        if getattr(self.args, "strict_reproducibility", False):
            configure_strict_torch(self.torch)
        payload = self._observation_payload_batched(obs, include_images=True)
        if not env_ids and seed is not None and self.reset_observation_capture is not None:
            self.reset_observation_capture.record(int(seed), payload)
        payload["info"] = {"supports_action_chunking": False}
        return payload

    def _handle_step_batched(self, message: dict[str, Any]) -> dict[str, Any]:
        np, num_envs = self.np, self.num_envs
        action = np.asarray(message["action"], dtype=np.float32) * self.args.action_scale
        if action.ndim != 2 or action.shape[0] != num_envs:
            raise ValueError(
                f"batched action must be [num_envs={num_envs}, {POLICY_ACTION_DIM}], got {action.shape}"
            )
        action_t = self.torch.as_tensor(action, dtype=self.torch.float32, device=self.device)
        t_start = time.perf_counter()
        obs, reward, terminated, truncated, _ = self.env.step(self._to_wbc50(action_t))
        # The host copies below wait for the GPU, so the sim time measured
        # here covers physics, render and the whole-body controller.
        done = self._to_numpy(terminated).reshape(num_envs).astype(bool)
        trunc = self._to_numpy(truncated).reshape(num_envs).astype(bool)
        # The success termination term is computed before the internal
        # auto-reset, so it is True exactly for the envs that terminated in
        # success this step ("success" presence is validated at boot).
        success = (
            self._to_numpy(self.env.unwrapped.termination_manager.get_term("success"))
            .reshape(num_envs)
            .astype(bool)
        )
        t_sim_done = time.perf_counter()
        # Images dominate the wire payload; skip them on intermediate chunk
        # steps, but always ship them when an episode ended (the collector
        # consumes that step's observation).
        include_images = bool(message.get("need_obs", True)) or bool(
            done.any() or trunc.any()
        )
        payload = self._observation_payload_batched(obs, include_images)
        payload.update(
            reward=self._to_numpy(reward).reshape(num_envs).astype(np.float32),
            done=done,
            truncated=trunc,
            success=success,
            num_steps=np.ones(num_envs, dtype=np.int64),
            info={},
        )
        t_payload_done = time.perf_counter()
        summary = self.step_timer.record(t_sim_done - t_start, t_payload_done - t_sim_done, include_images)
        if summary is not None:
            print(summary, flush=True)
        # Echoed so the client can split its round trip into server work and
        # wire time (serialization + transfer + latency) without a shared clock.
        payload["server_s"] = t_payload_done - t_start
        return payload

    def _handle_reset(self, message: dict[str, Any]) -> dict[str, Any]:
        seed = message.get("seed")
        obs = self._reset_full(seed)
        if getattr(self.args, "strict_reproducibility", False):
            configure_strict_torch(self.torch)
        payload = self._observation_payload(obs)
        if seed is not None and self.reset_observation_capture is not None:
            self.reset_observation_capture.record(int(seed), payload)
        payload["info"] = {"supports_action_chunking": True}
        return payload

    def _handle_step(self, message: dict[str, Any]) -> dict[str, Any]:
        np = self.np
        action = np.asarray(message["action"], dtype=np.float32) * self.args.action_scale
        action_t = self.torch.as_tensor(action, dtype=self.torch.float32, device=self.device)
        if action_t.numel() == 0:
            raise ValueError("action chunk must contain at least one action")
        num_steps = 0
        if action_t.ndim == 2:
            # Action chunk [chunk, dim]: execute sequentially, stop early on
            # episode end.
            result = self._step_once(action_t[0])
            num_steps = 1
            for i in range(1, action_t.shape[0]):
                if bool(result[2].any()) or bool(result[3].any()):
                    break
                result = self._step_once(action_t[i])
                num_steps += 1
            obs, reward, terminated, truncated, _ = result
        else:
            obs, reward, terminated, truncated, _ = self._step_once(action_t)
            num_steps = 1
        payload = self._observation_payload(obs)
        payload.update(
            reward=float(self._to_numpy(reward).reshape(-1)[0]),
            done=bool(np.any(self._to_numpy(terminated))),
            truncated=bool(np.any(self._to_numpy(truncated))),
            success=self._extract_success(obs),
            num_steps=num_steps,
            info={},
        )
        return payload

    def _handle_spec(self) -> dict[str, Any]:
        args, wbc = self.args, self.wbc
        state_dim = None
        if wbc.get("policy_cfg"):
            state_dim = sum(len(wbc["policy_cfg"][group]) for group in STATE_GROUP_ORDER)
        side = args.image_resolution if args.image_resolution > 0 else None  # None: native
        # Sizes this server cannot fix stay named, so a client reports them as unverified.
        emitted = ObservationSpec(
            {
                "ego_view": ArraySpec((side, side, 3), "uint8"),
                "state": ArraySpec(("state_dim" if state_dim is None else state_dim,), "float32"),
            },
            frozenset({"prompt"}),
        )
        accepted = ActionSpec(width=POLICY_ACTION_DIM if wbc else "act_dim", dtype="float32")
        return {
            "protocol": PROTOCOL_VERSION,
            "device": str(self.device),
            "task": args.task,
            "embodiment": args.embodiment,
            "num_envs": self.num_envs,
            "prompt": self.task_description,
            "specs": spec_payload(emitted, accepted),
            "spaces": {
                "observation": {
                    "ego_view": {
                        "dtype": "uint8",
                        "hw": None if side is None else [side, side],
                        # shortest side after the server-side shrink; None: native frames
                        "shortest_edge": args.image_shortest_edge if args.image_shortest_edge > 0 else None,
                    },
                    "state": {"dtype": "float32", "dim": state_dim},
                },
                "action": {"dtype": "float32", "dim": POLICY_ACTION_DIM if wbc else None},  # None: raw env space
            },
        }


def main() -> int:
    """Boot the simulator, bind the transport, and serve until interrupted."""
    args = parse_args()
    if args.image_resolution > 0 and args.image_shortest_edge > 0:
        raise SystemExit(
            "--image-resolution and --image-shortest-edge are exclusive: letterbox to a square "
            "or shrink the shortest edge, not both"
        )

    def _on_sigterm(signum: int, frame: Any) -> None:
        # Raise SystemExit so the simulation context and the socket server
        # unwind through their normal exit paths (a plain `kill` suffices).
        print(f"{LOG_PREFIX} SIGTERM received; shutting down", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    bootstrap_arena_runtime()
    sim_device = resolve_sim_device()

    # The Arena CLI stack parses sys.argv itself (application launcher
    # options first, then the environment subcommand and its options).
    sim_argv = [
        sys.argv[0],
        "--enable_cameras",
        *(["--viz", os.environ["SIM_VIZ"]] if os.environ.get("SIM_VIZ") else []),  # no --viz: headless
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
    ]
    env_spacing = resolve_env_spacing(args.num_envs, args.env_spacing)
    if env_spacing is not None:
        sim_argv += ["--env_spacing", str(env_spacing)]
        print(f"{LOG_PREFIX} env_spacing={env_spacing} m", flush=True)
    if sim_device is not None:
        sim_argv += ["--device", sim_device]
    # The G1 factory LM* tasks are not in the Arena's built-in example-environment
    # registry; register the task class by path so the CLI adds it, then select it
    # positionally by class name.
    sim_argv += [
        "--external_environment_class_path",
        f"{args.task_module}.{args.task}:{args.task}",
    ]
    sim_argv += [args.task, "--embodiment", args.embodiment]
    if str(args.object).lower() not in ("", "none"):
        # Some tasks define their own object and reject --object.
        sim_argv += ["--object", args.object]
    sys.argv = sim_argv

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )

    base_parser = get_isaaclab_arena_cli_parser()
    launcher_args, _ = base_parser.parse_known_args()
    strict_reproducibility = configure_strict_launcher(
        launcher_args, args.strict_reproducibility
    )
    renderer_antialiasing = configure_renderer_antialiasing(
        launcher_args, args.renderer_antialiasing
    )
    if renderer_antialiasing is not None:
        print(
            f"{LOG_PREFIX} renderer_antialiasing={args.renderer_antialiasing} "
            f"(SimulationApp value {renderer_antialiasing})",
            flush=True,
        )

    with SimulationAppContext(launcher_args):
        # Imports below require an initialized simulation app.
        import cv2
        import numpy as np
        import torch

        if strict_reproducibility:
            from isaaclab.app.settings_manager import get_settings_manager

            configure_strict_renderer(get_settings_manager())
        if args.disable_texture_streaming:
            _disable_texture_streaming()

        env_parser = get_isaaclab_arena_environments_cli_parser(base_parser)
        env_args = env_parser.parse_args()
        arena_builder = get_arena_builder_from_cli(env_args)
        if strict_reproducibility:
            install_strict_environment_callback(arena_builder.arena_env)
            configure_strict_torch(torch)
            print(
                f"{LOG_PREFIX} strict reproducibility enabled: deterministic "
                "RTX without tone-map dithering, PhysX enhanced determinism, "
                "and deterministic torch",
                flush=True,
            )
        run_at_50hz(arena_builder.arena_env)
        env, _ = arena_builder.make_registered_and_return_cfg()
        reset_with_index_tensors(env)
        device = env.unwrapped.device
        print(f"{LOG_PREFIX} environment device: {device}", flush=True)

        task_description = resolve_task_prompt(env, args.task, args.task_module, LOG_PREFIX)
        print(f"{LOG_PREFIX} task prompt: {task_description!r}", flush=True)

        render_once_prims = resolve_render_once_prims(
            args.num_envs, args.render_once_prims
        )
        if render_once_prims:
            render_prims_once(env, args.num_envs, render_once_prims)
            # Stage edits reach the renderer's scene on the next Kit updates; pump
            # them now so the first frame a client sees already reflects the edit
            # (headless env.step does not run the app loop except to render).
            import omni.kit.app

            for _ in range(3):
                omni.kit.app.get_app().update()

        wbc = _load_wbc_configs(args)
        if args.num_envs > 1:
            _validate_batched_support(env, wbc, args.num_envs)
        handler, step_timer = build_handler(
            env, args, device, task_description, wbc, np, torch, cv2
        )

        server = make_server(args.transport, args.host, args.port)
        print(f"{LOG_PREFIX} listening on {args.host}:{args.port}", flush=True)
        print(
            f"{LOG_PREFIX} task={args.task} embodiment={args.embodiment} "
            f"num_envs={args.num_envs} action_scale={args.action_scale} "
            f"image_resolution={args.image_resolution} image_shortest_edge={args.image_shortest_edge} "
            f"transport={args.transport}",
            flush=True,
        )
        server.serve_forever(
            handler,
            on_response=lambda timing: step_timer.record_send(timing["encode_s"], timing["send_s"]),
            on_connect=lambda addr: print(
                f"{LOG_PREFIX} client connected: {addr}", flush=True
            ),
            on_disconnect=lambda addr, exc: print(
                f"{LOG_PREFIX} client disconnected: {addr}"
                + (f" (error: {exc})" if exc is not None else ""),
                flush=True,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
