# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration: path bootstrap, markers, seeding, and mocks.

This module provides the deterministic test scaffolding used across the unit
and integration suites:

* path bootstrap so ``import isaac_rl_posttraining`` works from a raw checkout,
* a collection hook (:func:`pytest_collection_modifyitems`) that skips
  hardware/sim-marked tests (``gpu`` / ``arena`` / ``libero``) with a clear
  reason when their GPU, IsaacLab-Arena checkout, or LIBERO stack is
  unavailable — the markers themselves are registered in ``pyproject.toml``,
* seeding fixtures (an explicit ``seed_everything`` callable plus an autouse
  baseline seed so every test starts from the same RNG state),
* the scripted ``MockEnv`` / ``MockPolicy`` / ``StubBasePolicy`` trio used by
  the rollout-worker and evaluator tests (``UnbatchedMockEnv`` is the one-env
  view a ``GymRolloutWorker`` drives),
* the in-process ``FakeDreamZeroServer`` / ``FakeDreamZeroConnection`` pair
  implementing the DreamZero wire protocol for the base_model/contract tests,
* recording doubles (``RecordingBuffer``, ``ScriptedRuntimeSource``,
  ``ScriptedLearnerGroup``) for the training-loop semantics tests,
* subprocess helpers for the launcher dry-run tests and for booting the
  external IsaacLab-Arena env server in the integration suite.
"""

import importlib.util
import os
import random
import re
import shutil
import subprocess
import sys
import time as _time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest

# Allow `import isaac_rl_posttraining` from a raw checkout, without an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: Import root of the package under test.
PACKAGE_NAME = "isaac_rl_posttraining"

#: Absolute repository root (directory that contains the package and scripts/).
REPO_ROOT = _REPO_ROOT

#: The observation key mapping installed on Arena envs: the SAC actor/critic
#: consume {state, dino_embedding}; language is None (single-task); the frozen
#: base policy consumes the raw ego image.
ARENA_STEERING_KEY_MAPPING = {
    "state": "state",
    "language": None,
    "image": ["ego_view_image"],
}

#: The arena venv's python, relative to an IsaacLab-Arena checkout.
ARENA_VENV_PYTHON = (".venv", "bin", "python")


def _cuda_gpu_available() -> bool:
    """True only if torch is importable and reports a usable CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    # A broken CUDA runtime counts as unavailable.
    except Exception:  # noqa: BLE001
        return False


def _module_available(*names: str) -> bool:
    """True only if every named module resolves an import spec in this venv."""
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _distribution_version_matches(distribution: str, expected: str) -> bool:
    try:
        return version(distribution) == expected
    except PackageNotFoundError:
        return False


def _arena_checkout_available() -> bool:
    """True only if an external IsaacLab-Arena checkout is configured on disk."""
    path = os.environ.get("ISAACLAB_ARENA_PATH")
    return bool(path and os.path.isdir(path))


def pytest_collection_modifyitems(config, items):
    """Skip hardware/sim-marked tests when their dependency or environment is
    absent, so the ``gpu`` / ``arena`` / ``libero`` markers actually gate
    execution instead of merely labelling it.

    Marker expressions (e.g. CI's ``-m "not gpu and not arena"``) deselect
    tests before this runs; anything that survives selection but cannot run
    here is turned into a visible skip with a concrete reason.
    """
    gpu_ok = _cuda_gpu_available()
    arena_ok = _arena_checkout_available()
    libero_ok = _module_available("libero", "robosuite")
    rlinf_ok = _distribution_version_matches("rlinf", "0.3.0")
    # An installed Ray is not a usable cluster: on a workstation without one
    # the RLInf runtime tests block inside native Ray calls, so they run only
    # when a lane asks for them.
    rlinf_cluster_ok = os.environ.get("ISAAC_RL_POSTTRAINING_RLINF_CLUSTER") == "1"
    native_ray_ok = _module_available("ray")
    gates = {
        "gpu": (gpu_ok, "no CUDA-capable GPU available"),
        "arena": (
            arena_ok,
            "IsaacLab-Arena checkout not available (set ISAACLAB_ARENA_PATH)",
        ),
        "libero": (libero_ok, "LIBERO stack not installed (needs libero + robosuite)"),
        "rlinf": (rlinf_ok, "RLInf runtime stack not installed"),
        "rlinf_cluster": (
            rlinf_cluster_ok,
            "starts a real Ray/RLInf runtime (set ISAAC_RL_POSTTRAINING_RLINF_CLUSTER=1 to run it)",
        ),
        "native_ray": (native_ray_ok, "Ray is not installed (the ray extra)"),
    }
    for item in items:
        for marker, (available, reason) in gates.items():
            # get_closest_marker, not `marker in item.keywords`: keywords also carry
            # directory names, and simulators/libero/ shares its name with a marker.
            if item.get_closest_marker(marker) is not None and not available:
                item.add_marker(pytest.mark.skip(reason=f"{marker}: {reason}"))


@pytest.fixture
def seed_everything():
    """Return a callable that seeds random, numpy, and (if present) torch."""

    def _seed(seed: int = 0) -> int:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        return seed

    return _seed


@pytest.fixture(autouse=True, scope="session")
def _hydra_schemas():
    """Register the ConfigStore schemas once: the group yamls list them (base_<name>) in their defaults."""
    from isaac_rl_posttraining.configs import register_configs

    register_configs()


@pytest.fixture(autouse=True)
def _served_simulator_addresses(monkeypatch):
    """Stand in for the launcher: a served simulator's plan refuses to build without its server addresses."""
    for variable, port in (("ARENA_SERVER_URL", 7300), ("ARENA_EVAL_SERVER_URL", 7301)):
        if variable not in os.environ:
            monkeypatch.setenv(variable, f"tcp://127.0.0.1:{port}")


@pytest.fixture(autouse=True)
def _seed_baseline():
    """Start every test from the same RNG state (tests may re-seed as needed)."""
    random.seed(0)
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Scripted environment / policy mocks
# ---------------------------------------------------------------------------


class MockEnv:
    """Scripted stand-in for the (vectorized, batch=1) Arena env the workers drive.

    ``reset() -> (obs, info)`` and ``step(a) -> (obs, reward[1], terminated[1],
    truncated[1], [info])`` with numpy-batched values, matching the interface
    the evaluation and rollout workers consume. Episode outcomes are scripted
    via ``episode_outcomes`` (cycled per reset):

    * ``"success"``: terminates at ``horizon`` with ``info["success"] = True``,
    * ``"fail"``:    terminates at ``horizon`` without success,
    * ``"never"``:   never terminates; ends only via the ``max_episode_steps``
      truncation cap (mirroring the client-side Arena episode cap) or the
      caller's own step cap.

    ``step_rewards`` (optional list, indexed by step-in-episode) overrides the
    default outcome-based reward so reward-aggregation semantics are
    observable. ``hard_step_limit`` turns a caller that ignores
    terminated/truncated into a fast, explicit test failure instead of a hang.
    """

    def __init__(
        self,
        episode_outcomes=("success", "fail"),
        horizon=5,
        max_episode_steps=None,
        state_dim=4,
        image_hw=None,
        dino_dim=None,
        action_dim=35,
        hard_step_limit=None,
        step_rewards=None,
        reset_info=None,
    ):
        self.episode_outcomes = list(episode_outcomes)
        self.horizon = int(horizon)
        self.max_episode_steps = max_episode_steps
        self.state_dim = int(state_dim)
        self.image_hw = image_hw
        self.dino_dim = dino_dim
        self.action_dim = int(action_dim)
        self.hard_step_limit = hard_step_limit
        self.step_rewards = list(step_rewards) if step_rewards is not None else None
        self.reset_info = (
            {"supports_action_chunking": False}
            if reset_info is None
            else dict(reset_info)
        )
        self.episode_idx = -1
        self.t = 0
        self.total_steps = 0
        self.reset_count = 0
        self.actions_received = []
        self.need_obs = True
        self.steering_key_mapping = dict(ARENA_STEERING_KEY_MAPPING)

    def _obs(self):
        obs = {"state": np.zeros((1, self.state_dim), dtype=np.float32)}
        obs["state"][0, 0] = float(
            self.t
        )  # step-coded so observations are distinguishable
        if self.image_hw is not None:
            h, w = self.image_hw
            obs["ego_view_image"] = np.zeros((1, h, w, 3), dtype=np.uint8)
        if self.dino_dim is not None:
            obs["dino_embedding"] = np.zeros((1, self.dino_dim), dtype=np.float32)
        return obs

    @property
    def _outcome(self):
        return self.episode_outcomes[self.episode_idx % len(self.episode_outcomes)]

    def reset(self, **kwargs):
        self.episode_idx += 1
        self.reset_count += 1
        self.t = 0
        return self._obs(), dict(self.reset_info)

    def _episode_end(self):
        """``(terminated, truncated, success)`` after the current step under the scripted outcome."""
        terminated = truncated = success = False
        if self._outcome == "success" and self.t >= self.horizon:
            terminated, success = True, True
        elif self._outcome == "fail" and self.t >= self.horizon:
            terminated = True
        if (
            self.max_episode_steps is not None
            and not terminated
            and self.t >= self.max_episode_steps
        ):
            truncated = True
        return terminated, truncated, success

    def step(self, action):
        self.t += 1
        self.total_steps += 1
        self.actions_received.append(np.array(action, copy=True))
        if self.hard_step_limit is not None and self.t > self.hard_step_limit:
            raise AssertionError(
                f"MockEnv stepped past its hard limit ({self.hard_step_limit}); "
                "the caller is not honoring terminated/truncated."
            )
        terminated, truncated, success = self._episode_end()
        if self.step_rewards is not None:
            reward = float(self.step_rewards[(self.t - 1) % len(self.step_rewards)])
        else:
            reward = -1.0 if (terminated or truncated) and not success else 0.0
        info = {"success": success, "is_success": success}
        return (
            self._obs(),
            np.array([reward], dtype=np.float32),
            np.array([terminated]),
            np.array([truncated]),
            [info],
        )


class UnbatchedMockEnv:
    """The one-environment view of a ``MockEnv`` that a ``GymRolloutWorker`` drives.

    ``MockEnv`` speaks the batch-1 vector API; the rollout worker steps one
    unbatched environment, so this view strips the leading axis both ways.
    """

    def __init__(self, env):
        self.env = env

    def reset(self, *, seed=None):
        del seed
        observation, info = self.env.reset()
        return {key: value[0] for key, value in observation.items()}, info

    def step(self, action):
        observation, reward, terminated, truncated, infos = self.env.step(action)
        return (
            {key: value[0] for key, value in observation.items()},
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            infos[0],
        )

    def close(self):
        # Test double: holds no resources to release.
        pass


class MockPolicy:
    """Minimal actor stub compatible with the rollout policies.

    ``act(obs, deterministic=..., actor_state=...) -> (torch tensor of
    ``action_shape``, actor_state)`` and records the ``deterministic`` flag of
    every call, so tests can pin that steered evaluation samples the latent
    actor stochastically. ``normalize_actions`` is False, matching the DSRL
    latent actor the steering policy wraps.
    """

    def __init__(self, action_shape=(1, 35), constant=0.25):
        import torch  # lazy: keeps conftest importable without torch

        self._torch = torch
        self.action_shape = tuple(action_shape)
        self.constant = float(constant)
        self.training = False
        self.normalize_actions = False
        self.calls = []  # deterministic flag per act() call

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def act(self, obs, deterministic=False, actor_state=None):
        self.calls.append(bool(deterministic))
        return self._torch.full(self.action_shape, self.constant), actor_state


class StubBasePolicy:
    """Stub of the frozen base policy (the steered VLA ``infer`` surface).

    Implements ``infer(observations, noise) -> {"actions": [B, horizon,
    action_dim]}``. The decode is deterministic and position-coded: action row
    ``t`` is ``(t + 1) * ones``, so tests can verify exactly which chunk rows
    reached the env and in what order. Every received latent is recorded in
    ``self.noises`` (None is recorded for native-mode calls).
    """

    def __init__(self, horizon=40, action_dim=35):
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.noises = []

    @property
    def action_horizon(self):
        return self.horizon

    def infer(self, observations, noise=None):
        self.noises.append(None if noise is None else np.array(noise, dtype=np.float32))
        actions = np.tile(
            (np.arange(self.horizon, dtype=np.float32) + 1.0)[:, None],
            (1, self.action_dim),
        )
        return {"actions": actions[None]}  # [1, horizon, action_dim]


class FakeDreamZeroConnection:
    """In-process double of one WebSocket connection to a DreamZero server.

    Implements the client-side surface (``send`` / ``recv`` / ``close``) the
    DreamZero client drives, against a scripted :class:`FakeDreamZeroServer`.
    Mirrors the reference protocol exactly: the server's metadata handshake
    frame is pending from the moment the connection opens, and every
    ``send`` enqueues exactly one response frame. Raw sent frames are kept
    in ``sent_frames`` so tests can pin the wire encoding itself.
    """

    def __init__(self, server):
        self._server = server
        self._pending = [server.handshake_frame()]
        self.sent_frames = []
        self.closed = False

    def send(self, data):
        if self._server.drop_next_sends > 0:
            self._server.drop_next_sends -= 1
            raise ConnectionError(
                "connection dropped (scripted by FakeDreamZeroServer)"
            )
        self.sent_frames.append(data)
        self._pending.append(self._server.handle_frame(data))

    def recv(self):
        if not self._pending:
            raise AssertionError(
                "recv() with no pending frame: the DreamZero protocol is strictly "
                "one request, one response (plus the initial handshake)"
            )
        return self._pending.pop(0)

    def close(self):
        self.closed = True


class FakeDreamZeroServer:
    """Scripted DreamZero policy server (exact reference wire semantics).

    Implements the server side of the protocol documented in
    ``isaac_rl_posttraining/policies/dreamzero/client.py``: a msgpack-numpy
    metadata frame on connect, then endpoint-dispatched request/response —
    ``"infer"`` answered with a binary msgpack action-chunk frame (or a text
    frame carrying a scripted error traceback), ``"reset"`` answered with a
    plain text acknowledgement (or a scripted error). The ``"endpoint"``
    key is popped before the request is recorded, as the reference server
    does.

    Responses are deterministic and call-coded: chunk ``n`` (1-based) is
    ``arange(H * A).reshape(H, A) / 1000 + n``, so tests can distinguish
    successive chunks and recover positions. Failure injection: set
    ``fail_next_infer`` or ``fail_next_reset`` to an error string (served as
    the text frame the real server sends before closing), or
    ``drop_next_sends`` to force connection drops.

    Use ``connect`` as the client's ``open_connection``; every call
    counts a fresh connection with a fresh handshake, exactly like a
    re-dial against the real server.
    """

    def __init__(self, action_horizon=24, action_dim=8, metadata=None):
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        # The reference server's handshake payload (its serving configuration).
        self.metadata = (
            dict(metadata)
            if metadata is not None
            else {
                "image_resolution": (180, 320),
                "needs_wrist_camera": True,
                "n_external_cameras": 2,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }
        )
        self.infer_requests = []  # decoded obs dicts, endpoint already popped
        self.reset_requests = []  # decoded reset dicts, endpoint already popped
        self.connections = 0
        self.fail_next_infer = None
        self.fail_next_reset = None
        self.drop_next_sends = 0

    def _codec(self):
        # Lazy: keeps conftest importable without msgpack installed.
        from isaac_rl_posttraining.policies.dreamzero import client as dz_client

        return dz_client

    def connect(self, server_url=None, connect_timeout=None):
        """Connection factory: one fake connection per (re-)dial."""
        self.connections += 1
        return FakeDreamZeroConnection(self)

    def handshake_frame(self):
        """The metadata frame the server sends when a connection opens."""
        return self._codec().pack_payload(self.metadata)

    def handle_frame(self, data):
        """Dispatch one request frame and produce the response frame."""
        codec = self._codec()
        request = codec.unpack_payload(data)
        endpoint = request.pop("endpoint")  # the reference server pops it too
        if endpoint == "reset":
            self.reset_requests.append(request)
            if self.fail_next_reset is not None:
                error, self.fail_next_reset = self.fail_next_reset, None
                return error
            return "reset successful"  # text frame, as the reference server replies
        if endpoint != "infer":
            raise AssertionError(f"unknown endpoint in request frame: {endpoint!r}")
        self.infer_requests.append(request)
        if self.fail_next_infer is not None:
            error, self.fail_next_infer = self.fail_next_infer, None
            return error  # text frame = server-side traceback
        chunk = np.arange(
            self.action_horizon * self.action_dim, dtype=np.float32
        ).reshape(self.action_horizon, self.action_dim) / 1000.0 + float(
            len(self.infer_requests)
        )
        return codec.pack_payload(chunk)


# ---------------------------------------------------------------------------
# Training-loop doubles
# ---------------------------------------------------------------------------


class RecordingBuffer:
    """Replay-buffer double: records every add() and delegates length to a callable."""

    reward_channel = "env"
    reward_aggregation = "last"

    def __init__(self, len_fn=None):
        self.adds = []
        self._len_fn = len_fn

    def add(self, transition):
        self.adds.append(transition)

    def __len__(self):
        if self._len_fn is not None:
            return int(self._len_fn())
        return len(self.adds)

    def get_statistics(self):
        return {}

    def reward_statistics(self):
        return {}

    def transitions(self):
        return list(self.adds)


def make_transition(
    *,
    episode_id=0,
    step_in_episode=0,
    obs=None,
    next_obs=None,
    action=None,
    action_kind="latent",
    env_action=None,
    reward=0.0,
    rewards=None,
    steps=1,
    done=False,
    truncated=False,
    success=None,
    info=None,
    behavior_source="policy",
    log_prob=None,
    value=None,
    next_value=None,
    features=None,
    channels=None,
):
    """A typed transition with the defaults the doubles need: a flat state observation and an ``env`` reward.

    ``reward`` lands on the last executed step (``rewards`` gives every step);
    ``success`` defaults to the ``info`` flag of an ending step, as the workers
    read it; ``action`` is the learner action, ``env_action`` the executed
    commands (a zero row per step when neither is given).
    """
    from isaac_rl_posttraining.data import (
        DynamicObservation,
        EnvAction,
        LearnerAction,
        Transition,
        TransitionMetadata,
    )

    def observation(value):
        if value is None:
            value = np.zeros(1, dtype=np.float32)
        if not isinstance(value, dict):
            value = {"state": np.asarray(value)}
        return DynamicObservation.from_mapping(value)

    if rewards is None:
        rewards = np.zeros(steps, dtype=np.float32)
        rewards[-1] = reward
    rewards = np.asarray(rewards, dtype=np.float32)
    steps = len(rewards)
    info = {} if info is None else dict(info)
    if success is None:
        success = bool(done or truncated) and bool(
            info.get("is_success", info.get("success", False))
        )

    def flags(flag):
        return np.asarray([False] * (steps - 1) + [bool(flag)])

    learner_action = (
        None
        if action is None
        else LearnerAction(action_kind, np.asarray(action, dtype=np.float32))
    )
    if env_action is None and learner_action is None:
        env_action = np.zeros((steps, 1), dtype=np.float32)
    executed = None
    if env_action is not None:
        commands = np.asarray(env_action, dtype=np.float32)
        executed = EnvAction(commands=commands, execution_length=int(commands.shape[0]))
    return Transition(
        episode_id=episode_id,
        observation=observation(obs),
        next_observation=observation(next_obs),
        learner_action=learner_action,
        env_action=executed,
        rewards={"env": rewards, **(channels or {})},
        terminated=flags(done),
        truncated=flags(truncated),
        success=flags(success),
        behavior_source=behavior_source,
        metadata=TransitionMetadata(step_in_episode=step_in_episode, extra=info),
        policy_outputs=_policy_outputs(log_prob, value, next_value, features),
    )


def _policy_outputs(log_prob, value, next_value, features):
    """The behaviour policy's recorded outputs, keeping only the ones given."""
    from isaac_rl_posttraining.data import PolicyOutputs

    outputs = {}
    for name, entry in (
        ("log_prob", log_prob),
        ("value", value),
        ("next_value", next_value),
    ):
        if entry is not None:
            outputs[name] = np.asarray(entry, dtype=np.float32)
    if features is not None:
        outputs["features"] = features
    return PolicyOutputs(**outputs)


# ---------------------------------------------------------------------------
# Launcher dry-run helpers (unit) and Arena server helpers (integration)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
# The DSRL launchers are self-contained in the g1_arena simulator.
_G1_ARENA_SCRIPTS_DIR = os.path.join(
    REPO_ROOT, "example_workflows", "g1_arena_posttraining", "scripts"
)
TRAIN_LAUNCHER = os.path.join(_G1_ARENA_SCRIPTS_DIR, "train.sh")
EVAL_LAUNCHER = os.path.join(_G1_ARENA_SCRIPTS_DIR, "eval.sh")
ARENA_SERVER_SCRIPT = os.path.join(REPO_ROOT, "simulators", "g1_arena", "env_server.py")

#: Environment variables that must never leak from the developer's shell into
#: a launcher dry-run (the tests control them explicitly).
_LAUNCHER_CLEARED_VARS = (
    "BASE_MODEL",
    "GR00T_MULTITASK_DIRS",
    "GR00T_TASK_DIR",
    "SIMULATOR",
    "TRAIN_SIM_DEVICE",
    "EVAL_SIM_DEVICE",
    "ARENA_SIM_DEVICE",
    "DSRL_EVAL_BASE",
    "TRAIN_MAXSTEPS",
    "EVAL_MAXSTEPS",
    "NUM_ROLLOUTS",
    "LEARNER_PYTHON",
    "GPU",
    "SIM_STARTUP_LOCK",
)

#: The 9 G1 Factory benchmark tasks and their launcher episode caps
#: (demo max + margin; the plan's per-task max-steps map).
TASK_MAX_STEPS = {
    "LMPushButton": 350,
    "LMBoxLift": 450,
    "LMDrillLift": 450,
    "LMPickDrillFromHolder": 500,
    "LMDrillLiftObs": 650,
    "LMDrillPnP": 650,
    "LMBoxLiftFloor": 750,
    "LMBoxTableToShelfPnP": 900,
    "LMPushShelfForward": 1000,
}


@pytest.fixture
def launcher_sandbox(tmp_path):
    """A sandbox env for ISAAC_RL_POSTTRAINING_DRYRUN launcher runs (fake arena/ckpt/data trees)."""
    arena = tmp_path / "arena"
    sim_py = arena.joinpath(*ARENA_VENV_PYTHON)
    sim_py.parent.mkdir(parents=True)
    sim_py.write_text("#!/bin/sh\nexit 0\n")
    sim_py.chmod(0o755)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    data = tmp_path / "data"
    for task in TASK_MAX_STEPS:
        (data / task / "lerobot").mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    return {
        "tmp": tmp_path,
        "arena": str(arena),
        "ckpt": str(ckpt),
        "data": str(data),
        "logs": str(logs),
    }


def run_launcher_dryrun(script_path, args, sandbox, extra_env=None, timeout=90):
    """Run a launcher with ISAAC_RL_POSTTRAINING_DRYRUN=1 and return the CompletedProcess.

    The dry-run contract: the launcher prints its resolved plan (task,
    max-steps, devices, hydra overrides) and exits 0 without starting any
    simulator or learner process.
    """
    assert os.path.isfile(script_path), f"launcher not found at {script_path}"
    env = os.environ.copy()
    for var in _LAUNCHER_CLEARED_VARS:
        env.pop(var, None)
    env.update(
        {
            "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
            "ISAACLAB_ARENA_PATH": sandbox["arena"],
            "POSTTRAIN_BASE_CKPT": sandbox["ckpt"],
            "POSTTRAIN_DATA_ROOT": sandbox["data"],
            "LOG_DIR": sandbox["logs"],
            "HF_TOKEN": "",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script_path)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(sandbox["tmp"]),
        check=False,
    )


def dryrun_text(result):
    """stdout+stderr of a dry-run, for tolerant plan parsing."""
    return (result.stdout or "") + "\n" + (result.stderr or "")


# --- Arena integration helpers (stdlib only; used by tests/integration) ---


def arena_checkout():
    """Path of the external IsaacLab-Arena checkout, or None."""
    path = os.environ.get("ISAACLAB_ARENA_PATH")
    if path and os.path.isdir(path):
        return path
    return None


def arena_sim_python(arena_path):
    """The arena venv's python (Isaac must run under the ARENA venv, not ours)."""
    override = os.environ.get("SIM_PYTHON")
    if override:
        return override
    return os.path.join(arena_path, *ARENA_VENV_PYTHON)


def gpu_is_free(max_used_mb=2000):
    """True if at least one visible GPU is essentially idle (do not disturb running jobs)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    # A failing nvidia-smi means no idle GPU.
    except Exception:  # noqa: BLE001
        return False
    if out.returncode != 0:
        return False
    for line in out.stdout.strip().splitlines():
        try:
            if int(line.strip()) < max_used_mb:
                return True
        except ValueError:
            continue
    return False


def require_arena_prereqs():
    """Skip unless the external arena checkout, server script, and an idle GPU exist."""
    arena = arena_checkout()
    if arena is None:
        pytest.skip(
            "ISAACLAB_ARENA_PATH not set (external IsaacLab-Arena checkout required)"
        )
    if not os.path.isfile(ARENA_SERVER_SCRIPT):
        pytest.skip(f"arena env server script not found at {ARENA_SERVER_SCRIPT}")
    sim_py = arena_sim_python(arena)
    if not os.path.isfile(sim_py):
        pytest.skip(f"arena venv python not found at {sim_py}")
    if not gpu_is_free():
        pytest.skip("no idle GPU available (refusing to disturb running jobs)")
    return arena


def boot_arena_server(
    arena_path,
    port,
    log_path,
    task="LMDrillLift",
    sim_device=None,
    image_resolution=0,
    boot_timeout=None,
    extra_args=(),
):
    """Boot the Arena env server under the external arena venv, through a PTY.

    Isaac Sim dies silently without a TTY, so the server runs under
    ``script -qec`` (the known-good PTY shim). EULA acceptance env vars are
    exported, and readiness is detected by the ``listening on`` line; boots
    routinely take 1-3 minutes, so the default timeout is generous.

    Returns the Popen handle. Raises AssertionError with the log tail if the
    server errors out or never becomes ready.
    """
    if boot_timeout is None:
        boot_timeout = int(
            os.environ.get("ISAAC_RL_POSTTRAINING_ARENA_BOOT_TIMEOUT", "600")
        )
    env = os.environ.copy()
    env.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "ISAACLAB_ARENA_PATH": arena_path,
            "ARENA_DISABLE_TEXTURE_STREAMING": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if sim_device is not None:
        env["ARENA_SIM_DEVICE"] = str(sim_device)
    pythonpath = [REPO_ROOT]
    if env.get("GR00T_REPO"):
        pythonpath.append(env["GR00T_REPO"])
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    sim_py = arena_sim_python(arena_path)
    inner = (
        f"cd {arena_path} && "
        f"{sim_py} {ARENA_SERVER_SCRIPT} --task {task} --embodiment g1_wbc_joint "
        f"--port {port} --action-scale 1.0 --image-resolution {image_resolution} "
        + " ".join(str(a) for a in extra_args)
    )
    # Handle outlives this scope.
    log_file = open(log_path, "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        ["script", "-qec", inner, "/dev/null"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    try:
        wait_for_log_line(
            log_path,
            pattern=r"listening on",
            timeout=boot_timeout,
            fail_patterns=(
                r"Traceback \(most recent call last\)",
                r"ModuleNotFoundError",
            ),
            proc=proc,
        )
    except Exception:
        stop_process(proc)
        raise
    return proc


def wait_for_log_line(log_path, pattern, timeout, fail_patterns=(), proc=None):
    """Poll a log file until ``pattern`` appears; raise on error patterns/exit/timeout."""
    deadline = _time.time() + timeout
    rx = re.compile(pattern)
    fail_rx = [re.compile(p) for p in fail_patterns]
    while _time.time() < deadline:
        text = ""
        if os.path.exists(log_path):
            with open(log_path, "rb") as fh:
                text = fh.read().decode("utf-8", errors="replace")
        if rx.search(text):
            return text
        for frx in fail_rx:
            if frx.search(text):
                raise AssertionError(
                    f"server log matched failure pattern {frx.pattern!r}; tail:\n{text[-4000:]}"
                )
        if proc is not None and proc.poll() is not None:
            raise AssertionError(
                f"server exited early (rc={proc.returncode}); log tail:\n{text[-4000:]}"
            )
        _time.sleep(5)
    tail = ""
    if os.path.exists(log_path):
        with open(log_path, "rb") as fh:
            tail = fh.read().decode("utf-8", errors="replace")[-4000:]
    raise AssertionError(
        f"timed out ({timeout}s) waiting for {pattern!r}; log tail:\n{tail}"
    )


def stop_process(proc, term_timeout=90):
    """SIGTERM then (if needed) SIGKILL a spawned server process tree."""
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    deadline = _time.time() + term_timeout
    while _time.time() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        _time.sleep(2)
    proc.kill()
    proc.wait(timeout=30)
    return proc.returncode


def single_learner_batch():
    """The one-row learner batch the loop doubles draw per update."""
    from isaac_rl_posttraining.algorithms.replay_learner import pack_batches

    return pack_batches(
        [
            {
                "replay": {
                    "obs": {"state": np.zeros((1, 1), dtype=np.float32)},
                    "action": np.zeros((1, 1), dtype=np.float32),
                    "reward": np.zeros(1, dtype=np.float32),
                    "next_obs": {"state": np.zeros((1, 1), dtype=np.float32)},
                    "done": np.zeros(1, dtype=np.float32),
                    "truncated": np.zeros(1, dtype=np.float32),
                    "bootstrap_steps": np.ones(1, dtype=np.int64),
                }
            }
        ]
    )


class _SubmittedRound:
    """The scripted source's asynchronous call: the entry is consumed when ``finish`` ingests it, and ends at once."""

    def __init__(self):
        self._finished_at = _time.time()

    def result(self):
        return ()

    def finished_at(self):
        return self._finished_at


class ScriptedRuntimeSource:
    """``DataSource`` double driven by a script of collection iterations.

    Each ``run()`` (or ``start()`` + ``finish()`` pair under the overlapped
    schedule) consumes one ``{"steps", "episodes"[, "decisions"]}`` entry and
    reports cumulative ``total_steps``; ``set_weights`` counts actor refreshes.
    ``events`` records every call in order (``("run", can_train)``,
    ``("start", can_train)``, ``("finish",)``, ``("publish", version)``) so a
    test can check what the trainer did between them; ``learner_rows`` holds
    the ``log_learner`` calls.
    """

    max_policy_lag = None

    def __init__(self, script, action_exec_len):
        self.script = list(script)
        self.script_pos = 0
        self.cum_steps = 0
        self.total_episodes = 0
        self.action_exec_len = int(action_exec_len)
        self.num_envs = 1
        self.decisions_collected_last_run = 0
        self.last_transitions = ()
        self.can_train_flags = []
        self.actor_updates = 0
        self.published_versions = []
        self.events = []
        self._policy_version = 0
        self._in_flight = None
        self.dropped_stale_last_run = 0
        self.dropped_stale_total = 0
        self.learner_rows = []

    def run(self, can_train, *, learner_version=None):
        self.events.append(("run", bool(can_train)))
        return self._consume(can_train)

    def start(self, can_train):
        from isaac_rl_posttraining.training import PendingCollection

        if self._in_flight is not None:
            raise AssertionError("the trainer started a round while one was in flight")
        self.events.append(("start", bool(can_train)))
        started = _time.time()
        self._in_flight = PendingCollection(
            call=_SubmittedRound(),
            policy_version=self._policy_version,
            mode="train" if can_train else "warmup",
            started=started,
        )
        return self._in_flight

    def finish(self, pending, *, learner_version=None):
        if pending is not self._in_flight:
            raise AssertionError("the trainer finished a round it did not start")
        self._in_flight = None
        self.events.append(("finish",))
        return self._consume(pending.mode == "train")

    def _consume(self, can_train):
        if self.script_pos >= len(self.script):
            raise AssertionError(
                "scripted source exhausted: the trainer requested more rollout "
                "iterations than the test scripted (step accounting is off)"
            )
        self.can_train_flags.append(bool(can_train))
        entry = self.script[self.script_pos]
        self.script_pos += 1
        self.cum_steps += int(entry["steps"])
        self.total_episodes += int(entry["episodes"])
        self.decisions_collected_last_run = int(entry.get("decisions", 1))
        return {
            "total_steps": self.cum_steps,
            "num_episodes": int(entry["episodes"]),
            "ep_avg_reward": 0.0,
            "success_rate": 0.0,
        }

    def set_weights(self, weights):
        self.publish(weights.resolve())

    def publish(self, state):
        self.actor_updates += 1
        self.published_versions.append(state.version)
        self.events.append(("publish", state.version))
        self._policy_version = state.version

    def log_learner(self, step, env_steps, metrics, *, publish_version=None):
        self.learner_rows.append((step, env_steps, dict(metrics), publish_version))


class _ScriptedHookDispatcher:
    def __init__(self, group):
        self._group = group

    def query(self, name, *args):
        return self._group._hook(name, *args)

    def notify(self, name, *args):
        return self._group._hook(name, *args)


class ScriptedLearnerGroup:
    """Learner-group double: counts updates, publishes a fixed policy, saves via torch.save.

    ``save_checkpoint`` writes ``algo_state.pt`` through ``torch.save`` on
    purpose, so tests can monkeypatch ``torch.save`` to simulate a filesystem
    outage and exercise the checkpoint store's resilience.
    """

    num_learner_ranks = 1

    def __init__(self, step_at_call=None, policy_module=None):
        import torch

        self._torch = torch
        self.train_step_calls = []  # value of step_at_call() at each train_step
        self.save_dirs = []  # every attempted checkpoint dir (including failed ones), without the staging suffix
        self._step_at_call = step_at_call
        self._policy_module = (
            policy_module if policy_module is not None else torch.nn.Linear(1, 1)
        )
        self.policy_versions = []
        self.observed_reward_streams = []
        self.target_update_count_calls = []
        self.evaluations = []

    def train_step(self, batches):
        from isaac_rl_posttraining.runtimes.learner import LearnerStepResult

        at = self._step_at_call() if self._step_at_call is not None else None
        self.train_step_calls.append(at)
        return (
            LearnerStepResult(
                learner_rank=0,
                step=len(self.train_step_calls),
                samples=batches[0].samples,
                metrics={
                    "critic_loss": 0.0,
                    "actor_loss": 0.0,
                    "ent_coef": 0.1,
                    "q_values_mean": 0.0,
                },
            ),
        )

    def policy_state(self, *, version):
        from isaac_rl_posttraining.runtimes.policy_snapshot import (
            policy_state_from_tensors,
        )

        self.policy_versions.append(version)
        return policy_state_from_tensors(
            self._policy_module.state_dict(), version=version
        )

    def save_checkpoint(self, directory):
        self.save_dirs.append(directory.removesuffix(".tmp"))
        os.makedirs(directory, exist_ok=True)
        self._torch.save(
            {"train_steps": len(self.train_step_calls)},
            os.path.join(directory, "algo_state.pt"),
        )

    def load_checkpoint(self, directory):
        raise NotImplementedError("the scripted learner group never resumes")

    def hooks(self):
        from isaac_rl_posttraining.runtimes.learner import GroupLearnerHooks

        return GroupLearnerHooks(_ScriptedHookDispatcher(self))

    def _hook(self, method, *args):
        if method == "gradient_steps":
            return len(self.train_step_calls)
        if method == "target_update_count":
            self.target_update_count_calls.append(args[0])
            return None
        if method == "observe_rewards":
            self.observed_reward_streams.append(args)
            return None
        if method == "on_evaluation":
            self.evaluations.append(args[0])
            return {}
        raise AttributeError(method)

    def close(self):
        # Test double: holds no resources to release.
        pass


def make_recording_runtime_evaluator(record, source, metrics=None):
    """Evaluator double for OffPolicyTrainer tests: records the source's step count per call."""
    out_metrics = {
        "avg_reward": 0.0,
        "std_reward": 0.0,
        "success_rate": 1.0,
        "avg_steps": 1.0,
    }
    if metrics:
        out_metrics.update(metrics)

    class _RecordingEvaluator:
        def __init__(self):
            self.policy_versions = []

        def evaluate(self, policy_state, *, env_steps=0):
            self.policy_versions.append(policy_state.version)
            record.append(source.cum_steps)
            return dict(out_metrics)

    return _RecordingEvaluator()
