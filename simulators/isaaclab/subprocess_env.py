# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host an Isaac Lab simulation in a fresh child process and drive it over a pipe.

Isaac Sim must start in a process that has not yet imported PyTorch, NumPy's
BLAS or Ray: their fork handlers and thread pools crash Kit's startup and its
USD module imports (Isaac Lab's own launcher defers the torch import for this
reason). A runtime rollout worker is a Ray actor with all three loaded, so it
places the simulation in a spawned child whose interpreter starts Kit first
and imports the environment module afterwards. RLinf's Isaac Lab workers use
the same arrangement. Observations, rewards and flags travel as NumPy arrays
(about 6 MB per step for 16 scenes), which costs a few milliseconds per step.

Kit start-up is not fully reliable when several simulations come up on one
node (native crashes while loading USD modules were observed in roughly one of
five launches), so a start is retried and a dead child can be replaced with
:meth:`SubprocessIsaacLabEnv.restart`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from loguru import logger

from isaac_rl_posttraining.runtimes.base import RolloutInfrastructureError

__all__ = [
    "IsaacLabChildError",
    "IsaacLabEnvSpec",
    "SubprocessIsaacLabEnv",
    "serve_environment",
    "startup_delay_seconds",
]

_STARTUP_TIMEOUT_SECONDS = 1800.0
_RETRY_DELAY_SECONDS = 30.0
_FIRST_FOLLOWER_DELAY_SECONDS = 180.0
_PER_WORKER_DELAY_SECONDS = 20.0


_SPAWN_LOCK = threading.Lock()


@contextmanager
def _without_main_module_replay() -> Iterator[None]:
    """Spawn a child that replays no main module, so it imports Isaac Lab only after Kit.

    ``spawn`` sends the child the path of the launcher's ``__main__``, and the
    child executes that file before it unpickles the target. Isaac Lab's RL
    entry point imports ``isaaclab.envs`` at module level, so that replay loads
    the Isaac Lab packages into a process that has not started Kit. Whatever is
    loaded by then is what ``AppLauncher`` drops from ``sys.modules`` while Kit
    starts and puts back afterwards, so Kit's own import of the same extensions
    leaves a second set of config classes behind: the task config then carries
    an ``ArticulationCfg`` the scene builder does not recognise (``Unknown asset
    config type for robot``). Spawn has nothing to replay when ``__main__`` has
    no ``__file__`` and no ``__spec__``, and the child needs only the pickled
    target, which lives in this module. The lock keeps concurrent worker starts
    from restoring each other's placeholder.
    """
    with _SPAWN_LOCK:
        original = sys.modules.get("__main__")
        sys.modules["__main__"] = types.ModuleType("__main__")
        try:
            yield
        finally:
            if original is None:
                sys.modules.pop("__main__", None)
            else:
                sys.modules["__main__"] = original


def startup_delay_seconds(worker_index: int) -> float:
    """Stagger simulator starts on one node: worker 0 warms the shared asset cache alone."""
    if worker_index == 0:
        return 0.0
    return _FIRST_FOLLOWER_DELAY_SECONDS + _PER_WORKER_DELAY_SECONDS * worker_index


class IsaacLabChildError(RolloutInfrastructureError):
    """The simulation child failed to start, died, or reported an error.

    An infrastructure failure, so a rollout worker may replace the child and
    redo the episode rather than record the outcome.
    """


@dataclass(frozen=True)
class IsaacLabEnvSpec:
    """Constructor arguments of the batched environment built in the child.

    ``startup_delay_seconds`` staggers simulations that share a node (the first
    one warms the asset cache the others read); ``startup_attempts`` bounds the
    Kit start retries.
    """

    task_id: str
    num_envs: int
    image_size: int
    action_dim: int
    seed: int | None
    max_episode_steps: int | None = None
    kit_log_path: str | None = None
    startup_delay_seconds: float = 0.0
    startup_attempts: int = 3

    def __post_init__(self) -> None:
        if self.startup_delay_seconds < 0.0:
            raise ValueError("startup_delay_seconds must be non-negative")
        if isinstance(self.startup_attempts, bool) or not isinstance(self.startup_attempts, int) or self.startup_attempts < 1:
            raise ValueError("startup_attempts must be a positive integer")


def serve_environment(connection: Connection, env: Any) -> None:
    """Answer ``reset`` / ``step`` / ``close`` requests on ``connection`` until closed.

    Every reply is ``("ok", result)``; a failure replies ``("error", traceback)``
    and ends the loop so the parent can raise with the child's traceback.
    """
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("ok", env.reset(seed=payload)))
            elif command == "step":
                connection.send(("ok", env.step(payload)))
            elif command == "close":
                env.close()
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown environment command {command!r}")
    except EOFError:
        return
    # The parent re-raises with this traceback.
    except Exception:  # noqa: BLE001
        connection.send(("error", traceback.format_exc()))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise  # SystemExit / KeyboardInterrupt still end the child after reporting


def _serve(connection: Connection, spec_fields: Mapping[str, Any], kit_log_path: str | None) -> None:
    """Child entry point: start Kit, then build and serve the environment."""
    spec = IsaacLabEnvSpec(**spec_fields)
    try:
        # Kit parses the process arguments; the child's are multiprocessing's.
        sys.argv = sys.argv[:1]
        os.environ.pop("DISPLAY", None)
        from isaaclab.app import AppLauncher

        launcher_args: dict[str, Any] = {"headless": True, "enable_cameras": True}
        if kit_log_path is not None:
            launcher_args["kit_args"] = f"--/log/file={kit_log_path}"
        AppLauncher(launcher_args)
        from simulators.isaaclab.env import APP_LAUNCHED_FLAG, IsaacLabBatchedGr00tEnv, mapping_for_task

        os.environ[APP_LAUNCHED_FLAG] = "1"
        env = IsaacLabBatchedGr00tEnv(
            spec.task_id,
            num_envs=spec.num_envs,
            mapping=mapping_for_task(spec.task_id),
            max_episode_steps=spec.max_episode_steps,
            image_size=spec.image_size,
            action_dim=spec.action_dim,
            seed=spec.seed,
        )
    # The parent re-raises with this traceback.
    except Exception:  # noqa: BLE001
        connection.send(("error", traceback.format_exc()))
        return
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise  # SystemExit / KeyboardInterrupt still end the child after reporting
    connection.send(
        (
            "ready",
            {
                "video_keys": env.video_keys,
                "state_keys": env.state_keys,
                "single_observation_space": env.single_observation_space,
                "single_action_space": env.single_action_space,
            },
        )
    )
    serve_environment(connection, env)


class SubprocessIsaacLabEnv:
    """The batched Isaac Lab environment, served from a spawned child process.

    Presents the ``reset`` / ``step`` / ``close`` surface of
    :class:`~simulators.isaaclab.env.IsaacLabBatchedGr00tEnv` plus the
    ``video_keys`` and ``state_keys`` the policy needs and the per-scene
    ``single_observation_space`` / ``single_action_space`` the launcher sizes
    the learner from. Errors of the child surface as
    :class:`IsaacLabChildError` carrying its traceback.
    """

    def __init__(self, spec: IsaacLabEnvSpec) -> None:
        self.spec = spec
        self.num_envs = spec.num_envs
        self._connection: Connection | None = None
        self._process: mp.process.BaseProcess | None = None
        self._closed = False
        self._launches = 0
        if spec.startup_delay_seconds > 0:
            time.sleep(spec.startup_delay_seconds)
        self._start_with_retries()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _start_with_retries(self) -> None:
        last_error: IsaacLabChildError | None = None
        for attempt in range(1, self.spec.startup_attempts + 1):
            try:
                self._start()
                return
            except IsaacLabChildError as exc:
                last_error = exc
                logger.warning(
                    f"Isaac Lab child start attempt {attempt}/{self.spec.startup_attempts} failed: "
                    f"{str(exc).splitlines()[-1] if str(exc) else exc!r}"
                )
                if attempt < self.spec.startup_attempts:
                    time.sleep(_RETRY_DELAY_SECONDS)
        raise IsaacLabChildError(
            f"Isaac Lab failed to start after {self.spec.startup_attempts} attempts for {self.spec}"
        ) from last_error

    def _start(self) -> None:
        self._launches += 1
        kit_log_path = self.spec.kit_log_path
        if kit_log_path is not None and self._launches > 1:
            kit_log_path = f"{kit_log_path}.launch{self._launches}"
        context = mp.get_context("spawn")
        connection, child = context.Pipe(duplex=True)
        process = context.Process(target=_serve, args=(child, asdict(self.spec), kit_log_path), daemon=True)
        with _without_main_module_replay():
            process.start()
        child.close()
        self._connection, self._process = connection, process
        if not connection.poll(_STARTUP_TIMEOUT_SECONDS):
            self._terminate()
            raise IsaacLabChildError(f"Isaac Lab did not start within {_STARTUP_TIMEOUT_SECONDS:.0f}s")
        try:
            status, payload = connection.recv()
        except EOFError as exc:
            self._terminate()
            raise IsaacLabChildError("the Isaac Lab child died while starting (Kit crash); see its log") from exc
        if status != "ready":
            self._terminate()
            raise IsaacLabChildError(f"Isaac Lab failed to start in the child process:\n{payload}")
        self.video_keys: list[str] = list(payload["video_keys"])
        self.state_keys: list[str] = list(payload["state_keys"])
        self.single_observation_space = payload["single_observation_space"]
        self.single_action_space = payload["single_action_space"]

    def restart(self) -> None:
        """Replace a dead or wedged child with a fresh simulation."""
        if self._closed:
            raise RuntimeError("subprocess Isaac Lab environment is closed")
        self._terminate()
        self._start_with_retries()

    def _terminate(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=30)
            self._process = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None and self._process is not None and self._process.is_alive():
            try:
                self._connection.send(("close", None))
                self._connection.poll(120)
            except (EOFError, OSError):
                pass
        self._terminate()

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------
    def _request(self, command: str, payload: Any = None) -> Any:
        if self._closed:
            raise RuntimeError("subprocess Isaac Lab environment is closed")
        if self._connection is None:
            raise IsaacLabChildError("the Isaac Lab child is not running; call restart()")
        try:
            self._connection.send((command, payload))
            status, result = self._connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise IsaacLabChildError("the Isaac Lab child process died (Kit crash); see its log") from exc
        if status != "ok":
            raise IsaacLabChildError(f"Isaac Lab child failed during {command!r}:\n{result}")
        return result

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every scene; ``seed`` makes the new initial states reproducible."""
        return self._request("reset", seed)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        return self._request("step", np.asarray(actions, dtype=np.float32))
