# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report, then end, an env-server request that never returns.

Seen on the final G1 Factory: rarely, ``omni.kit.app.update()`` never returns inside
omni.syntheticdata's on-demand post-process graph evaluation while a camera observation is
rendered, with the GPU idle. The server then neither answers nor fails, and its client waits.
:class:`StallDump` dumps where the server is stuck and exits it, so clients fail fast.
"""

from __future__ import annotations

import faulthandler
import os
import shutil
import subprocess
import sys
import threading
from typing import Self

__all__ = ["STALL_EXIT_CODE", "StallDump"]

STALL_EXIT_CODE = 70
"""Exit status of a server ended by :class:`StallDump`."""


class StallDump:
    """Context manager around one request: dump stacks after ``seconds``, exit after ``exit_seconds``.

    faulthandler's dump runs on a C thread, so it fires even when the stuck call holds the GIL;
    the snapshot and the exit run on Python timer threads (the observed hang releases the GIL).
    A value of 0 disables that stage.
    """

    def __init__(
        self, seconds: float, exit_seconds: float = 0.0, log_prefix: str = "[stall]"
    ) -> None:
        self.seconds = seconds
        self.exit_seconds = exit_seconds
        self.log_prefix = log_prefix
        self._timers: list[threading.Timer] = []

    @classmethod
    def from_environment(cls, log_prefix: str) -> StallDump:
        """Thresholds from ``ARENA_STALL_DUMP_S`` (default 90 s) and ``ARENA_STALL_EXIT_S`` (default 300 s)."""
        return cls(
            float(os.environ.get("ARENA_STALL_DUMP_S", "90")),
            float(os.environ.get("ARENA_STALL_EXIT_S", "300")),
            log_prefix,
        )

    def __enter__(self) -> Self:
        if self.seconds > 0:
            faulthandler.dump_traceback_later(self.seconds, exit=False, file=sys.stderr)
            self._timers.append(threading.Timer(self.seconds + 1, self._snapshot))
        if self.exit_seconds > 0:
            self._timers.append(threading.Timer(self.exit_seconds, self._exit))
        for timer in self._timers:
            timer.daemon = True
            timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    def _snapshot(self) -> None:
        print(
            f"{self.log_prefix} STALL: request still running after {self.seconds:.0f} s",
            flush=True,
        )
        commands = [
            ["nvidia-smi"],
            ["free", "-g"],
            ["cat", "/sys/fs/cgroup/memory.events"],
        ]
        if shutil.which("py-spy"):
            commands.insert(
                0, ["py-spy", "dump", "--native", "--pid", str(os.getpid())]
            )
        for command in commands:
            try:
                out = subprocess.run(
                    command, capture_output=True, text=True, timeout=60, check=False
                )
            except (OSError, subprocess.SubprocessError) as exc:  # diagnostics only
                print(
                    f"{self.log_prefix} STALL $ {' '.join(command)} failed: {exc}",
                    flush=True,
                )
                continue
            print(
                f"{self.log_prefix} STALL $ {' '.join(command)}\n{out.stdout}{out.stderr}",
                flush=True,
            )

    def _exit(self) -> None:
        print(
            f"{self.log_prefix} STALL: request still running after {self.exit_seconds:.0f} s; "
            f"exiting with code {STALL_EXIT_CODE} so clients fail fast",
            flush=True,
        )
        sys.stderr.flush()
        os._exit(STALL_EXIT_CODE)
