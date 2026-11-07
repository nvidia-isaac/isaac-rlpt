# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side Ray setup shared by the runtimes that start or join Ray from this process."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Iterator

__all__ = ["keep_driver_interpreter", "learner_gpu_last", "visible_devices"]


def visible_devices() -> tuple[str, ...] | None:
    """The driver's ``CUDA_VISIBLE_DEVICES`` entries, or None when it sees every GPU."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or not value.strip():
        return None
    return tuple(part.strip() for part in value.split(","))


def keep_driver_interpreter() -> None:
    """Run every actor on this interpreter, not on a fresh ``uv run`` environment.

    Under ``uv run`` Ray rebuilds the project environment from the lock for its workers by default,
    which drops packages installed beside the lock (GR00T, simulators, Ray itself). An explicit
    ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` in the environment is left as the user set it.
    """
    flag = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    if "RAY_ENABLE_UV_RUN_RUNTIME_ENV" in os.environ:
        return
    constants = importlib.import_module("ray._private.ray_constants")
    if hasattr(constants, flag):
        setattr(constants, flag, False)


@contextlib.contextmanager
def learner_gpu_last() -> Iterator[None]:
    """Start Ray with this process's GPU 0, where the controller's learner trains, listed last.

    Ray hands out a node's GPUs in the order it was started with, so rollout workers take the
    other GPUs first and share the learner's only when the rest are taken.
    """
    devices = visible_devices()
    if devices is None:
        import torch

        devices = tuple(str(index) for index in range(torch.cuda.device_count()))
    if len(devices) < 2:
        yield
        return
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[1:] + devices[:1])
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
