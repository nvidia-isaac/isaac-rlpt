# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment logging backends."""

from isaac_rl_posttraining.loggers.base import Logger

__all__ = ["Logger", "WandbLogger"]


def __getattr__(name: str):
    """Lazily import :class:`WandbLogger` so ``wandb`` stays optional."""
    if name == "WandbLogger":
        from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger

        return WandbLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
