# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment-logger interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

__all__ = ["Histogram", "Logger"]


@dataclass(frozen=True)
class Histogram:
    """A distribution of values to log; each backend renders it natively."""

    values: np.ndarray


class Logger(abc.ABC):
    """Interface for metric/video/artifact logging backends.

    Args:
        exp_name: Experiment/run name.
        log_dir: Directory the backend writes into (also used by callers to
            place associated files such as evaluation videos).
        prefix: Default metric-name prefix when a call does not supply one.
    """

    def __init__(self, exp_name: str, log_dir: Optional[str], prefix: Optional[str] = None) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.prefix = prefix

    def set_prefix(self, prefix: str) -> None:
        """Change the default metric-name prefix."""
        self.prefix = prefix

    @abc.abstractmethod
    def initialize(self) -> None:
        """Set up the backend (create the run, open files, ...)."""

    @abc.abstractmethod
    def log_scalar(
        self, name: str, value: float, step: Optional[int] = None, prefix: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log a single scalar."""

    @abc.abstractmethod
    def log_video(
        self,
        name: str,
        video: torch.Tensor,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a video tensor of shape ``[N, T, C, H, W]``."""

    @abc.abstractmethod
    def log_hparams(self, cfg: dict) -> None:
        """Log the experiment configuration."""

    @abc.abstractmethod
    def log_dict(
        self, dictionary: dict, step: Optional[int] = None, prefix: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log a dict of metrics (scalars or :class:`Histogram` values)."""

    @abc.abstractmethod
    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        type: str = "checkpoint",
        metadata: Optional[dict] = None,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Log a file or directory as an artifact.

        Args:
            path: File or directory to log.
            name: Artifact name (derived when omitted).
            type: Artifact category, e.g. ``"checkpoint"``.
            metadata: Metadata to attach.
            aliases: Aliases such as ``["latest"]``.
        """

    def log(self, dictionary: dict, step: Optional[int] = None, prefix: Optional[str] = None, **kwargs: Any) -> None:
        """Convenience alias for :meth:`log_dict`."""
        self.log_dict(dictionary, step, prefix, **kwargs)

    @abc.abstractmethod
    def finish(self) -> None:
        """Finalize/close the run."""
