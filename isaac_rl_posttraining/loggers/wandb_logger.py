# Copyright (c) Meta Platforms, Inc. and affiliates.
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
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from TorchRL's wandb logger,
# Copyright (c) Meta Platforms, Inc. and affiliates (MIT License).
# Modified; see THIRD_PARTY_NOTICES.md.
#
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Weights & Biases logging backend."""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Any, Optional

from torch import Tensor

from isaac_rl_posttraining.loggers.base import Histogram, Logger

try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False

__all__ = ["WandbLogger", "wandb_run_dir"]

_ARTIFACT_NAME_MAX_LENGTH = 128
#: Where run files go when neither ``logging.wandb_log_dir_base`` nor ``WANDB_DIR`` says.
DEFAULT_WANDB_LOG_DIR_BASE = "logs/wandb"


def wandb_run_dir(base: Optional[str], run_name: str) -> Optional[str]:
    """The directory a run's W&B files go to.

    ``base`` (``logging.wandb_log_dir_base``) wins when set: ``<base>/<run_name>``.
    Otherwise ``WANDB_DIR``, when set, is left to W&B (``None``); otherwise
    ``logs/wandb/<run_name>``.
    """
    if base is not None:
        return os.path.join(str(base), run_name)
    if os.environ.get("WANDB_DIR"):
        return None
    return os.path.join(DEFAULT_WANDB_LOG_DIR_BASE, run_name)


class WandbLogger(Logger):
    """Logger backed by a W&B run.

    Args:
        exp_name: Run name (ignored when resuming by ``id``).
        offline: Run W&B offline (``mode="offline"`` for this run only; the
            process environment is left alone).
        id: Existing run id to attach to (enables resume).
        project: W&B project.
        log_dir: Local directory for W&B files (``None``: W&B's own choice,
            ``WANDB_DIR`` when set); see :func:`wandb_run_dir`.
        prefix: Default metric-name prefix.
        entity: W&B entity.
        group: Run group.
        job_type: Run job type.
        tags: Run tags.
        notes: Run notes.
        **kwargs: Extra ``wandb.init`` arguments.

    Raises:
        ImportError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        id: Optional[str] = None,
        project: Optional[str] = None,
        log_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not _has_wandb:
            raise ImportError("wandb is not installed")

        self.offline = offline
        self.id = id
        self.project = project
        self.entity = entity
        self.group = group
        self.job_type = job_type
        self.tags = tags
        self.notes = notes

        # Attaching to an existing run id implies resuming unless the caller
        # chose a policy explicitly.
        if id is not None and "resume" not in kwargs and not offline:
            kwargs["resume"] = "allow"
        if offline:
            kwargs.setdefault("mode", "offline")
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        base_kwargs: dict[str, Any] = {
            "dir": log_dir,
            "id": id,
            "project": project,
            **({"entity": entity} if entity is not None else {}),
            **kwargs,
        }
        if group is not None:
            base_kwargs["group"] = group
        if job_type is not None:
            base_kwargs["job_type"] = job_type
        if tags is not None:
            base_kwargs["tags"] = tags
        if notes is not None:
            base_kwargs["notes"] = notes
        if id is None:
            base_kwargs["name"] = exp_name
        self._wandb_kwargs = base_kwargs

        super().__init__(exp_name=exp_name, log_dir=log_dir, prefix=prefix)

        self._has_imported_moviepy = False
        self.initialize()

    @staticmethod
    def _native(value: Any) -> Any:
        """Render logger-level media types as their W&B counterparts."""
        if isinstance(value, Histogram):
            return wandb.Histogram(value.values)
        return value

    @property
    def run_id(self) -> Optional[str]:
        """The underlying run id, when available."""
        try:
            return getattr(self.logger, "id", None)
        except Exception:
            return None

    @property
    def run_url(self) -> Optional[str]:
        """The underlying run URL, when available."""
        try:
            return getattr(self.logger, "url", None)
        except Exception:
            return None

    def initialize(self) -> None:
        """Create the W&B run."""
        self.logger = wandb.init(**self._wandb_kwargs)

    def log_scalar(
        self, name: str, value: float, step: Optional[int] = None, prefix: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log one scalar, with the step as a sibling metric."""
        if prefix is None:
            prefix = self.prefix
        if step is not None:
            self.logger.log({f"{prefix}/{name}": value, f"{prefix}/step": step})
        else:
            self.logger.log({f"{self.prefix}/{name}": value})

    def log_video(
        self,
        name: str,
        video: Tensor,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a video tensor of shape ``[N, T, C, H, W]``.

        Args:
            name: Metric name.
            video: Video tensor; the channel dimension must be 1 or 3.
            step: Logged as ``<prefix>/step``, like :meth:`log_dict`.
            prefix: Metric-name prefix override.
            **kwargs: ``fps`` (default 6) and ``format`` (default ``mp4``)
                are consumed; the rest is dropped.

        Raises:
            ValueError: On a malformed video tensor.
        """
        if video.dim() != 5 or video.size(dim=2) not in {1, 3}:
            raise ValueError("Wrong format of the video tensor. Should be ((N), T, C, H, W)")
        if not self._has_imported_moviepy:
            try:
                # wandb.Video encodes through this module, which also needs imageio.
                import moviepy.video.io.ImageSequenceClip  # noqa: F401

                self._has_imported_moviepy = True
            except ImportError as exc:
                # A missing moviepy should not kill a training run — the mp4
                # is still written to disk by the caller; only the upload is
                # skipped.
                warnings.warn(
                    f"moviepy is unusable ({exc}); skipping wandb video upload (eval mp4 still saved to disk)."
                )
                return
        fps = kwargs.pop("fps", 6)
        fmt = kwargs.pop("format", "mp4")
        kwargs.pop("commit", None)
        if prefix is None:
            prefix = self.prefix
        data = {f"{prefix}/{name}": wandb.Video(video, fps=fps, format=fmt)}
        if step is not None:
            data[f"{prefix}/step"] = step
        self.logger.log(data)

    def log_dict(
        self, dictionary: dict, step: Optional[int] = None, prefix: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log a metrics dict under ``prefix``, with the step as a metric.

        The step is logged as ``<prefix>/step`` instead of W&B's global step
        to avoid out-of-order warnings; map it via ``define_metric`` when
        needed.
        """
        if prefix is None:
            prefix = self.prefix
        dictionary = {f"{prefix}/{k}": self._native(v) for k, v in dictionary.items()}
        if step is not None:
            dictionary[f"{prefix}/step"] = step
        self.logger.log(dictionary, **kwargs)

    def log_hparams(self, cfg: dict) -> None:
        """Merge the experiment config into the run config."""
        self.logger.config.update(cfg, allow_val_change=True)

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        type: str = "checkpoint",
        metadata: Optional[dict] = None,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Upload a file or directory as a W&B artifact.

        Args:
            path: File or directory to upload (missing paths are skipped
                with a warning).
            name: Artifact name suffix; combined with the run name.
            type: Artifact category.
            metadata: Metadata to attach.
            aliases: Aliases; defaults to ``["latest"]``.
        """
        if not os.path.exists(path):
            warnings.warn(f"Artifact path does not exist: {path}. Skipping.")
            return

        # <run>-<basename>, plus the suffix when one is given. W&B caps artifact
        # names at 128 characters, so nothing else goes into the name.
        base_name = os.path.basename(path.rstrip("/"))
        artifact_name = f"{self.exp_name}-{base_name}" if name is None else f"{self.exp_name}-{base_name}-{name}"
        if len(artifact_name) > _ARTIFACT_NAME_MAX_LENGTH:
            # Preserve distinct checkpoint suffixes when shortening long run names.
            digest = hashlib.sha256(artifact_name.encode()).hexdigest()[:16]
            prefix_length = _ARTIFACT_NAME_MAX_LENGTH - len(digest) - 1
            artifact_name = f"{artifact_name[:prefix_length]}-{digest}"
        artifact = wandb.Artifact(name=artifact_name, type=type, metadata=metadata or {})
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.logger.log_artifact(artifact, aliases=aliases or ["latest"])

    def finish(self) -> None:
        """Close the W&B run (never raises)."""
        try:
            if hasattr(self, "logger") and hasattr(self.logger, "finish"):
                self.logger.finish()
            else:
                wandb.finish()
        except Exception:
            try:
                wandb.finish()
            except Exception:
                pass
