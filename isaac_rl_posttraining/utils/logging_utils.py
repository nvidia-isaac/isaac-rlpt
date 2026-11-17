# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Console/file logging setup shared by the training entry points."""

from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger as _logger

__all__ = ["setup_loguru_logging", "get_logger"]


def setup_loguru_logging(log_level: str = "INFO", output_dir: str | None = None) -> Any:
    """Configure loguru to log to stderr and, optionally, a run log file.

    Args:
        log_level: Minimum level for both sinks (e.g. ``"INFO"``, ``"DEBUG"``).
        output_dir: If given, also append records to ``<output_dir>/run.log``
            (the directory is created if needed; file-sink failures are
            non-fatal so a read-only filesystem cannot kill a run).

    Returns:
        The configured loguru logger.
    """
    level = str(log_level).upper()
    try:
        _logger.remove()
    except ValueError:
        pass
    _logger.add(sys.stderr, level=level)
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            _logger.add(os.path.join(output_dir, "run.log"), level=level)
        except OSError:
            pass
    return _logger


def get_logger() -> Any:
    """Return the process-wide loguru logger."""
    return _logger
