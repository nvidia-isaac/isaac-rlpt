# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two launcher helpers this module uses, on both Isaac Lab layouts.

Isaac Lab 3.0.0-beta2 provides them in the ``common`` module of its reinforcement-learning scripts.
Isaac Lab release/3.0.0 removed that file: the launcher arguments live in ``isaaclab.app`` and ``--headless`` is gone (headless is the default).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["add_isaaclab_launcher_args", "import_local_module"]

try:
    from common import add_isaaclab_launcher_args, import_local_module
except ImportError:
    from isaaclab.app import add_launcher_args

    def add_isaaclab_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Add Isaac Lab's launcher arguments, and accept ``--headless`` as a no-op."""
        add_launcher_args(parser)
        if "--headless" not in parser._option_string_actions:
            parser.add_argument(
                "--headless", action="store_true", help=argparse.SUPPRESS
            )

    def import_local_module(module_name: str, module_path: Path) -> ModuleType:
        """Import a module from an explicit file path."""
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load module {module_name!r} from {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
