# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL post-training for robot foundation policies on Isaac simulation environments.

``from isaac_rl_posttraining import launch, train`` is the Python entry point
(:mod:`isaac_rl_posttraining.train`); it is imported on first use, so importing
the package stays light.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

#: Names re-exported lazily from isaac_rl_posttraining.train.
_FACADE = ("Env", "Run", "launch")


def __getattr__(name: str) -> Any:
    if name == "train":
        # The facade module is callable, so `train(...)` works whichever object this import yields.
        return importlib.import_module("isaac_rl_posttraining.train")
    if name in _FACADE:
        return getattr(importlib.import_module("isaac_rl_posttraining.train"), name)
    raise AttributeError(f"module 'isaac_rl_posttraining' has no attribute {name!r}")
