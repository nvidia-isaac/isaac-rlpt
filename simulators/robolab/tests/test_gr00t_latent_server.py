# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for the lightweight GR00T latent server entry point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _server_module():
    path = Path(__file__).parents[1] / "scripts" / "gr00t_latent_server.py"
    spec = importlib.util.spec_from_file_location("test_gr00t_latent_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_size_accepts_wrapped_gr00t_sim_observation():
    module = _server_module()
    observation = {
        "video": {"exterior_image_1_left": np.zeros((2, 1, 4, 4, 3), dtype=np.uint8)},
        "state": {"joint_position": torch.zeros((2, 1, 7))},
        "language": {
            "language_instruction": [["pick up the banana"], ["pick up the banana"]]
        },
    }

    assert module._batch_size(observation) == 2
