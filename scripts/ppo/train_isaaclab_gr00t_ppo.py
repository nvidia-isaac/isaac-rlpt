#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retired entry: forwards to ``scripts/train.py algorithm=ppo`` for one release.

``scripts/ppo/train_isaaclab_gr00t_ppo.py [overrides]`` is now ``scripts/train.py algorithm=ppo
[overrides]``; ``ppo.num_learner_ranks`` / ``ppo.local_learner_device`` are translated to
``placement.learner.ranks`` / ``learner.device``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.legacy_entry import forward

if __name__ == "__main__":
    forward(
        sys.argv[1:], algorithm="ppo", entry="scripts/ppo/train_isaaclab_gr00t_ppo.py"
    )
