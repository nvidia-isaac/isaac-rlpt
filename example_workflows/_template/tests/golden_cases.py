# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``).

Copy this file with the workflow: list each recipe your launchers run (``launcher_overrides`` reads
a launcher's dry run), then pin them with
``uv run python -m example_workflows._common.golden <workflow> --write``.
"""

from __future__ import annotations

from pathlib import Path

#: The template's one recipe: the smoke check's dry run on the mock base policy.
RECIPES: dict[str, list[str]] = {
    "dry_run": [
        "algorithm=dsrl_sac",
        "base_model=mock",
        "+simulator.entry=simulators._template.runtime:MyRobot",
        "env.env_name=my_robot/reach",
        "env.max_episode_steps=50",
        "base_model.action_dim=7",
        "base_model.action_horizon=16",
        "dsrl.noise_dim=7",
        "training.chunk_size=16",
    ],
}


def cases(work: Path) -> dict[str, list[str]]:
    del work
    return {name: list(overrides) for name, overrides in RECIPES.items()}
