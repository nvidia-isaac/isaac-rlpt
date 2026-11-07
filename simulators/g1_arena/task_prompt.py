# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The language instruction a G1 Factory simulator sends with its observations."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


def resolve_task_prompt(
    env: Any, task: str, task_module: str, log_prefix: str = ""
) -> str:
    """The task's language instruction, sent with every observation.

    The environment's own instruction wins; G1 Factory tasks carry none, so the
    instruction comes from the task's closed-loop policy config, as in G1 Factory's own evaluation.
    """
    get_instruction = getattr(env.unwrapped, "get_language_instruction", None)
    prompt = get_instruction() if get_instruction is not None else None
    if prompt:
        return str(prompt)
    package = importlib.import_module(task_module.split(".")[0])
    config = (
        Path(package.__file__).parent
        / "configs"
        / "gr00t"
        / "policy"
        / f"{task}_closedloop_config.yaml"
    )
    if config.is_file():
        prompt = (yaml.safe_load(config.read_text()) or {}).get("language_instruction")
    if not prompt:
        raise SystemExit(
            f"{log_prefix} no language instruction for {task}: the task has none and {config} gives none"
        )
    return str(prompt)
