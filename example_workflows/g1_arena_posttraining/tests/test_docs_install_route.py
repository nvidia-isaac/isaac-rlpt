# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The G1 Factory page's install steps follow the validated route: one workflow environment, built by fetch_externals.sh."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent
PAGE = WORKFLOW.parent.parent / "docs" / "pages" / "workflows" / "g1_arena.md"


def install_commands() -> list[str]:
    """The command lines in the code blocks of the page's Installation section."""
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("\n## Installation", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, re.DOTALL)
    return [line.strip() for block in blocks for line in block.splitlines()]


def test_install_builds_the_workflow_environment_with_fetch_externals() -> None:
    commands = install_commands()
    assert (
        "( cd example_workflows/g1_arena_posttraining && bash ./scripts/fetch_externals.sh && uv sync )"
        in commands
    )
    assert not [c for c in commands if "ISAACLAB_ARENA_PATH" in c and "uv sync" in c]
    assert not [
        c for c in commands if "GR00T" in c and re.search(r"\bpip install\b", c)
    ]


def test_the_page_states_the_locked_isaac_sim_version() -> None:
    lock = (WORKFLOW / "uv.lock").read_text(encoding="utf-8")
    locked = re.search(r'name = "isaacsim"\nversion = "(\d+\.\d+)', lock)
    assert locked is not None
    stated = set(re.findall(r"Isaac Sim (\d+\.\d+)", PAGE.read_text(encoding="utf-8")))
    assert stated == {locked.group(1)}
