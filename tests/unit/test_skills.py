# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent skills have one copy each, and the agent-specific folders only link to it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOTS = (REPO_ROOT / "skills", REPO_ROOT / ".agents" / "contributor-skills")
AGENT_ROOTS = (REPO_ROOT / ".claude" / "skills", REPO_ROOT / ".codex" / "skills")
SKILLS = sorted(
    path for root in SKILL_ROOTS for path in root.iterdir() if path.is_dir()
)


def _frontmatter(skill: Path) -> dict[str, str]:
    text = (skill / "SKILL.md").read_text()
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{skill.name}/SKILL.md has no frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines())


@pytest.mark.parametrize("agent_root", AGENT_ROOTS, ids=lambda p: p.parent.name)
def test_agent_folders_link_every_skill_and_hold_no_copies(agent_root):
    entries = sorted(agent_root.iterdir())
    assert all(entry.is_symlink() for entry in entries), [
        e.name for e in entries if not e.is_symlink()
    ]
    assert sorted(entry.resolve() for entry in entries) == [
        skill.resolve() for skill in SKILLS
    ]


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_skill_is_named_after_its_folder_and_ships_evals(skill):
    frontmatter = _frontmatter(skill)
    assert frontmatter["name"] == skill.name
    assert skill.name.startswith("isaac-rlpt-")
    assert (
        "Use when" in frontmatter["description"]
        and "Not for" in frontmatter["description"]
    )
    evals = json.loads((skill / "evals" / "evals.json").read_text())
    assert evals["skill_name"] == skill.name and evals["evals"]
    triggers = [case["should_trigger"] for case in evals["trigger_evals"]]
    assert True in triggers and False in triggers
    assert (skill / "BENCHMARK.md").is_file()
