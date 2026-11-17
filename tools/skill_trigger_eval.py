# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure how often an agent skill's description makes Claude Code load the skill.

For each query in the skill's ``evals/evals.json`` ``trigger_evals``, runs
``claude -p <query>`` several times in a scratch copy of the repository at a
git ref, with the description under test installed as the only project skill,
and counts the runs in which the agent calls the Skill tool on it within the
timeout. The skill folders under ``.claude/`` and ``.codex/`` are left out of
the copy so the installed skills do not compete with the description under test.

    uv run python tools/skill_trigger_eval.py skills/isaac-rlpt-add-simulator
    uv run python tools/skill_trigger_eval.py skills/isaac-rlpt-add-simulator \\
        --ref <git ref> --description-from <git ref>:<path to an older SKILL.md>   # compare a previous description

Needs the ``claude`` CLI, signed in; every run is one billed agent session.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True
    ).stdout


def _description(skill_md: str) -> str:
    match = re.search(r"^description: (.*)$", skill_md, re.MULTILINE)
    if match is None:
        raise ValueError("SKILL.md has no single-line description in its frontmatter")
    return match.group(1)


def _checkout(ref: str, root: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(_git("archive", ref)), mode="r:") as archive:
        archive.extractall(root, filter="data")
    for agent_folder in (root / ".claude" / "skills", root / ".codex"):
        subprocess.run(["rm", "-rf", str(agent_folder)], check=True)


def _loads_skill(root: Path, query: str, name: str, timeout: float) -> bool:
    env = {key: value for key, value in os.environ.items() if key != "CLAUDECODE"}
    process = subprocess.Popen(
        ["claude", "-p", query, "--output-format", "stream-json", "--verbose"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    try:
        for line in process.stdout:
            if time.monotonic() > deadline:
                return False
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "assistant":
                continue
            for item in event["message"].get("content", []):
                if (
                    item.get("type") == "tool_use"
                    and item["name"] == "Skill"
                    and item["input"].get("skill") == name
                ):
                    return True
        return False
    finally:
        process.kill()
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "skill", type=Path, help="skill folder holding SKILL.md and evals/evals.json"
    )
    parser.add_argument(
        "--ref", default="HEAD", help="git ref of the repository copy the agent runs in"
    )
    parser.add_argument(
        "--description-from",
        help="<ref>:<path> of the SKILL.md whose description is tested",
    )
    parser.add_argument("--runs", type=int, default=3, help="runs per query")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds per run")
    args = parser.parse_args()

    skill = args.skill.resolve().relative_to(REPO_ROOT)
    evals = json.loads((REPO_ROOT / skill / "evals" / "evals.json").read_text())
    name = evals["skill_name"]
    source = args.description_from or f"{args.ref}:{skill}/SKILL.md"
    description = _description(_git("show", source).decode())

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        _checkout(args.ref, root)
        commands = root / ".claude" / "commands"
        commands.mkdir(parents=True, exist_ok=True)
        (commands / f"{name}.md").write_text(
            f"---\ndescription: |\n  {description}\n---\n\n# {name}\n\n{description}\n"
        )
        jobs = [case for case in evals["trigger_evals"] for _ in range(args.runs)]
        with ThreadPoolExecutor(args.workers) as pool:
            hits = list(
                pool.map(
                    lambda case: _loads_skill(root, case["query"], name, args.timeout),
                    jobs,
                )
            )

    cases: dict[str, dict[str, int | bool]] = {}
    for case, hit in zip(jobs, hits, strict=True):
        row = cases.setdefault(
            case["query"],
            {"should_trigger": case["should_trigger"], "triggers": 0, "runs": 0},
        )
        row["triggers"] = int(row["triggers"]) + int(hit)
        row["runs"] = int(row["runs"]) + 1
    positives = [row for row in cases.values() if row["should_trigger"]]
    negatives = [row for row in cases.values() if not row["should_trigger"]]
    summary = {
        "skill": name,
        "ref": args.ref,
        "description_from": source,
        "positives_triggered": sum(
            int(row["triggers"]) * 2 > int(row["runs"]) for row in positives
        ),
        "positives": len(positives),
        "false_triggers": sum(
            int(row["triggers"]) * 2 > int(row["runs"]) for row in negatives
        ),
        "negatives": len(negatives),
        "cases": cases,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
