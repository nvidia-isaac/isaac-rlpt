# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every repo-relative path mentioned in configs, docs, and pyproject exists.

Guards against reorganizations leaving dangling references behind (deleted
docs still cited from config headers, moved scripts still named in comments).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _py_sources(root: Path) -> list[Path]:
    """Shipped (non-test) Python sources under ``root``.

    Test files intentionally name fake/sandbox/not-present paths, so scanning
    them for real references would produce false positives; they are excluded.
    """
    return [
        p
        for p in root.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]


#: Files whose text is scanned for repo-relative path references. Docs, configs,
#: launchers, AND shipped Python sources are all scanned (recursively) so a
#: dangling path left behind by a move -- a deleted doc still cited, a script
#: renamed in a comment or docstring -- is caught wherever it hides.
SCANNED = (
    list((REPO_ROOT / "isaac_rl_posttraining" / "configs").rglob("*.yaml"))
    + list((REPO_ROOT / "docs").rglob("*.md"))
    + list((REPO_ROOT / "simulators").rglob("*.md"))
    + list((REPO_ROOT / "simulators").rglob("*.yaml"))
    + list((REPO_ROOT / "simulators").rglob("*.sh"))
    + _py_sources(REPO_ROOT / "simulators")
    + list((REPO_ROOT / "scripts").rglob("*.sh"))
    + _py_sources(REPO_ROOT / "scripts")
    + list((REPO_ROOT / "example_workflows").rglob("*.md"))
    + list((REPO_ROOT / "example_workflows").rglob("*.yaml"))
    + list((REPO_ROOT / "example_workflows").rglob("*.sh"))
    + _py_sources(REPO_ROOT / "example_workflows")
    + _py_sources(REPO_ROOT / "tools")
    + [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", REPO_ROOT / "pyproject.toml"]
)
# Hidden folders hold fetched third-party checkouts and environments (a workflow's .externals/, .venv/).
SCANNED = [
    p
    for p in SCANNED
    if not any(part.startswith(".") for part in p.relative_to(REPO_ROOT).parts)
]

#: Repo-relative path shapes worth checking (concrete extensions only).
PATH_PATTERN = re.compile(
    r"(?<![\w/.-])((?:scripts|docs|simulators|tests|integrations|isaac_rl_posttraining|example_workflows|tools)/[\w./-]+\.(?:sh|py|md|yaml))"
)

#: Box-drawing glyphs used in ASCII directory trees, normalized to spaces so a
#: path drawn under tree branches (``├── simulators/foo.py``) still extracts
#: cleanly and a glyph can never fuse into an adjacent path token.
_TREE_GLYPHS = str.maketrans(dict.fromkeys("│├└─┬┼┤┌┐┘┴╭╮╰╯", " "))


def _referenced_paths() -> set[tuple[Path, str]]:
    """Every distinct (source, referenced path) pair across the scanned files."""
    refs = set()
    for source in SCANNED:
        text = source.read_text(encoding="utf-8", errors="ignore").translate(
            _TREE_GLYPHS
        )
        for match in PATH_PATTERN.finditer(text):
            path = match.group(1)
            if any(ch in path for ch in "<>{}$*"):
                continue  # placeholder, glob, or shell expansion
            refs.add((source.relative_to(REPO_ROOT), path))
    return refs


def _workflow_root(source: Path) -> Path | None:
    """The workflow folder a source under ``example_workflows/`` belongs to, if any."""
    parts = source.parts
    if not parts or parts[0] != "example_workflows" or len(parts) < 3:
        return None
    depth = 3 if parts[1] == "experimental" else 2
    if len(parts) <= depth:
        return None
    return REPO_ROOT / Path(*parts[:depth])


def test_every_referenced_path_exists():
    """Each referenced path resolves from the repository root or, inside a workflow, from that workflow's folder.

    One test over every reference so a reorganization reports all of its dangling paths at once.
    """
    refs = _referenced_paths()
    assert len(refs) > 100, (
        "the scan found suspiciously few references; check SCANNED and PATH_PATTERN"
    )
    missing = []
    for source, path in sorted(refs, key=str):
        roots = [REPO_ROOT]
        workflow = _workflow_root(source)
        if workflow is not None:
            roots.append(workflow)
        if not any((root / path).exists() for root in roots):
            missing.append(f"{source} references {path}")
    assert not missing, "dangling path references:\n  " + "\n  ".join(missing)
