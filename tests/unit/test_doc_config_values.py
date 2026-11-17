# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every ``group=value`` selection written down in the repo names a real config entry.

Guards against documentation running ahead of the code: a rename lands in the
docs before (or instead of) the config group, and every documented command
using the old name dies on a Hydra "Could not find" error before it starts.
The sibling ``test_doc_references.py`` checks that the *paths* a document names
exist; this one checks the *config group values*, the other half of a command a
reader copies.

Scanned: every tracked text file, ``internal/`` directories included. Those are
stripped for a release (``tools/strip_internal.py``) but until then they launch
real cluster jobs against these same config groups, so a stale override there
fails a job exactly the way it fails a reader. Tracked files only -- a walk
would also read a developer's finished runs, where Hydra writes the overrides
of the day under ``outputs/**/.hydra/``. Skill evals (``evals/`` and
``BENCHMARK.md``) are skipped: their prompts name entries an agent is asked to create.

Python sources contribute only their string literals. Docstrings, error
messages and literal override lists are where a Hydra override appears; a bare
``algorithm=<variable>`` in code is a keyword argument, not a config selection.

A token that cannot be a config file name is a placeholder rather than a
mistake, and is skipped: ``algorithm=$ALGO``, ``algorithm=${ALGO}``,
``base_model=<your-base-model>``, ``algorithm=NAME``, ``simulator=isaaclab_<task>``.
A command-line flag is not a group selection either, so the ``docker run
--runtime=nvidia`` in the station launchers is skipped on its leading dash,
not by excusing the value.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"


def _group_roots() -> list[Path]:
    """Directories Hydra composes config groups from.

    The package's ``configs/`` plus every ``pkg://`` entry of the launch
    config's ``hydra.searchpath``, resolved the way Hydra resolves it
    (``pkg://simulators.libero.configs`` -> ``simulators/libero/configs``).
    That search path is how the co-located ``simulator`` group is reached.
    """
    searchpath = re.findall(
        r"pkg://([\w.]+)", (CONFIGS_DIR / "train.yaml").read_text(encoding="utf-8")
    )
    roots = [CONFIGS_DIR] + [
        REPO_ROOT.joinpath(*dotted.split(".")) for dotted in searchpath
    ]
    return [root for root in roots if root.is_dir()]


def _config_groups() -> dict[str, frozenset[str]]:
    """Every selectable config group and the entries it offers, derived from the config tree.

    Derived rather than listed, so a new group directory is covered the day it
    appears and a deleted one stops being checked.
    """
    groups: dict[str, set[str]] = {}
    for root in _group_roots():
        for directory in sorted(root.iterdir()):
            if directory.is_dir() and not directory.name.startswith("__"):
                groups.setdefault(directory.name, set()).update(
                    entry.stem for entry in directory.glob("*.yaml")
                )
    return {group: frozenset(entries) for group, entries in sorted(groups.items())}


#: Config groups a launch command can select from, with each group's entries.
GROUPS = _config_groups()

#: Shape of a config entry name. Anything else in value position is a
#: placeholder or a shell variable; ``test_config_entries_are_lowercase``
#: pins the convention that makes the distinction safe.
ENTRY_NAME = r"[a-z][a-z0-9_]*"

#: A group selection on a command line -- ``algorithm=dsrl_sac``, or the
#: ``a|b|c`` enumeration prose uses to list a group's options. The lookbehind
#: drops flags (``--runtime=nvidia``) and nested field overrides
#: (``algorithm.policy=...``); the lookahead drops a name cut short by a
#: placeholder (``simulator=isaaclab_<task>``) or a shell expansion.
OVERRIDE = re.compile(
    r"(?<![\w./-])("
    + "|".join(GROUPS)
    + r")=("
    + ENTRY_NAME
    + r"(?:\|"
    + ENTRY_NAME
    + r")*)(?![\w<${*])"
)


def _tracked_sources() -> list[tuple[Path, str]]:
    """``(path, text)`` for every tracked file in the repository that decodes as UTF-8."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git is needed to list the repository's tracked files: {exc}")
    sources = []
    for name in filter(None, listing.split("\0")):
        path = REPO_ROOT / name
        try:
            sources.append((path, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue  # binary, or a file the checkout does not materialize
    return sources


def _searchable_chunks(source: Path, text: str) -> list[tuple[int, int, str]]:
    """``(first line, last line, text)`` chunks of one file to search.

    Every file but a Python source is searched whole; a Python source
    contributes its string literals only, so that keyword arguments spelled
    like a group (``algorithm=<variable>``) never read as a config selection.
    """
    if source.suffix != ".py":
        return [(1, text.count("\n") + 1, text)]
    return [
        (node.lineno, node.end_lineno or node.lineno, node.value)
        for node in ast.walk(ast.parse(text, filename=str(source)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _line_of(text: str, lines: list[str], first: int, last: int) -> int:
    """The line between ``first`` and ``last`` that shows ``text``, else ``first``.

    Whole-file chunks land on ``first`` straight away. A python literal starts
    at its opening quote, which is a line or two early when the source
    concatenates fragments; searching its span closes that gap.
    """
    return next(
        (n for n in range(first, last + 1) if text in lines[n - 1]),
        first,
    )


def _selections() -> list[tuple[str, int, str, str]]:
    """Every ``(file, line, group, value)`` group selection written down in the repository."""
    selections = set()
    for source, text in _tracked_sources():
        relative = str(source.relative_to(REPO_ROOT))
        # Skill evals ask an agent to create entries that do not exist yet.
        if source.name == "BENCHMARK.md" or source.parent.name == "evals":
            continue
        lines = text.splitlines()
        for first, last, chunk in _searchable_chunks(source, text):
            for match in OVERRIDE.finditer(chunk):
                start = first + chunk[: match.start()].count("\n")
                line = _line_of(match.group(0), lines, start, last)
                for value in match.group(2).split("|"):
                    selections.add((relative, line, match.group(1), value))
    return sorted(selections)


def test_config_entries_are_lowercase():
    """Config entries stay lowercase snake_case, which is what tells them from placeholders.

    The scan reads any other token in value position as a placeholder --
    ``algorithm=$ALGO``, ``base_model=<your-base-model>``, ``algorithm=NAME`` -- so an
    entry that broke the convention would be skipped everywhere it is named.
    """
    assert GROUPS, f"no config groups found under {CONFIGS_DIR}"
    odd = [
        f"{group}/{entry}.yaml"
        for group, entries in GROUPS.items()
        for entry in sorted(entries)
        if not re.fullmatch(ENTRY_NAME, entry)
    ]
    assert not odd, (
        "config entries must be lowercase snake_case for the override scan to tell "
        "them from placeholders:\n  " + "\n  ".join(odd)
    )


def test_every_written_config_value_resolves():
    """Each ``group=value`` written anywhere in the repository names a real entry of that group.

    One test over every selection, so a rename reports all of its stale commands at once.
    """
    assert GROUPS, f"no config groups found under {CONFIGS_DIR}"
    selections = _selections()
    assert len(selections) > 100, (
        "the scan found suspiciously few group selections; check GROUPS and OVERRIDE"
    )
    unknown = [
        f"{source}:{line}: {group}={value} is not an entry of the {group!r} config "
        f"group ({', '.join(sorted(GROUPS[group]))})"
        for source, line, group, value in selections
        if value not in GROUPS[group]
    ]
    assert not unknown, "config values Hydra cannot resolve:\n  " + "\n  ".join(unknown)
