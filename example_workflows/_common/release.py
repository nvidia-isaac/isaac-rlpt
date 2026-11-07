# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``internal/`` policy: what may only live under an ``internal/`` directory, and the scan that enforces it.

release-scan: exempt (this module spells out the terms it looks for)

Anything that works only inside NVIDIA (cluster job templates, container registry paths, shared-filesystem
paths, internal hostnames) lives under a workflow's ``internal/`` directory so ``tools/strip_internal.py``
can remove it for a release and the remaining tree still works.

A feature the internal repository keeps but a release leaves out is listed in :data:`RELEASE_FEATURES_FILE`
(itself internal, so a release never names it) with the paths it owns and the terms that name it. Lines of
shared files that belong to it are fenced: a line carrying ``release-exclude: <feature>`` goes, and so does
every line from ``release-exclude-begin: <feature>`` through ``release-exclude-end: <feature>``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

INTERNAL_DIR_NAME = "internal"

#: Spellings that only make sense inside the company. Word-bounded where a term could hide in another word; a
#: character class keeps a few terms from appearing literally in the released tree.
INTERNAL_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cluster scheduler name", re.compile(r"\bosmo\b", re.IGNORECASE)),
    ("internal container registry", re.compile(r"nvcr\.io", re.IGNORECASE)),
    (
        "object-store path",
        re.compile(r"\bs3://|swift://|pdx\.s8k\.io|\bAUTH_team", re.IGNORECASE),
    ),
    ("shared filesystem name", re.compile(r"\bamlfs\b", re.IGNORECASE)),
    ("GPU pool tooling", re.compile(r"\bgear ray\b", re.IGNORECASE)),
    ("internal source host", re.compile(r"gitlab[-]master", re.IGNORECASE)),
    ("internal registry org", re.compile(r"\bnvidian\b", re.IGNORECASE)),
    ("internal user namespace", re.compile(r"\bmm[o]ghani\b", re.IGNORECASE)),
    ("internal branch name", re.compile(r"\bmasou[d]/", re.IGNORECASE)),
    ("internal-only wording", re.compile(r"\bnvidia[- ]internal\b", re.IGNORECASE)),
    (
        "internal host",
        re.compile(
            r"\b(?:confluence|sharepoint)\.nvidia\.com|\bnvidia\.slack\.com|\bnvbugs\b",
            re.IGNORECASE,
        ),
    ),
)

#: The public NGC side of nvcr.io: images under the ``nvidia`` org and the registry login for them.
PUBLIC_NGC = re.compile(r"nvcr\.io/nvidia/|docker login nvcr\.io\b")

#: A path into an ``internal/`` directory (``internal/jobs/train.yaml``); a bare ``internal/`` mention is fine.
INTERNAL_PATH = re.compile(r"(?<![\w-])internal/[./-]*\w")

_TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".dockerfile",
}
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv"}
#: A file that spells out the internal terms on purpose (the policy and its tests) says so in its first lines.
EXEMPT_MARKER = "release-scan: exempt"

#: Paths (relative to the repository root) that serve only the internal repository and cannot live under an
#: ``internal/`` directory: its CI (SonarQube through NVIDIA's Jenkins, GitLab CI) and the release skill, which
#: must sit beside the other skills. A release drops them, and the scan skips them.
INTERNAL_ONLY_PATHS = (
    "sonar-project.properties",
    "Jenkinsfile",
    ".agents/contributor-skills/isaac-rlpt-release",
    ".claude/skills/isaac-rlpt-release",
    ".codex/skills/isaac-rlpt-release",
)


#: The features a release leaves out: ``[features.<name>]`` tables with ``paths`` (removed) and ``terms``
#: (case-insensitive regular expressions no remaining line may match).
RELEASE_FEATURES_FILE = "tools/internal/release_features.toml"

#: A fence marker: ``release-exclude: <feature>`` (this line) or ``release-exclude-begin|end: <feature>`` (a block).
_FENCE = re.compile(r"release-exclude(?:-(begin|end))?:\s*([\w-]+)")


@dataclass(frozen=True)
class ReleaseFeature:
    name: str
    paths: tuple[str, ...]
    terms: tuple[re.Pattern[str], ...]


def release_features(root: Path) -> dict[str, ReleaseFeature]:
    """The features of :data:`RELEASE_FEATURES_FILE` under ``root`` (none once a release has removed the file)."""
    listing = root / RELEASE_FEATURES_FILE
    if not listing.is_file():
        return {}
    tables = tomllib.loads(listing.read_text(encoding="utf-8")).get("features", {})
    return {
        name: ReleaseFeature(
            name,
            tuple(table.get("paths", ())),
            tuple(re.compile(term, re.IGNORECASE) for term in table.get("terms", ())),
        )
        for name, table in tables.items()
    }


def fenced_lines(lines: list[str], features: Collection[str]) -> list[bool]:
    """Which of ``lines`` a release removes: fenced for one of ``features``.

    Raises ``ValueError`` on a fence for an unlisted feature, a nested or unclosed block, or a stray end.
    """
    removed, block = [], None
    for number, line in enumerate(lines, start=1):
        match = _FENCE.search(line)
        if match is None:
            removed.append(block is not None)
            continue
        kind, name = match.groups()
        if name not in features:
            raise ValueError(
                f"line {number}: release fence for unlisted feature {name!r}"
            )
        if kind == "begin" and block is not None:
            raise ValueError(
                f"line {number}: release fence opens inside the {block!r} block"
            )
        if kind == "end" and block != name:
            raise ValueError(
                f"line {number}: release fence closes a {name!r} block that is not open"
            )
        block = name if kind == "begin" else None if kind == "end" else block
        removed.append(True)
    if block is not None:
        raise ValueError(f"release fence for {block!r} never closes")
    return removed


def strip_fences(text: str, features: Collection[str]) -> str:
    """``text`` without its lines fenced for ``features``."""
    lines = text.splitlines(keepends=True)
    return "".join(
        line for line, gone in zip(lines, fenced_lines(lines, features)) if not gone
    )


@dataclass(frozen=True)
class InternalReference:
    path: Path
    line: int
    label: str
    text: str

    def describe(self, root: Path) -> str:
        return f"{self.path.relative_to(root).as_posix()}:{self.line}: {self.label}: {self.text.strip()[:100]}"


def internal_only_paths(repo_root: Path) -> list[Path]:
    """The :data:`INTERNAL_ONLY_PATHS` present under ``repo_root`` (files, directories or symlinks)."""
    return [
        repo_root / name
        for name in INTERNAL_ONLY_PATHS
        if os.path.lexists(repo_root / name)
    ]


def is_internal(
    path: Path, root: Path, features: Collection[ReleaseFeature] = ()
) -> bool:
    """Whether a release removes ``path``: under an ``internal/`` directory below ``root``, one of
    :data:`INTERNAL_ONLY_PATHS`, or a path of one of ``features``."""
    relative = path.relative_to(root)
    owned = [
        *INTERNAL_ONLY_PATHS,
        *(name for feature in features for name in feature.paths),
    ]
    return INTERNAL_DIR_NAME in relative.parts[:-1] or any(
        relative.is_relative_to(name) for name in owned
    )


def files(root: Path) -> list[Path]:
    """The files of ``root``: the ones git tracks or would track when ``root`` is in a checkout, else all."""
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0:
        candidates = [root / name for name in listed.stdout.split("\0") if name]
    else:
        candidates = list(root.rglob("*"))
    return sorted(
        path
        for path in candidates
        if path.is_file()
        and not any(part in _SKIP_DIRS for part in path.relative_to(root).parts)
    )


def internal_dirs(root: Path) -> list[Path]:
    """Every ``internal/`` directory under ``root``, outermost first."""
    found: set[Path] = set()
    for path in files(root):
        parts = path.relative_to(root).parts[:-1]
        if INTERNAL_DIR_NAME in parts:
            found.add(root.joinpath(*parts[: parts.index(INTERNAL_DIR_NAME) + 1]))
    return sorted(found)


def text_files(root: Path):
    for path in files(root):
        if (
            path.suffix.lower() in _TEXT_SUFFIXES
            or path.name in {"Dockerfile", "Makefile"}
            or path.name.startswith("Dockerfile.")
        ):
            yield path


def is_exempt(path: Path) -> bool:
    """A file that spells out the internal terms on purpose carries :data:`EXEMPT_MARKER` near its top."""
    with path.open(encoding="utf-8", errors="ignore") as handle:
        head = "".join(handle.readline() for _ in range(10))
    return EXEMPT_MARKER in head


def _line_references(
    path: Path, number: int, line: str, features: Collection[ReleaseFeature] = ()
) -> list[InternalReference]:
    published = PUBLIC_NGC.sub("", line)
    labels = [label for label, pattern in INTERNAL_TERMS if pattern.search(published)]
    if INTERNAL_PATH.search(line):
        labels.append("path into an internal/ directory")
    labels += [
        f"release-excluded feature {feature.name}"
        for feature in features
        if any(term.search(line) for term in feature.terms)
    ]
    return [InternalReference(path, number, label, line) for label in labels]


def internal_references(root: Path) -> list[InternalReference]:
    """What would dangle in the release tree: internal terms, paths into ``internal/``, malformed fences, and
    terms of a release-excluded feature, in the lines of every file a release keeps; and feature paths that no
    longer exist."""
    features = release_features(root)
    findings = [
        InternalReference(
            root / RELEASE_FEATURES_FILE,
            0,
            f"release-excluded feature {feature.name}",
            f"lists the missing path {name}",
        )
        for feature in features.values()
        for name in feature.paths
        if not os.path.lexists(root / name)
    ]
    for path in text_files(root):
        if is_internal(path, root, features.values()) or is_exempt(path):
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        try:
            removed = fenced_lines(lines, features)
        except ValueError as error:
            findings.append(
                InternalReference(path, 0, "malformed release fence", str(error))
            )
            continue
        for number, (line, gone) in enumerate(zip(lines, removed), start=1):
            if not gone:
                findings.extend(_line_references(path, number, line, features.values()))
    return findings
