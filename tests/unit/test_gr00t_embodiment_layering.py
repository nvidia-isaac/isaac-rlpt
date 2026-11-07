# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layering guardrail: the GR00T policy stays embodiment-agnostic.

``isaac_rl_posttraining/policies/gr00t/`` must load a checkpoint and process
image/state/language into actions without any G1-Factory-whole-body-control
knowledge baked in. That knowledge lives in the sim/embodiment layer
(``simulators/g1_arena/embodiment.py``); the policy receives
it through an injected dataset adapter and the checkpoint's baked modality.

Two invariants are pinned here:

* the policy package imports nothing from ``isaac_rl_posttraining.envs``
  (the dependency points from the env-aware caller into the policy, never
  back), and
* no source token names a specific embodiment (``wbc`` / ``teleop`` / a
  ``leg`` or ``g1`` identifier part), scanning code and string literals but
  not comments or docstrings.
"""

from __future__ import annotations

import ast
import os
import re

from conftest import PACKAGE_NAME, REPO_ROOT

GR00T_DIR = os.path.join(REPO_ROOT, PACKAGE_NAME, "policies", "gr00t")

#: Substrings that never occur in ordinary English or unrelated code.
_FORBIDDEN_SUBSTRINGS = ("wbc", "teleop")
#: Identifier *parts* (split on non-alphanumerics) that flag an embodiment
#: without tripping on words like "delegate"/"legacy" or names like "arg1".
_FORBIDDEN_PARTS = ("leg", "g1")


def _gr00t_sources() -> list[str]:
    files = [
        os.path.join(GR00T_DIR, name)
        for name in sorted(os.listdir(GR00T_DIR))
        if name.endswith(".py")
    ]
    assert files, f"no python sources found under {GR00T_DIR}"
    return files


def _flag(text: str) -> bool:
    low = text.lower()
    if any(sub in low for sub in _FORBIDDEN_SUBSTRINGS):
        return True
    parts = [p for p in re.split(r"[^a-z0-9]+", low) if p]
    return any(part in _FORBIDDEN_PARTS for part in parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of the string-constant nodes that are docstrings (module/def/class)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _embodiment_tokens(source: str) -> list[str]:
    """Identifiers and non-docstring string literals that name an embodiment.

    Comments are absent from the AST and docstrings are skipped by node id,
    so only live code (names, attributes, arguments, keywords, and string
    literals used as values) is inspected.
    """
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            candidates = [node.id]
        elif isinstance(node, ast.Attribute):
            candidates = [node.attr]
        elif isinstance(node, ast.arg):
            candidates = [node.arg]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates = [node.name]
        elif isinstance(node, ast.keyword) and node.arg:
            candidates = [node.arg]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            candidates = [node.value]
        else:
            continue
        offenders.extend(tok for tok in candidates if _flag(tok))
    return offenders


def _env_imports(tree: ast.AST) -> list[str]:
    """The ``from ...`` / ``import ...`` statements in ``tree`` that reach the env layer."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            f"{PACKAGE_NAME}.envs"
        ):
            found.append(f"from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_NAME}.envs"):
                    found.append(f"import {alias.name}")
    return found


def test_gr00t_policy_does_not_import_envs():
    """No module under policies/gr00t imports isaac_rl_posttraining.envs."""
    violations: list[str] = []
    for path in _gr00t_sources():
        with open(path) as handle:
            tree = ast.parse(handle.read())
        violations.extend(
            f"{os.path.basename(path)}: {statement}" for statement in _env_imports(tree)
        )
    assert not violations, (
        "policies/gr00t must not import the env layer:\n" + "\n".join(violations)
    )


def test_gr00t_policy_has_no_embodiment_tokens():
    """No live source token names the G1-WBC embodiment (comments/docstrings ok)."""
    offenders: dict[str, list[str]] = {}
    for path in _gr00t_sources():
        with open(path) as handle:
            found = _embodiment_tokens(handle.read())
        if found:
            offenders[os.path.basename(path)] = sorted(set(found))
    assert not offenders, (
        "embodiment-specific tokens leaked into the generic GR00T policy; move "
        f"them to simulators/g1_arena/embodiment.py:\n{offenders}"
    )
