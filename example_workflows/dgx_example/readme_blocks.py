# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated blocks of a README, between ``<!-- <name>:start -->`` and ``<!-- <name>:end -->`` markers.

The same shape as the results table's markers in ``_common/results_table.py``, for the other tables
this workflow renders from its data files.
"""

from __future__ import annotations

from pathlib import Path


def markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def _split(text: str, name: str, readme: Path) -> tuple[str, str, str]:
    start_marker, end_marker = markers(name)
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{readme} needs a '{start_marker}' ... '{end_marker}' block")
    head = text[: start + len(start_marker)]
    return head, text[start + len(start_marker) : end], text[end:]


def write_block(readme: Path, name: str, body: str) -> None:
    """Replace the block ``name`` of ``readme`` with ``body``."""
    head, _, tail = _split(readme.read_text(encoding="utf-8"), name, readme)
    text = f"{head}\n{body}\n{tail}"
    readme.write_text(text, encoding="utf-8")  # NOSONAR - user-named CLI path


def block_matches(readme: Path, name: str, body: str) -> bool:
    """Whether the block ``name`` of ``readme`` already reads ``body``."""
    _, current, _ = _split(readme.read_text(encoding="utf-8"), name, readme)
    return current.strip() == body.strip()
