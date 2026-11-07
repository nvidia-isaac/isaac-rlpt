# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shims for the oldest Python this package supports (3.10)."""

from __future__ import annotations


def add_note(exc: BaseException, note: str) -> None:
    """Attach ``note`` to ``exc``.

    Uses :meth:`BaseException.add_note` where it exists (3.11+). On 3.10 the
    note is kept in ``__notes__`` for callers that read it and appended to the
    message so it still shows in the traceback.
    """
    native = getattr(exc, "add_note", None)
    if native is not None:
        native(note)
        return
    notes = getattr(exc, "__notes__", None)
    if notes is None:
        notes = []
        exc.__notes__ = notes
    notes.append(note)
    exc.args = (*exc.args, note) if exc.args else (note,)
