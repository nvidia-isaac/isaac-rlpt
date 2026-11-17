# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add_note attaches context on a real exception and on one without the native method (Python 3.10)."""

from __future__ import annotations

from isaac_rl_posttraining.utils.compat import add_note


def test_add_note_on_a_real_exception():
    exc = ValueError("boom")
    add_note(exc, "while closing the simulator")
    assert exc.__notes__ == ["while closing the simulator"]
    assert exc.args == ("boom",)


def test_add_note_falls_back_to_the_message_without_the_native_method():
    class LegacyError(Exception):
        add_note = None  # as on Python 3.10, where exceptions have args but no add_note

    exc = LegacyError("boom")
    add_note(exc, "first")
    add_note(exc, "second")
    assert exc.args == ("boom", "first", "second")
    assert exc.__notes__ == ["first", "second"]
