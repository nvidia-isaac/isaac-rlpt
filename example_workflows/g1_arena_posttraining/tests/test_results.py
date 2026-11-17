# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reported numbers follow the evaluation protocol (well-formedness and the README table are checked
for every workflow by tests/unit/test_example_workflows.py)."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common.results_table import load_results

WORKFLOW = Path(__file__).resolve().parent.parent


def test_base_and_released_checkpoint_share_the_held_out_episodes() -> None:
    """The base and the released checkpoint are scored on the same 300 seeded episodes."""
    results = load_results(WORKFLOW / "results" / "results.json")
    assert results.rows, "a validated workflow reports at least one row"
    for row in results.rows:
        assert row.peak is not None, row
        assert row.base.episodes == row.peak.episodes == 300, row
