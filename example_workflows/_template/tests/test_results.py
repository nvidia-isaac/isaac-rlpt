# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape every workflow test has: runs without hardware or external repositories."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common.results_table import load_results, table_matches

WORKFLOW = Path(__file__).resolve().parent.parent


def test_results_are_well_formed_and_rendered() -> None:
    results = load_results(WORKFLOW / "results" / "results.json")
    assert results.rows, "a workflow reports at least one row"
    assert table_matches(results, WORKFLOW / "README.md")
