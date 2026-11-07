# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: every recipe in golden_cases.py resolves to the config pinned in golden_config.json."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden

WORKFLOW = Path(__file__).resolve().parent.parent.name


def test_every_recipe_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow(WORKFLOW, tmp_path)
