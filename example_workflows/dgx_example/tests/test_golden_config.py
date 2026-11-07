# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: the learner of every DGX Station recipe resolves to the pinned config."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common import golden


def test_every_station_recipe_resolves_to_its_pinned_config(tmp_path: Path) -> None:
    golden.check_workflow("dgx_example", tmp_path)
