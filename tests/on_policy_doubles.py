# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data source an ``OnPolicyTrainer`` collects through, opened as the on-policy launchers open it."""

from __future__ import annotations

from typing import Any

from isaac_rl_posttraining.runtimes import RolloutGroup
from isaac_rl_posttraining.training import DataSource


def episode_source(
    group: RolloutGroup, episodes_per_environment: int = 1, **lineage: Any
) -> DataSource:
    """A buffer-less data source over ``group`` collecting ``episodes_per_environment`` episodes a round."""
    return DataSource(
        group,
        None,
        count_by="episode",
        request_count=episodes_per_environment,
        reward_aggregation="sum",
        **lineage,
    )
