# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule specs: kinds and lag bounds."""

from __future__ import annotations

import pytest

from isaac_rl_posttraining.training.schedule import ScheduleSpec


def test_defaults_and_lag_bounds():
    assert ScheduleSpec().kind == "sync" and ScheduleSpec().policy_lag_bound == 0
    assert ScheduleSpec("overlapped").policy_lag_bound == 1
    assert ScheduleSpec("async").policy_lag_bound is None
    assert (
        ScheduleSpec("async", max_policy_lag=2, publish_every=4).policy_lag_bound == 2
    )
    assert (
        not ScheduleSpec().overlaps_collection
        and ScheduleSpec("async").overlaps_collection
    )


def test_only_the_three_schedule_names_are_accepted():
    for removed in ("serial", "lookahead"):
        with pytest.raises(ValueError, match="sync"):
            ScheduleSpec(removed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "pipelined"}, "schedule must be one of"),
        ({"kind": "sync", "max_policy_lag": 1}, "must be 0"),
        ({"kind": "overlapped", "max_policy_lag": 0}, ">= 1"),
        ({"kind": "async", "max_policy_lag": -1}, ">= 0"),
        ({"kind": "sync", "publish_every": 2}, "async schedule only"),
        ({"kind": "async", "publish_every": 0}, ">= 1"),
    ],
)
def test_invalid_specs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ScheduleSpec(**kwargs)
