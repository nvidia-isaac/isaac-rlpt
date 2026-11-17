# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The telemetry schema: tables, row validation and the null sink."""

from __future__ import annotations

import pytest

from isaac_rl_posttraining.training.telemetry import (
    LEARNER,
    ROUNDS,
    SCHEMA_VERSION,
    TABLES,
    TRANSITIONS,
    NullTelemetrySink,
    TelemetrySink,
)


def test_tables_and_columns():
    assert SCHEMA_VERSION == 1
    assert set(TABLES) == {"transitions", "episodes", "rounds", "learner"}
    for table in TABLES.values():
        assert table.names[0] == "run_id"
        assert len(set(table.names)) == len(table.names)
        assert all(column.description for column in table.columns)
    assert {"policy_version", "policy_lag", "episode_id", "t_obs", "t_received"} <= set(
        TRANSITIONS.names
    )


def _learner_row(**overrides):
    row = {
        "run_id": "r",
        "learner_step": 1,
        "env_steps": 64,
        "wall_time": 1.5,
        "metrics": {"loss": 0.1},
    }
    row.update(overrides)
    return row


def test_row_validation():
    LEARNER.validate(_learner_row())
    LEARNER.validate(_learner_row(publish_version=None, env_steps=64))
    with pytest.raises(ValueError, match="no columns"):
        LEARNER.validate(_learner_row(extra=1))
    with pytest.raises(ValueError, match="required"):
        LEARNER.validate(_learner_row(metrics=None))
    with pytest.raises(TypeError, match="int64"):
        LEARNER.validate(_learner_row(learner_step=True))
    with pytest.raises(TypeError, match="float64"):
        LEARNER.validate(_learner_row(wall_time="now"))
    with pytest.raises(ValueError, match="required"):
        ROUNDS.validate({"run_id": "r"})


def test_null_sink_accepts_everything():
    sink = NullTelemetrySink()
    assert isinstance(sink, TelemetrySink)
    sink.write_transitions([{"anything": 1}])
    sink.write_episodes([])
    sink.write_rounds([])
    sink.write_learner([_learner_row()])
    sink.flush()
    sink.close()
