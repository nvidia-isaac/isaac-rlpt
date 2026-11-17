# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``placement.policy`` decides policy serving on the native and RLInf runtimes."""

from __future__ import annotations

import pytest

from isaac_rl_posttraining.configs.schema import (
    PlacementConfig,
    PolicyPlacementConfig,
    RuntimeConfig,
)
from isaac_rl_posttraining.runtimes import _policy_serving_settings as serving_settings
from isaac_rl_posttraining.runtimes import create_runtime_from_config


def _placement(**policy) -> PlacementConfig:
    return PlacementConfig(policy=PolicyPlacementConfig(**policy))


def test_a_served_policy_on_native_is_one_in_process_server_with_the_placement_batching():
    runtime = create_runtime_from_config(
        RuntimeConfig(
            name="native",
            native={
                "device": "cpu",
                "concurrent_collection": True,
                "serve_evaluation_unbatched": False,
            },
        ),
        _placement(mode="served", max_batch=8, max_wait_ms=1.5),
    )
    config = runtime.config
    assert (
        config.policy_mode,
        config.policy_max_batch,
        config.policy_max_wait_ms,
    ) == ("served", 8, 1.5)
    assert config.concurrent_collection and not config.policy_serve_eval_unbatched


def test_native_defaults_without_a_placement_keep_todays_serving():
    config = create_runtime_from_config(
        RuntimeConfig(name="native", native={"device": "cpu"})
    ).config
    assert (
        config.policy_mode,
        config.policy_max_batch,
        config.policy_max_wait_ms,
    ) == ("auto", None, 5.0)
    # Workers that collect one after another have nobody to wait for; evaluation stays unbatched.
    assert config.batcher(concurrent=False).max_wait_ms == 0.0
    assert config.batcher().max_wait_ms == 5.0 and config.batcher().serve_eval_unbatched
    local = create_runtime_from_config(
        RuntimeConfig(name="native"), _placement(mode="local")
    ).config
    assert local.policy_mode == "local"


def test_native_needs_the_plan_for_several_replicas_and_serves_the_base_model_alone():
    with pytest.raises(ValueError, match="placement plan"):
        create_runtime_from_config(
            RuntimeConfig(name="native"), _placement(mode="served", replicas=2)
        )
    parts = PlacementConfig(
        parts={"base": PolicyPlacementConfig(mode="served", max_batch=8)}
    )
    native = create_runtime_from_config(RuntimeConfig(name="native"), parts)
    assert native.config.base_mode == "served"
    assert native.config.policy_mode == "local"
    assert native.config.policy_max_batch == 8
    assert serving_settings("rlinf", parts) == {
        "policy_mode": "local",
        "base_mode": "served",
        "num_policy_servers": None,
        "policy_batch_size": 8,
    }


def test_rlinf_takes_the_placement_words():
    runtime = create_runtime_from_config(
        RuntimeConfig(name="rlinf", rlinf={"device": "cpu"}),
        _placement(mode="served", replicas=2),
    )
    assert (runtime.config.policy_mode, runtime.config.base_mode) == ("served", "local")
    assert runtime.config.num_policy_servers == 2


def test_rlinf_policy_servers_come_from_the_placement():
    assert serving_settings(
        "rlinf", _placement(mode="served", replicas=3, max_batch=16, max_wait_ms=10.0)
    ) == {
        "policy_mode": "served",
        "num_policy_servers": 3,
        "policy_batch_size": 16,
        "policy_batch_wait_seconds": 0.01,
    }
    # Unset batching keeps the runtime's own defaults.
    assert serving_settings("rlinf", _placement()) == {
        "policy_mode": "auto",
        "num_policy_servers": None,
    }
