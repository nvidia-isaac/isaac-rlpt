# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf serves a steering policy's base model alone: policy-server workers decode, rollout workers run the actor.

The ``rlinf_cluster`` marker keeps the test out of the CPU lane unless
``ISAAC_RL_POSTTRAINING_RLINF_CLUSTER=1`` requests it.
"""

from __future__ import annotations

import pytest
from base_model_doubles import (
    DrawingBaseModel,
    batched_spec,
    collect,
    gym_spec,
    steering_spec,
)

from isaac_rl_posttraining.runtimes import (
    NativeRuntime,
    NativeRuntimeConfig,
    RolloutRequest,
)

SEEDED_EVALUATION = (
    RolloutRequest(episodes_per_environment=2, mode="eval", reset_seed=11),
)


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.parametrize("worker", ["gym", "batched"])
def test_rlinf_serves_the_base_model_to_workers_that_run_the_actor(worker) -> None:
    """Seeded evaluation matches an all-local run in one process; two groups share the servers."""
    from isaac_rl_posttraining.runtimes.rlinf import RLInfRuntime, RLInfRuntimeConfig

    num_envs, workers = (4, 2) if worker == "batched" else (3, 3)
    make = batched_spec if worker == "batched" else gym_spec

    def spec():
        return make(steering_spec(DrawingBaseModel(), "rfs"), num_envs)

    single = collect(
        NativeRuntime(NativeRuntimeConfig(device="cpu", policy_mode="local")),
        spec(),
        workers=workers,
        num_envs=num_envs,
        requests=SEEDED_EVALUATION,
    )
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="cpu",
            base_mode="served",
            num_policy_servers=2,
            policy_request_timeout_seconds=120.0,
        )
    )
    shared = spec()
    try:
        runtime.open_rollout_group(
            shared, num_rollout_workers=workers, num_envs=num_envs, name="other"
        )
        (service,) = runtime._base_services.values()
        assert service.handle().replicas == 2
    except BaseException:
        runtime.close()
        raise
    several = collect(
        runtime, shared, workers=workers, num_envs=num_envs, requests=SEEDED_EVALUATION
    )
    assert several == single
