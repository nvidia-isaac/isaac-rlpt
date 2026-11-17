# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``open_policy_service`` behaves the same on the native runtime (Ray actors) and on RLInf.

Both start Ray: the native case needs the ``ray`` extra, the RLInf case the
``rlinf`` extra and ``ISAAC_RL_POSTTRAINING_RLINF_CLUSTER=1``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from serving_doubles import REPLICA_STATS, ScaledPolicy

from isaac_rl_posttraining.runtimes import create_runtime
from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef

pytestmark = pytest.mark.timeout(600)


@pytest.mark.parametrize(
    ("runtime_name", "settings"),
    [
        pytest.param("native", {"device": "cpu"}, marks=pytest.mark.native_ray),
        pytest.param("rlinf", {}, marks=[pytest.mark.rlinf, pytest.mark.rlinf_cluster]),
    ],
)
def test_a_standalone_service_looks_the_same_on_every_runtime(runtime_name, settings):
    runtime = create_runtime(runtime_name, **settings)
    try:
        service = runtime.open_policy_service(
            ScaledPolicy,
            name="jet_policies",
            replicas=2,
            max_batch=8,
            max_wait_ms=3_000.0,
        )
        assert service.handle().name == "jet_policies"
        assert service.ping() == (0, 1)
        assert service.features().multi_session
        clients = [service.client(route_index) for route_index in (0, 1)]
        started = time.monotonic()
        chunks = [
            client.action_chunk({"state": np.full(2, 3.0, np.float32)}, mode="eval")
            for client in clients
        ]
        # Each replica's only client is answered without waiting out max_wait_ms.
        assert time.monotonic() - started < 3.0
        assert [float(chunk.learner_action[0, 0]) for chunk in chunks] == [3.0, 3.0]
        weights = PolicyState(version=1, tensors={"scale": np.array([2.0], np.float32)})
        assert service.set_weights(WeightsRef.of(weights)) == (1, 1)
        chunk = clients[1].action_chunk({"state": np.ones(2, np.float32)}, mode="eval")
        assert (float(chunk.learner_action[0, 0]), chunk.policy_version) == (2.0, 1)
        # Statistics are collected without asking, one entry per replica.
        replicas = service.stats()["replicas"]
        assert [replica["policy_server_index"] for replica in replicas] == [0, 1]
        assert [replica["batch_sizes"] for replica in replicas] == [(1,), (1, 1)]
        assert all(set(replica) == REPLICA_STATS for replica in replicas)
        assert [replica["policy_metadata"] for replica in replicas] == [
            {"replica": 0},
            {"replica": 1},
        ]
        service.close()
    finally:
        runtime.close()
