# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy-server request/response messages and the service handle."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.serving import (
    PolicyRequest,
    PolicyResponse,
    PolicyService,
    ServiceHandle,
    ServingEngine,
)


def _chunk() -> ActionChunk:
    return ActionChunk(
        learner_action=np.zeros((1, 2), dtype=np.float32),
        learner_action_kind="latent",
        environment_actions=np.zeros((1, 4, 3), dtype=np.float32),
        execution_length=4,
        policy_version=1,
    )


def test_requests_validate_kind_and_mode():
    request = PolicyRequest(
        "act", session_id=3, observation={"state": np.zeros(2)}, mode="train"
    )
    assert pickle.loads(pickle.dumps(request)).session_id == 3
    PolicyRequest("encode", session_id=3, observation=None)
    with pytest.raises(ValueError, match="decision mode"):
        PolicyRequest("act_and_value", session_id=0, observation=None)
    with pytest.raises(ValueError, match="kind"):
        PolicyRequest("successor", session_id=0, observation=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        PolicyRequest("act", session_id=0, observation=None, mode="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request_id"):
        PolicyRequest("encode", session_id=0, observation=None, request_id=-1)


def test_responses_carry_what_their_kind_needs():
    PolicyResponse(request_id=1, kind="act", policy_version=1, action_chunk=_chunk())
    PolicyResponse(
        request_id=1,
        kind="act_and_value",
        policy_version=1,
        action_chunk=_chunk(),
        value=0.5,
    )
    PolicyResponse(
        request_id=1, kind="value", policy_version=1, observation=np.zeros(2), value=0.5
    )
    with pytest.raises(ValueError, match="action chunk"):
        PolicyResponse(request_id=1, kind="act", policy_version=1)
    with pytest.raises(ValueError, match="action chunk"):
        PolicyResponse(
            request_id=1, kind="encode", policy_version=1, action_chunk=_chunk()
        )
    with pytest.raises(ValueError, match="needs a value"):
        PolicyResponse(request_id=1, kind="value", policy_version=1)


def test_service_handle_routing():
    handle = ServiceHandle("policy", replicas=3)
    assert [handle.replica_for(i) for i in range(5)] == [0, 1, 2, 0, 1]
    with pytest.raises(ValueError, match="request time"):
        ServiceHandle("policy", replicas=2, routing="least_loaded").replica_for(0)
    with pytest.raises(ValueError):
        ServiceHandle("policy", replicas=0)
    with pytest.raises(ValueError):
        ServiceHandle("", replicas=1)


def test_protocols_are_structural():
    class _Engine:
        features = None
        version = 0

        def load(self, spec, context): ...
        def run(self, kind, observations, sessions, modes):
            return []

        def reset_sessions(self, session_ids, seeds): ...
        def load_weights(self, ref):
            return ref.version

        def metadata(self):
            return {}

        def close(self): ...

    class _Service:
        def handle(self):
            return ServiceHandle("p", 1)

        def client(self, route_index=0): ...

        def ping(self):
            return (0,)

        def features(self): ...

        def set_weights(self, ref):
            return (ref.version,)

        def stats(self):
            return {"replicas": ()}

        def reset_stats(self): ...

        def close(self): ...

    assert isinstance(_Engine(), ServingEngine)
    assert isinstance(_Service(), PolicyService)
    assert not isinstance(_Engine(), PolicyService)


def test_a_policy_server_timeout_names_the_timeout_of_either_runtime(monkeypatch):
    import sys
    import types

    from isaac_rl_posttraining.runtimes.serving.ray_client import (
        PolicyServerTimeoutError,
        RayTransport,
    )

    class GetTimeoutError(Exception):
        pass

    def get(reference, timeout):
        raise GetTimeoutError

    ray = types.SimpleNamespace(
        get=get, exceptions=types.SimpleNamespace(GetTimeoutError=GetTimeoutError)
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    actor = types.SimpleNamespace(
        connect=types.SimpleNamespace(remote=lambda **kwargs: None)
    )
    with pytest.raises(PolicyServerTimeoutError, match="within 5.0 s") as raised:
        RayTransport(actor, timeout_seconds=5.0)
    for key in ("runtime.native", "runtime.rlinf"):
        assert f"{key}.policy_request_timeout_seconds" in str(raised.value)


def test_learner_ranks_help_covers_off_policy_algorithms():
    from dataclasses import fields

    from isaac_rl_posttraining.configs.runtime_schema import LearnerPlacementConfig

    (ranks,) = [
        field for field in fields(LearnerPlacementConfig) if field.name == "ranks"
    ]
    assert "on-policy algorithm" not in ranks.metadata["help"]
    assert "runtime.native.processes" in ranks.metadata["help"]


def test_both_runtimes_host_policies_through_one_policy_service_class():
    from isaac_rl_posttraining.runtimes.base import PolicyState, WeightsRef
    from isaac_rl_posttraining.runtimes.native_ray import serving as native_serving
    from isaac_rl_posttraining.runtimes.rlinf import runtime as rlinf_runtime
    from isaac_rl_posttraining.runtimes.serving import HostedPolicy

    assert native_serving.HostedPolicy is rlinf_runtime.HostedPolicy is HostedPolicy

    class _Servers:
        def __init__(self):
            self.calls = []

        def set_weights(self, ref, *, policy):
            self.calls.append(("set_weights", policy))
            return [ref.version, ref.version - 1]

        def stats(self, *, policy):
            return [{"replica": 0}, {"replica": 1}]

        def reset_stats(self, *, policy): ...

        def remove_policy(self, name):
            self.calls.append(("remove_policy", name))

    servers = _Servers()
    hosted = HostedPolicy(servers, "train", ServiceHandle("train", 2))
    assert isinstance(hosted, PolicyService)
    assert hosted.stats() == {"replicas": ({"replica": 0}, {"replica": 1})}
    with pytest.raises(
        RuntimeError, match=r"acknowledged versions \(3, 2\), expected 3"
    ):
        hosted.set_weights(WeightsRef.of(PolicyState(version=3, tensors={})))
    hosted.close()
    hosted.close()
    assert servers.calls == [("set_weights", "train"), ("remove_policy", "train")]
