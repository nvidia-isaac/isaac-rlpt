# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every runtime implements the whole RolloutRuntime protocol; a missing capability is a clear NotImplementedError."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from serving_doubles import REPLICA_STATS, ScaledPolicy

from isaac_rl_posttraining.runtimes import (
    NativeRuntime,
    NativeRuntimeConfig,
    create_runtime,
    learner,
)
from isaac_rl_posttraining.runtimes.base import (
    PolicyState,
    WeightsRef,
    check_runtime,
)
from isaac_rl_posttraining.runtimes.serving import PolicyService


def test_native_implements_every_member_and_declares_its_features():
    runtime = check_runtime(NativeRuntime(NativeRuntimeConfig(device="cpu")))
    assert NativeRuntime.features.distributed_learner
    # One process: services in it, but no nodes or learner ranks of their own.
    assert runtime.features.standalone_policy_service
    assert not (runtime.features.distributed_learner or runtime.features.multi_node)


def test_rlinf_opens_standalone_policy_services():
    assert create_runtime("rlinf").features.standalone_policy_service


def test_one_native_replica_serves_in_this_process_with_the_whole_service_surface():
    runtime = create_runtime("native", device="cpu")
    try:
        service = runtime.open_policy_service(
            ScaledPolicy, name="in_process", max_wait_ms=5_000.0
        )
        assert isinstance(service, PolicyService)
        assert (service.handle().name, service.ping()) == ("in_process", (0,))
        assert service.features().multi_session
        client = service.client(3)
        started = time.monotonic()
        for _ in range(3):
            client.action_chunk({"state": np.ones(2, np.float32)}, mode="eval")
        assert time.monotonic() - started < 2.0  # a lone client never waits 5 s
        state = PolicyState(version=1, tensors={"scale": np.array([2.0], np.float32)})
        assert service.set_weights(WeightsRef.of(state)) == (1,)
        (replica,) = service.stats()["replicas"]
        assert set(replica) == REPLICA_STATS and replica["batch_sizes"] == (1, 1, 1)
    finally:
        runtime.close()


def test_a_runtime_that_omits_a_method_is_refused():
    class _Partial:
        name = "partial"
        features = None

        def open_rollout_group(self, spec, *, num_rollout_workers, num_envs, name): ...

        def close(self): ...

    with pytest.raises(
        TypeError, match="lacks open_learner_group, open_policy_service"
    ):
        check_runtime(_Partial())


@pytest.mark.parametrize("name", ["native", "rlinf"])
def test_one_learner_rank_trains_in_this_process_on_every_runtime(monkeypatch, name):
    def no_group(*args, **kwargs):
        raise AssertionError("one rank must not open a learner group")

    runtime = SimpleNamespace(
        name=name,
        features=SimpleNamespace(distributed_learner=True),
        open_learner_group=no_group,
    )
    monkeypatch.setattr(
        learner, "LocalLearnerGroup", lambda spec, *, device: ("local", device)
    )
    group = learner.open_learner_group(
        runtime, lambda *_: None, num_learner_ranks=1, name="x", local_device="cpu"
    )
    assert group == ("local", "cpu")
