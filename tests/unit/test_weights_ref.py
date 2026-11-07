# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WeightsRef resolution and the declared runtime features."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_rl_posttraining.runtimes import NativeRuntime, RuntimeFeatures, WeightsRef
from isaac_rl_posttraining.runtimes.base import PolicyState, register_weights_resolver


def _state(version: int = 3) -> PolicyState:
    return PolicyState(version=version, tensors={"w": np.ones(2, dtype=np.float32)})


class _Handle:
    def __init__(self, state: PolicyState) -> None:
        self.state = state


class _SubHandle(_Handle):
    pass


def test_inline_reference_resolves_to_its_state():
    state = _state()
    ref = WeightsRef.of(state, parts=["actor"])
    assert ref.version == 3 and ref.parts == ("actor",)
    assert ref.resolve() is state


def test_reference_validation():
    with pytest.raises(ValueError, match="version"):
        WeightsRef(version=2, inline=_state(3))
    with pytest.raises(ValueError, match="not both"):
        WeightsRef(version=3, inline=_state(3), handle=object())
    with pytest.raises(ValueError):
        WeightsRef(version=-1)
    with pytest.raises(ValueError, match="version-only"):
        WeightsRef(version=4).resolve()


def test_handles_need_a_registered_resolver():
    with pytest.raises(TypeError, match="no weights resolver"):
        WeightsRef(version=3, handle=object()).resolve()
    register_weights_resolver(_Handle, lambda handle: handle.state)
    state = _state()
    assert WeightsRef(version=3, handle=_SubHandle(state)).resolve() is state
    with pytest.raises(ValueError, match="resolved weights carry version 3"):
        WeightsRef(version=4, handle=_Handle(state)).resolve()


def test_runtime_features():
    native = NativeRuntime.features
    assert native.name == "native" and native.async_collect
    assert native.policy_service and native.served_batched_collectors
    with pytest.raises(ValueError, match="policy_service"):
        RuntimeFeatures(
            name="x",
            multi_node=False,
            policy_service=False,
            served_batched_collectors=False,
            distributed_learner=False,
            async_collect=False,
            weights_by_reference=False,
            serving_engines=("builtin",),
        )
