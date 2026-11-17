# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf rollout calls must not delegate failure handling to RLInf waiters."""

from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest

from isaac_rl_posttraining.runtimes import RolloutRequest, RolloutResult
from isaac_rl_posttraining.runtimes.rlinf.rollout import RLInfRolloutGroup


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, *args):
        return self._function(*args)


class _Actor:
    def __init__(self, worker_index: int, channel) -> None:
        self.collect_to_channel = _RemoteMethod(
            lambda request, output_channel, key: self._collect(
                request, output_channel, key
            )
        )
        self._worker_index = worker_index
        self._channel = channel

    def _collect(self, request, output_channel, key):
        assert request.decisions_per_environment == 1
        output_channel.values[key] = RolloutResult(
            worker_index=self._worker_index,
            decisions=(),
            env_steps=0,
            episodes_completed=0,
            policy_version=0,
        )
        return self._worker_index, time.time()


class _Channel:
    def __init__(self) -> None:
        self.values = {}

    def get(self, *, key):
        return self.values[key]


def test_collection_uses_actor_refs_instead_of_suicidal_waiters(monkeypatch) -> None:
    channel = _Channel()
    actors = [_Actor(index, channel) for index in range(2)]
    provider_group = SimpleNamespace(
        worker_info_list=[SimpleNamespace(worker=actor) for actor in actors]
    )
    group = RLInfRolloutGroup(
        provider_group,
        2,
        num_envs=2,
        assignments=((0,), (1,)),
        output_channel=channel,
    )
    fake_ray = SimpleNamespace(get=lambda refs: refs)
    monkeypatch.setattr(
        "isaac_rl_posttraining.runtimes.rlinf.rollout.importlib.import_module",
        lambda name: fake_ray if name == "ray" else importlib.import_module(name),
    )

    before = time.time()
    call = group.collect_async([RolloutRequest(decisions_per_environment=1)] * 2)
    results = call.result()

    assert [result.worker_index for result in results] == [0, 1]
    # The round's end is stamped by the actors themselves, so a late reader still gets its duration.
    finished_at = call.finished_at()
    assert finished_at is not None and before <= finished_at <= time.time()


def test_collection_rejects_missing_actor_before_submission() -> None:
    group = RLInfRolloutGroup(
        SimpleNamespace(worker_info_list=[]),
        1,
        num_envs=1,
        assignments=((0,),),
        output_channel=_Channel(),
    )

    with pytest.raises(RuntimeError, match="0 actors, expected 1"):
        group.collect_async([RolloutRequest(decisions_per_environment=1)])


class _Publisher:
    """A worker that records when it takes weights; ``frozen`` never answers (its collection is stuck)."""

    def __init__(self, index: int, calls: list, *, frozen: bool = False) -> None:
        self.set_policy_state = _RemoteMethod(
            lambda state: (
                calls.append(f"worker{index}") or ("late" if frozen else state.version)
            )
        )


def _publishing_group(
    monkeypatch, calls: list, *, frozen: bool = False
) -> RLInfRolloutGroup:
    workers = [_Publisher(0, calls), _Publisher(1, calls, frozen=frozen)]
    servers = SimpleNamespace(set_weights=lambda weights: calls.append("servers"))
    group = RLInfRolloutGroup(
        SimpleNamespace(
            worker_info_list=[SimpleNamespace(worker=worker) for worker in workers]
        ),
        2,
        num_envs=2,
        assignments=((0,), (1,)),
        output_channel=_Channel(),
        policy=servers,
        rollout_timeout_seconds=5.0,
    )
    fake_ray = SimpleNamespace(
        get=lambda refs: refs,
        wait=lambda refs, num_returns, timeout: (
            [ref for ref in refs if ref != "late"],
            [ref for ref in refs if ref == "late"],
        ),
    )
    monkeypatch.setattr(
        "isaac_rl_posttraining.runtimes.rlinf.rollout.importlib.import_module",
        lambda name: fake_ray if name == "ray" else importlib.import_module(name),
    )
    return group


def test_weights_reach_the_servers_only_after_every_worker_took_them(
    monkeypatch,
) -> None:
    from isaac_rl_posttraining.runtimes import PolicyState

    calls: list[str] = []
    group = _publishing_group(monkeypatch, calls)
    assert group.set_policy_state(PolicyState(version=3, tensors={})) == (3, 3)
    # A worker takes weights only after its collection in flight, which still calls the old servers.
    assert calls == ["worker0", "worker1", "servers"]


def test_a_frozen_worker_fails_the_publish_naming_it(monkeypatch) -> None:
    from isaac_rl_posttraining.runtimes import PolicyState
    from isaac_rl_posttraining.runtimes.watchdog import RolloutTimeoutError

    calls: list[str] = []
    group = _publishing_group(monkeypatch, calls, frozen=True)
    with pytest.raises(
        RolloutTimeoutError,
        match=r"publishing weights: no result within 5 s from RLInf rollout worker 1",
    ):
        group.set_policy_state(PolicyState(version=3, tensors={}))
    assert "servers" not in calls and group._frozen
