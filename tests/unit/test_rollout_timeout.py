# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A frozen simulator fails the run with the late rollout worker named, instead of hanging it."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from entry_doubles import scripted_worker_spec

from isaac_rl_posttraining.configs.schema import RuntimeConfig
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    NativeRuntime,
    NativeRuntimeConfig,
    PolicyState,
    RolloutInfrastructureError,
    RolloutRequest,
    create_runtime_from_config,
)
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.watchdog import (
    FROZEN_ROLLOUT_EXIT_CODE,
    CollectionWatchdog,
    RolloutTimeoutError,
    collection_timeout,
    on_frozen_rollout,
)
from isaac_rl_posttraining.training.checkpoints import save_aborted_checkpoint


class _Env:
    """Counts its steps; ``freeze_at`` makes the step that reaches it spin in Python forever."""

    def __init__(self, freeze_at: int | None = None) -> None:
        self.freeze_at = freeze_at
        self.t = 0
        self.closed = False

    def reset(self, *, seed=None):
        self.t = 0
        return {"state": np.asarray([0.0], np.float32)}, {}

    def step(self, action):
        self.t += 1
        while self.freeze_at is not None and self.t >= self.freeze_at:
            time.sleep(0.01)
        return {"state": np.asarray([float(self.t)], np.float32)}, 1.0, False, False, {}

    def close(self):
        self.closed = True


class _Policy(RolloutPolicy):
    features = PolicyFeatures(encodes_observations=False)

    def action_chunks(self, observations, *, mode, session_ids):
        return tuple(
            ActionChunk.from_arrays(
                learner_action=np.zeros((1, 1), np.float32),
                environment_actions=np.zeros((1, 1, 1), np.float32),
                execution_length=1,
                learner_action_kind="latent",
                policy_version=0,
            )
            for _ in observations
        )

    def set_policy_state(self, state: PolicyState) -> int:
        return state.version


def _group(timeout: float | None, envs: dict[int, _Env]):
    runtime = NativeRuntime(
        NativeRuntimeConfig(device="cpu", rollout_timeout_seconds=timeout)
    )
    spec = scripted_worker_spec(
        lambda context: envs[context.environment_indices[0]],
        lambda context: _Policy(),
        num_envs=len(envs),
    )
    return runtime, runtime.open_rollout_group(
        spec, num_rollout_workers=len(envs), num_envs=len(envs), name="sims"
    )


def test_fast_collections_are_unchanged_by_the_time_limit():
    results = []
    for timeout in (None, 30.0):
        runtime, group = _group(timeout, {0: _Env(), 1: _Env()})
        try:
            collected = group.collect([RolloutRequest(decisions_per_environment=3)] * 2)
        finally:
            runtime.close()
        results.append(
            [(r.worker_index, len(r.decisions), r.env_steps) for r in collected]
        )
    assert results[0] == results[1] == [(0, 3, 3), (1, 3, 3)]


class _SlowEpisodes(_Env):
    """Two-step episodes of 0.1 s steps: slow but alive."""

    def step(self, action):
        time.sleep(0.1)
        obs, reward, _, truncated, info = super().step(action)
        return obs, reward, self.t >= 2, truncated, {**info, "is_success": False}


def test_the_limit_is_per_requested_episode_so_a_long_evaluation_finishes():
    runtime, group = _group(0.3, {0: _SlowEpisodes()})
    try:
        (result,) = group.collect(
            [RolloutRequest(episodes_per_environment=4, mode="eval")]
        )
    finally:
        runtime.close()
    assert result.env_steps == 8


def test_the_collection_limit_scales_with_the_episodes_requested():
    assert (
        collection_timeout(None, [RolloutRequest(episodes_per_environment=5)]) is None
    )
    assert (
        collection_timeout(10.0, [RolloutRequest(decisions_per_environment=7)]) == 10.0
    )
    assert (
        collection_timeout(
            10.0,
            [
                RolloutRequest(episodes_per_environment=2),
                RolloutRequest(episodes_per_environment=5),
            ],
        )
        == 50.0
    )


def test_a_frozen_worker_fails_the_collection_naming_it_and_is_not_closed():
    envs = {0: _Env(), 1: _Env(freeze_at=2)}
    runtime, group = _group(0.5, envs)
    started = time.monotonic()
    with pytest.raises(
        RolloutTimeoutError, match=r"rollout worker 1 \(cpu\).*may be frozen"
    ) as raised:
        group.collect([RolloutRequest(decisions_per_environment=3)] * 2)
    assert time.monotonic() - started < 5.0
    assert isinstance(raised.value, RolloutInfrastructureError)
    assert "rollout worker 0" not in str(raised.value)
    runtime.close()
    # The healthy simulator is closed; the frozen one is left alone (closing it could hang too).
    assert envs[0].closed and not envs[1].closed


def test_an_overlapped_collection_times_out_on_its_helper_thread():
    runtime, group = _group(0.5, {0: _Env(freeze_at=1)})
    try:
        pending = group.collect_async([RolloutRequest(decisions_per_environment=2)])
        with pytest.raises(RolloutTimeoutError, match="rollout worker 0"):
            pending.result()
    finally:
        runtime.close()


def test_a_collection_stuck_in_native_code_saves_and_exits_the_process():
    """No exception reaches a thread blocked in native code: the handlers run, then the process exits."""
    release = threading.Event()
    exits: list[int] = []
    saved: list[BaseException] = []
    raised: list[BaseException] = []

    def exit_process(code: int) -> None:
        exits.append(code)
        release.set()

    watchdog = CollectionWatchdog(0.2, grace_seconds=0.2, exit_process=exit_process)

    def collect() -> None:
        try:
            watchdog.run(
                release.wait,
                describe=lambda: "rollout worker 3 (cuda:2)",
                what="rollout group 'train'",
            )
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    with on_frozen_rollout(saved.append):
        thread = threading.Thread(target=collect)
        thread.start()
        thread.join(timeout=10)
    assert exits == [FROZEN_ROLLOUT_EXIT_CODE]
    (error,) = saved
    assert isinstance(
        error, RolloutTimeoutError
    ) and "rollout worker 3 (cuda:2)" in str(error)
    # Once the thread runs Python again it unwinds with the same error.
    assert raised and raised[0] is error


def test_the_aborted_checkpoint_is_written_once():
    class _Trainer:
        def __init__(self) -> None:
            self.saves: list[str] = []

        def save_checkpoint(self, tag: str) -> None:
            self.saves.append(tag)

    trainer = _Trainer()
    error = RolloutTimeoutError("frozen")
    save_aborted_checkpoint(trainer, error)
    save_aborted_checkpoint(trainer, error)
    assert trainer.saves == ["aborted"]


def test_the_time_limit_is_a_runtime_key_for_every_runtime():
    assert (
        create_runtime_from_config(
            RuntimeConfig(name="native")
        ).config.rollout_timeout_seconds
        == 1800.0
    )
    runtime = create_runtime_from_config(
        RuntimeConfig(name="native", rollout_timeout_seconds=None)
    )
    assert runtime.config.rollout_timeout_seconds is None
    with pytest.raises(ValueError, match="rollout_timeout_seconds"):
        RuntimeConfig(name="native", rollout_timeout_seconds=0)
    with pytest.raises(ValueError, match="rollout_timeout_seconds"):
        CollectionWatchdog(-1.0)
