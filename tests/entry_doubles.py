# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulator entry over a scripted environment, for tests that build their own environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
)
from isaac_rl_posttraining.runtimes import GymWorkerSpec, RolloutWorkerContext

__all__ = ["ScriptedSimulator", "scripted_environment", "scripted_worker_spec"]


class ScriptedSimulator(SimulatorEntry):
    """Builds the test's environment on the worker from a ``context -> environment`` callable."""

    name = "scripted"

    def __init__(self, make_env: Callable[[RolloutWorkerContext], Any]) -> None:
        super().__init__()
        self._make_env = make_env

    def make_environment(
        self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext
    ) -> Any:
        del spec, phase
        return self._make_env(context)


def scripted_environment(
    num_envs: int = 1, *, max_episode_steps: int = 1000, seed: int = 0
) -> EnvironmentSpec:
    """The environment description of a scripted simulator (nothing reads it but the spec)."""
    return EnvironmentSpec(
        env_name="scripted/task",
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )


def scripted_worker_spec(
    make_env: Callable[[RolloutWorkerContext], Any],
    policy: Callable[..., Any],
    *,
    phase: Phase = "train",
    num_envs: int = 1,
    **options: Any,
) -> GymWorkerSpec:
    """A Gym worker spec over a scripted environment and a policy spec."""
    return GymWorkerSpec(
        entry=ScriptedSimulator(make_env),
        environment=scripted_environment(num_envs),
        phase=phase,
        policy=policy,
        **options,
    )
