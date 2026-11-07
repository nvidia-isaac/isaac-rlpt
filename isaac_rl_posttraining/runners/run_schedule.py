# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an RL run trains: its schedule, the learner ranks of an off-policy run, and config-only checks."""

from __future__ import annotations

import functools
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import algorithm_spec
from isaac_rl_posttraining.algorithms.replay_learner import (
    ReplayLearner,
    replica_learner_spec,
)
from isaac_rl_posttraining.configs import selected_algorithm_name
from isaac_rl_posttraining.runtimes import RolloutRuntime
from isaac_rl_posttraining.runtimes.learner import (
    LearnerGroup,
    LocalLearnerGroup,
    open_distributed_learner_group,
)
from isaac_rl_posttraining.training.schedule import (
    ScheduleSpec,
    on_policy_schedule,
    schedule_from_config,
)

__all__ = [
    "check_run_config",
    "open_replay_learner_group",
    "schedule_spec",
]


def schedule_spec(cfg: DictConfig, *, on_policy: bool = False) -> ScheduleSpec:
    """The run's schedule from its ``schedule`` section (on-policy: data at most one version old)."""
    section = OmegaConf.select(cfg, "schedule", default=None)
    if isinstance(section, DictConfig):
        section = OmegaConf.to_container(section, resolve=True)
    spec = schedule_from_config(section)
    return on_policy_schedule(spec) if on_policy else spec


def check_run_config(cfg: DictConfig, *, on_policy: bool) -> None:
    """Checks that need only the config, run before anything loads (and by the dry run).

    The schedule must be valid for the family (on-policy data lags at most
    one version); an off-policy algorithm config must construct (for example
    step-scheduled settings need ``algorithm.respect_step_schedules``; an edit
    algorithm's single MLP covers the whole decoded chunk).
    """
    schedule_spec(cfg, on_policy=on_policy)
    if not on_policy:
        name = selected_algorithm_name(cfg)
        if algorithm_spec(name).action_space == "edit" and cfg.training.chunk_size is not None:
            raise ValueError(
                f"{name} uses a single MLP over the full action chunk; set training.chunk_size=null"
            )
        from isaac_rl_posttraining.runners.off_policy_launch import (
            algorithm_config_from,
        )

        algorithm_config_from(cfg)


def open_replay_learner_group(
    cfg: DictConfig, runtime: RolloutRuntime, algorithm: Any, device: torch.device
) -> LearnerGroup:
    """One learner rank in this process, or ``placement.learner.ranks`` ranks on a runtime with a distributed learner.

    The ranks train replicas of the algorithm (without the replay buffer,
    which stays here) on their slices of each controller draw.
    """
    num_ranks = int(cfg.placement.learner.ranks)
    if num_ranks == 1:
        return LocalLearnerGroup(
            functools.partial(ReplayLearner, algorithm=algorithm), device=str(device)
        )
    logger.info(f"Off-policy learner: {num_ranks} ranks on the {runtime.name} runtime")
    return open_distributed_learner_group(
        runtime,
        replica_learner_spec(algorithm),
        num_learner_ranks=num_ranks,
        name="replay_learner",
    )
