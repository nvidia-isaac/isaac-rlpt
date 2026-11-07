# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training loops and what they share.

``OffPolicyTrainer`` (replay buffer, bursts of updates per collected decision)
and ``OnPolicyTrainer`` (iterations of fresh episodes) both run over runtime
rollout and learner groups. Checkpoints, weight fingerprints, evaluation, the
replay data source and the trajectory structures live here because both loops
use them.
"""

from isaac_rl_posttraining.training.checkpoints import (
    CheckpointStore,
    FingerprintSink,
    policy_state_fingerprint,
)
from isaac_rl_posttraining.training.data_source import (
    DataSource,
    PendingCollection,
    PersistSpec,
    RuntimeDataSource,
)
from isaac_rl_posttraining.training.evaluation import (
    EpisodeOutcome,
    RolloutEvaluator,
    episode_outcomes,
    summarize_episodes,
)
from isaac_rl_posttraining.training.off_policy import (
    OffPolicyConfig,
    OffPolicyDataSource,
    OffPolicyTrainer,
    PolicyEvaluator,
    ReplayLearnerGroup,
)
from isaac_rl_posttraining.training.on_policy import (
    ITERATION_CHECKPOINT_PREFIX,
    OnPolicyAlgorithm,
    OnPolicyConfig,
    OnPolicyTrainer,
    checkpoint_iteration,
)
from isaac_rl_posttraining.training.recording import RolloutRecorder
from isaac_rl_posttraining.training.schedule import (
    SCHEDULES,
    Schedule,
    ScheduleSpec,
    schedule_from_config,
)
from isaac_rl_posttraining.training.telemetry import (
    ParquetTelemetrySink,
    open_run_telemetry,
    read_telemetry,
)
from isaac_rl_posttraining.training.trajectories import (
    ChunkTrajectory,
    DiscountPer,
    trajectories_from_results,
)

__all__ = [
    "ITERATION_CHECKPOINT_PREFIX",
    "SCHEDULES",
    "CheckpointStore",
    "ChunkTrajectory",
    "DataSource",
    "DiscountPer",
    "EpisodeOutcome",
    "FingerprintSink",
    "OffPolicyConfig",
    "OffPolicyDataSource",
    "OffPolicyTrainer",
    "OnPolicyAlgorithm",
    "OnPolicyConfig",
    "OnPolicyTrainer",
    "ParquetTelemetrySink",
    "PendingCollection",
    "PersistSpec",
    "PolicyEvaluator",
    "ReplayLearnerGroup",
    "RolloutEvaluator",
    "RolloutRecorder",
    "RuntimeDataSource",
    "Schedule",
    "ScheduleSpec",
    "checkpoint_iteration",
    "episode_outcomes",
    "open_run_telemetry",
    "policy_state_fingerprint",
    "read_telemetry",
    "schedule_from_config",
    "summarize_episodes",
    "trajectories_from_results",
]
