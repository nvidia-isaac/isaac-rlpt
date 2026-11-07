# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names shared across algorithm checkpoints: components and state files."""

from __future__ import annotations

__all__ = [
    "ACTOR",
    "ACTOR_BEHAVIOR_ANCHOR",
    "ACTOR_BEHAVIOR_ANCHOR_CAPTURED",
    "ACTOR_OPTIMIZER",
    "CRITIC",
    "CRITIC_OPTIMIZER",
    "CRITIC_TARGET",
    "ENT_COEF_OPTIMIZER",
    "INPUT_NORMALIZER",
    "LOG_ENT_COEF",
    "OPTIMIZER_COMPONENT_SUFFIX",
    "REWARD_SCALER",
    "TRAINING_STATE_FILE",
]


# Registered component names; each doubles as the checkpoint file stem (<name>.pt).
ACTOR = "actor"
CRITIC = "critic"
CRITIC_TARGET = "critic_target"
ACTOR_OPTIMIZER = "actor_optimizer"
CRITIC_OPTIMIZER = "critic_optimizer"
ENT_COEF_OPTIMIZER = "ent_coef_optimizer"
LOG_ENT_COEF = "log_ent_coef"
ACTOR_BEHAVIOR_ANCHOR = "actor_behavior_anchor"
ACTOR_BEHAVIOR_ANCHOR_CAPTURED = "actor_behavior_anchor_captured"
REWARD_SCALER = "reward_scaler"
INPUT_NORMALIZER = "input_normalizer"

#: Components with this suffix hold optimizer state and are restored only with it.
OPTIMIZER_COMPONENT_SUFFIX = "_optimizer"
#: JSON sidecar next to the component files: step counter and training-state scalars.
TRAINING_STATE_FILE = "training_state.json"
