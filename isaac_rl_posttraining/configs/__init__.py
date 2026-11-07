# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema, Hydra registration, and the canonical YAML configs."""

from isaac_rl_posttraining.configs.normalize import register_resolvers
from isaac_rl_posttraining.configs.registry import register_configs
from isaac_rl_posttraining.configs.schema import RuntimeConfig, TrainConfig
from isaac_rl_posttraining.configs.selection import (
    evaluation_enabled,
    selected_algorithm_name,
    selected_base_model_name,
)

register_resolvers()

__all__ = [
    "RuntimeConfig",
    "TrainConfig",
    "evaluation_enabled",
    "register_configs",
    "selected_algorithm_name",
    "selected_base_model_name",
]
