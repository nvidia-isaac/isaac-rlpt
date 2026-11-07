# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration of structured configs with Hydra's ConfigStore."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from isaac_rl_posttraining.algorithms.registry import (
    algorithm_spec,
    available_algorithms,
    get_algorithm_config,
)
from isaac_rl_posttraining.configs.schema import (
    BaseModelConfig,
    PolicyConfig,
    PPOTrainingConfig,
    RuntimeConfig,
    TrainConfig,
    ValueFunctionConfig,
)

__all__ = ["register_configs"]


def register_configs() -> None:
    """Register the schema and config groups with Hydra.

    Schemas are stored as ``base_<name>`` entries that each group yaml lists
    first in its own defaults list (Hydra 1.1+ explicit schema matching); a
    schema stored under a yaml's own name would only be matched through the
    deprecated implicit path.

    Idempotent; safe to call from both the CLI entry point and tests.
    """
    cs = ConfigStore.instance()

    cs.store(name="train_config", node=TrainConfig)

    # Config group "algorithm". Each registered algorithm with a config class
    # contributes the YAML-settable part of its schema; algorithms/registry.py is
    # the single source of truth for what exists. The on-policy options
    # (configs/algorithm/ppo.yaml, dsrl_ppo.yaml) compose the "ppo" schema below.
    for algorithm_name in available_algorithms():
        if algorithm_spec(algorithm_name).config is None:
            continue
        cs.store(
            group="algorithm",
            name=f"base_{algorithm_name}",
            node=get_algorithm_config(algorithm_name).yaml_schema(),
        )

    # Config groups "policy" and "value_function" (the transformer networks;
    # architecture switching happens via training.chunk_size / training.use_rnn).
    cs.store(group="policy", name="base_policy", node=PolicyConfig)
    cs.store(group="value_function", name="base_value_function", node=ValueFunctionConfig)

    # Config group "base_model" (the pretrained policy; see configs/base_model/).
    cs.store(group="base_model", name="base_base_model", node=BaseModelConfig)

    # Config group "runtime" (execution provider; see configs/runtime/ and
    # docs/runtimes.md). Selected independently of simulator, base model, and algorithm.
    cs.store(group="runtime", name="base_runtime", node=RuntimeConfig)

    # The "ppo" section of the on-policy algorithms (schedule and update; see docs/ppo.md).
    cs.store(group="ppo", name="base_ppo", node=PPOTrainingConfig)
