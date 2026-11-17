# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer actor/critic family for chunked action spaces."""

from isaac_rl_posttraining.modules.transformer.actor import TransformerActor, TransformerActorConfig
from isaac_rl_posttraining.modules.transformer.critic import TransformerCritic, TransformerCriticConfig
from isaac_rl_posttraining.modules.transformer.feature_extractor import (
    PositionalEncoding,
    TransformerFeatureExtractor,
)

__all__ = [
    "TransformerActor",
    "TransformerActorConfig",
    "TransformerCritic",
    "TransformerCriticConfig",
    "PositionalEncoding",
    "TransformerFeatureExtractor",
]
