# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer critic scoring an observation plus a chunk of actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from isaac_rl_posttraining.modules.base import BaseCritic, BaseCriticConfig
from isaac_rl_posttraining.modules.transformer.feature_extractor import (
    PositionalEncoding,
    TransformerFeatureExtractor,
)
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["TransformerCriticConfig", "TransformerCritic"]


@dataclass
class TransformerCriticConfig(BaseCriticConfig):
    """Configuration for :class:`TransformerCritic`.

    Attributes:
        d_model: Transformer token width.
        nhead: Attention heads.
        num_encoder_layers: Encoder depth.
        transformer_dropout: Dropout inside the transformer.
        transformer_activation: ``relu`` or ``gelu``.
        chunk_size: Expected action-chunk length for Q-functions.
        feature_hidden_dims: Shared MLP over concatenated per-key features.
        output_hidden_dims: MLP applied to the pooled representation.
        activation: Activation for the MLP parts.
        use_layer_norm: LayerNorm inside the MLP parts.
        dropout_rate: Dropout inside the MLP parts.
        positional_dropout: Dropout inside the positional encoding.
        action_embedding_dim: Width of the per-step action embedding.
        pooling_strategy: How to pool transformer tokens into one vector —
            ``first`` (observation token), ``attention`` (learned pooling),
            ``weighted_mean`` (recency-weighted), or ``mean``.
        featurizer: Per-key featurizer spec for dict observations.
        image_encoder_type: ``"impala"``/``"dinov2"`` for image keys.
        image_feature_dim: Fallback image embedding width.
        use_language_embeddings: Encode string observations with MiniLM.
        lang_embedding_dim: Language embedding width.
        lang_embedding_device: Device for the language encoder.
        dinov2_model: Loaded DINOv2 model (not a model id).
        dinov2_processor: Matching image processor.
        impala_nn_scale: IMPALA channel-width multiplier.
        impala_num_blocks_per_stack: IMPALA residual blocks per stack.
        impala_use_smaller: Use the reduced-depth IMPALA variant.
        impala_output_dim: IMPALA embedding projection width.
        impala_bounded_head: Append LayerNorm + tanh to IMPALA embeddings.
    """

    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 1
    transformer_dropout: float = 0.1
    transformer_activation: str = "gelu"

    chunk_size: int = 10

    feature_hidden_dims: Optional[Sequence[int]] = (128,)
    output_hidden_dims: Optional[list[int]] = None

    activation: str = "relu"
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    positional_dropout: float = 0.0
    action_embedding_dim: int = 128
    pooling_strategy: str = "attention"

    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[list[Any]] = None

    image_encoder_type: Optional[str] = None
    image_feature_dim: int = 128

    use_language_embeddings: bool = True
    lang_embedding_dim: int = 384
    lang_embedding_device: str = "cpu"

    dinov2_model: Any = None
    dinov2_processor: Any = None

    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: Optional[int] = None
    impala_bounded_head: bool = False

    def __post_init__(self) -> None:
        """Fill list defaults and validate transformer geometry."""
        if self.feature_hidden_dims is None:
            self.feature_hidden_dims = []
        if self.output_hidden_dims is None:
            self.output_hidden_dims = []
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be positive, got {self.nhead}")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        if self.transformer_activation not in ("relu", "gelu"):
            raise ValueError(f"transformer_activation must be 'relu' or 'gelu', got {self.transformer_activation}")
        if self.num_encoder_layers <= 0:
            raise ValueError(f"num_encoder_layers must be positive, got {self.num_encoder_layers}")
        if self.action_embedding_dim <= 0:
            raise ValueError(f"action_embedding_dim must be positive, got {self.action_embedding_dim}")

    @property
    def critic_class(self) -> type:
        return TransformerCritic


class TransformerCritic(BaseCritic):
    """Sequence critic over ``[obs_token, action_1, ..., action_K]`` tokens.

    For Q-functions the observation embedding becomes the first token and
    each chunk step contributes one action token; the pooled transformer
    output feeds a scalar value head. With ``use_action=False`` the critic
    scores the observation token alone (a V-function).
    """

    def __init__(self, config: TransformerCriticConfig) -> None:
        super().__init__(config)
        self.config = config
        self.use_action = config.use_action

        self.obs_feature_extractor = TransformerFeatureExtractor(
            observation_space=config.observation_space,
            featurizer_cfg=config.featurizer,
            feature_hidden_dims=config.feature_hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            preprocess_obs_transform=config.preprocess_obs_transform,
            image_encoder_type=config.image_encoder_type,
            image_feature_dim=config.image_feature_dim,
            dinov2_model=config.dinov2_model,
            dinov2_processor=config.dinov2_processor,
            impala_nn_scale=config.impala_nn_scale,
            impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
            impala_use_smaller=config.impala_use_smaller,
            impala_output_dim=config.impala_output_dim,
            impala_bounded_head=config.impala_bounded_head,
            use_language_embeddings=config.use_language_embeddings,
            lang_embedding_dim=config.lang_embedding_dim,
            lang_embedding_device=config.lang_embedding_device,
        )

        if self.use_action:
            if isinstance(config.action_space, gym.spaces.Box):
                action_dim = int(np.prod(config.action_space.shape))
            elif isinstance(config.action_space, gym.spaces.Discrete):
                action_dim = config.action_space.n
            else:
                raise ValueError(f"Unsupported action space: {config.action_space}")
            self.action_embedding = nn.Linear(action_dim, config.action_embedding_dim)
            self.action_projection = nn.Linear(config.action_embedding_dim, config.d_model)

        if (
            self.obs_feature_extractor.output_dim is not None
            and self.obs_feature_extractor.output_dim != config.d_model
        ):
            self.obs_projection: Optional[nn.Linear] = nn.Linear(
                self.obs_feature_extractor.output_dim, config.d_model
            )
        else:
            self.obs_projection = None

        self.input_norm = nn.LayerNorm(config.d_model)
        self.position_embedding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.chunk_size + 1,  # observation token + chunk tokens
            dropout=config.positional_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.transformer_dropout,
            activation=config.transformer_activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        if config.output_hidden_dims:
            self.output_mlp: Optional[nn.Module] = build_mlp(
                config.d_model,
                config.output_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            final_hidden_dim = config.output_hidden_dims[-1]
        else:
            self.output_mlp = None
            final_hidden_dim = config.d_model

        self.pooling_strategy = config.pooling_strategy
        if self.pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.Tanh(),
                nn.Linear(config.d_model, 1, bias=False),
            )

        self.value_head = nn.Linear(final_hidden_dim, 1)

    def _process_actions(self, obs_features: torch.Tensor, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Build the ``[B, 1 + K, d_model]`` token sequence from obs + actions.

        Args:
            obs_features: Observation features ``[B, feature_dim]``.
            action: Actions ``[B, K, A]`` (a 2-D ``[B, A]`` batch is treated
                as a length-1 chunk).
            batch_size: Batch size.

        Raises:
            ValueError: If ``action`` is not 2-D or 3-D.
        """
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if action.dim() != 3:
            raise ValueError(f"Action must be 3D [batch, chunk_size, action_dim], got shape {action.shape}")

        chunk_size = action.size(1)

        obs_token = self.obs_projection(obs_features) if self.obs_projection is not None else obs_features
        obs_token = obs_token.unsqueeze(1)  # [B, 1, d_model]

        action_flat = action.reshape(-1, action.size(-1))
        action_tokens = self.action_projection(self.action_embedding(action_flat))
        action_tokens = action_tokens.view(batch_size, chunk_size, -1)

        return torch.cat([obs_token, action_tokens], dim=1)

    def _pool_transformer_output(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, T, d_model]`` tokens to ``[B, d_model]`` per the config."""
        if self.pooling_strategy == "first":
            # The observation token — keeps state information undiluted.
            return transformer_output[:, 0, :]

        if self.pooling_strategy == "attention":
            attention_scores = self.attention_pool(transformer_output)  # [B, T, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            return (transformer_output * attention_weights).sum(dim=1)

        if self.pooling_strategy == "weighted_mean":
            # Later tokens weigh more; the observation token keeps top weight.
            seq_len = transformer_output.size(1)
            positions = torch.arange(seq_len, device=transformer_output.device, dtype=transformer_output.dtype)
            weights = torch.exp(positions * 0.1)
            weights[0] = weights.max()
            weights = (weights / weights.sum()).view(1, -1, 1)
            return (transformer_output * weights).sum(dim=1)

        return transformer_output.mean(dim=1)

    def forward(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return values of shape ``[B, 1]``."""
        return self.value_head(self.compute_pooled(obs, action))

    def compute_pooled(self, obs: Any, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the pooled representation feeding the value head.

        Ensembles that share this critic's trunk call ``compute_pooled`` once
        and then apply each member's ``value_head`` separately.
        """
        obs = self._remove_obs_keys(obs)
        obs_features = self.obs_feature_extractor(obs)
        batch_size = int(obs_features.size(0))

        if self.use_action and action is not None:
            input_sequence = self._process_actions(obs_features, action, batch_size)
        else:
            obs_token = self.obs_projection(obs_features) if self.obs_projection is not None else obs_features
            input_sequence = obs_token.unsqueeze(1)

        input_sequence = self.input_norm(input_sequence)
        sequence_with_pos = self.position_embedding(input_sequence)
        transformer_output = self.transformer_encoder(sequence_with_pos)

        pooled_output = self._pool_transformer_output(transformer_output)
        if self.output_mlp is not None:
            pooled_output = self.output_mlp(pooled_output)
        return pooled_output
