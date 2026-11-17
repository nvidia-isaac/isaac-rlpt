# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer chunking actor: one observation in, a chunk of actions out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.modules.base import BaseActor, BaseActorConfig
from isaac_rl_posttraining.modules.base.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from isaac_rl_posttraining.modules.transformer.feature_extractor import (
    PositionalEncoding,
    TransformerFeatureExtractor,
)
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = ["TransformerActorConfig", "TransformerActor"]


@dataclass
class TransformerActorConfig(BaseActorConfig):
    """Configuration for :class:`TransformerActor`.

    Attributes:
        d_model: Transformer token width.
        nhead: Attention heads.
        num_encoder_layers: Encoder depth.
        transformer_dropout: Dropout inside the transformer.
        transformer_activation: ``relu`` or ``gelu``.
        chunk_size: Number of action steps predicted per observation.
        feature_hidden_dims: Shared MLP applied to concatenated per-key
            features before the transformer (empty = none).
        output_hidden_dims: MLP applied per token after the transformer
            (empty = none).
        activation: Activation for the MLP parts.
        use_layer_norm: LayerNorm inside the MLP parts.
        dropout_rate: Dropout inside the MLP parts.
        use_tanh_output: Squash continuous actions through ``tanh``.
        log_std_init: Initial bias of the ``log_std`` head.
        log_std_min: Lower clamp for predicted ``log_std``.
        log_std_max: Upper clamp for predicted ``log_std``.
        official_policy_init: Make the initial distribution the frozen policy's
            observation-independent Gaussian noise prior.
        positional_dropout: Dropout inside the positional encoding.
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

    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 1
    transformer_dropout: float = 0.1
    transformer_activation: str = "gelu"

    chunk_size: int = 10

    feature_hidden_dims: Optional[list[int]] = None
    output_hidden_dims: Optional[list[int]] = None

    activation: str = "relu"
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    use_tanh_output: bool = True
    log_std_init: float = 0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    official_policy_init: bool = False

    positional_dropout: float = 0.0

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
    impala_use_smaller: bool = True
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

    @property
    def actor_class(self) -> type:
        return TransformerActor


def _tensor_batch_size(v: Any) -> Optional[int]:
    """Batch size implied by one dict entry, or ``None`` if it is not clearly batched."""
    if not isinstance(v, torch.Tensor):
        return None
    if v.dim() >= 4:
        return v.size(0)
    if v.dim() == 3:
        # HWC/CHW image-like tensors are single frames.
        if v.size(0) == 3 or v.size(-1) == 3:
            return 1
        return v.size(0)
    if v.dim() == 2:
        return v.size(0)
    return None


class TransformerActor(BaseActor):
    """Chunking policy: predicts a ``[B, chunk_size, A]`` action sequence.

    The current observation embedding is repeated once per chunk step; the
    sinusoidal positional encoding is the only signal that differentiates
    steps, and the transformer encoder mixes information across them. A
    single mean/log-std head is applied per token, so every step shares
    parameters but produces its own distribution.
    """

    def __init__(self, config: TransformerActorConfig) -> None:
        super().__init__(config)
        self.config = config

        self.feature_extractor = TransformerFeatureExtractor(
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

        if self.feature_extractor.output_dim != config.d_model:
            self.obs_projection: Optional[nn.Linear] = nn.Linear(self.feature_extractor.output_dim, config.d_model)
        else:
            self.obs_projection = None

        self.input_norm = nn.LayerNorm(config.d_model)
        self.position_embedding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.chunk_size,
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

        self.pre_output_norm = nn.LayerNorm(final_hidden_dim)

        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))
            self.mean_processor = nn.Sequential(
                nn.Linear(final_hidden_dim, final_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(final_hidden_dim // 2, action_dim),
            )
            self.log_std_processor = nn.Sequential(
                nn.Linear(final_hidden_dim, final_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(final_hidden_dim // 2, action_dim),
            )
            with torch.no_grad():
                self.log_std_processor[-1].weight.fill_(0.0)
                self.log_std_processor[-1].bias.fill_(config.log_std_init)
                if config.official_policy_init:
                    self.mean_processor[-1].weight.fill_(0.0)
                    self.mean_processor[-1].bias.fill_(0.0)

            if config.use_tanh_output:
                self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
            else:
                self.action_dist = DiagGaussianDistribution(action_dim=action_dim)
        else:
            action_dim = config.action_space.n
            self.logits_processor = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    @staticmethod
    def _infer_batch_size(obs: Any) -> int:
        """Infer the batch size from the first clearly batched tensor."""
        if isinstance(obs, dict):
            for v in obs.values():
                batch_size = _tensor_batch_size(v)
                if batch_size is not None:
                    return batch_size
            return 1
        if isinstance(obs, torch.Tensor):
            return obs.size(0) if obs.dim() >= 2 else 1
        return 1

    def _forward(
        self, obs: Any, actor_state: Any = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-step distribution parameters for the whole chunk.

        Args:
            obs: Observation batch (dict or tensor).
            actor_state: Unused; the transformer actor is stateless.

        Returns:
            ``(mean, log_std)`` of shape ``[B, chunk_size, A]`` for continuous
            actions, or logits ``[B, chunk_size, n]`` for discrete ones.
        """
        batch_size = self._infer_batch_size(obs)

        obs_features = self.feature_extractor(obs)  # [B, feature_dim]
        if self.obs_projection is not None:
            obs_features = self.obs_projection(obs_features)  # [B, d_model]

        # One token per chunk step: same observation, distinguished by the
        # positional encoding added below.
        repeated_obs = obs_features.unsqueeze(1).repeat(1, self.config.chunk_size, 1)
        repeated_obs = self.input_norm(repeated_obs)
        sequence_with_pos = self.position_embedding(repeated_obs)

        transformer_output = self.transformer_encoder(src=sequence_with_pos)  # [B, K, d_model]

        if self.output_mlp is not None:
            flat = transformer_output.view(-1, transformer_output.size(-1))
            output_features = self.output_mlp(flat).view(batch_size, self.config.chunk_size, -1)
        else:
            output_features = transformer_output
        output_features = self.pre_output_norm(output_features)

        output_flat = output_features.view(-1, output_features.size(-1))
        if self.is_continuous:
            mean = self.mean_processor(output_flat)
            log_std = torch.clamp(
                self.log_std_processor(output_flat), self.config.log_std_min, self.config.log_std_max
            )
            mean = mean.view(batch_size, self.config.chunk_size, -1)
            log_std = log_std.view(batch_size, self.config.chunk_size, -1)
            return mean, log_std
        logits = self.logits_processor(output_flat)
        return logits.view(batch_size, self.config.chunk_size, -1)

    def _act(self, obs: Any, deterministic: bool = False, actor_state: Any = None) -> tuple[torch.Tensor, Any]:
        """Sample (or take the mode of) a full action chunk.

        Returns:
            Tuple of ``(actions [B, chunk_size, A], None)`` — the actor is
            stateless.
        """
        mean, log_std, _ = self.get_action_dist_params(obs, actor_state)
        if not self.is_continuous:
            raise NotImplementedError("Discrete action spaces are not supported by the transformer actor")
        actions = self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic)
        return actions, None

    def get_action_dist_params(
        self, obs: Any, hidden: Any = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Return ``(mean, log_std, extra_kwargs)`` for the chunk distribution."""
        if self.is_continuous:
            mean, log_std = self._forward(obs, actor_state=hidden)
            return mean, log_std, {}
        logits = self._forward(obs, actor_state=hidden)
        return logits, None, {}

    def get_initial_state(self) -> dict[str, Any]:
        """Return an empty state (the transformer actor is stateless)."""
        return {}
