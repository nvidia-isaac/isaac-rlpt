# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation feature extraction for the transformer actor/critic.

Turns a dict observation batch into a single ``[B, D]`` feature vector:
each key runs through its own encoder (an MLP for vector keys; optionally an
IMPALA CNN or a frozen DINOv2 model for image keys; a sentence encoder for
language strings) and the per-key features are concatenated, then optionally
passed through a shared feature MLP.
"""

from __future__ import annotations

from typing import Any, Optional

import copy
import math

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch import nn

from isaac_rl_posttraining.utils.featurizers import identify_image_keys
from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = [
    "PositionalEncoding",
    "MiniLMLangEncoder",
    "build_featurizers",
    "TransformerFeatureExtractor",
]


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """Args:
        d_model: Token embedding width.
        max_len: Maximum supported sequence length.
        dropout: Dropout applied after adding the encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the encoding to a ``[B, T, d_model]`` sequence."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MiniLMLangEncoder:
    """Sentence encoder producing fixed-width language embeddings."""

    def __init__(self, device: str = "cpu", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Args:
        device: Device the encoder runs on.
        model_name: SentenceTransformer model id.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for language encoding: pip install sentence-transformers"
            ) from exc

        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.encode = self.model.encode

    def get_lang_emb(self, lang_strings: str | list[str]) -> torch.Tensor:
        """Embed one or more instruction strings to ``[N, D]``."""
        if isinstance(lang_strings, str):
            lang_strings = [lang_strings]
        return self.model.encode(lang_strings, convert_to_tensor=True, device=self.device)


def build_featurizers(
    featurizer_cfg: dict[str, Any],
    observation_space: gym.Space,
    activation: str,
    use_layer_norm: bool = False,
    dropout_rate: float = 0.0,
) -> nn.ModuleDict:
    """Build per-key featurizer modules from a config mapping.

    Args:
        featurizer_cfg: Maps observation keys to either a list/tuple of MLP
            hidden dims or a ready ``nn.Module``.
        observation_space: Dict space used to size the MLP inputs.
        activation: Activation for MLP featurizers.
        use_layer_norm: LayerNorm inside MLP featurizers.
        dropout_rate: Dropout inside MLP featurizers.

    Returns:
        A ``ModuleDict`` of featurizers keyed by observation key.
    """
    featurizers = nn.ModuleDict()
    for key, value in featurizer_cfg.items():
        if isinstance(value, (list, tuple)):
            if isinstance(observation_space, gym.spaces.Dict) and key in observation_space.spaces:
                obs_dim = int(np.prod(observation_space.spaces[key].shape))
            else:
                raise ValueError(f"Cannot determine observation dimension for key {key!r}")
            featurizers[key] = build_mlp(obs_dim, value, activation, use_layer_norm, dropout_rate)
        elif isinstance(value, nn.Module):
            featurizers[key] = value
        else:
            raise ValueError(f"Featurizer for key {key!r} must be list/tuple or nn.Module")
    return featurizers


def _infer_batch_size(obs: dict[str, Any], keys: list[str]) -> int:
    """Batch size from the first non-empty tensor/array among ``keys``."""
    for k in keys:
        v = obs.get(k)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return v.size(0) if v.dim() > 1 else 1
        if isinstance(v, np.ndarray) and v.size > 0:
            return v.shape[0] if v.ndim > 1 else 1
    raise ValueError("Could not determine batch size from observations")


class TransformerFeatureExtractor(nn.Module):
    """Per-key observation encoder feeding the transformer actor/critic.

    Keys are processed in sorted order (the intersection of the observation
    space's keys and the keys actually present) so the concatenated layout is
    deterministic across rollout, training, and evaluation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        featurizer_cfg: Optional[dict[str, Any]] = None,
        feature_hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        preprocess_obs_transform: Optional[list[Any]] = None,
        image_encoder_type: Optional[str] = None,
        image_feature_dim: int = 128,
        dinov2_model: Any = None,
        dinov2_processor: Any = None,
        impala_nn_scale: int = 1,
        impala_num_blocks_per_stack: int = 2,
        impala_use_smaller: bool = False,
        impala_output_dim: Optional[int] = None,
        impala_bounded_head: bool = False,
        use_language_embeddings: bool = True,
        lang_embedding_dim: int = 384,
        lang_embedding_device: str = "cpu",
    ) -> None:
        """Build per-key encoders.

        Args:
            observation_space: Dict observation space (a flat space is
                treated as a single ``"obs"`` key).
            featurizer_cfg: Per-key encoder spec (list of MLP dims or module).
            feature_hidden_dims: Optional shared MLP applied after
                concatenation.
            activation: Activation for MLP featurizers.
            use_layer_norm: LayerNorm inside MLP featurizers.
            dropout_rate: Dropout inside MLP featurizers.
            preprocess_obs_transform: Callables applied to raw observations.
            image_encoder_type: ``"impala"`` or ``"dinov2"`` for image keys;
                ``None`` flattens image keys like any other.
            image_feature_dim: Fallback embedding width when the image
                encoder cannot report one.
            dinov2_model: Loaded DINOv2 model (``image_encoder_type="dinov2"``).
            dinov2_processor: Matching image processor.
            impala_nn_scale: IMPALA channel-width multiplier.
            impala_num_blocks_per_stack: IMPALA residual blocks per stack.
            impala_use_smaller: Use the reduced-depth IMPALA variant.
            impala_output_dim: IMPALA embedding projection width.
            impala_bounded_head: Append LayerNorm + tanh to IMPALA embeddings.
            use_language_embeddings: Encode string observations with MiniLM.
            lang_embedding_dim: Width of language embeddings (zero-fill width
                when encoding is unavailable).
            lang_embedding_device: Device for the language encoder.
        """
        super().__init__()
        self.observation_space = observation_space
        self.featurizer_cfg = copy.deepcopy(featurizer_cfg) or {}
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.preprocess_obs_transform = preprocess_obs_transform or []

        self.use_language_embeddings = use_language_embeddings
        self.lang_embedding_dim = lang_embedding_dim
        self.lang_embedding_cache: dict[str, torch.Tensor] = {}
        self.lang_encoder = MiniLMLangEncoder(device=lang_embedding_device) if use_language_embeddings else None

        self.image_encoder_type = image_encoder_type
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        self.dinov2_feature_dim = (
            getattr(dinov2_model.config, "hidden_size", image_feature_dim)
            if dinov2_model is not None
            else image_feature_dim
        )

        if isinstance(observation_space, gym.spaces.Dict):
            self.obs_keys = list(observation_space.spaces.keys())
            self._expected_keys: Optional[set[str]] = set(self.obs_keys)
        else:
            self.obs_keys = ["obs"]
            self._expected_keys = None
        self.image_keys = identify_image_keys(self.obs_keys)
        self.keys_to_ignore: set[str] = set()
        self._key_mismatch_warned = False

        if self.image_encoder_type == "resnet":
            raise ValueError(
                "image_encoder_type='resnet' is not supported; use 'impala' or 'dinov2'."
            )

        # Image keys get dedicated encoders that bypass the per-key MLPs.
        self.image_encoders: Optional[dict[str, nn.Module]] = None
        if self.image_encoder_type in ("small", "impala") and self.image_keys:
            self.image_encoders = {}
            for img_key in self.image_keys:
                encoder = self._build_image_encoder(
                    img_key,
                    impala_nn_scale=impala_nn_scale,
                    impala_num_blocks_per_stack=impala_num_blocks_per_stack,
                    impala_use_smaller=impala_use_smaller,
                    impala_output_dim=impala_output_dim,
                    impala_bounded_head=impala_bounded_head,
                )
                self.image_encoders[img_key] = encoder
                # Register in the config so parameters and width accounting
                # include the encoder.
                self.featurizer_cfg[img_key] = encoder
        elif self.image_encoder_type == "dinov2" and (self.dinov2_model is None or self.dinov2_processor is None):
            logger.warning("image_encoder_type='dinov2' but model/processor missing; image keys will be flattened.")

        self.featurizers = build_featurizers(
            self.featurizer_cfg, observation_space, activation, use_layer_norm, dropout_rate
        )

        self.feature_hidden_dims = feature_hidden_dims
        self._set_feature_mlp(self._infer_feature_dim())

    def _build_image_encoder(
        self,
        key: str,
        *,
        impala_nn_scale: int,
        impala_num_blocks_per_stack: int,
        impala_use_smaller: bool,
        impala_output_dim: Optional[int],
        impala_bounded_head: bool,
    ) -> nn.Module:
        """Build the ``small`` or ``impala`` CNN encoder for one image key."""
        input_shape = self._image_input_shape(key)
        if self.image_encoder_type == "small":
            from isaac_rl_posttraining.modules.cnn import SmallConvEncoder

            return SmallConvEncoder(input_shape)

        from isaac_rl_posttraining.modules.cnn import ImpalaEncoder, SmallerImpalaEncoder

        if impala_use_smaller:
            return SmallerImpalaEncoder(
                input_shape=input_shape,
                nn_scale=impala_nn_scale,
                output_dim=impala_output_dim,
                bounded_head=impala_bounded_head,
            )
        return ImpalaEncoder(
            input_shape=input_shape,
            nn_scale=impala_nn_scale,
            num_blocks_per_stack=impala_num_blocks_per_stack,
            output_dim=impala_output_dim,
            bounded_head=impala_bounded_head,
        )

    @property
    def device(self) -> torch.device:
        """Device of the extractor's parameters (CPU when parameterless)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _image_input_shape(self, key: str) -> tuple[int, ...]:
        """Best-effort image shape for ``key`` from the observation space."""
        default_shape = (128, 128, 3)
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return default_shape
        shape = getattr(self.observation_space.spaces.get(key), "shape", None)
        if shape is None:
            return default_shape
        if len(shape) == 4 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) in (2, 3):
            return tuple(shape)
        logger.warning(f"Irregular image shape for {key!r}: {shape}; assuming {default_shape}")
        return default_shape

    @staticmethod
    def _cfg_width(value: Any) -> int:
        """Feature width of one ``featurizer_cfg`` entry."""
        if hasattr(value, "output_dim"):
            return int(value.output_dim)
        if isinstance(value, (list, tuple)):
            return int(value[-1])
        return int(value)

    def _infer_feature_dim(self) -> Optional[int]:
        """Width of the concatenated per-key features (mirrors ``forward``)."""
        if not isinstance(self.observation_space, gym.spaces.Dict):
            if self.featurizer_cfg:
                return sum(self._cfg_width(v) for v in self.featurizer_cfg.values())
            shape = getattr(self.observation_space, "shape", None)
            return int(np.prod(shape)) if shape is not None else None
        total = sum(self._key_width(k) for k in self.obs_keys)
        return total or None

    def _key_width(self, key: str) -> int:
        """Feature width contributed by one dict key (0 if it yields no features)."""
        cfg = (self.featurizer_cfg or {}).get(key)
        if cfg is not None:
            return self._cfg_width(cfg)
        space = self.observation_space.spaces.get(key)
        if isinstance(space, gym.spaces.Text):
            return self.lang_embedding_dim
        if key in self.image_keys and self.image_encoder_type == "dinov2":
            return self.dinov2_feature_dim
        if space is not None and getattr(space, "shape", None) is not None:
            return int(np.prod(space.shape))
        return 0

    def _keys_to_process(self, obs: dict[str, Any]) -> list[str]:
        """Sorted intersection of expected and provided keys (warn once on drift)."""
        input_keys = set(obs.keys())
        if self._expected_keys is not None:
            missing = self._expected_keys - input_keys
            extra = input_keys - self._expected_keys
            if (missing or extra) and not self._key_mismatch_warned:
                logger.warning(
                    "Observation key mismatch: "
                    f"extra={sorted(extra)} missing={sorted(missing)} expected={sorted(self._expected_keys)}"
                )
                self._key_mismatch_warned = True
            keys = sorted(self._expected_keys & input_keys)
        else:
            keys = sorted(input_keys)
        return [k for k in keys if k not in self.keys_to_ignore]

    def _coerce_value(self, value: Any, batch_size: int) -> Any:
        """Convert an observation value to a float32 device tensor when possible.

        Strings pass through for the language path; unsupported types return
        ``None`` (skipped with a warning at the call site).
        """
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if value and all(isinstance(item, torch.Tensor) for item in value):
                value = torch.stack(value)
            else:
                return None
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            return None
        value = value.to(device=self.device, dtype=torch.float32)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.size(0) != batch_size and value.size(0) == 1:
            value = value.expand(batch_size, *value.shape[1:])
        return value

    def _encode_image(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Encode an image key with its dedicated encoder, or flatten."""
        if self.image_encoders is not None and key in self.image_encoders:
            return self.image_encoders[key](value)
        if (
            self.image_encoder_type == "dinov2"
            and self.dinov2_model is not None
            and self.dinov2_processor is not None
        ):
            return self._compute_dinov2_features(value)
        if value.max() > 1.0:
            value = value / 255.0
        return value.reshape(value.size(0), -1)

    def _compute_dinov2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images with the shared frozen DINOv2 model to ``[B, D]``."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D image tensor for DINOv2 embedding, got {tuple(x.shape)}")
        if x.size(-1) != 3 and x.size(1) in (1, 3):
            x = x.permute(0, 2, 3, 1)

        images = x.detach().to("cpu")
        img_np_list = []
        for i in range(images.size(0)):
            arr = images[i].numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255.0 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
            img_np_list.append(arr)

        with torch.no_grad():
            processed = self.dinov2_processor(images=img_np_list, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(self.device)
            outputs = self.dinov2_model(pixel_values=pixel_values)
        return outputs.pooler_output.to(self.device, dtype=torch.float32)

    def _encode_language(self, key: str, value: str, batch_size: int) -> torch.Tensor:
        """Embed an instruction string (cached), zero-filling on failure."""
        if self.use_language_embeddings and self.lang_encoder is not None:
            try:
                cached = self.lang_embedding_cache.get(value)
                if cached is None:
                    with torch.no_grad():
                        cached = self.lang_encoder.get_lang_emb([value])[0].cpu()
                    self.lang_embedding_cache[value] = cached
                emb = cached.to(self.device).unsqueeze(0)
                if emb.size(0) != batch_size:
                    emb = emb.expand(batch_size, -1)
                return emb
            except Exception as exc:
                logger.warning(f"Language encoding failed for key {key!r}: {exc}")
        return torch.zeros(batch_size, self.lang_embedding_dim, device=self.device, dtype=torch.float32)

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        """Encode a dict observation batch into ``[B, output_dim]`` features.

        Raises:
            ValueError: If ``obs`` is not a dict, the batch size cannot be
                inferred, or no key produced features.
        """
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        keys = self._keys_to_process(obs)

        batch_size = _infer_batch_size(obs, keys)

        feats = []
        for k in keys:
            value = self._coerce_value(obs[k], batch_size)
            if value is None:
                logger.warning(f"Skipping observation key {k!r} with unsupported type {type(obs[k])}")
                continue
            feats.append(self._encode_key(k, value, batch_size))

        if not feats:
            raise ValueError("No valid features extracted from observations")

        obs_flat = torch.cat(feats, dim=1).to(dtype=torch.float32)

        # If the realized width diverges from the configured one (e.g. keys
        # supplied at runtime differ from the space), rebuild the shared MLP
        # to match rather than crashing mid-rollout.
        if self.obs_dim is None or obs_flat.size(-1) != self.obs_dim:
            self._set_feature_mlp(obs_flat.size(-1))
            if self.feature_mlp is not None:
                self.feature_mlp.to(obs_flat.device)

        if self.feature_mlp is not None:
            obs_flat = self.feature_mlp(obs_flat)
        return obs_flat

    def _encode_key(self, key: str, value: Any, batch_size: int) -> torch.Tensor:
        """Encode one coerced observation value into ``[B, D]`` features."""
        if isinstance(value, str):
            return self._encode_language(key, value, batch_size)
        if key in self.image_keys:
            return self._encode_image(key, value)
        v_flat = value.reshape(value.size(0), -1)
        if key in self.featurizers:
            return self.featurizers[key](v_flat)
        return v_flat

    def _set_feature_mlp(self, obs_dim: Optional[int]) -> None:
        """(Re)build the shared feature MLP for a concatenated input width."""
        self.obs_dim = obs_dim
        self.feature_mlp: Optional[nn.Module]
        if self.feature_hidden_dims:
            self.feature_mlp = build_mlp(
                self.obs_dim,
                self.feature_hidden_dims,
                self.activation,
                self.use_layer_norm,
                self.dropout_rate,
            )
            self.output_dim = self.feature_hidden_dims[-1]
        else:
            self.feature_mlp = None
            self.output_dim = self.obs_dim
