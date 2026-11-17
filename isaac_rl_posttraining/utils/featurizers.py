# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-key observation featurizers for dict observation spaces.

Actor/critic networks in :mod:`isaac_rl_posttraining.modules` consume a flat
feature vector. :class:`ObservationFeaturizer` maps each observation key
through its own small encoder (an MLP for vector keys, optionally an IMPALA
CNN for image keys) and concatenates the results.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

import copy

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.utils.networks import build_mlp

__all__ = [
    "identify_image_keys",
    "FlattenFeaturizerWrapper",
    "ImpalaImageFeaturizer",
    "build_impala_featurizers",
    "ObservationFeaturizer",
]

_IMAGE_KEYWORDS = ("image", "rgb", "camera", "vision", "visual")


def identify_image_keys(obs_keys: list[str]) -> list[str]:
    """Return the observation keys that look like image streams by name."""
    return [k for k in obs_keys if any(word in k.lower() for word in _IMAGE_KEYWORDS)]


class FlattenFeaturizerWrapper(nn.Module):
    """Flatten inputs to ``[B, D]`` before delegating to the wrapped module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
        return self.module(x)


class ImpalaImageFeaturizer(nn.Module):
    """IMPALA CNN featurizer producing a ``[B, D]`` embedding from images.

    Accepts ``[B, H, W, C]``, ``[B, C, H, W]``, or single unbatched frames;
    layout conversion and value normalization happen inside the encoder.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_scale: int = 1,
        num_blocks_per_stack: int = 2,
        use_smaller: bool = False,
        requires_grad: bool = True,
        output_dim: Optional[int] = None,
        bounded_head: bool = False,
    ) -> None:
        """Create the underlying encoder.

        Args:
            input_shape: Image shape, ``(H, W, C)`` or ``(C, H, W)``.
            nn_scale: Channel-width multiplier for the encoder stacks.
            num_blocks_per_stack: Residual blocks per stack (full variant).
            use_smaller: Use the reduced-depth encoder variant.
            requires_grad: Train the encoder (``False`` freezes it).
            output_dim: Optional linear projection width for the embedding.
        """
        super().__init__()
        from isaac_rl_posttraining.modules.cnn import ImpalaEncoder, SmallerImpalaEncoder

        if use_smaller:
            self.encoder = SmallerImpalaEncoder(
                bounded_head=bounded_head,
                input_shape=input_shape, nn_scale=nn_scale, output_dim=output_dim
            )
        else:
            self.encoder = ImpalaEncoder(
                bounded_head=bounded_head,
                input_shape=input_shape,
                nn_scale=nn_scale,
                num_blocks_per_stack=num_blocks_per_stack,
                output_dim=output_dim,
            )

        for p in self.encoder.parameters():
            p.requires_grad = requires_grad
        if not requires_grad:
            self.encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("ImpalaImageFeaturizer expects torch.Tensor inputs")
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
        return self.encoder(x.to(self.device))


def build_impala_featurizers(
    observation_space: gym.Space,
    keys: Optional[list[str]] = None,
    nn_scale: int = 1,
    num_blocks_per_stack: int = 2,
    use_smaller: bool = False,
    requires_grad: bool = True,
    output_dim: Optional[int] = None,
    bounded_head: bool = False,
) -> dict[str, ImpalaImageFeaturizer]:
    """Build IMPALA featurizers for the image-like keys of an observation space.

    Args:
        observation_space: The (usually ``Dict``) observation space.
        keys: Explicit keys to build encoders for; auto-detected when ``None``.
        nn_scale: Channel-width multiplier.
        num_blocks_per_stack: Residual blocks per stack (full variant).
        use_smaller: Use the reduced-depth encoder variant.
        requires_grad: Train the encoders.
        output_dim: Optional projection width for each embedding.

    Returns:
        Mapping from observation key to featurizer, suitable for the
        ``featurizer`` field of an actor/critic config.
    """
    if hasattr(observation_space, "spaces"):
        obs_keys = list(observation_space.spaces.keys())
    else:
        obs_keys = ["obs"]

    image_keys = keys if keys is not None else identify_image_keys(obs_keys)
    featurizers: dict[str, ImpalaImageFeaturizer] = {}
    for k in image_keys:
        if isinstance(observation_space, gym.spaces.Dict):
            input_shape = observation_space.spaces[k].shape
        else:
            input_shape = observation_space.shape
        featurizers[k] = ImpalaImageFeaturizer(
            input_shape=input_shape,
            nn_scale=nn_scale,
            num_blocks_per_stack=num_blocks_per_stack,
            use_smaller=use_smaller,
            requires_grad=requires_grad,
            output_dim=output_dim,
            bounded_head=bounded_head,
        )
    return featurizers


class ObservationFeaturizer(nn.Module):
    """Flatten and featurize (possibly dict) observations into ``[B, D]``.

    When a ``featurizer_cfg`` is given, each configured key runs through its
    own encoder — a list/tuple entry builds an MLP over the flattened key,
    an ``nn.Module`` entry is used directly — and the per-key features are
    concatenated. Without a config, keys are flattened and concatenated
    as-is (missing keys are zero-filled to keep the layout stable).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        featurizer_cfg: Optional[dict[str, Union[list, tuple, nn.Module]]] = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        image_encoder_type: Optional[str] = None,
        impala_nn_scale: int = 1,
        impala_num_blocks_per_stack: int = 2,
        impala_use_smaller: bool = False,
        impala_output_dim: Optional[int] = None,
        impala_bounded_head: bool = False,
        small_bottleneck_dim: int = 50,
    ) -> None:
        """Build per-key featurizers from the config.

        Args:
            observation_space: The observation space to featurize.
            featurizer_cfg: Per-key encoder spec (see class docstring).
            activation: Activation for MLP featurizers.
            use_layer_norm: LayerNorm inside MLP featurizers.
            dropout_rate: Dropout inside MLP featurizers.
            image_encoder_type: ``"impala"`` adds IMPALA encoders for image
                keys not already covered by ``featurizer_cfg``.
            impala_nn_scale: IMPALA channel-width multiplier.
            impala_num_blocks_per_stack: IMPALA residual blocks per stack.
            impala_use_smaller: Use the reduced-depth IMPALA variant.
            impala_output_dim: Projection width for IMPALA embeddings.
            small_bottleneck_dim: Bottleneck width of the ``"small"``
                per-image-key conv encoders.
        """
        super().__init__()
        self.observation_space = observation_space
        self.featurizer_cfg = copy.deepcopy(featurizer_cfg)

        if image_encoder_type == "small":
            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError("'small' encoder requires a Dict observation space")
            from isaac_rl_posttraining.modules.cnn import SmallConvEncoder

            self._register_image_encoders(
                {
                    key: SmallConvEncoder(
                        observation_space.spaces[key].shape,
                        bottleneck_dim=small_bottleneck_dim,
                    )
                    for key in identify_image_keys(
                        list(observation_space.spaces.keys())
                    )
                }
            )
        elif image_encoder_type == "impala":
            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError("IMPALA encoder requires a Dict observation space")
            impala_featurizers = build_impala_featurizers(
                observation_space=observation_space,
                keys=None,
                nn_scale=impala_nn_scale,
                num_blocks_per_stack=impala_num_blocks_per_stack,
                use_smaller=impala_use_smaller,
                requires_grad=True,
                output_dim=impala_output_dim,
                bounded_head=impala_bounded_head,
            )
            self._register_image_encoders(impala_featurizers)
        else:
            self.featurizers = nn.ModuleDict() if self.featurizer_cfg else None

        if self.featurizer_cfg:
            self._build_configured_featurizers(activation, use_layer_norm, dropout_rate)

        self._flatten_keys: Optional[list[str]] = self._resolve_flatten_keys()
        self._output_dim = self._compute_output_dim()

    def _resolve_flatten_keys(self) -> Optional[list[str]]:
        """Keys concatenated by :meth:`flatten_obs`, in order (``None`` for a non-dict space)."""
        if self.featurizer_cfg:
            return list(self.featurizer_cfg.keys())
        if isinstance(self.observation_space, gym.spaces.Dict):
            return [
                k
                for k, space in self.observation_space.spaces.items()
                if getattr(space, "shape", None) is not None
            ]
        return None

    def _register_image_encoders(self, encoders: Mapping[str, nn.Module]) -> None:
        """Install image encoders and record their output widths in ``featurizer_cfg``."""
        if self.featurizer_cfg is None:
            self.featurizer_cfg = {}
        self.featurizers = nn.ModuleDict()
        for key, encoder in encoders.items():
            self.featurizer_cfg[key] = encoder.output_dim
            self.featurizers[key] = encoder

    def _build_configured_featurizers(
        self, activation: str, use_layer_norm: bool, dropout_rate: float
    ) -> None:
        """Build an encoder for every ``featurizer_cfg`` key not already registered."""
        for key, value in self.featurizer_cfg.items():
            if key in self.featurizers:
                continue
            if isinstance(value, (list, tuple)):
                obs_dim = int(np.prod(self.observation_space.spaces[key].shape))
                mlp = build_mlp(
                    obs_dim, value, activation, use_layer_norm, dropout_rate
                )
                self.featurizers[key] = FlattenFeaturizerWrapper(mlp)
            elif isinstance(value, nn.Module):
                self.featurizers[key] = value
            elif isinstance(value, int):
                # Placeholder width recorded for an already-registered module.
                continue
            else:
                raise ValueError(
                    f"Featurizer for key {key!r} must be list/tuple or nn.Module"
                )

    def _create_example_obs(self) -> Union[dict[str, torch.Tensor], torch.Tensor, None]:
        """Create a zero observation matching the space, for shape probing."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            keys = self._flatten_keys if self.featurizer_cfg else list(self.observation_space.spaces.keys())
            example_obs = {}
            for key in keys:
                space = self.observation_space.spaces.get(key)
                shape = getattr(space, "shape", None)
                if shape is None:
                    continue
                example_obs[key] = torch.zeros((1,) + shape, dtype=torch.float32)
            return example_obs or None
        shape = getattr(self.observation_space, "shape", None)
        if shape is None:
            return None
        return torch.zeros((1,) + shape, dtype=torch.float32)

    def _compute_output_dim(self) -> int:
        """Infer the width of the concatenated feature vector."""
        if self.featurizer_cfg:
            example_obs = self._create_example_obs()
            if example_obs is None:
                return 0
            with torch.no_grad():
                flattened = self.flatten_obs(example_obs)
            return int(flattened.shape[-1]) if flattened.numel() > 0 else 0
        if isinstance(self.observation_space, gym.spaces.Dict):
            if self._flatten_keys is None:
                return 0
            return sum(int(np.prod(self.observation_space.spaces[k].shape)) for k in self._flatten_keys)
        return int(np.prod(self.observation_space.shape))

    @property
    def output_dim(self) -> int:
        """Width of the concatenated feature vector."""
        return self._output_dim

    def flatten_obs(
        self,
        obs: Union[dict[str, torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Featurize/flatten an observation batch into a single tensor.

        Args:
            obs: Dict observation batch or a raw tensor.
            device: Optional device for the resulting features.

        Returns:
            Tensor of shape ``[B, output_dim]``.
        """
        if isinstance(obs, dict):
            if self.featurizer_cfg:
                return self._featurize_configured_keys(obs, device)
            if self._flatten_keys:
                return self._flatten_space_keys(obs, device)
            feats = [
                _to_device(_flatten_batch(v), device)
                for v in obs.values()
                if v is not None
            ]
            return torch.cat(feats, dim=-1) if feats else torch.empty(0)

        return _to_device(_flatten_raw_obs(obs), device)

    def _featurize_configured_keys(
        self, obs: dict[str, torch.Tensor], device: Optional[torch.device]
    ) -> torch.Tensor:
        """Run each configured key through its encoder (or flatten it) and concatenate."""
        feats = []
        for k in self._flatten_keys:
            if k not in obs:
                continue
            v = obs[k]
            if k in self.featurizers:
                if device is not None:
                    v = v.to(device).float()
                feats.append(self.featurizers[k](v))
            else:
                feats.append(_to_device(_flatten_batch(v), device))
        if not feats:
            raise ValueError(
                "No featurizable keys present in observation dict. "
                f"Keys present: {list(obs.keys())}, "
                f"expected: {list(self.featurizers.keys()) if self.featurizers else None}"
            )
        return torch.cat(feats, dim=-1)

    def _flatten_space_keys(
        self, obs: dict[str, torch.Tensor], device: Optional[torch.device]
    ) -> torch.Tensor:
        """Flatten the space's keys in order, zero-filling missing ones."""
        feats = []
        batch_size: Optional[int] = None
        for k in self._flatten_keys:
            v = obs.get(k)
            flat_dim = int(np.prod(self.observation_space.spaces[k].shape))
            if v is None:
                if batch_size is None:
                    batch_size = _infer_batch_size(obs)
                feats.append(_to_device(torch.zeros(batch_size, flat_dim), device))
            else:
                feats.append(_to_device(_flatten_batch(v), device))
        return torch.cat(feats, dim=-1) if feats else torch.empty(0)


def _flatten_batch(v: torch.Tensor) -> torch.Tensor:
    """Flatten to ``[B, -1]``; a 1-D tensor becomes a single-row batch."""
    return v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)


def _flatten_raw_obs(obs: torch.Tensor) -> torch.Tensor:
    """Present a raw tensor observation as ``[B, D]``."""
    if obs.dim() > 2:
        return obs.reshape(obs.size(0), -1)
    if obs.dim() == 1:
        return obs.unsqueeze(0)
    if obs.dim() == 2 and obs.size(1) == 1 and obs.size(0) != 1:
        # A column vector is (features, 1); present it as (1 x features).
        return obs.t()
    return obs


def _to_device(t: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
    return t.to(device) if device is not None else t


def _infer_batch_size(obs: dict[str, torch.Tensor]) -> int:
    """Batch size of the first present value (1 when none is present)."""
    for other in obs.values():
        if other is not None:
            return (other.size(0) if other.dim() > 1 else 1) or 1
    return 1
