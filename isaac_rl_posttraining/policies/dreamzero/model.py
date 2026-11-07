# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from RLinf (Apache-2.0) and modified; see THIRD_PARTY_NOTICES.md.
#
# SPDX-FileCopyrightText: Copyright 2026 The RLinf Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading a frozen DreamZero (WAN world-action) model for in-process inference.

Builds the model from a **bundled SFT checkpoint** — a directory holding
``config.json`` plus ``model.safetensors`` (or a sharded index). Component
cold-start (assembling WAN/T5/VAE from their separate releases) is a
training-pipeline concern and is deliberately not supported here: this repo
only ever steers finished checkpoints.

Ported from RLinf's validated ``get_model`` (``rlinf/models/embodiment/
dreamzero/__init__.py``), trimmed of its FSDP/SFT branches; the DSRL noise
patch and the memory techniques it installs live in
:mod:`~isaac_rl_posttraining.policies.dreamzero.patch` and
:mod:`~isaac_rl_posttraining.policies.dreamzero.memory`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from isaac_rl_posttraining.policies.dreamzero.layouts import (
    EmbodimentSpec,
    get_embodiment_spec,
)
from isaac_rl_posttraining.policies.dreamzero.memory import park_text_encoder
from isaac_rl_posttraining.policies.dreamzero.patch import (
    apply_dsrl_noise_patch,
    compile_causal_attention_hot_paths,
    raise_dynamo_limits,
)
from isaac_rl_posttraining.policies.dreamzero.transforms import (
    build_composed_transform,
    load_metadata,
)

__all__ = ["LoadedDreamZero", "load_dreamzero_model"]

#: Real-DiT-compute counts with a pinned step mask in the WAN head; 16 = the
#: full scheduler (no masking). Chosen via the NUM_DIT_STEPS env var read at
#: head construction (wan_flow_matching_action_tf.py).
_VALID_NUM_DIT_STEPS = (5, 6, 7, 8, 16)


class LoadedDreamZero:
    """Everything :class:`~.policy.DreamZeroLocalPolicy` needs, in one place.

    Attributes:
        model: The frozen groot ``VLA`` (IdentityBackbone + WANPolicyHead),
            in eval mode on the target device.
        transforms: The checkpoint-faithful ``ComposedModalityTransform``,
            metadata-bound and in eval mode.
        metadata: The checkpoint's ``DatasetMetadata`` (normalization stats,
            per-key action shapes).
        spec: The embodiment layout the checkpoint was trained with.
        head_action_dim: The WAN head's latent action width (32) — the DSRL
            steering width.
        head_action_horizon: The head's action-chunk length (16 / 24).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Any,
        metadata: Any,
        spec: EmbodimentSpec,
        head_action_dim: int,
        head_action_horizon: int,
    ) -> None:
        self.model = model
        self.transforms = transforms
        self.metadata = metadata
        self.spec = spec
        self.head_action_dim = head_action_dim
        self.head_action_horizon = head_action_horizon


def _read_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Read and adapt the checkpoint's ``config.json`` for inference loading."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"DreamZero checkpoint config not found: {config_path}. "
            "base_model.ckpt must point at a bundled SFT checkpoint "
            "(config.json + model.safetensors)."
        )
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    head_config = config_dict.get("action_head_cfg", {}).get("config")
    if not isinstance(head_config, dict):
        raise ValueError(
            f"{config_path} carries no action_head_cfg.config mapping; this is "
            "not a DreamZero checkpoint config."
        )
    # The bundled safetensors carry every weight: LoRA is merged (or absent)
    # and no component downloads should happen at construction.
    head_config["defer_lora_injection"] = False
    head_config["skip_component_loading"] = True
    return config_dict


def _load_bundled_weights(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Load ``model.safetensors`` (single file or sharded index) into the model."""
    from safetensors.torch import load_file

    single = checkpoint_dir / "model.safetensors"
    index = checkpoint_dir / "model.safetensors.index.json"
    state_dict: dict[str, torch.Tensor] = {}
    if index.is_file():
        with open(index, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        for shard_file in sorted(set(weight_map.values())):
            state_dict.update(load_file(str(checkpoint_dir / shard_file)))
    elif single.is_file():
        state_dict.update(load_file(str(single)))
    else:
        raise FileNotFoundError(
            f"No model.safetensors under {checkpoint_dir}. base_model.ckpt "
            "must point at a bundled SFT checkpoint; component cold-start is "
            "not supported."
        )
    # LoRA-era checkpoints store the frozen weight under `.base_layer.`.
    if any(".base_layer." in key for key in state_dict):
        state_dict = {
            key.replace(".base_layer.", "."): value
            for key, value in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"DreamZero checkpoint {checkpoint_dir} is missing "
            f"{len(missing)} model weights (first few: {sorted(missing)[:5]}); "
            "the checkpoint does not match this groot version."
        )
    if unexpected:
        logger.warning(
            f"DreamZero checkpoint carries {len(unexpected)} unused keys "
            f"(first few: {sorted(unexpected)[:5]}); continuing."
        )


def _set_num_dit_steps(num_dit_steps: int) -> None:
    """Validate ``num_dit_steps`` and export it for the WAN head."""
    if int(num_dit_steps) not in _VALID_NUM_DIT_STEPS:
        raise ValueError(
            f"base_model.num_dit_steps must be one of {_VALID_NUM_DIT_STEPS} "
            f"(pinned step masks in the WAN head); got {num_dit_steps}."
        )
    # The head picks its dit_step_mask from this env var at construction;
    # the mask table stays single-sourced in groot.
    os.environ["NUM_DIT_STEPS"] = str(int(num_dit_steps))


def _apply_target_video(
    config_dict: dict[str, Any],
    head_config: dict[str, Any],
    target_video: tuple[int, int],
) -> None:
    """Override the video resolution in both the top-level and head configs."""
    height, width = target_video
    config_dict["target_video_height"] = int(height)
    config_dict["target_video_width"] = int(width)
    head_config["target_video_height"] = int(height)
    head_config["target_video_width"] = int(width)


def _check_head_geometry(head_config: dict[str, Any]) -> None:
    """Require the geometry keys the head config must carry."""
    # The head config is the authoritative geometry (it is what WANPolicyHead
    # consumes); top-level copies vary across checkpoint producers (the
    # released 14B omits the max_* duplicates that RLinf-SFT configs carry).
    for required in ("action_horizon", "action_dim", "max_state_dim", "max_action_dim"):
        if head_config.get(required) is None:
            raise ValueError(
                f"DreamZero checkpoint action_head_cfg.config is missing "
                f"{required!r}; this is not a complete SFT checkpoint config."
            )


def load_dreamzero_model(
    checkpoint_dir: str,
    *,
    embodiment_tag: str,
    metadata_json_path: str | None,
    tokenizer_path: str,
    device: torch.device | str,
    text_encoder_cpu_offload: bool,
    torch_compile: bool,
    num_dit_steps: int | None,
    target_video: tuple[int, int] | None,
    precision: torch.dtype = torch.bfloat16,
) -> LoadedDreamZero:
    """Load a frozen DreamZero checkpoint with the DSRL noise seam installed.

    Args:
        checkpoint_dir: Bundled SFT checkpoint directory.
        embodiment_tag: groot embodiment tag (e.g. ``libero_sim``).
        metadata_json_path: The dataset ``metadata.json``; ``None`` uses the
            checkpoint's ``experiment_cfg/metadata.json``.
        tokenizer_path: The umt5 tokenizer (HF id or local path).
        device: Target device for the frozen model.
        text_encoder_cpu_offload: Park umt5-xxl on CPU with per-prompt
            embedding caching (~10.6 GiB of VRAM back on 14B bundles).
        torch_compile: Compile the causal-attention hot paths
            (``reduce-overhead``); disable when debugging or on incompatible
            stacks.
        num_dit_steps: Real DiT computes per 16-step denoise (5-8, or 16 for
            the full scheduler); ``None`` keeps the model default (8).
        target_video: Optional ``(height, width)`` override of the head's
            video resolution (the 5B/LIBERO recipe uses 160x320).
        precision: Frozen-model dtype; bf16 is what the recipes validate.

    Returns:
        The loaded bundle (model + transforms + metadata + spec + geometry).
    """
    spec = get_embodiment_spec(embodiment_tag)
    checkpoint = Path(checkpoint_dir).expanduser()

    raise_dynamo_limits()
    apply_dsrl_noise_patch()
    if torch_compile:
        compile_causal_attention_hot_paths()
    if num_dit_steps is not None:
        _set_num_dit_steps(num_dit_steps)

    config_dict = _read_checkpoint_config(checkpoint)
    head_config = config_dict["action_head_cfg"]["config"]
    if target_video is not None:
        _apply_target_video(config_dict, head_config, target_video)
    _check_head_geometry(head_config)
    head_action_horizon = int(head_config["action_horizon"])
    head_action_dim = int(head_config["action_dim"])
    # Keep the top-level pair (read by groot's VLA for its output validation)
    # in lockstep with the head.
    config_dict["action_horizon"] = head_action_horizon
    config_dict["action_dim"] = head_action_dim

    if metadata_json_path is None:
        metadata_json_path = str(checkpoint / "experiment_cfg" / "metadata.json")
    metadata = load_metadata(metadata_json_path, spec)

    # 512 is groot's own tokenizer default; SFT configs may legitimately
    # omit max_seq_len.
    max_seq_len = head_config.get("max_seq_len") or config_dict.get("max_seq_len") or 512
    transforms = build_composed_transform(
        spec,
        tokenizer_path,
        action_horizon=head_action_horizon,
        max_state_dim=int(head_config["max_state_dim"]),
        max_action_dim=int(head_config["max_action_dim"]),
        max_seq_len=int(max_seq_len),
    )
    transforms.set_metadata(metadata)
    transforms.eval()

    from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

    class _DreamZeroVLAConfig(VLAConfig):
        model_type = "dreamzero"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    logger.info(f"Loading DreamZero checkpoint from {checkpoint} ...")
    model = VLA(_DreamZeroVLAConfig(**config_dict))
    _load_bundled_weights(model, checkpoint)

    for param in model.parameters():
        param.requires_grad_(False)
    action_head = model.action_head
    # The head's inference path branches on these TensorRT handles; a python
    # construction (no TRT export) leaves them undefined.
    if not hasattr(action_head, "trt_engine"):
        action_head.trt_engine = None
    if not hasattr(action_head, "trt_context"):
        action_head.trt_context = None

    model = model.to(precision)
    if text_encoder_cpu_offload:
        # Owns device placement: the encoder pins to CPU first so the
        # ~10.6 GiB tower never transits GPU memory during the move.
        park_text_encoder(model, torch.device(device))
    else:
        model = model.to(torch.device(device))
    model = model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"DreamZero ready: {total_params / 1e9:.2f}B params, embodiment "
        f"{spec.tag!r}, chunk {head_action_horizon} x latent {head_action_dim}, "
        f"{precision}, text encoder "
        f"{'CPU-parked' if text_encoder_cpu_offload else 'on-device'}."
    )
    return LoadedDreamZero(
        model=model,
        transforms=transforms,
        metadata=metadata,
        spec=spec,
        head_action_dim=head_action_dim,
        head_action_horizon=head_action_horizon,
    )
