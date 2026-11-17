# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema of the ``base_model`` config group: the union of every base model's fields.

Each ``configs/base_model/<name>.yaml`` fills the fields its base model consumes and
leaves the rest ``None``; a field missing here is dropped from the composed config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["BaseModelConfig"]


@dataclass
class BaseModelConfig:
    """Base-model selection group (``base_model`` in the defaults tree).

    One schema serves every base model; each ``configs/base_model/<name>.yaml``
    fills the fields its base model consumes and leaves the rest ``None``.
    """

    name: str = field(
        default="gr00t",
        metadata={"help": "Registry name of the base model (mock, gr00t, pi0, dreamzero, dreamzero_server)"},
    )
    # Checkpoint-loaded base models (gr00t, pi0).
    ckpt: Optional[str] = field(
        default=None, metadata={"help": "Base checkpoint directory (HF layout)"}
    )
    backbone_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional local directory or HF id overriding the GR00T VLM backbone"},
    )
    bc_ckpt: Optional[str] = field(
        default=None, metadata={"help": "Optional task-BC checkpoint loaded on top of the base"}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "Dataset directory for modality config and stats"}
    )
    multitask_dirs: Optional[Any] = field(
        default=None, metadata={"help": "Multi-task dataset roots (list, brace/space list, or parent dir)"}
    )
    task_dir: Optional[str] = field(
        default=None, metadata={"help": "Task dataset dir supplying the steered task's language"}
    )
    embodiment_tag: Optional[str] = field(
        default=None,
        metadata={
            "help": "Embodiment tag the checkpoint was trained under; with no dataset "
            "configured, its baked processor config supplies modality + statistics"
        },
    )
    dataset_adapter: str = field(
        default="g1_wbc",
        metadata={
            "help": "Embodiment adapter mapping a dataset's raw episodes to the buffer's "
            "contiguous state/action arrays and modality: 'g1_wbc' (G1 whole-body-control) "
            "or 'none' (raw observation.state/action passthrough). Ignored on the "
            "datasetless baked path (no dataset_dir/multitask_dirs)."
        },
    )
    task_instruction: Optional[str] = field(
        default=None,
        metadata={"help": "Default task language when observations carry no prompt (baked path)"},
    )
    # GR00T N1.5 checkpoints (model_type gr00t_n1_5; self-describing, no dataset).
    data_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "GR00T N1.5 only (required there): the data config the checkpoint was "
            "finetuned with -- a gr00t DATA_CONFIG_MAP key or a 'module:Class' path"
        },
    )
    denoising_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "GR00T N1.5 only: flow denoising steps (action_head.num_inference_timesteps); "
            "null keeps the checkpoint's value"
        },
    )
    # pi0 fine-tuning (policies/pi0/flow_sde.py).
    openpi_config: Optional[str] = field(
        default=None,
        metadata={"help": "pi0 fine-tuning only: the openpi training config the checkpoint was trained with"},
    )
    observation_keys: Optional[dict[str, str]] = field(
        default=None,
        metadata={
            "help": "pi0 fine-tuning only: observation key (video.<cam> / state.<part>) to the openpi "
            "input key the checkpoint's transforms expect"
        },
    )
    # Server-backed base models (dreamzero_server).
    server_url: Optional[str] = field(
        default=None, metadata={"help": "Policy-server WebSocket URL (ws://host:port)"}
    )
    # In-process DreamZero (dreamzero).
    metadata_json_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset metadata.json (normalization stats, action shapes); "
            "null uses <ckpt>/experiment_cfg/metadata.json"
        },
    )
    tokenizer_path: Optional[str] = field(
        default=None, metadata={"help": "Text-encoder tokenizer (HF id or local path)"}
    )
    text_encoder_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Park the umt5-xxl text encoder (~10.6 GiB) on CPU with "
            "per-prompt embedding caching"
        },
    )
    torch_compile: bool = field(
        default=True,
        metadata={"help": "torch.compile the WAN causal-attention hot paths"},
    )
    num_dit_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Real DiT computes per 16-step denoise (5-8, or 16 for the "
            "full scheduler); null keeps the model default (8)"
        },
    )
    predict_micro_batch_size: int = field(
        default=1,
        metadata={"help": "Env rows per DreamZero model call (keep 1; see base model)"},
    )
    target_video_height: Optional[int] = field(
        default=None, metadata={"help": "Override of the head's video height"}
    )
    target_video_width: Optional[int] = field(
        default=None, metadata={"help": "Override of the head's video width"}
    )
    # Shape overrides for base models that cannot read them from a checkpoint.
    action_horizon: Optional[int] = field(
        default=None, metadata={"help": "Environment steps produced per inference call"}
    )
    action_dim: Optional[int] = field(
        default=None, metadata={"help": "Raw action width per step"}
    )
