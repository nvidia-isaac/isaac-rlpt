# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frozen policy and image encoder a steering or edit rollout policy decodes with."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

from isaac_rl_posttraining.runtimes.base import PolicyContext

__all__ = ["FrozenBaseModelSpec", "build_dino_wrapper"]

#: Base models built in this policy-server process, by spec key and device.
_SERVED_BASE_MODELS: dict[tuple[str, str], Any] = {}


def build_dino_wrapper(
    device: torch.device,
    *,
    dino_model_name: str | None,
    dino_wrapper: Any,
    image_keys: Sequence[str],
) -> Any:
    """The DINO observation encoder a rollout policy embeds camera frames with.

    Either a wrapper already loaded by the controller is reused, or the named
    model is loaded onto ``device``; ``None`` when the learner takes no DINO
    embedding.
    """
    if dino_model_name is None:
        return dino_wrapper
    if dino_wrapper is not None:
        raise ValueError("provide dino_model_name or dino_wrapper, but not both")
    from transformers import AutoImageProcessor, AutoModel

    from isaac_rl_posttraining.envs.dino_wrapper import DinoBatchObservationEncoder

    model = AutoModel.from_pretrained(dino_model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(dino_model_name)
    return DinoBatchObservationEncoder(model, processor, image_keys=list(image_keys))


@dataclass(frozen=True)
class FrozenBaseModelSpec:
    """The frozen policy a rollout worker decodes with.

    Either ``config`` (the resolved run config) rebuilds the base model inside the
    worker's own process, or ``prebuilt`` hands over the controller's instance
    when the workers share its process (the native runtime). A policy-server
    process builds a config's base model once for all the policies it serves.
    """

    config: dict[str, Any] | None = None
    prebuilt: Any = None

    def __post_init__(self) -> None:
        if (self.config is None) == (self.prebuilt is None):
            raise ValueError("provide exactly one of config or prebuilt")

    def key(self) -> str:
        """Equal for specs that build the same base model: the same prebuilt object, or equal configs."""
        if self.prebuilt is not None:
            return f"prebuilt:{id(self.prebuilt)}"
        text = json.dumps(self.config, sort_keys=True, default=repr)
        return "config:" + hashlib.sha256(text.encode()).hexdigest()

    def __call__(self, context: PolicyContext):
        if self.prebuilt is not None:
            return self.prebuilt
        if context.placement != "policy_server":
            return self._build(context)
        # Policies served by one server process (a run's training and evaluation
        # policies, or a base model alone) share one copy of the base model.
        key = (self.key(), str(context.device))
        base_model = _SERVED_BASE_MODELS.get(key)
        if base_model is None:
            base_model = _SERVED_BASE_MODELS[key] = self._build(context)
        return base_model

    def _build(self, context: PolicyContext):
        from isaac_rl_posttraining.envs.embodiments import get_dataset_adapter
        from isaac_rl_posttraining.policies import get_base_model, is_gr00t_base_model

        cfg = OmegaConf.create(copy.deepcopy(self.config))
        base_model_name = str(OmegaConf.select(cfg, "base_model.name"))
        base_model = get_base_model(base_model_name)
        extra: dict[str, Any] = {}
        if is_gr00t_base_model(base_model_name):
            extra["dataset_adapter"] = get_dataset_adapter(
                OmegaConf.select(cfg, "base_model.dataset_adapter", default="g1_wbc")
            )
        return base_model.from_dsrl_config(cfg, context.device, **extra)
