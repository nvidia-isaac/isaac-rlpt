# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a frozen policy reads and emits, known before it loads.

The launch plan compares the simulator's declared observation and action with
the policy's. A policy's widths usually live in its checkpoint; the ones that
can be read cheaply (a config field, a JSON file next to the weights) are
gathered here into a :class:`PolicyGeometry`, and everything else is left
unknown with the reason. Nothing here imports a model stack or reads weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

__all__ = ["PolicyGeometry", "policy_geometry"]


@dataclass(frozen=True)
class PolicyGeometry:
    """The shapes a frozen policy works with, as far as its config and checkpoint metadata say.

    Attributes:
        source: Where the numbers come from (for the plan line).
        state_width: Width of the state vector the policy slices.
        action_width: Width of one decoded action step.
        decoded_horizon: Action steps one decode emits.
        latent_horizon: Steering latents one decode accepts (the flow-noise window).
        latent_width: Widest steering latent the flow head accepts per step.
        exact_latent_width: The head accepts only latents exactly ``latent_width`` wide.
        horizon_key: How the plan names ``latent_horizon`` in a message.
        resizes_images: The policy resizes camera frames itself, so any frame size fits.
        unknown: Why the sizes left ``None`` are known only once the policy loads.
    """

    source: str
    state_width: int | None = None
    action_width: int | None = None
    decoded_horizon: int | None = None
    latent_horizon: int | None = None
    latent_width: int | None = None
    exact_latent_width: bool = False
    horizon_key: str = "base_model.action_horizon"
    resizes_images: bool = False
    unknown: str = ""


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _mock_geometry(cfg: DictConfig) -> PolicyGeometry:
    """The mock base model decodes ``base_model.action_dim``-wide chunks of ``base_model.action_horizon`` steps."""
    width = int(OmegaConf.select(cfg, "base_model.action_dim", default=None) or 35)
    horizon = int(OmegaConf.select(cfg, "base_model.action_horizon", default=None) or 40)
    return PolicyGeometry(
        source="base model mock config",
        action_width=width,
        decoded_horizon=horizon,
        latent_horizon=horizon,
        latent_width=width,
        exact_latent_width=True,
        resizes_images=True,
        unknown="base model mock reads a state of any width",
    )


def _config_geometry(cfg: DictConfig, name: str) -> PolicyGeometry:
    """A base model whose checkpoint is opaque before load: only the ``base_model`` fields the config pins."""
    width = _int_or_none(OmegaConf.select(cfg, "base_model.action_dim", default=None))
    horizon = _int_or_none(
        OmegaConf.select(cfg, "base_model.action_horizon", default=None)
    )
    unset = [
        key
        for key, value in (
            ("base_model.action_dim", width),
            ("base_model.action_horizon", horizon),
        )
        if value is None
    ]
    unknown = f"base model {name} states its sizes once loaded" + (
        f" ({', '.join(unset)} unset)" if unset else ""
    )
    return PolicyGeometry(
        source=f"base model {name} config",
        action_width=width,
        decoded_horizon=horizon,
        latent_horizon=horizon,
        unknown=unknown,
    )


def policy_geometry(cfg: DictConfig, *, dataset_adapter: Any = None) -> PolicyGeometry:
    """The selected base model's geometry from its config and checkpoint metadata, without loading it.

    ``dataset_adapter`` is the GR00T embodiment adapter the launcher would
    hand the base model (its layout applies to a checkpoint that bakes none).
    """
    from isaac_rl_posttraining.policies import is_gr00t_base_model

    name = str(cfg.base_model.name)
    if name == "mock":
        return _mock_geometry(cfg)
    if is_gr00t_base_model(name):
        from isaac_rl_posttraining.policies.gr00t.checkpoint_layout import (
            DEFAULT_EMBODIMENT_TAG,
            checkpoint_geometry,
        )

        tag = OmegaConf.select(cfg, "base_model.embodiment_tag", default=None)
        return checkpoint_geometry(
            OmegaConf.select(cfg, "base_model.ckpt", default=None),
            DEFAULT_EMBODIMENT_TAG if tag is None else str(tag),
            dataset_dir=OmegaConf.select(cfg, "base_model.dataset_dir", default=None),
            multitask=OmegaConf.select(cfg, "base_model.multitask_dirs", default=None)
            is not None,
            dataset_adapter=dataset_adapter,
        )
    return _config_geometry(cfg, name)
