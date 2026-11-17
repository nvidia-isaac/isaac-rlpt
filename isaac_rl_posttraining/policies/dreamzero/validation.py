# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time geometry validation for DreamZero DSRL runs.

The SAC noise must exactly match the WAN head's action latent
``[B, action_horizon, action_dim]``; a mismatch otherwise only surfaces as a
ValueError minutes into the first rollout — after the model spent minutes
loading. Validate everything at construction instead (fail before the run,
not after).
"""

from __future__ import annotations

__all__ = ["validate_dreamzero_dsrl_geometry"]


def validate_dreamzero_dsrl_geometry(
    *,
    head_action_dim: int,
    head_action_horizon: int,
    env_action_dim: int,
    noise_dim: int,
    chunk_size: int | None,
    action_exec_len: int,
    base_model_action_horizon: int | None,
    base_model_action_dim: int | None,
) -> None:
    """Check the DSRL steering geometry against the loaded checkpoint.

    Pins the RLinf-validated recipe shape: SAC emits one latent per decision
    in the WAN head's padded action space (``noise_dim == head action_dim``,
    32 for the released checkpoints), tiled across the full chunk
    (``chunk_size`` unset), and the environment executes the whole chunk per
    decision (``action_exec_len == head action_horizon``).

    Args:
        head_action_dim: The WAN action head's latent action width (from the
            checkpoint config; 32 for the released checkpoints).
        head_action_horizon: The head's action-chunk length (16 or 24).
        env_action_dim: Decoded environment action width (from metadata;
            the checkpoint's decoded env width).
        noise_dim: ``dsrl.noise_dim`` — the SAC latent width.
        chunk_size: ``training.chunk_size`` — must be ``None`` (single
            latent per decision).
        action_exec_len: ``dsrl.action_exec_len`` — env steps executed per
            decision.
        base_model_action_horizon: Optional ``base_model.action_horizon`` pin;
            must match the checkpoint when set.
        base_model_action_dim: Optional ``base_model.action_dim`` pin; must match
            the decoded env action width when set.

    Raises:
        ValueError: On any geometry mismatch, naming the config key to fix.
    """
    if noise_dim != head_action_dim:
        raise ValueError(
            f"dsrl.noise_dim ({noise_dim}) must equal the WAN action head's "
            f"action_dim ({head_action_dim}): DSRL steers the head's padded "
            "action-latent space, not the decoded env action space."
        )
    if chunk_size is not None:
        raise ValueError(
            f"training.chunk_size must be null for base_model=dreamzero_local (got "
            f"{chunk_size}): the validated recipe emits one {head_action_dim}-dim "
            "latent per decision and tiles it across the chunk."
        )
    if action_exec_len != head_action_horizon:
        raise ValueError(
            f"dsrl.action_exec_len ({action_exec_len}) must equal the checkpoint's "
            f"action horizon ({head_action_horizon}): the validated recipe makes "
            "one SAC decision per full action chunk."
        )
    if base_model_action_horizon is not None and base_model_action_horizon != head_action_horizon:
        raise ValueError(
            f"base_model.action_horizon ({base_model_action_horizon}) does not match the "
            f"checkpoint's action horizon ({head_action_horizon}); drop the pin or "
            "fix it."
        )
    if base_model_action_dim is not None and base_model_action_dim != env_action_dim:
        raise ValueError(
            f"base_model.action_dim ({base_model_action_dim}) does not match the decoded "
            f"env action width from the checkpoint metadata ({env_action_dim}); "
            "drop the pin or fix it."
        )
