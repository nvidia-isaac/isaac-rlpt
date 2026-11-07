# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime patches on the ``groot`` (DreamZero) model stack.

The DSRL seam: ``WANPolicyHead.lazy_joint_video_action`` calls
``generate_noise`` twice — once for the video latent (5-D shape) and once
for the action latent (3-D ``[B, action_horizon, action_dim]``). The patch
below intercepts **only the 3-D action draw**, via the
``_dsrl_external_action_noise`` side-channel attribute the policy sets
before each predict; the video draw and every other caller keep the
original seeded-randn behavior, so the patch is inert outside steered
calls. ``_dsrl_noise_consumed`` lets the caller verify the injection
actually happened — without it, an upstream refactor that stopped calling
``generate_noise`` would leave SAC training on latents that never reached
the sampler.

Verified against groot pin ``ab790c19``.
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_dsrl_noise_patch",
    "compile_causal_attention_hot_paths",
    "raise_dynamo_limits",
]


def _dsrl_generate_noise(
    self,
    shape: tuple,
    seed: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Drop-in replacement for ``WANPolicyHead.generate_noise`` with DSRL injection.

    When ``_dsrl_external_action_noise`` is set on the head and the requested
    shape is the 3-D action latent, returns the externally supplied noise and
    marks ``_dsrl_noise_consumed`` so the caller can verify the injection
    happened. All other draws keep the original seeded behavior.
    """
    external_action_noise = getattr(self, "_dsrl_external_action_noise", None)
    if external_action_noise is not None and len(shape) == 3:
        if tuple(external_action_noise.shape) != tuple(shape):
            raise ValueError(
                "DSRL external action noise shape "
                f"{tuple(external_action_noise.shape)} does not match the "
                f"action-latent shape {tuple(shape)} requested by the sampler. "
                "Check that dsrl.noise_dim and the base model's action horizon "
                "match the checkpoint's action head."
            )
        self._dsrl_noise_consumed = True
        return external_action_noise.to(device=device, dtype=dtype)
    generator = None if seed is None else torch.Generator(device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def apply_dsrl_noise_patch() -> None:
    """Install the DSRL noise injection on ``WANPolicyHead.generate_noise``.

    Verifies the patch target exists first: after an upstream rename the
    assignment would otherwise succeed and the injection would silently
    never run. Idempotent.

    Raises:
        RuntimeError: If the installed groot version no longer exposes
            ``WANPolicyHead.generate_noise``.
    """
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
        WANPolicyHead,
    )

    if not callable(getattr(WANPolicyHead, "generate_noise", None)):
        raise RuntimeError(
            "WANPolicyHead.generate_noise not found in the installed dreamzero "
            "(groot) package — the DSRL noise-injection patch target has moved. "
            "Update isaac_rl_posttraining/policies/dreamzero/patch.py for this "
            "groot version."
        )
    if WANPolicyHead.generate_noise is _dsrl_generate_noise:
        return
    WANPolicyHead.generate_noise = _dsrl_generate_noise


def raise_dynamo_limits() -> None:
    """Raise Dynamo recompile limits for DreamZero's compiled flow scheduler.

    ``FlowUniPCMultistepScheduler.multistep_uni_p_bh_update`` is compiled with
    ``fullgraph=True, dynamic=False``; the UniPC history tensors change rank
    across denoising steps (3-D action vs 5-D video), so the default
    ``recompile_limit`` (8) trips ``FailOnRecompileLimitHit`` during rollout.
    Same mitigation as the reference server
    (``dreamzero/eval_utils/serve_dreamzero_wan22.py``).
    """
    dynamo_config = torch._dynamo.config
    if hasattr(dynamo_config, "cache_size_limit"):
        dynamo_config.cache_size_limit = max(
            getattr(dynamo_config, "cache_size_limit", 8), 1000
        )
    if hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = max(
            getattr(dynamo_config, "recompile_limit", 8), 800
        )
    if hasattr(dynamo_config, "accumulated_cache_size_limit"):
        dynamo_config.accumulated_cache_size_limit = max(
            getattr(dynamo_config, "accumulated_cache_size_limit", 8), 1000
        )
    if hasattr(dynamo_config, "accumulated_recompile_limit"):
        dynamo_config.accumulated_recompile_limit = max(
            getattr(dynamo_config, "accumulated_recompile_limit", 8), 2000
        )


_COMPILED_ATTENTION_METHODS = (
    "_process_clean_image_only",
    "_process_state_blocks",
    "_process_noisy_image_blocks",
    "_process_noisy_action_blocks",
)


def compile_causal_attention_hot_paths() -> None:
    """Wrap the causal WAN self-attention hot paths with ``torch.compile``.

    The four block-processing methods dominate inference latency;
    ``mode="reduce-overhead"`` (CUDA graphs) is the setting RLinf validated
    for rollout. Idempotent.

    Raises:
        RuntimeError: If a method is missing on this groot version — a
            silent skip would just leave inference slow, so it fails loudly
            for an explicit decision (recompile or disable
            ``base_model.torch_compile``).
    """
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanSelfAttention,
    )

    for method_name in _COMPILED_ATTENTION_METHODS:
        method = getattr(CausalWanSelfAttention, method_name, None)
        if method is None:
            raise RuntimeError(
                f"CausalWanSelfAttention.{method_name} not found in the installed "
                "groot package — the torch.compile wrapper targets have moved. "
                "Update isaac_rl_posttraining/policies/dreamzero/patch.py or set "
                "base_model.torch_compile=false."
            )
        if getattr(method, "_torchdynamo_orig_callable", None) is not None:
            continue  # already wrapped
        setattr(
            CausalWanSelfAttention,
            method_name,
            torch.compile(method, mode="reduce-overhead"),
        )
