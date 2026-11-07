# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSRL-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DSRLSteeringConfig"]


@dataclass
class DSRLSteeringConfig:
    """The ``dsrl:`` config section — how the frozen policy is steered.

    These knobs describe the steering *interface* (latent geometry, execution
    chunking, and which frozen policy to load). The latent-injection
    mechanics themselves (tile/broadcast across the flow horizon, additive
    vs. replace, the +/-clip at injection) are implemented base-model-side in
    :mod:`isaac_rl_posttraining.policies.gr00t`.

    Attributes:
        noise_dim: Width of the per-step steering latent the actor emits.
        noise_action_bound: Symmetric bound of the latent action space.
        action_exec_len: Decoded actions executed per chunk (receding
            horizon).
        warmup_exploration: Latent distribution before ``learning_starts``:
            ``"uniform"`` over the bounded action box, or ``"normal"`` --
            unbounded N(0, 1), the flow's native noise (the official DSRL
            pi0 bootstrap), so warmup episodes behave near the base policy
            and seed the buffer with successes.
        use_vlm_features: Feed base model VLM features to the actor/critic
            (unsupported by the GR00T base model; its critic uses the DINO
            embedding instead).
        batched_actor: A served policy runs the actor once for every session
            of a batch instead of once per session (latent-sampling actors only).

    Checkpoint, dataset, and embodiment selection live in the ``base_model``
    config group (:class:`~isaac_rl_posttraining.configs.schema.BaseModelConfig`).
    """

    noise_dim: int = field(default=35, metadata={"help": "Per-step steering-latent width"})
    noise_action_bound: float = field(default=1.0, metadata={"help": "Symmetric latent action bound"})
    action_exec_len: int = field(default=8, metadata={"help": "Actions executed per decoded chunk"})
    warmup_exploration: str = field(
        default="uniform",
        metadata={"help": "Warmup latent distribution: 'uniform' (action box) or 'normal' (N(0,1))"},
    )
    use_vlm_features: bool = field(
        default=False, metadata={"help": "Feed base model VLM features to the actor/critic (gr00t: unsupported)"}
    )
    batched_actor: bool = field(
        default=False, metadata={"help": "A served policy runs the actor once per batch, not once per session"}
    )
