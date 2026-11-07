# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSRL — Diffusion Steering RL over a base model.

DSRL (Wagenmaker et al., 2025) post-trains a pretrained flow/diffusion
policy WITHOUT touching its weights: a small SAC agent acts in the policy's
latent-noise space, and the frozen policy decodes each latent into an action
chunk. Learning happens entirely in the latent space, so the base policy's
competence is preserved and improvement comes from steering its sampler.

The implementation is deliberately split into three layers:

* **SAC core** (:mod:`isaac_rl_posttraining.algorithms.sac`) — generic
  REDQ-style SAC (ensemble, target nets, entropy auto-tuning, optional
  prior/snapshot KL anchors). It never learns what a "latent" is; the
  steering latent is simply its action. Re-exported here because DSRL's
  optimizer *is* this core.
* **This layer** — the DSRL-specific piece that is separable without
  refactoring: :class:`DSRLSteeringConfig` (latent geometry, execution
  chunking, frozen-checkpoint selection). The receding-horizon discount
  (``gamma**action_exec_len``, applied because DSRL stores one transition per
  executed chunk) is the shared
  :func:`~isaac_rl_posttraining.algorithms.discounting.chunk_discounted_gamma`.
* **Base model** (:mod:`isaac_rl_posttraining.policies.gr00t`) — the
  latent-injection mechanics live with the model they steer: tiling or
  broadcasting the ``[B, K, noise_dim]`` latent across the flow horizon,
  additive-vs-replace injection, the +/-clip at injection time, and fresh
  N(0,1) resampling of non-steered dimensions. See
  ``policies/gr00t`` for those knobs; they are intentionally not mirrored
  here.

Rollout/evaluation wiring (the steering rollout policy in
:mod:`isaac_rl_posttraining.runtimes.steering_policy` over
:class:`~isaac_rl_posttraining.rollouts.steering_step.SteeringStep`) is
shared machinery for any latent-steering method, not DSRL-specific.
"""

from isaac_rl_posttraining.algorithms.dsrl.config import DSRLSteeringConfig
from isaac_rl_posttraining.algorithms.sac import SAC, SACConfig

__all__ = ["SAC", "SACConfig", "DSRLSteeringConfig"]
