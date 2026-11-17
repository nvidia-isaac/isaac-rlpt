# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit policy capability handshakes.

The harness must not infer a policy's control surface from a class name or a
checkpoint name.  This small immutable record is the compatibility boundary
between policy adapters and runtime components.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyCapabilities:
    """Capabilities exposed by one policy endpoint.

    ``action_horizon`` may be unknown before the first successful request.
    Some public policy servers expose only action chunks, not their model
    configuration, so adapters should fill it from a validated response rather
    than guess.
    """

    name: str
    native_actions: bool
    latent_steering: bool
    future_prediction: bool
    value_prediction: bool
    stateful_episode: bool
    supports_batching: bool
    action_horizon: int | None
    action_dim: int | None
    latent_horizon: int | None
    latent_dim: int | None
    expected_observation_keys: tuple[str, ...]
    max_batch_size: int
    typical_latency_ms: float | None
