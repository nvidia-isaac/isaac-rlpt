# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 DROID metadata for the RoboLab harness."""

from __future__ import annotations

from typing import Any

from simulators.robolab.policies.adapter import RoboLabHookClient, RoboLabPolicyAdapter
from simulators.robolab.runtime.capabilities import PolicyCapabilities


GR00T_N17_DROID_CAPABILITIES = PolicyCapabilities(
    name="gr00t_n1.7_droid",
    native_actions=True,
    # The stock RoboLab endpoint exposes decoded chunks only.  A custom
    # server must explicitly change this after it accepts flow latents.
    latent_steering=False,
    future_prediction=False,
    value_prediction=False,
    stateful_episode=False,
    supports_batching=False,
    # The public server does not advertise its full prediction horizon.
    # The adapter fills both action fields from the first validated response.
    action_horizon=None,
    action_dim=None,
    latent_horizon=None,
    latent_dim=None,
    expected_observation_keys=("image_obs", "proprio_obs"),
    max_batch_size=1,
    typical_latency_ms=None,
)


class Gr00tRoboLabAdapter(RoboLabPolicyAdapter):
    """Adapter for RoboLab's ``GR00TDroidJointposClient``.

    The type annotation is structural on purpose: importing this class does
    not import RoboLab, ZeroMQ, OpenCV, Isaac Lab, or GR00T.  The real client
    is constructed by the RoboLab launch wrapper and passed in here.
    """

    def __init__(self, client: RoboLabHookClient) -> None:
        super().__init__(client, GR00T_N17_DROID_CAPABILITIES)


def make_gr00t_adapter(client: RoboLabHookClient, **_: Any) -> Gr00tRoboLabAdapter:
    """Construct the standard GR00T adapter without importing heavy extras."""
    return Gr00tRoboLabAdapter(client)
