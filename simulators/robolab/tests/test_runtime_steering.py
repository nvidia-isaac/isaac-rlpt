# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the server-side latent safety boundary."""

from __future__ import annotations

import numpy as np
import pytest

from simulators.robolab.runtime.steering import bound_latent


def test_bound_latent_clips_once_and_exposes_trace_metadata():
    """The effective latent and provenance agree at the server boundary."""
    bounded = bound_latent(np.array([-4.0, -1.0, 0.5, 5.0]), clip=3.0)
    np.testing.assert_array_equal(bounded.effective, np.array([-3.0, -1.0, 0.5, 3.0]))
    assert bounded.clipped_entries == 2
    assert bounded.metadata(additive=True) == {
        "requested_shape": [4],
        "latent_clip": 3.0,
        "clipped_entries": 2,
        "effective_abs_max": 3.0,
        "additive": True,
    }


@pytest.mark.parametrize("latent", [[], [float("nan")], [[float("inf")]]])
def test_bound_latent_rejects_empty_or_nonfinite_values(latent):
    """Malformed controls never reach the flow model."""
    with pytest.raises(ValueError):
        bound_latent(latent, clip=1.0)
