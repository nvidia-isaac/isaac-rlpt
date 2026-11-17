# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0 base model.

Experimental -- ported, not yet validated in this repo.

Requires the optional ``openpi`` / ``openpi_client`` packages at policy
construction time; importing this package (and :mod:`isaac_rl_posttraining`
as a whole) works without them, and :func:`is_available` reports whether
the base model can actually be constructed.
"""

from __future__ import annotations

import importlib.util

from isaac_rl_posttraining.policies.pi0.base_model import Pi0BaseModelPolicy
from isaac_rl_posttraining.policies.pi0.policy import (
    Pi0Policy,
    build_noise_chunk,
    extract_vlm_features,
    infer_with_noise,
    load_pi0_policy,
)

__all__ = [
    "Pi0BaseModelPolicy",
    "Pi0Policy",
    "build_noise_chunk",
    "extract_vlm_features",
    "infer_with_noise",
    "is_available",
    "load_pi0_policy",
]


def is_available() -> bool:
    """Whether the optional pi0 dependencies are installed."""
    try:
        return (
            importlib.util.find_spec("openpi") is not None
            and importlib.util.find_spec("openpi_client") is not None
        )
    except (ImportError, ValueError):
        return False
