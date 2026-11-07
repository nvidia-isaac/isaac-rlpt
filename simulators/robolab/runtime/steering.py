# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit helpers shared by runtime steering endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundedLatent:
    """A requested steering latent and the server-authoritative clipped value."""

    requested: np.ndarray
    effective: np.ndarray
    clip: float

    @property
    def clipped_entries(self) -> int:
        """Number of entries modified by clipping."""
        return int(np.count_nonzero(self.requested != self.effective))

    def metadata(self, *, additive: bool) -> dict[str, float | int | bool | list[int]]:
        """Return JSON-safe provenance for a runtime trace."""
        return {
            "requested_shape": list(self.requested.shape),
            "latent_clip": float(self.clip),
            "clipped_entries": self.clipped_entries,
            "effective_abs_max": float(np.max(np.abs(self.effective))),
            "additive": bool(additive),
        }


def bound_latent(latent: object, *, clip: float) -> BoundedLatent:
    """Convert a numeric latent to float32 and clip it at one explicit boundary."""
    if clip <= 0:
        raise ValueError(f"clip must be positive; got {clip}")
    requested = np.asarray(latent, dtype=np.float32)
    if requested.ndim not in (1, 2, 3):
        raise ValueError(
            "latent must have shape [D], [B, D], or [B, K, D]; "
            f"got {requested.shape}"
        )
    if requested.size == 0:
        raise ValueError("latent must not be empty")
    if not np.isfinite(requested).all():
        raise ValueError("latent must contain only finite values")
    return BoundedLatent(
        requested=requested,
        effective=np.clip(requested, -clip, clip),
        clip=float(clip),
    )
