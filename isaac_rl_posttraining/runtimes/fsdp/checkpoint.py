# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest of a sharded learner checkpoint (``manifest.json``), shared by every FSDP host."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["LEARNER_CHECKPOINT_SCHEMA_VERSION", "validate_learner_checkpoint_manifest"]

LEARNER_CHECKPOINT_SCHEMA_VERSION = 1


def validate_learner_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    world_size: int,
    module_names: Sequence[str],
    optimizer_names: Sequence[str],
) -> tuple[str, int]:
    required = {
        "schema_version",
        "checkpoint_id",
        "world_size",
        "module_names",
        "optimizer_names",
        "learner_step",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(
            "distributed learner checkpoint manifest is missing: " + ", ".join(missing)
        )
    if manifest["schema_version"] != LEARNER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported distributed learner checkpoint schema "
            f"{manifest['schema_version']!r}"
        )
    if manifest["world_size"] != world_size:
        raise ValueError(
            "distributed learner checkpoint world size mismatch: "
            f"saved {manifest['world_size']}, current {world_size}"
        )
    expected_modules = sorted(module_names)
    if manifest["module_names"] != expected_modules:
        raise ValueError(
            "distributed learner checkpoint module topology mismatch: "
            f"saved {manifest['module_names']}, current {expected_modules}"
        )
    expected_optimizers = sorted(optimizer_names)
    if manifest["optimizer_names"] != expected_optimizers:
        raise ValueError(
            "distributed learner checkpoint optimizer topology mismatch: "
            f"saved {manifest['optimizer_names']}, current {expected_optimizers}"
        )
    checkpoint_id = manifest["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("distributed learner checkpoint ID must be non-empty")
    learner_step = manifest["learner_step"]
    if (
        isinstance(learner_step, bool)
        or not isinstance(learner_step, int)
        or learner_step < 0
    ):
        raise ValueError("distributed learner checkpoint step must be non-negative")
    return checkpoint_id, learner_step
