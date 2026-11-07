# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-one observation and request helpers shared by the runtime policy families."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.runtimes.base import DECISION_MODES, DecisionMode
from isaac_rl_posttraining.runtimes.policy import Observation

__all__ = [
    "batch_observation",
    "combine_observations",
    "numpy_observation",
    "split_observation",
    "validate_session_requests",
]


def validate_session_requests(
    observations: Sequence[Observation],
    mode: Sequence[DecisionMode],
    session_ids: Sequence[Hashable],
) -> None:
    """Check a batched decision request: one known mode and one hashable session per observation."""
    if not (len(observations) == len(mode) == len(session_ids)):
        raise ValueError("observations, mode, and session_ids must have equal lengths")
    for session_id in session_ids:
        if not isinstance(session_id, Hashable):
            raise TypeError("policy-service session ids must be hashable")
    for decision_mode in mode:
        if decision_mode not in DECISION_MODES:
            raise ValueError(
                f"decision mode must be one of {DECISION_MODES}, got {decision_mode!r}"
            )


def batch_observation(observation: Observation) -> dict[str, Any]:
    """Add a leading batch axis of one to every array of a mapping observation."""
    if not isinstance(observation, Mapping):
        raise TypeError("steering policies require a mapping observation")
    batched: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            batched[key] = value[None, ...]
        else:
            batched[key] = value
    return batched


def numpy_observation(
    observation: Mapping[str, torch.Tensor], *, unbatch: bool
) -> dict[str, np.ndarray]:
    """Copy a batch-one learner observation to NumPy, optionally dropping the batch axis."""
    converted: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        array = value.detach().cpu().numpy()
        if unbatch:
            if array.ndim < 1 or array.shape[0] != 1:
                raise ValueError(
                    f"prepared learner observation {key!r} must have batch size 1, "
                    f"got {array.shape}"
                )
            array = array[0]
        converted[key] = np.array(array, copy=True)
    return converted


def combine_observations(
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Combine batch-one observations without changing shared metadata."""
    if not observations:
        raise ValueError("at least one observation is required")
    keys = set(observations[0])
    if any(set(observation) != keys for observation in observations[1:]):
        raise ValueError("batched policy observations must have identical keys")
    combined: dict[str, Any] = {}
    for key in observations[0]:
        values = [observation[key] for observation in observations]
        if all(isinstance(value, np.ndarray) for value in values):
            combined[key] = np.concatenate(values, axis=0)
        elif all(torch.is_tensor(value) for value in values):
            combined[key] = torch.cat(values, dim=0)
        else:
            try:
                shared = all(value == values[0] for value in values[1:])
            except (TypeError, ValueError):
                shared = False
            combined[key] = values[0] if shared else values
    return combined


def split_observation(
    observation: dict[str, Any], index: int, batch_size: int
) -> dict[str, Any]:
    """Select one sample while preserving a batch-one leading axis."""
    split: dict[str, Any] = {}
    for key, value in observation.items():
        if (
            (isinstance(value, np.ndarray) or torch.is_tensor(value))
            and value.ndim > 0
            and value.shape[0] == batch_size
            or isinstance(value, (list, tuple))
            and len(value) == batch_size
        ):
            split[key] = value[index : index + 1]
        else:
            split[key] = value
    return split
