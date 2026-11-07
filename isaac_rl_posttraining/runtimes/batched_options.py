# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The batched rollout worker's environment protocols and collection options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np

__all__ = [
    "BOOTSTRAP_VALUES",
    "INACTIVE_ACTIONS",
    "BatchedEnvironment",
    "BootstrapValues",
    "InactiveAction",
    "ObservationCaptureControl",
    "RestartableBatchedEnvironment",
    "SuccessorValues",
    "accepts_active_mask",
    "resolve_bootstrap_values",
]

#: Where a valued decision's bootstrap value (the value of the observation its
#: chunk reached) comes from. ``evaluate_next_observation`` asks the policy to
#: encode and value that observation after every chunk (one extra policy pass);
#: decisions without a value only have it encoded for replay.
#: ``reuse_next_decision`` takes the next decision's own value in the same
#: episode and 0 at the episode's end, whatever ended it, so no decision costs
#: a second pass, the replayed next observation stays raw, and a time-limit
#: truncation is never bootstrapped.
BootstrapValues = Literal["evaluate_next_observation", "reuse_next_decision"]
BOOTSTRAP_VALUES: tuple[BootstrapValues, ...] = (
    "evaluate_next_observation",
    "reuse_next_decision",
)
#: The names before the rename, kept for one release.
_BOOTSTRAP_VALUE_ALIASES: dict[str, BootstrapValues] = {
    "policy": "evaluate_next_observation",
    "next_decision": "reuse_next_decision",
}
SuccessorValues = BootstrapValues

#: What an environment whose episode or chunk has ended executes while the
#: others finish their chunk: ``zeros`` a zero command, ``hold`` its last action.
InactiveAction = Literal["zeros", "hold"]
INACTIVE_ACTIONS: tuple[InactiveAction, ...] = ("zeros", "hold")


def resolve_bootstrap_values(
    bootstrap_values: str, successor_values: str | None = None
) -> BootstrapValues:
    """The bootstrap-value source, from the current name or its old aliases."""
    if successor_values is not None:
        if successor_values not in _BOOTSTRAP_VALUE_ALIASES:
            raise ValueError(
                f"successor_values must be one of {tuple(_BOOTSTRAP_VALUE_ALIASES)}, "
                f"got {successor_values!r}"
            )
        resolved = _BOOTSTRAP_VALUE_ALIASES[successor_values]
        if bootstrap_values not in (BOOTSTRAP_VALUES[0], resolved):
            raise ValueError(
                f"successor_values={successor_values!r} contradicts "
                f"bootstrap_values={bootstrap_values!r}; set bootstrap_values only"
            )
        return resolved
    if bootstrap_values in _BOOTSTRAP_VALUE_ALIASES:
        return _BOOTSTRAP_VALUE_ALIASES[bootstrap_values]
    if bootstrap_values not in BOOTSTRAP_VALUES:
        raise ValueError(
            f"bootstrap_values must be one of {BOOTSTRAP_VALUES}, got {bootstrap_values!r}"
        )
    return cast(BootstrapValues, bootstrap_values)


class BatchedEnvironment(Protocol):
    """Natively batched simulator API required by the rollout worker."""

    num_envs: int

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        Mapping[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Sequence[Mapping[str, Any]],
    ]: ...

    def close(self) -> None: ...


@runtime_checkable
class RestartableBatchedEnvironment(Protocol):
    """A batched environment that can replace a crashed simulation and continue."""

    def restart(self) -> None: ...


class ObservationCaptureControl(Protocol):
    """A batched environment that can skip building observations it will not use."""

    need_obs: bool


def accepts_active_mask(env: object) -> bool:
    """Whether a batched environment's ``step`` takes the ``active`` mask (declared by the environment)."""
    accepts = getattr(env, "accepts_active_mask", False)
    if not isinstance(accepts, bool):
        raise TypeError("a batched environment's accepts_active_mask must be a boolean")
    return accepts
