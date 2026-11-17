# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V2D success results and terminal-height diagnostics for the tissue-box workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from example_workflows.v2d_box_lift_posttraining.config import SuccessDefinition
from simulators.v2d.contract import object_height


class LiftHoldResult(Protocol):
    success: bool
    lift_hold_success: bool
    max_lift_m: float
    max_hold_steps: int
    sample_count: int


class LiftHoldTracker(Protocol):
    def update(self, object_z: np.ndarray) -> None: ...

    def complete(self, env_index: int, *, next_initial_z: float) -> LiftHoldResult: ...


@dataclass(frozen=True)
class SuccessResult:
    success: bool
    combined_success: bool
    lift_hold_success: bool
    final_height_ok: bool
    initial_object_z_m: float
    final_object_z_m: float
    max_lift_m: float
    max_hold_steps: int
    sample_count: int
    failure_reason: str | None
    evaluator: str
    success_definition_sha256: str

    def task_metrics(self) -> dict[str, str | int | float | bool | None]:
        return {
            "success": self.success,
            "combined_success": self.combined_success,
            "lift_hold_success": self.lift_hold_success,
            "final_height_ok": self.final_height_ok,
            "initial_object_z_m": self.initial_object_z_m,
            "final_object_z_m": self.final_object_z_m,
            "max_lift_m": self.max_lift_m,
            "max_hold_steps": self.max_hold_steps,
            "sample_count": self.sample_count,
            "failure_reason": self.failure_reason,
            "evaluator": self.evaluator,
            "success_definition_sha256": self.success_definition_sha256,
        }


class CombinedSuccessTracker:
    """Report V2D success alongside post-action terminal-height diagnostics."""

    def __init__(
        self,
        initial_record: dict[str, np.ndarray],
        *,
        object_index: int,
        definition: SuccessDefinition,
        lift_tracker: LiftHoldTracker,
    ) -> None:
        self.object_index = object_index
        self.definition = definition
        self.lift_tracker = lift_tracker
        self._initial_z = object_height(initial_record, object_index)
        self._final_z = self._initial_z.copy()
        self._num_envs = len(self._initial_z)

    def sample_pre_action(
        self, record: dict[str, np.ndarray], active: np.ndarray
    ) -> None:
        """Sample at the released pre-action timing."""
        if active.shape != (self._num_envs,):
            raise ValueError("active mask has the wrong shape")
        # V2D's tracker is vectorized. Inactive environments are never scored
        # again before reset, so sampling their autoreset state is immaterial.
        self.lift_tracker.update(object_height(record, self.object_index))

    def update_post_action(self, terminal_record: dict[str, np.ndarray]) -> None:
        """Capture the actual post-action state before simulator autoreset."""
        final_z = object_height(terminal_record, self.object_index)
        if final_z.shape != (self._num_envs,):
            raise ValueError("terminal object height has the wrong shape")
        self._final_z = final_z

    def complete(
        self,
        env_index: int,
        *,
        next_initial_record: dict[str, np.ndarray],
    ) -> SuccessResult:
        next_initial_z = object_height(next_initial_record, self.object_index)
        lift = self.lift_tracker.complete(
            env_index, next_initial_z=float(next_initial_z[env_index])
        )
        initial_z = float(self._initial_z[env_index])
        final_z = float(self._final_z[env_index])
        # Post-action height is diagnostic; V2D owns the success decision.
        final_height_ok = (
            final_z >= initial_z - self.definition.final_height_tolerance_m
        )
        success = bool(lift.success)
        if success:
            failure_reason = None
        elif not lift.lift_hold_success:
            failure_reason = "lift"
        else:
            failure_reason = "final_height"
        result = SuccessResult(
            success=success,
            combined_success=success,
            lift_hold_success=bool(lift.lift_hold_success),
            final_height_ok=bool(final_height_ok),
            initial_object_z_m=initial_z,
            final_object_z_m=final_z,
            max_lift_m=float(lift.max_lift_m),
            max_hold_steps=int(lift.max_hold_steps),
            sample_count=int(lift.sample_count),
            failure_reason=failure_reason,
            evaluator=self.definition.evaluator,
            success_definition_sha256=self.definition.sha256,
        )
        self._initial_z[env_index] = next_initial_z[env_index]
        self._final_z[env_index] = next_initial_z[env_index]
        return result
