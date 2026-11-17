# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated tissue-box task and server-launch configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)
from simulators.v2d.contract import JsonValue

EVALUATOR_NAME = "lift_hold"


@dataclass(frozen=True)
class ResetPerturbation:
    arm_joint_noise_rad: float
    finger_joint_noise_rad: float
    object_xy_noise_m: float
    object_yaw_noise_rad: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class SuccessDefinition:
    """Evaluator identity and the separate terminal-height diagnostic tolerance."""

    evaluator: str
    final_height_tolerance_m: float

    def __post_init__(self) -> None:
        if self.evaluator != EVALUATOR_NAME:
            raise ValueError(f"Unsupported success evaluator: {self.evaluator}")
        if (
            not math.isfinite(self.final_height_tolerance_m)
            or self.final_height_tolerance_m <= 0
        ):
            raise ValueError("final_height_tolerance_m must be finite and positive")

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    gym_task: str
    task_profile: str
    embodiment: str
    episode_horizon: int
    control_hz: int
    execution_length: int
    visual_mode: str
    visual_warmup_steps: int
    reset_frame: int
    reset_finger_openness: float
    use_primitive_urdfs: bool
    success_definition: SuccessDefinition
    reset_perturbation: ResetPerturbation
    fixed_reset: bool = False
    reset_validation: ResetValidationConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_reset, bool):
            raise TypeError("fixed_reset must be a boolean")
        if self.embodiment != "vega_sharpa_joint":
            raise ValueError(
                "The workflow supports only the 58-D vega_sharpa_joint contract"
            )
        if self.episode_horizon not in (519, 550):
            raise ValueError(
                "The tissue-box workflow supports 519- and 550-step task horizons"
            )
        if self.control_hz != 20 or self.execution_length != 4:
            raise ValueError("The tissue-box workflow requires 20 Hz control and E=4")
        if self.visual_mode not in ("off", "training_matched"):
            raise ValueError("visual_mode must be off or training_matched")
        if self.visual_warmup_steps < 0:
            raise ValueError("visual_warmup_steps cannot be negative")
        if self.reset_frame != 0 or self.reset_finger_openness != 0:
            raise ValueError(
                "Only released frame-zero, finger-open resets are supported"
            )
        if not self.task_id or not self.gym_task or not self.task_profile:
            raise ValueError("Task identifiers and profile are required")

    @property
    def sha256(self) -> str:
        values = asdict(self)
        if self.reset_validation is None:
            # Preserve the identity of historical profiles without collision checks.
            del values["reset_validation"]
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CapabilityLaunchConfig:
    task_config: Path
    v2d_root: Path
    motion_file: Path
    expected_v2d_revision: str


def _load_json(path: Path) -> dict[str, JsonValue]:
    value = cast(JsonValue, json.loads(path.read_text()))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_task(path: Path) -> TaskConfig:
    """Load a checked-in task profile and its success definition."""
    values = dict(_load_json(path))
    reset = values.pop("reset_perturbation")
    if not isinstance(reset, dict):
        raise TypeError("reset_perturbation must be an object")
    values["reset_perturbation"] = ResetPerturbation(**reset)
    if "reset_validation" in values:
        values["reset_validation"] = ResetValidationConfig.from_json(
            values["reset_validation"]
        )
    success = values.pop("success_definition")
    if not isinstance(success, dict):
        raise TypeError("success_definition must be an object")
    values["success_definition"] = SuccessDefinition(**success)
    return TaskConfig(**values)


def load_capability_launch(path: Path) -> CapabilityLaunchConfig:
    """Load the generated server input without hiding a missing field."""
    values = _load_json(path)
    required = {"task_config", "v2d_root", "motion_file", "expected_v2d_revision"}
    if set(values) != required:
        raise ValueError(
            f"capability launch config requires exactly {sorted(required)}"
        )
    for name in ("task_config", "v2d_root", "motion_file"):
        if not isinstance(values[name], str) or not values[name]:
            raise TypeError(f"{name} must be a non-empty path string")
    revision = values["expected_v2d_revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("expected_v2d_revision must be a full Git commit")
    return CapabilityLaunchConfig(
        task_config=Path(values["task_config"]),
        v2d_root=Path(values["v2d_root"]),
        motion_file=Path(values["motion_file"]),
        expected_v2d_revision=revision,
    )
