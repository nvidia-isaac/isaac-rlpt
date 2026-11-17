# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision checks required before accepting a sampled reset pose."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from simulators.v2d.contract import JsonValue


@dataclass(frozen=True)
class ResetValidationConfig:
    enabled: bool
    max_attempts: int
    max_penetration_m: float
    allowed_contact_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("reset_validation.enabled must be a boolean")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("reset_validation.max_attempts must be a positive integer")
        tolerance = self.max_penetration_m
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("max_penetration_m must be a finite non-negative number")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("max_penetration_m must be a finite non-negative number")
        for pair in self.allowed_contact_pairs:
            if len(pair) != 2 or any(
                not isinstance(name, str) or not name for name in pair
            ):
                raise ValueError(
                    "allowed_contact_pairs must contain pairs of body names"
                )
            if pair[0] == pair[1]:
                raise ValueError(
                    "An allowed contact pair must name two different bodies"
                )

    def as_dict(self) -> dict[str, JsonValue]:
        value = asdict(self)
        value["allowed_contact_pairs"] = [
            list(pair) for pair in self.allowed_contact_pairs
        ]
        return value

    @classmethod
    def from_json(cls, value: JsonValue) -> ResetValidationConfig:
        if not isinstance(value, dict) or set(value) != {
            "enabled",
            "max_attempts",
            "max_penetration_m",
            "allowed_contact_pairs",
        }:
            raise ValueError("Invalid reset_validation fields")
        pairs = value["allowed_contact_pairs"]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) for pair in pairs
        ):
            raise ValueError("allowed_contact_pairs must be a list of pairs")
        return cls(
            **{**value, "allowed_contact_pairs": tuple(tuple(pair) for pair in pairs)}
        )
