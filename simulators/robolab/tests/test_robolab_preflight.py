# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the non-mutating RoboLab host preflight helpers."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PREFLIGHT = Path(__file__).parents[1] / "scripts" / "preflight.py"
MODULE = runpy.run_path(str(PREFLIGHT))


def test_driver_check_accepts_the_pinned_driver_major() -> None:
    check = MODULE["driver_check"]("580.65.06\n580.65.06\n", expected_major=580)

    assert check.ok
    assert "observed=580.65.06" in check.detail


def test_driver_check_rejects_a_mismatched_or_mixed_driver_major() -> None:
    check = MODULE["driver_check"]("595.71.05\n580.65.06\n", expected_major=580)

    assert not check.ok
    assert "observed=580.65.06,595.71.05" in check.detail


def test_parse_args_allows_a_gr00t_only_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--robolab-root",
            "/external/robolab",
            "--gr00t-root",
            "/external/isaac-gr00t",
        ],
    )

    args = MODULE["parse_args"]()

    assert args.gr00t_root == Path("/external/isaac-gr00t")


def test_vulkan_check_requires_a_successful_instance(monkeypatch) -> None:
    class CompletedProcess:
        returncode = 0
        stdout = "Vulkan Instance Version: 1.3.280\n"
        stderr = ""

    monkeypatch.setattr(
        MODULE["subprocess"], "run", lambda *args, **kwargs: CompletedProcess()
    )

    check = MODULE["vulkan_check"]("vulkaninfo")

    assert check.ok
    assert "Vulkan Instance Version" in check.detail


def test_vulkan_check_rejects_an_incompatible_driver(monkeypatch) -> None:
    class CompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER\n"

    monkeypatch.setattr(
        MODULE["subprocess"], "run", lambda *args, **kwargs: CompletedProcess()
    )

    check = MODULE["vulkan_check"]("vulkaninfo")

    assert not check.ok
    assert "ERROR_INCOMPATIBLE_DRIVER" in check.detail
