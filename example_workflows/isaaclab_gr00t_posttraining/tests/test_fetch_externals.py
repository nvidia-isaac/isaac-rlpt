# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/fetch_externals.sh`` reads every pin from ``externals.lock.yaml``, so the lock is the one place to change one."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parent.parent
SCRIPT = WORKFLOW / "scripts" / "fetch_externals.sh"


def _pin(name: str) -> list[str]:
    """The script's own ``pin`` function, run against the workflow's lock."""
    pin_def = next(
        line for line in SCRIPT.read_text().splitlines() if line.startswith("pin()")
    )
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'LOCK="$1"; {pin_def}; pin "$2"',
            "_",
            str(WORKFLOW / "externals.lock.yaml"),
            name,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.split()


def test_the_fetch_script_reads_each_pin_from_the_lock() -> None:
    externals = yaml.safe_load((WORKFLOW / "externals.lock.yaml").read_text())[
        "externals"
    ]
    assert {entry["name"] for entry in externals} == {"IsaacLab", "Isaac-GR00T"}
    for entry in externals:
        assert _pin(entry["name"]) == [entry["url"], str(entry["commit"])]
