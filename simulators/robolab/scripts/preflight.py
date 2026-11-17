#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate local external checkouts and capacity before a RoboLab run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


#: Every checked-out source tree must be an installable project.
PROJECT_MANIFEST = "pyproject.toml"


@dataclass(frozen=True)
class Check:
    """One human-readable preflight result."""

    name: str
    ok: bool
    detail: str


def parse_args() -> argparse.Namespace:
    """Parse local paths without assuming an external-install location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--gr00t-root", type=Path, required=True)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=80.0,
        help="Minimum available disk capacity before model/simulator setup (default: 80).",
    )
    parser.add_argument(
        "--expected-driver-major",
        type=int,
        default=580,
        help=(
            "Required NVIDIA driver major version for the Isaac Sim camera stack "
            "(default: 580)."
        ),
    )
    parser.add_argument(
        "--require-vulkan",
        action="store_true",
        help="Require a Vulkan instance backed by an NVIDIA GPU before evaluation.",
    )
    parser.add_argument(
        "--vulkaninfo",
        default="vulkaninfo",
        help="Vulkan diagnostic executable to use with --require-vulkan (default: vulkaninfo).",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return parser.parse_args()


def required_paths(root: Path, name: str, paths: Iterable[str]) -> list[Check]:
    """Check that one external checkout exposes its expected integration files."""
    checks = [Check(name=f"{name}_root", ok=root.is_dir(), detail=str(root))]
    for relative_path in paths:
        path = root / relative_path
        checks.append(Check(name=f"{name}:{relative_path}", ok=path.exists(), detail=str(path)))
    return checks


def disk_check(path: Path, minimum_gb: float) -> Check:
    """Ensure setup cannot silently fill the filesystem."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1024**3
    return Check(
        name="disk_free",
        ok=free_gb >= minimum_gb,
        detail=f"free_gb={free_gb:.1f} required_gb={minimum_gb:.1f} path={path}",
    )


def gpu_check() -> Check:
    """Report GPU free memory without treating an occupied GPU as an error."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return Check(name="nvidia_smi", ok=False, detail=str(error))
    return Check(name="nvidia_smi", ok=bool(result.stdout.strip()), detail=result.stdout.strip())


def driver_check(output: str, expected_major: int) -> Check:
    """Validate that every visible GPU reports the baseline driver major version."""
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
    observed_majors = sorted({version.split(".", maxsplit=1)[0] for version in versions})
    ok = bool(versions) and observed_majors == [str(expected_major)]
    return Check(
        name="nvidia_driver",
        ok=ok,
        detail=(
            f"observed={','.join(versions) or 'none'} required_major={expected_major}; "
            "Isaac Sim RTX camera startup is blocked on unvalidated driver majors."
        ),
    )


def nvidia_driver_check(expected_major: int) -> Check:
    """Read the active driver version without changing the host configuration."""
    command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return Check(name="nvidia_driver", ok=False, detail=str(error))
    return driver_check(result.stdout, expected_major)


def vulkan_check(executable: str) -> Check:
    """Ensure the renderer can create a Vulkan instance, not merely a CUDA context."""
    try:
        result = subprocess.run(
            [executable, "--summary"],
            capture_output=True,
            check=False,
            env=os.environ.copy(),
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return Check(name="vulkan", ok=False, detail=str(error))

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    summary = next(
        (line.strip() for line in output.splitlines() if "Vulkan Instance Version" in line),
        " ".join(line.strip() for line in output.splitlines()[-3:]),
    )
    return Check(
        name="vulkan",
        ok=result.returncode == 0 and "Vulkan Instance Version" in output,
        detail=f"exit_code={result.returncode} {summary}".strip(),
    )


def main() -> None:
    """Run all non-mutating preflight checks."""
    args = parse_args()
    checks: list[Check] = []
    checks.extend(
        required_paths(
            args.robolab_root,
            "robolab",
            (
                PROJECT_MANIFEST,
                "policies/gr00t/client.py",
            ),
        )
    )
    checks.extend(
        required_paths(
            args.gr00t_root,
            "gr00t",
            (PROJECT_MANIFEST, "gr00t/eval/run_gr00t_server.py"),
        )
    )
    checks.append(disk_check(args.robolab_root.parent, args.min_free_gb))
    checks.append(gpu_check())
    checks.append(nvidia_driver_check(args.expected_driver_major))
    if args.require_vulkan:
        checks.append(vulkan_check(args.vulkaninfo))

    if args.json:
        print(json.dumps([asdict(check) for check in checks], sort_keys=True))
    else:
        for check in checks:
            state = "PASS" if check.ok else "FAIL"
            print(f"{state} {check.name}: {check.detail}")
    raise SystemExit(0 if all(check.ok for check in checks) else 1)


if __name__ == "__main__":
    main()
