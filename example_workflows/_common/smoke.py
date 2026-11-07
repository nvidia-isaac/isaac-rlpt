# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every workflow's ``smoke.sh``; the check CI's ``test:examples`` job runs.

python -m example_workflows._common.smoke            # run them all, exit 1 if any fails
python -m example_workflows._common.smoke --list     # print the workflows that have a smoke check
python -m example_workflows._common.smoke --only g1_arena_posttraining
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLES_ROOT.parent
DEFAULT_TIMEOUT_SECONDS = 600
_TAIL_LINES = 40


@dataclass(frozen=True)
class SmokeOutcome:
    workflow: str
    passed: bool
    seconds: float
    output: str


def workflow_dirs(root: Path = EXAMPLES_ROOT) -> list[Path]:
    """Workflow directories under ``root``: top-level and ``experimental/`` entries, skipping ``_``-prefixed ones."""
    found = []
    for parent in (root, root / "experimental"):
        if not parent.is_dir():
            continue
        for candidate in sorted(parent.iterdir()):
            if (
                not candidate.is_dir()
                or candidate.name.startswith(("_", "."))
                or candidate.name == "experimental"
            ):
                continue
            if (candidate / "README.md").exists() or (candidate / "smoke.sh").exists():
                found.append(candidate)
    return found


def workflow_name(path: Path, root: Path = EXAMPLES_ROOT) -> str:
    return path.relative_to(root).as_posix()


def _text(captured: str | bytes | None) -> str:
    if captured is None:
        return ""
    return (
        captured.decode(errors="replace") if isinstance(captured, bytes) else captured
    )


def run_smoke(
    path: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS, root: Path = EXAMPLES_ROOT
) -> SmokeOutcome:
    """Run one workflow's ``smoke.sh`` from the repository root and capture its outcome."""
    name = workflow_name(path, root)
    script = path / "smoke.sh"
    if not script.exists():
        return SmokeOutcome(name, False, 0.0, f"{script} does not exist")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=root.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = _text(exc.stdout) + _text(exc.stderr)
        return SmokeOutcome(
            name,
            False,
            time.monotonic() - started,
            f"{output}\ntimed out after {timeout:.0f} s",
        )
    return SmokeOutcome(
        name,
        completed.returncode == 0,
        time.monotonic() - started,
        completed.stdout + completed.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="list the workflows and exit"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="WORKFLOW",
        help="run only these workflows",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds per smoke check",
    )
    args = parser.parse_args(argv)

    workflows = workflow_dirs()
    if args.only:
        workflows = [path for path in workflows if workflow_name(path) in args.only]
        missing = set(args.only) - {workflow_name(path) for path in workflows}
        if missing:
            print(f"unknown workflows: {sorted(missing)}", file=sys.stderr)
            return 2
    if args.list:
        for path in workflows:
            print(workflow_name(path))
        return 0
    if not workflows:
        print("no workflows with a smoke check yet")
        return 0

    failures = 0
    for path in workflows:
        outcome = run_smoke(path, timeout=args.timeout)
        status = "PASS" if outcome.passed else "FAIL"
        print(f"{status} {outcome.workflow} ({outcome.seconds:.1f} s)", flush=True)
        if not outcome.passed:
            failures += 1
            tail = outcome.output.splitlines()[-_TAIL_LINES:]
            print("\n".join(f"    {line}" for line in tail), flush=True)
    print(f"{len(workflows) - failures}/{len(workflows)} smoke checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
