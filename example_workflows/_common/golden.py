# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-config parity: the resolved ``train`` config of every workflow recipe, pinned in a file.

A workflow lists its recipes in a ``golden_cases.py`` module of its tests folder, as
``cases(work) -> {name: overrides}`` (``work`` is a scratch directory the recipes may use), and pins
what each resolves to in ``golden_config.json`` next to it. :func:`resolved` composes ``train.yaml`` as ``scripts/train.py``
does, normalizes it (older key spellings land on their canonical keys), resolves every
interpolation and flattens it to ``{"section.key": value}``, keeping only the selected runtime's settings. The older spellings themselves are
dropped: the run reads only the canonical keys. Check every workflow (checking is the default), or
regenerate one workflow's file after an intended change::

    uv run python -m example_workflows._common.golden --all
    uv run python -m example_workflows._common.golden g1_arena_posttraining --write
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLES_ROOT.parent
GOLDEN_FILE = "golden_config.json"
CASES_MODULE = "golden_cases"
#: The line prefix a launcher's dry run gives each override it passes to ``scripts/train.py``.
LAUNCH_OVERRIDE = "launch.override="

Cases = Mapping[str, Sequence[str]]
#: Launcher knobs a developer's shell may carry; a case sets the ones it needs.
_LAUNCHER_KNOBS = frozenset(
    [
        "ALGO",
        "BASE_MODEL",
        "EVAL_MAXSTEPS",
        "EVAL_NUM_ENVS",
        "EVAL_SEED",
        "GPU",
        "GPUS",
        "IMAGE_SHORTEST_EDGE",
        "LEARNER_PYTHON",
        "NUM_ENVS",
        "NUM_ROLLOUTS",
        "NUM_SERVERS",
        "PPO_LEARNER_RANKS",
        "RECIPE",
        "RUNTIME",
        "SCHEDULE",
        "SIMULATOR",
        "SUITE",
        "TRAINER",
        "TRAIN_MAXSTEPS",
        "VARIANT",
        "WANDB_NAME",
    ]
)


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` as ``{"a.b": 1}``; lists stay values."""
    if isinstance(node, Mapping) and node:
        flat: dict[str, Any] = {}
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: node}


def _renamed_keys() -> set[str]:
    from isaac_rl_posttraining.configs.normalize import CONFIG_ALIASES

    return {alias.old for alias in CONFIG_ALIASES}


def resolved(
    overrides: Sequence[str], placeholders: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The flattened, normalized and resolved ``train`` config ``overrides`` compose.

    ``hydra.*`` overrides only steer Hydra's own output and are left out; ``placeholders``
    replaces machine-specific path prefixes in string values (``{"/tmp/x": "<work>"}``).
    """
    from omegaconf import OmegaConf

    from isaac_rl_posttraining.train import compose_train_config

    cfg = compose_train_config(
        [item for item in overrides if not item.startswith("hydra.")]
    )
    flat = flatten(OmegaConf.to_container(cfg, resolve=True))
    # Only the selected runtime's settings reach the run, and a release may leave other runtimes out.
    selected = f"runtime.{flat.get('runtime.name')}."
    flat = {
        key: value
        for key, value in flat.items()
        if not key.startswith("runtime.")
        or key.count(".") < 2
        or key.startswith(selected)
    }
    for key in _renamed_keys():
        flat.pop(key, None)
    substitutions = sorted((placeholders or {}).items(), key=lambda item: -len(item[0]))
    for key, value in flat.items():
        if isinstance(value, str):
            for path, token in substitutions:
                value = value.replace(path, token)
            flat[key] = value
    return flat


def launcher_overrides(
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> list[str]:
    """The overrides a launcher's dry run passes to ``scripts/train.py`` (its ``launch.override=`` lines)."""
    run_env = {
        key: value
        for key, value in os.environ.items()
        if key not in _LAUNCHER_KNOBS
        and not key.startswith(("POSTTRAIN_", "ISAACLAB_", "ARENA_"))
    }
    run_env.update(
        {"ISAAC_RL_POSTTRAINING_DRYRUN": "1", "LEARNER_PYTHON": sys.executable}
    )
    run_env.update(env or {})
    completed = subprocess.run(
        list(argv),
        cwd=cwd or REPO_ROOT,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(argv)} exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    lines = completed.stdout.splitlines()
    return [
        line[len(LAUNCH_OVERRIDE) :]
        for line in lines
        if line.startswith(LAUNCH_OVERRIDE)
    ]


def run_parallel(jobs: Mapping[str, Callable[[], list[str]]]) -> dict[str, list[str]]:
    """Run independent launcher dry runs side by side; ``{name: overrides}`` in ``jobs``' order."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def snapshot(
    cases: Cases, placeholders: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """``{case: resolved config}`` of every case."""
    return {
        name: resolved(overrides, placeholders) for name, overrides in cases.items()
    }


_ABSENT = "<absent>"


def pack(snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The file form: the first case in full under ``base``, every case as its keys that differ from it."""
    names = list(snapshots)
    base = dict(sorted(snapshots[names[0]].items())) if names else {}
    cases = {}
    for name in names:
        config = snapshots[name]
        changed = {
            key: value
            for key, value in config.items()
            if base.get(key, _ABSENT) != value
        }
        changed.update({key: _ABSENT for key in base if key not in config})
        cases[name] = dict(sorted(changed.items()))
    return {"base": base, "cases": cases}


def unpack(packed: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configs = {}
    for name, changed in packed["cases"].items():
        config = {**packed["base"], **changed}
        configs[name] = {
            key: value for key, value in config.items() if value != _ABSENT
        }
    return configs


def differences(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Readable ``key: expected -> actual`` lines for every key whose value moved."""
    return [
        f"{key}: {expected.get(key, _ABSENT)!r} -> {actual.get(key, _ABSENT)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key, _ABSENT) != actual.get(key, _ABSENT)
    ]


def compare(golden: Path, snapshots: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every difference between the pinned and the current configs, prefixed with the case name."""
    expected = unpack(json.loads(golden.read_text(encoding="utf-8")))
    problems = [
        f"{name}: no pinned config" for name in snapshots if name not in expected
    ]
    problems += [
        f"{name}: pinned but no longer a case"
        for name in expected
        if name not in snapshots
    ]
    for name in snapshots.keys() & expected.keys():
        problems += [
            f"{name}: {line}" for line in differences(expected[name], snapshots[name])
        ]
    return problems


def write(golden: Path, snapshots: Mapping[str, Mapping[str, Any]]) -> None:
    golden.write_text(
        json.dumps(pack(snapshots), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def workflow_snapshots(workflow: str, work: Path) -> dict[str, dict[str, Any]]:
    """Resolve every case the workflow's ``golden_cases`` module declares."""
    module = importlib.import_module(
        f"example_workflows.{workflow}.tests.{CASES_MODULE}"
    )
    placeholders = {
        str(work): "<work>",
        str(REPO_ROOT): "<repo>",
        str(Path.home()): "<home>",
    }
    return snapshot(module.cases(work), placeholders)


def golden_path(workflow: str) -> Path:
    return EXAMPLES_ROOT / workflow / "tests" / GOLDEN_FILE


def check_workflow(workflow: str, work: Path) -> None:
    """Fail with every moved value when a recipe of ``workflow`` no longer resolves to its pinned config."""
    problems = compare(golden_path(workflow), workflow_snapshots(workflow, work))
    assert not problems, "\n".join(
        [
            *problems,
            f"(after an intended change: python -m example_workflows._common.golden {workflow} --write)",
        ]
    )


def check_same_config(
    workflow: str, case: str, overrides: Sequence[str], placeholders: Mapping[str, str]
) -> None:
    """Fail unless ``overrides`` (a Python launch, say) resolve to the pinned config of ``case``."""
    expected = unpack(json.loads(golden_path(workflow).read_text(encoding="utf-8")))[
        case
    ]
    problems = differences(expected, resolved(overrides, placeholders))
    assert not problems, f"{workflow}/{case}:\n" + "\n".join(problems)


def workflows_with_cases() -> list[str]:
    return sorted(
        path.parent.parent.name
        for path in EXAMPLES_ROOT.glob(f"*/tests/{CASES_MODULE}.py")
    )


def argument_parser() -> argparse.ArgumentParser:
    """The command line: workflows to check (or ``--all``), and ``--write`` to re-pin them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "workflows", nargs="*", help="workflow folder names (default: --all)"
    )
    parser.add_argument(
        "--all", action="store_true", help="every workflow with a golden_cases module"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the pinned file instead of checking it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    names = workflows_with_cases() if args.all or not args.workflows else args.workflows
    failed = False
    for name in names:
        with tempfile.TemporaryDirectory() as scratch:
            snapshots = workflow_snapshots(name, Path(scratch).resolve())
        if args.write:
            write(golden_path(name), snapshots)
            print(f"{name}: pinned {len(snapshots)} recipe config(s)")
            continue
        problems = compare(golden_path(name), snapshots)
        failed |= bool(problems)
        print(
            "\n".join(problems)
            if problems
            else f"{name}: {len(snapshots)} recipe config(s) match"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
