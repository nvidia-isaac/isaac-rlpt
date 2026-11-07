# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DGX Station recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``).

A recipe's learner runs inside a container: ``train_eval.sh`` and ``layout_sweep.sh`` hand
``station_train.sh`` a container command, and ``run_container_train.sh`` in that container passes
the overrides to ``scripts/train.py``. Each case follows that chain on a fake station tree: the
dry run's learner container commands, then the container launcher's dry run on each.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from example_workflows._common.golden import REPO_ROOT, launcher_overrides, run_parallel

WORKFLOW = Path(__file__).resolve().parent.parent
CONTAINER_LAUNCHER = (
    REPO_ROOT
    / "example_workflows/g1_arena_posttraining/scripts/station/run_container_train.sh"
)

#: name -> (script and arguments, extra environment); every learner container command becomes one case.
RECIPES: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "train_eval_single": (
        ("train_eval.sh", "DrillLift"),
        {"SIMULATOR_GPU": "0", "LEARNER_GPU": "0"},
    ),
    "train_eval_split_rlinf_overlapped": (
        ("train_eval.sh", "DrillLift"),
        {
            "SIMULATOR_GPU": "1",
            "LEARNER_GPU": "0",
            "RUNTIME": "rlinf",
            "SCHEDULE": "overlapped",
            "NUM_SERVERS": "2",
            "NUM_ENVS": "256",
            "MAX_UPDATES_PER_BURST": "310",
            "EVAL_FREQ": "400000",
            "EVAL_NUM_ENVS": "25",
        },
    ),
    "sweep": (("layout_sweep.sh",), {"LAYOUTS": "two_cards:1:0", "NS": "64"}),
}


def station_tree(work: Path) -> dict[str, str]:
    """A fake deployed station (the G1 Factory tree with this repository as ``posttrain``) and its exports."""
    for path in ("ckpt", "data/LMDrillLift/lerobot", "logs", "g1f", "out", "hf"):
        (work / path).mkdir(parents=True, exist_ok=True)
    if not (work / "g1f" / "posttrain").exists():
        (work / "g1f" / "posttrain").symlink_to(REPO_ROOT)
    return {
        "G1F": str(work / "g1f"),
        "HF_CACHE": str(work / "hf"),
        "OUT_DIR": str(work / "out"),
        "LOG_DIR": str(work / "logs"),
        "POSTTRAIN_BASE_CKPT": str(work / "ckpt"),
        "POSTTRAIN_DATA_ROOT": str(work / "data"),
        "PYTHON": sys.executable,
        "HF_TOKEN": "",
    }


def learner_commands(argv: list[str], env: dict[str, str]) -> list[str]:
    """The container commands of the learner runs a dry run prints (simulator-only containers skipped)."""
    run_env = {**os.environ, **env, "ISAAC_RL_POSTTRAINING_DRYRUN": "1"}
    completed = subprocess.run(
        argv, cwd=REPO_ROOT, env=run_env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(argv)} exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    prefix = "launch.container_cmd="
    commands = [
        line[len(prefix) :]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    return [command for command in commands if "SIMS_ONLY=1" not in command]


def container_overrides(command: str, work: Path) -> list[str]:
    """What ``run_container_train.sh`` passes to ``scripts/train.py`` for one container command."""
    words = shlex.split(command)
    split = words.index("bash")
    env = dict(word.split("=", 1) for word in words[:split])
    extras = words[split + 2 :]
    env.update({"WORKDIR": str(work / "g1f"), "PY": sys.executable})
    return launcher_overrides(["bash", str(CONTAINER_LAUNCHER), *extras], env)


def cases(work: Path) -> dict[str, list[str]]:
    exports = station_tree(work)
    jobs = {}
    for name, (argv, env) in RECIPES.items():
        script = WORKFLOW / "scripts" / argv[0]
        for index, command in enumerate(
            learner_commands(["bash", str(script), *argv[1:]], {**exports, **env})
        ):
            # A dry run names the checkpoint the training run would write with a <timestamp> placeholder.
            command = command.replace("<timestamp>", "timestamp")
            jobs[f"{name}_{index}"] = lambda command=command: container_overrides(
                command, work
            )
    return run_parallel(jobs)
