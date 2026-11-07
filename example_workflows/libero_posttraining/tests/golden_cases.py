# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LIBERO recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``)."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common.golden import REPO_ROOT, launcher_overrides, run_parallel

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
#: What a user exports before launching; fixed so the pinned paths do not depend on the machine.
ENV = {
    "POSTTRAIN_BASE_CKPT": "/ckpt/GR00T-N1.7-LIBERO/libero_10",
    "WANDB_NAME": "golden",
}
PI0_TASK = "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray"

#: name -> (script and arguments, extra environment).
RECIPES: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "train_gr00t": (("train.sh", "8"), {}),
    "train_gr00t_rlinf_4_envs": (
        ("train.sh", "3"),
        {"RUNTIME": "rlinf", "NUM_ENVS": "4"},
    ),
    "train_gr00t_spatial": (("train.sh", "0"), {"SUITE": "libero_spatial"}),
    "train_dsrl_ppo": (("train_ppo.sh", "8"), {}),
    "train_dsrl_ppo_2_ranks": (
        ("train_ppo.sh", "8"),
        {"RUNTIME": "rlinf", "PPO_LEARNER_RANKS": "2"},
    ),
    "train_pi0": (
        ("train.sh", PI0_TASK, f"env.env_name=libero_pi0/libero_90/{PI0_TASK}"),
        {
            "BASE_MODEL": "pi0",
            "TRAIN_MAXSTEPS": "400",
            "SUITE": "libero_90",
            "SIMULATOR": "libero_10",
        },
    ),
    "train_sft": (
        ("train.sh", "8"),
        {"ALGO": "sft", "POSTTRAIN_DATASET_DIR": "/data/libero_10"},
    ),
    "train_iql": (
        ("train.sh", "8"),
        {"ALGO": "iql", "POSTTRAIN_DATASET_DIR": "/data/libero_10"},
    ),
    "train_iql_pi0": (
        ("train.sh", "8"),
        {
            "ALGO": "iql",
            "BASE_MODEL": "pi0",
            "POSTTRAIN_DATASET_DIR": "/data/libero_10_pi0",
        },
    ),
    "train_dreamzero": (
        ("train.sh", "0"),
        {"BASE_MODEL": "dreamzero_local", "SUITE": "libero_spatial"},
    ),
    "train_dreamzero_pixel": (
        ("train.sh", "0"),
        {
            "BASE_MODEL": "dreamzero_local",
            "SUITE": "libero_spatial",
            "RECIPE": "pixel",
            "NUM_ENVS": "8",
        },
    ),
}


def cases(work: Path) -> dict[str, list[str]]:
    del work

    def job(script: str, args: tuple[str, ...], env: dict[str, str]):
        argv = ["bash", str(SCRIPTS / script), *args]
        return lambda: launcher_overrides(argv, {**ENV, **env}, cwd=REPO_ROOT)

    return run_parallel(
        {name: job(argv[0], argv[1:], env) for name, (argv, env) in RECIPES.items()}
    )
