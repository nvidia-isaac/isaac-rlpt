# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The G1 Factory recipes whose resolved config ``golden_config.json`` pins (see ``_common/golden.py``)."""

from __future__ import annotations

from pathlib import Path

from example_workflows._common.golden import REPO_ROOT, launcher_overrides, run_parallel

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
#: What a user exports before launching (the checkpoint is a directory of the scratch tree).
ENV = {"POSTTRAIN_DATA_ROOT": "/data/g1", "WANDB_NAME": "golden"}

#: name -> (script and arguments, extra environment); ``{checkpoint}`` is a checkpoint directory.
RECIPES: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "train_dsrl": (("train.sh", "DrillLift"), {}),
    "train_dsrl_8_envs": (("train.sh", "PushButton"), {"NUM_ENVS": "8"}),
    "train_dsrl_rlinf": (
        ("train.sh", "BoxLift"),
        {"RUNTIME": "rlinf", "NUM_ROLLOUTS": "1650"},
    ),
    "train_dsrl_ppo": (("train_ppo.sh", "DrillLift"), {}),
    "train_dsrl_ppo_2_ranks": (
        ("train_ppo.sh", "DrillLift"),
        {"RUNTIME": "rlinf", "PPO_LEARNER_RANKS": "2"},
    ),
    "finetune_ppo": (("finetune_ppo.sh", "PushButton"), {}),
    "train_sft": (("train.sh", "DrillLift"), {"ALGO": "sft"}),
    "train_iql": (("train.sh", "DrillLift"), {"ALGO": "iql"}),
    "eval_iql_policy": (
        ("eval.sh", "DrillLift", "policy", "100", "{checkpoint}"),
        {"ALGO": "iql"},
    ),
    "eval_base": (("eval.sh", "DrillLift", "base", "50"), {}),
    "eval_policy": (("eval.sh", "DrillLift", "policy", "100", "{checkpoint}"), {}),
    "eval_parallel": (
        ("eval.sh", "PushButton", "base", "100"),
        {"EVAL_NUM_ENVS": "10"},
    ),
    "zoo_rfs": (("zoo_recipes.sh", "rfs", "DrillPnP", "1"), {}),
    "zoo_rfs_pc": (("zoo_recipes.sh", "rfs_pc", "DrillPnP", "1"), {}),
    "zoo_rfs_nb2": (("zoo_recipes.sh", "rfs_nb2", "DrillPnP", "1"), {}),
    "zoo_sac_edit": (("zoo_recipes.sh", "sac_edit", "DrillPnP", "1"), {}),
    "repro_train": (("repro_sep2026.sh", "train", "DrillPnP", "1"), {}),
    "repro_train_b2": (
        ("repro_sep2026.sh", "train", "DrillLift"),
        {"VARIANT": "b2"},
    ),
    "repro_train_b2_succ": (
        ("repro_sep2026.sh", "train", "PushButton"),
        {"VARIANT": "b2_succ"},
    ),
    "repro_base": (("repro_sep2026.sh", "base", "DrillPnP"), {}),
    "repro_base_seeded": (
        ("repro_sep2026.sh", "base", "DrillPnP"),
        {"EVAL_SEED": "1000"},
    ),
    "repro_rescore": (("repro_sep2026.sh", "rescore", "DrillPnP", "{checkpoint}"), {}),
    "repro_oct_dsrl": (("reproduce_results.sh", "dsrl", "PushButton", "1"), {}),
    "repro_oct_ppo": (("reproduce_results.sh", "ppo", "PushButton", "1"), {}),
    "repro_oct_base": (("reproduce_results.sh", "base", "PushButton"), {}),
    "repro_oct_rescore": (
        ("reproduce_results.sh", "rescore", "PushButton", "{checkpoint}"),
        {},
    ),
    "repro_oct_rescore_ppo": (
        ("reproduce_results.sh", "rescore_ppo", "PushButton", "{iteration}"),
        {},
    ),
}


def base_checkpoint(work: Path) -> Path:
    """A stand-in GR00T N1.7 checkpoint: the dry run reads only its ``config.json``."""
    path = work / "GR00T-N1.7-G1"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    return path


def cases(work: Path) -> dict[str, list[str]]:
    checkpoint = work / "checkpoints" / "16088"
    checkpoint.mkdir(parents=True, exist_ok=True)
    iteration = work / "ppo" / "iteration_40"
    iteration.mkdir(parents=True, exist_ok=True)
    env = {**ENV, "POSTTRAIN_BASE_CKPT": str(base_checkpoint(work))}

    def job(script: str, args: tuple[str, ...], env: dict[str, str]):
        argv = [
            "bash",
            str(SCRIPTS / script),
            *(
                arg.replace("{checkpoint}", str(checkpoint)).replace(
                    "{iteration}", str(iteration)
                )
                for arg in args
            ),
        ]
        return lambda: launcher_overrides(argv, env, cwd=REPO_ROOT)

    return run_parallel(
        {
            name: job(argv[0], argv[1:], {**env, **extra})
            for name, (argv, extra) in RECIPES.items()
        }
    )
