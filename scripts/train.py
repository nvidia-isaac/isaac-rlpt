#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training entry point: ``python scripts/train.py algorithm=<name> [overrides]``.

``algorithm`` selects the method: ``dsrl_sac``, ``rfs`` and ``sac_edit`` steer or
edit a FROZEN policy with an off-policy learner, ``ppo`` fine-tunes GR00T's action
head with flow-SDE PPO, ``dsrl_ppo`` trains the steering actor with PPO, and
``iql`` / ``sft`` fine-tune the base model from demonstrations (offline).
``simulator`` and ``base_model`` pick the task family and the policy,
``runtime`` where the rollouts run. Evaluation-only off-policy runs are
``training.num_rollouts=0 eval.eval_on_first_step=true`` (plus
``eval.eval_base=true`` for the native base policy); on-policy ones
``ppo.eval_only=true``.

``ISAAC_RL_POSTTRAINING_DRYRUN=1`` prints the launch plan and the resolved
config, then exits before any policy, simulator or learner loads. The plan
ends with ``plan.checks=``: how many checks passed, what stays unverified
until load (and why), and every failed check; failed checks then end the
run with their numbered list and exit status 1.
``ISAAC_RL_POSTTRAINING_OUTPUT_DIR`` sets the run-output root (default
``~/isaac_rl_outputs``); everything tunable is a config field.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.configs.normalize import resolved_without_old_keys
from isaac_rl_posttraining.runners.build import build_run, plan_report
from isaac_rl_posttraining.runners.off_policy_launch import configure_runtime_env
from isaac_rl_posttraining.utils.config_utils import refuse_removed_overrides


def run_from_hydra_cfg(cfg: DictConfig) -> None:
    """Print the plan of a composed ``train`` config, then run it unless this is a dry run.

    The Hydra entry point and external launchers (the Isaac Lab
    ``--rl_library isaac_rlpt`` module) both come through here, so a config
    composed with ``hydra.compose`` behaves exactly like one from the CLI.
    """
    # Other processes load these directories, so a relative one is taken from the caller's directory here.
    for key in ("training.load_dir", "ppo.resume_from"):
        path = OmegaConf.select(cfg, key)
        if path:
            OmegaConf.update(cfg, key, os.path.abspath(path))
    report = plan_report(cfg)
    print("\n".join(report.lines), flush=True)
    # Every failed check is reported at once, after the lines show what did resolve.
    try:
        plan = report.raise_errors()
    except ValueError as error:
        # The problems name the keys to change; a traceback would only bury them.
        raise SystemExit(str(error)) from None
    if os.environ.get("ISAAC_RL_POSTTRAINING_DRYRUN") == "1":
        print(OmegaConf.to_yaml(resolved_without_old_keys(cfg)), flush=True)
        logger.info(f"DRY RUN: resolved launch plan for algorithm={plan.algorithm!r}; nothing will be launched")
        return
    try:
        run = build_run(cfg)
    except ValueError as error:
        # A launch input the dry run could only list (a server address, a runtime's package) is missing.
        raise SystemExit(str(error)) from None
    run()


@hydra_main(version_base=None, config_path="../isaac_rl_posttraining/configs", config_name="train")
def _hydra_entry(cfg: DictConfig) -> None:
    run_from_hydra_cfg(cfg)


def main() -> None:
    try:
        refuse_removed_overrides(sys.argv[1:])
    except ValueError as error:
        # The message names the key to use instead; a traceback would only bury it.
        raise SystemExit(str(error)) from None
    configure_runtime_env()
    register_configs()
    _hydra_entry()


if __name__ == "__main__":
    main()
