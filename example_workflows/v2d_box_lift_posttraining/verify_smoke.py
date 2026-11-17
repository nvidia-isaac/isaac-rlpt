# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the dry-run plan emitted by the V2D workflow launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    text = args.plan.read_text(encoding="utf-8")
    plan, end = json.JSONDecoder().raw_decode(text)
    if "DRY RUN" not in text[end:]:
        raise ValueError("launcher did not confirm dry-run mode")
    settings = plan["resolved_settings"]
    expected = {
        "training.num_envs": 128,
        "training.num_rollouts": 704000,
        "training.seed": 2,
        "training.chunk_size": 16,
        "training.rollout_count_by": "step",
        "training.train_after_episode": False,
        "training.max_updates_per_burst": 128,
        "training.save_interval": 10000,
        "training.policy_decode_batch_size": 16,
        "algorithm.learning_starts": 70400,
        "algorithm.batch_size": 64,
        "algorithm.num_updates_per_train_step": 1,
        "algorithm.num_critic_updates_per_actor_update": 4,
        "algorithm.critic_target_reduction": "mean",
        "algorithm.ent_coef": 0.005,
        "algorithm.td_target_clip": [0.0, 1.0],
        "eval.eval_freq": 0,
        "eval.eval_on_first_step": False,
        "eval.eval_base": False,
        "eval.eval_num_envs": 32,
        "eval.eval_num_episodes": 32,
        "env.max_episode_steps": 550,
        "dsrl.action_exec_len": 4,
        "eval.eval_max_steps": 138,
        "+simulator.training_fixed_reset": True,
        "+simulator.evaluation_fixed_reset": None,
        "+simulator.reset_validation": None,
    }
    for key, value in expected.items():
        if settings[key] != value:
            raise ValueError(f"{key} resolved to {settings[key]!r}, expected {value!r}")
    if len(plan["servers"]) != 1 or plan["eval_url"] is not None:
        raise ValueError("default training requires only the training simulator")
    if plan["native_baseline_learner"] is not None:
        raise ValueError("default training must not run native evaluation")
    if len(plan["server_receipts"]) != len(plan["servers"]):
        raise ValueError("every simulator server requires a startup receipt")
    print("dry-run contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
