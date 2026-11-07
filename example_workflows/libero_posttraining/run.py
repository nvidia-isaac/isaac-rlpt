# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LIBERO GR00T steering recipe from Python: the run ``scripts/train.sh TASK`` launches (``BASE_MODEL=gr00t``).

The LIBERO simulator steps inside the learner, so this is the whole run::

    export POSTTRAIN_BASE_CKPT=/path/to/GR00T-N1.7-LIBERO/libero_10 LIBERO_CONFIG_PATH=...
    uv run python -m example_workflows.libero_posttraining.run 8 --dry-run
    uv run python -m example_workflows.libero_posttraining.run 8 algorithm.batch_size=256
"""

from __future__ import annotations

import argparse
import os

from isaac_rl_posttraining import Env, launch
from isaac_rl_posttraining.algorithms import DSRL, BaseModel

#: The GR00T-N1.7-LIBERO reference protocol: 720-step episodes (the simulator's gr00t_eval block).
EPISODE_CAP = 720


def recipe(
    task: str, base_ckpt: str, suite: str = "libero_10"
) -> tuple[DSRL, Env, list[str]]:
    """Algorithm, environments and remaining overrides of ``train.sh TASK`` with the frozen GR00T policy."""
    base = BaseModel("gr00t", base_ckpt, embodiment_tag="libero_sim")
    algorithm = DSRL(base=base, noise_dim=7, action_exec_len=8)
    env = Env(
        f"libero/{suite}/{task}",
        simulator=suite,
        max_episode_steps=EPISODE_CAP,
        image_keys=["video.image"],
    )
    return algorithm, env, ["training.num_rollouts=100000", "eval.eval_max_steps=200"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", help="LIBERO task name or its index in the suite")
    parser.add_argument(
        "overrides", nargs="*", help="further overrides, as train.sh takes them"
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and start nothing"
    )
    args = parser.parse_args()
    algorithm, env, overrides = recipe(
        args.task, os.environ["POSTTRAIN_BASE_CKPT"], args.suite
    )
    run = launch(
        algorithm,
        env=env,
        overrides=[*overrides, *args.overrides],
        dry_run=args.dry_run,
    )
    print("\n".join(run.plan.lines()))
    if not args.dry_run:
        run.loop()


if __name__ == "__main__":
    main()
