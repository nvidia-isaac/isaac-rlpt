# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the stack-cube SFT checkpoint with RLinf's own pipeline inside Isaac Lab 3.0 (baseline oracle).

RLinf's Isaac Lab example evaluates ``RLinf-Gr00t-SFT-Stack-cube`` on its own task id
``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0`` (Isaac Lab's ``...-Visuomotor-v0``
plus a dense PPO reward) with a four-GPU placement. Isaac Lab 3.0.0-beta2 does not register
that id and the job has one GPU, so this wrapper runs Isaac Lab's ``play_rlinf.py`` with:

* an RLinf extension module (this file; ``register()``) that registers the ``-Rewarded-v0``
  id onto the plain visuomotor config -- RLinf's evaluation reward is the success
  termination, so no dense reward is needed -- and points the Franka USD at the asset
  server's ``Legacy/`` path (the 3.0.0 container's path is gone),
* Hydra composition patched to a single-GPU placement, no video, and
  ``ORACLE_EPISODES`` sequential episodes on one environment.

Usage (inside the container, after ``setup_rlinf.sh``; RLinf's repo provides the configs)::

    export EMBODIED_PATH=/path/to/RLinf/examples/embodiment RLINF_EXT_MODULE=rlinf_stack_cube_oracle
    export PYTHONPATH=$(dirname "$0"):$PYTHONPATH ORACLE_EPISODES=20
    python rlinf_stack_cube_oracle.py --config_path $EMBODIED_PATH/config \\
        --config_name isaaclab_franka_stack_cube_ppo_gr00t --model_path <ckpt> --num_envs 1
"""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
from pathlib import Path

REWARDED_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
VISUOMOTOR_CFG = (
    "isaaclab_tasks.manager_based.manipulation.stack.config.franka."
    "stack_ik_rel_visuomotor_env_cfg:FrankaCubeStackVisuomotorEnvCfg"
)
FRANKA_LEGACY_USD = "Robots/FrankaEmika/Legacy/panda_instanceable.usd"


def register() -> None:
    """RLinf extension hook (``RLINF_EXT_MODULE``): task alias plus the Franka asset fix."""
    import gymnasium as gym

    if REWARDED_TASK_ID not in gym.registry:
        gym.register(
            id=REWARDED_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": VISUOMOTOR_CFG},
            disable_env_checker=True,
        )
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab_assets.robots import franka

    for name in ("FRANKA_PANDA_CFG", "FRANKA_PANDA_HIGH_PD_CFG"):
        cfg = getattr(franka, name, None)
        if cfg is not None and "FrankaEmika" in str(cfg.spawn.usd_path):
            cfg.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_LEGACY_USD}"


def _patch_compose(episodes: int, log_root: Path) -> None:
    """Make ``hydra.compose`` return RLinf's stack-cube config adapted to one GPU."""
    import hydra
    from omegaconf import open_dict

    original = hydra.compose

    def compose(config_name=None, overrides=None, **kwargs):
        cfg = original(config_name=config_name, overrides=overrides, **kwargs)
        with open_dict(cfg):
            cfg.cluster.num_nodes = 1
            cfg.cluster.component_placement = {"actor,env,rollout": "all"}
            cfg.cluster.pop("node_groups", None)
            cfg.env.eval.total_num_envs = 1
            cfg.env.eval.rollout_epoch = (
                episodes  # one env -> one episode per rollout epoch
            )
            cfg.env.eval.video_cfg.save_video = False
            cfg.runner.logger.log_path = str(log_root)
        return cfg

    hydra.compose = compose


def main() -> None:
    scripts_dir = Path(os.environ["ISAAC_RLPT_RL_SCRIPTS"]) / "rlinf"
    episodes = int(os.environ.get("ORACLE_EPISODES", "10"))
    if "ORACLE_LOG_ROOT" in os.environ:
        log_root = Path(os.environ["ORACLE_LOG_ROOT"])
    else:
        log_root = Path(tempfile.mkdtemp(prefix="rlinf_oracle_"))
    log_root.mkdir(parents=True, exist_ok=True)
    os.environ["RLINF_EXT_MODULE"] = "rlinf_stack_cube_oracle"
    here = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = (
        here + os.pathsep + os.environ.get("PYTHONPATH", "")
    )  # Ray workers import this module
    sys.path.insert(0, here)
    sys.path.insert(0, str(scripts_dir))
    register()
    _patch_compose(episodes, log_root)
    sys.argv = [str(scripts_dir / "play_rlinf.py"), *sys.argv[1:]]
    runpy.run_path(str(scripts_dir / "play_rlinf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
