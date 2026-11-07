# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""50-step DSRL training smoke via the launcher (cluster integration, arena/gpu).

Drives ``example_workflows/g1_arena_posttraining/scripts/train.sh`` end to end -- two Arena env servers plus the
learner -- for a 50-env-step run with a tiny learning-starts threshold, and
pins the three artifacts a real run depends on:

* SAC updates are logged (critic/actor losses appear in the train log),
* a checkpoint is saved,
* a wandb offline run directory is written.

The launcher owns process lifecycle (PTY boot, EULA env, readiness waits,
SIGTERM teardown), so this test only prepares the environment and inspects
the outputs. Expect a long wall clock (two Isaac boots); the subprocess
timeout is generous.
"""

import glob
import os
import re
import subprocess

import pytest
from conftest import (
    TRAIN_LAUNCHER,
    require_arena_prereqs,
)

pytestmark = [
    pytest.mark.arena,
    pytest.mark.gpu,
    pytest.mark.timeout(0),
]  # boots Isaac Sim

TASK = os.environ.get("ISAAC_RL_POSTTRAINING_TEST_TASK", "DrillLift")
NUM_STEPS = 50
TIMEOUT_S = int(os.environ.get("ISAAC_RL_POSTTRAINING_SMOKE_TIMEOUT", str(45 * 60)))

#: Hydra overrides that shrink the run to a smoke: train immediately, no
#: initial eval, short episodes, offline wandb, one checkpoint before the end.
SMOKE_OVERRIDES = [
    f"training.num_rollouts={NUM_STEPS}",
    "training.save_interval=40",
    "algorithm.learning_starts=16",
    "algorithm.batch_size=8",
    "eval.eval_on_first_step=false",
    "eval.eval_freq=0",
    "eval.eval_num_episodes=1",
    "eval.eval_record_video=false",
    "env.max_episode_steps=60",
    "logging.wandb_offline=true",
]


def _require_training_prereqs():
    require_arena_prereqs()
    assert os.path.isfile(TRAIN_LAUNCHER), f"train launcher missing at {TRAIN_LAUNCHER}"
    for var in ("POSTTRAIN_BASE_CKPT", "POSTTRAIN_DATA_ROOT"):
        value = os.environ.get(var)
        if not value or not os.path.isdir(value):
            pytest.skip(
                f"{var} not set / not a directory (real checkpoint + dataset required)"
            )


def _find(root, patterns):
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return hits


def test_training_smoke_50_steps(tmp_path):
    """A 50-step launcher run logs SAC updates, saves a checkpoint, writes wandb-offline."""
    _require_training_prereqs()

    out_dir = tmp_path / "outputs"
    log_dir = tmp_path / "launcher_logs"
    out_dir.mkdir()
    log_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "NUM_ROLLOUTS": str(NUM_STEPS),
            "LOG_DIR": str(log_dir),
            "ISAAC_RL_POSTTRAINING_OUTPUT_DIR": str(out_dir),
            "RLPL_OUTPUT_DIR": str(out_dir),  # legacy output knob, pre-rename launchers
            "GPU": os.environ.get("ISAAC_RL_POSTTRAINING_TEST_GPU", "0"),
            "PORT_BASE": os.environ.get("ISAAC_RL_POSTTRAINING_TEST_PORT_BASE", "7500"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("ISAAC_RL_POSTTRAINING_DRYRUN", None)

    result = subprocess.run(
        ["bash", TRAIN_LAUNCHER, TASK] + SMOKE_OVERRIDES,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )

    log_text = (result.stdout or "") + (result.stderr or "")
    for log_file in _find(str(log_dir), ["*.log"]):
        try:
            with open(log_file, "rb") as fh:
                log_text += fh.read().decode("utf-8", errors="replace")
        except OSError:
            pass

    assert result.returncode == 0, (
        f"launcher smoke run failed (rc={result.returncode}); combined log tail:\n{log_text[-6000:]}"
    )

    # 1. SAC updates were logged.
    assert re.search(r"(critic_loss|actor_loss|Training \d+ updates)", log_text), (
        "no SAC update evidence in the training log (learning_starts plumbing or the "
        f"train loop is broken); log tail:\n{log_text[-6000:]}"
    )

    # 2. A checkpoint was saved.
    ckpt_hits = _find(str(tmp_path), ["*.pt", "*.pth", "*.safetensors"])
    ckpt_hits = [h for h in ckpt_hits if "wandb" not in h]
    assert ckpt_hits, (
        f"no checkpoint artifact found under {tmp_path}; log tail:\n{log_text[-4000:]}"
    )

    # 3. A wandb offline run directory was written.
    wandb_hits = _find(str(tmp_path), ["offline-run-*", "*.wandb"])
    assert wandb_hits, (
        f"no wandb offline run written under {tmp_path}; log tail:\n{log_text[-4000:]}"
    )
