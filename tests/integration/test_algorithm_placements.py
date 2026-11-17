# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every steering and edit algorithm trains through the entry point with every policy placement.

Two rollout workers in one process act with the policy local in each worker,
served by an in-process policy server, or with only the frozen base model
served (the actor stays in each worker), on the out-of-tree stand-in robot
and the mock base model. The same placements on Ray actors and RLInf are
tested at the runtime level (test_native_ray_mixed_placement.py,
test_rlinf_mixed_placement.py); DSRL-PPO also trains through the entry point
on RLInf (the RLInf CI lane).
"""

from __future__ import annotations

import json

import pytest
import wandb
from hydra import compose, initialize_config_dir

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger

from .test_out_of_tree_entry_launch import CONFIG_DIR, OVERRIDES, install_robot_package

#: The robot and model geometry every algorithm shares, without the algorithm and its schedule.
ROBOT = [
    override
    for override in OVERRIDES
    if override.split("=", 1)[0].split(".", 1)[0]
    in ("base_model", "env", "+simulator", "model")
    or override.startswith(
        ("policy.", "value_function.", "eval.eval_num", "eval.eval_record")
    )
]
OFF_POLICY = [
    "training.chunk_size=16",
    "dsrl.noise_dim=7",
    "dsrl.action_exec_len=8",
    "algorithm.num_critics=2",
    "algorithm.num_updates_per_train_step=2",
    "training.num_rollouts=48",
    "algorithm.learning_starts=16",
    "eval.eval_freq=24",
    "training.save_interval=24",
]
ALGORITHMS = {
    "dsrl_sac": ["algorithm=dsrl_sac", *OFF_POLICY],
    "rfs": ["algorithm=rfs", *OFF_POLICY],
    "rfs_pc": [
        "algorithm=rfs",
        *OFF_POLICY,
        "rfs.proposal_conditioned_residual=true",
        "rfs.reference_action_conditioning=true",
    ],
    "sac_edit": ["algorithm=sac_edit", *OFF_POLICY, "training.chunk_size=null"],
    "dsrl_ppo": [
        "algorithm=dsrl_ppo",
        "training.chunk_size=16",
        "dsrl.noise_dim=7",
        "dsrl.action_exec_len=8",
        "ppo.iterations=1",
        "ppo.eval_every=1",
        "ppo.global_batch_size=4",
        "ppo.update_epochs=1",
    ],
}
PLACEMENTS = {
    "local": "placement.policy.mode=local",
    "served": "placement.policy.mode=served",
    "base_served": "placement.parts.base.mode=served",
}


@pytest.fixture
def robot_package(tmp_path, monkeypatch):
    package = install_robot_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    # A W&B service left by an earlier offline run in this process would fail this run's logging.
    wandb.teardown()
    yield package
    providers._SIMULATOR_ENTRIES.pop("my_robot", None)


def _train(tmp_path, monkeypatch, run_overrides):
    """Run the entry point with two rollout workers and check it evaluated and saved."""
    from scripts.train import run_from_hydra_cfg

    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(tmp_path / "outputs"))
    # Offline W&B runs cannot follow each other in one process (wandb-core), so none is created.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("ISAAC_RL_POSTTRAINING_DRYRUN", raising=False)
    monkeypatch.chdir(tmp_path)
    overrides = [
        *ROBOT,
        *run_overrides,
        "training.num_envs=2",
        "placement.envs.workers=2",
        "logging.wandb_name=robot_under_test",
    ]
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        run_from_hydra_cfg(compose(config_name="train", overrides=overrides))

    (run,) = sorted((tmp_path / "outputs").glob("robot_under_test_*"))
    summary = json.loads((run / "checkpoints" / "evaluation_summary.json").read_text())
    assert summary["history"], "the run must evaluate"
    assert (run / "checkpoints" / "final").is_dir()


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_every_algorithm_trains_with_every_policy_placement(
    robot_package, tmp_path, monkeypatch, algorithm, placement
):
    """Two rollout workers in one process, the policy local, served, or with only its base model served."""
    _train(
        tmp_path,
        monkeypatch,
        [
            *ALGORITHMS[algorithm],
            "runtime.native.device=cpu",
            "runtime.native.processes=single",
            PLACEMENTS[placement],
        ],
    )


@pytest.mark.parametrize("algorithm", ["dsrl_sac", "sac_edit"])
def test_learner_rows_step_on_env_steps_and_count_updates(
    robot_package, tmp_path, monkeypatch, algorithm
):
    """Every ``learner/*`` row is logged at the run's environment steps and carries ``learner/updates``."""
    rows: list[tuple[int, dict]] = []
    log_dict = WandbLogger.log_dict

    def record(self, dictionary, step=None, prefix=None, **kwargs):
        if prefix == "learner":
            rows.append((step, dict(dictionary)))
        return log_dict(self, dictionary, step, prefix, **kwargs)

    monkeypatch.setattr(WandbLogger, "log_dict", record)
    _train(
        tmp_path,
        monkeypatch,
        [
            *ALGORITHMS[algorithm],
            "runtime.native.device=cpu",
            "runtime.native.processes=single",
        ],
    )

    steps = [step for step, _ in rows]
    assert len(rows) >= 2
    assert steps == sorted(steps) and steps[-1] == 48
    updates = [metrics["updates"] for _, metrics in rows]
    assert updates == sorted(updates) and updates[0] > 0


@pytest.mark.rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_dsrl_ppo_trains_on_rlinf(robot_package, tmp_path, monkeypatch, placement):
    """RLInf rollout workers on the CPU; the one PPO learner rank trains in this process."""
    _train(
        tmp_path,
        monkeypatch,
        [
            *ALGORITHMS["dsrl_ppo"],
            "runtime=rlinf",
            "runtime.rlinf.device=cpu",
            PLACEMENTS[placement],
        ],
    )
