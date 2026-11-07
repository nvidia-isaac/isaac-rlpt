# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The offline run on a toy policy: scheduled saves, the final checkpoint once, and an interrupted save."""

from __future__ import annotations

import os
import signal

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.runners import offline_launch
from isaac_rl_posttraining.runners.build import build_run
from isaac_rl_posttraining.runtimes.policy_snapshot import load_policy_snapshot
from tests.unit.test_train_entry import CONFIG_DIR


class _ToyPolicy(nn.Module):
    frozen_submodules = ("backbone",)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)
        self.value_head = nn.Linear(2, 1)

    def actor_bc_loss(self, inputs):
        return self.head(inputs["obs"]).pow(2).sum(dim=1)


class _Source:
    def __init__(self, interrupt_at: int | None) -> None:
        self.draws = 0
        self.interrupt_at = interrupt_at

    def sample(self, batch_size: int) -> dict:
        self.draws += 1
        if self.draws == self.interrupt_at:
            os.kill(os.getpid(), signal.SIGTERM)
        return {
            "inputs_cur": {"obs": torch.ones(batch_size, 2)},
            "action": torch.zeros(batch_size, 1, 2),
        }


class _Logger:
    def __init__(self, **kwargs) -> None:
        self.rows: list[tuple[int, str]] = []

    def log(self, metrics, *, step, prefix=None):
        self.rows.append((step, prefix))


def _run(monkeypatch, tmp_path, *overrides, interrupt_at=None):
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        offline_launch.TrainablePolicySpec, "build", lambda self, device: _ToyPolicy()
    )
    monkeypatch.setattr(
        offline_launch,
        "open_offline_source",
        lambda *args, **kwargs: _Source(interrupt_at),
    )
    from isaac_rl_posttraining import loggers

    monkeypatch.setattr(loggers, "WandbLogger", _Logger)
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "algorithm=sft",
                "base_model=gr00t_n17",
                "base_model.ckpt=/ckpt/n17",
                "base_model.embodiment_tag=new_embodiment",
                "base_model.action_horizon=16",
                "algorithm.batch_size=2",
                "eval.eval_freq=0",
                "eval.eval_on_first_step=false",
                *overrides,
            ],
        )
    return build_run(cfg)()


def test_derived_eval_freq_stays_at_least_one_update():
    cfg = OmegaConf.create(
        {
            "eval": {"eval_freq": None, "eval_on_first_step": False},
            "training": {"num_rollouts": 50},
        }
    )
    assert offline_launch._eval_freq(cfg) == 1
    cfg.training.num_rollouts = 250
    assert offline_launch._eval_freq(cfg) == 2
    cfg.eval.eval_freq = 0
    assert offline_launch._eval_freq(cfg) == 0


def test_offline_run_saves_on_schedule_and_the_final_checkpoint_once(
    monkeypatch, tmp_path
):
    run_dir = _run(
        monkeypatch, tmp_path, "training.num_rollouts=5", "training.save_interval=2"
    )
    save_dir = os.path.join(run_dir, "checkpoints")
    assert sorted(os.listdir(save_dir)) == ["2", "4", "final"]
    state = load_policy_snapshot(os.path.join(save_dir, "final"))
    # The snapshot holds the trained weights only: the frozen backbone stays out.
    assert sorted(state.tensors) == [
        "head.bias",
        "head.weight",
        "value_head.bias",
        "value_head.weight",
    ]


def test_sigterm_saves_the_interrupted_checkpoint(monkeypatch, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(
            monkeypatch,
            tmp_path,
            "training.num_rollouts=10",
            "training.save_interval=0",
            interrupt_at=4,
        )
    (run_dir,) = os.listdir(tmp_path)
    assert os.listdir(os.path.join(tmp_path, run_dir, "checkpoints")) == ["interrupted"]
