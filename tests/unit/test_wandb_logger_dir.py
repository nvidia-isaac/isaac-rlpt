# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""W&B run files go where the user said (logging.wandb_log_dir_base, else WANDB_DIR), and offline mode stays local."""

import os
from unittest.mock import Mock

import pytest

wandb = pytest.importorskip("wandb")

from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger, wandb_run_dir


def test_the_run_directory_prefers_the_config_then_wandb_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    assert wandb_run_dir(None, "run") == os.path.join("logs/wandb", "run")
    assert wandb_run_dir(str(tmp_path), "run") == str(tmp_path / "run")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "shared"))
    # W&B reads WANDB_DIR itself when no directory is passed.
    assert wandb_run_dir(None, "run") is None
    assert wandb_run_dir(str(tmp_path), "run") == str(tmp_path / "run")


def test_offline_runs_set_their_own_mode_without_touching_the_environment(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    init = Mock(return_value=Mock())
    monkeypatch.setattr(wandb, "init", init)
    WandbLogger(
        exp_name="run", offline=True, project="p", log_dir=wandb_run_dir(None, "run")
    )
    kwargs = init.call_args.kwargs
    assert kwargs["mode"] == "offline" and kwargs["dir"] is None
    assert "WANDB_MODE" not in os.environ
    WandbLogger(exp_name="run", project="p", log_dir=str(tmp_path / "explicit" / "run"))
    assert "mode" not in init.call_args.kwargs
    assert init.call_args.kwargs["dir"] == str(tmp_path / "explicit" / "run")
    assert (tmp_path / "explicit" / "run").is_dir()
