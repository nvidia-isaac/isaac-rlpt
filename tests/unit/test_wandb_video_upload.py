# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A video upload the environment cannot encode is skipped, never fatal to the run."""

import sys
import types
from unittest.mock import Mock

import pytest
import torch

wandb = pytest.importorskip("wandb")

from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger


def test_moviepy_without_imageio_skips_the_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    run = Mock()
    monkeypatch.setattr(wandb, "init", Mock(return_value=run))
    # moviepy imports, but the encoder wandb.Video uses fails (as when imageio is missing).
    monkeypatch.setitem(sys.modules, "moviepy", types.ModuleType("moviepy"))
    monkeypatch.setitem(sys.modules, "moviepy.video.io.ImageSequenceClip", None)
    logger = WandbLogger(exp_name="video-test", project="video-test")
    logger.initialize()
    with pytest.warns(UserWarning, match="skipping wandb video upload"):
        logger.log_video("eval", torch.zeros(1, 2, 3, 4, 4, dtype=torch.uint8), step=0)
    run.log.assert_not_called()
