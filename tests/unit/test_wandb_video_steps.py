# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video rows keep the step they are given, and loggers do not share state."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

wandb = pytest.importorskip("wandb")

from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger


def _logger(monkeypatch: pytest.MonkeyPatch) -> tuple[WandbLogger, Mock]:
    run = Mock()
    monkeypatch.setattr(wandb, "init", Mock(return_value=run))
    return WandbLogger(exp_name="run", project="p", offline=True), run


def test_video_rows_log_the_given_step_on_each_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "moviepy",
        "moviepy.video",
        "moviepy.video.io",
        "moviepy.video.io.ImageSequenceClip",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(
        wandb, "Video", Mock(side_effect=lambda video, **kwargs: "video")
    )
    video = torch.zeros(1, 2, 3, 4, 4)
    first, first_run = _logger(monkeypatch)
    second, second_run = _logger(monkeypatch)

    for step in (0, 2400, 4808):
        first.log_video("eval_video", video, step=step, prefix="eval")
    second.log_video("eval_video", video, step=0, prefix="eval")

    assert [call.args[0]["eval/step"] for call in first_run.log.call_args_list] == [
        0,
        2400,
        4808,
    ]
    assert second_run.log.call_args.args[0]["eval/step"] == 0
