# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run ends its W&B run before it shuts its runtimes down."""

from types import SimpleNamespace
from typing import Any, cast

from isaac_rl_posttraining.runners.off_policy_launch import close_run_resources
from isaac_rl_posttraining.runners.ppo_launch import close_resources


class _Recorder:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def close(self) -> None:
        self.events.append(f"close {self.name}")

    def finish(self) -> None:
        self.events.append(f"finish {self.name}")


def test_off_policy_run_finishes_wandb_before_closing_runtimes() -> None:
    events: list[str] = []
    components = SimpleNamespace(
        env=_Recorder(events, "env"), wandb_logger=_Recorder(events, "wandb")
    )
    runtimes: list[Any] = [_Recorder(events, "runtime")]

    close_run_resources(cast(Any, components), runtimes)

    assert events == ["finish wandb", "close runtime", "close env"]


def test_on_policy_run_finishes_wandb_before_closing_runtimes() -> None:
    events: list[str] = []

    close_resources(
        (_Recorder(events, "workers"), None, _Recorder(events, "runtime")),
        _Recorder(events, "wandb"),
        None,
    )

    assert events == ["finish wandb", "close workers", "close runtime"]
