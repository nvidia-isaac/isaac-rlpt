# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO's metric sink writes every iteration and logs evaluations like the off-policy trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from loguru import logger

from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.runners.ppo_launch import metric_sink


class _Logger:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int, str]] = []

    def log(self, values, step, prefix):
        self.logged.append((dict(values), step, prefix))


def test_evaluations_log_the_common_success_rate_line(tmp_path: Path) -> None:
    lines: list[str] = []
    handle = logger.add(lambda message: lines.append(message.record["message"]))
    try:
        write = metric_sink(tmp_path, cast(Logger, _Logger()))
        write({"train/success_rate": 0.5}, 1)
        write({"eval/success_rate": 0.75}, 2)
    finally:
        logger.remove(handle)

    assert "[EVAL] Success Rate: 75.0%" in lines
    assert sum(line.startswith("[EVAL]") for line in lines) == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "ppo_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [1, 2]
