# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint names must remain acceptable to the real W&B artifact SDK."""

from pathlib import Path
from unittest.mock import Mock

import pytest

wandb = pytest.importorskip("wandb")

from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger


@pytest.mark.parametrize(
    "run_name", ["short", "workflow-b128-noeval-xy1cm-20260917_2026-09-18_04-58-24" * 3]
)
def test_checkpoint_artifact_names_are_valid_distinct_and_stable(
    run_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
    run = Mock()
    monkeypatch.setattr(wandb, "init", Mock(return_value=run))
    logger = WandbLogger(exp_name=run_name, project="artifact-name-test")
    paths = []
    for tag in (66432, 132864):
        path = tmp_path / str(tag)
        path.mkdir()
        (path / "manifest.json").write_text("{}")
        paths.append(path)
    # Called the way the checkpoint store does: the logger names the artifact <run>-<tag> from the directory.
    for path in (paths[0], paths[1], paths[0]):
        logger.log_artifact(
            path=str(path),
            type="checkpoint",
            metadata={"env_steps": int(path.name)},
            aliases=[f"rollout-{path.name}"],
        )
    calls = run.log_artifact.call_args_list
    artifacts = [call.args[0] for call in calls]
    assert all(len(artifact.name) <= 128 for artifact in artifacts)
    assert artifacts[0].name != artifacts[1].name
    assert artifacts[0].name == artifacts[2].name
    assert artifacts[0].metadata == {"env_steps": 66432}
    assert calls[0].kwargs["aliases"] == ["rollout-66432"]
    if run_name == "short":
        assert artifacts[0].name == "short-66432"
