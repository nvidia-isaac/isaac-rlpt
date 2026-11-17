# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving the co-training source a config asks for.

Nothing here needs a dataset on disk beyond the miniature one each test
builds, so this file runs anywhere. What a real dataset would add -- that a
base model's encoding matches its own -- needs a checkpoint too, and belongs with
the base model that owns the processor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isaac_rl_posttraining.training.demonstrations import LeRobotDemonstrationReader

STATE_DIM = 28


def _write_dataset(root: Path, *, episodes: int = 3, length: int = 40) -> Path:
    """A minimal LeRobot v2.1 tree: metadata plus one parquet per episode."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": episodes,
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {
                    "left_arm": {"start": 0, "end": 7},
                    "right_arm": {"start": 7, "end": 14},
                    "left_hand": {"start": 14, "end": 21},
                    "right_hand": {"start": 21, "end": 28},
                },
                "action": {"left_arm": {"start": 0, "end": 7}},
                "video": {"room_view": {"original_key": "observation.images.cam_room"}},
                "annotation": {
                    "human.task_description": {"original_key": "task_index"}
                },
            }
        )
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "do the thing"}) + "\n"
    )
    for episode in range(episodes):
        path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "observation.state": [
                    np.full(STATE_DIM, float(i), np.float32) for i in range(length)
                ],
                "action": [
                    np.full(STATE_DIM, float(i) + 0.5, np.float32)
                    for i in range(length)
                ],
                "task_index": np.zeros(length, dtype=np.int64),
            }
        ).to_parquet(path)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path / "ds")


def test_an_action_horizon_longer_than_every_episode_is_rejected(dataset: Path):
    """Fail at construction, not on the first batch mid-run."""
    with pytest.raises(ValueError, match="exceeds every episode"):
        LeRobotDemonstrationReader(dataset, action_horizon=500)


def test_unknown_camera_is_rejected_with_the_available_ones(dataset: Path):
    with pytest.raises(ValueError, match="unknown camera"):
        LeRobotDemonstrationReader(dataset, action_horizon=8, cameras=("nope",))


def test_missing_dataset_is_reported_by_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="info.json"):
        LeRobotDemonstrationReader(tmp_path / "absent", action_horizon=8)


# -------------------------------------------------------------- iterable ----
# ----------------------------------------------------------- from config ----
def _cfg(**overrides):
    from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig

    return PPOConfig(**overrides)


def test_disabled_cotraining_opens_no_source(dataset: Path):
    """beta=0 must not read the dataset at all."""
    from isaac_rl_posttraining.training.demonstrations import open_demonstration_source

    assert open_demonstration_source(_cfg(), action_horizon=8) == (None, None)


def test_beta_without_a_dataset_path_is_rejected(dataset: Path):
    """The gap this closes: beta alone could not reach any demonstrations."""
    from isaac_rl_posttraining.training.demonstrations import open_demonstration_source

    with pytest.raises(ValueError, match="cotrain_dataset_path is unset"):
        open_demonstration_source(_cfg(cotrain_beta=0.1), action_horizon=8)


def test_a_dataset_path_without_beta_is_rejected(dataset: Path):
    from isaac_rl_posttraining.training.demonstrations import open_demonstration_source

    with pytest.raises(ValueError, match="would never be read"):
        open_demonstration_source(
            _cfg(cotrain_dataset_path=str(dataset)), action_horizon=8
        )


def test_shipped_ppo_yaml_carries_the_dataset_fields():
    """A run must be able to enable co-training from config alone."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    ppo = yaml.safe_load(
        (root / "isaac_rl_posttraining/configs/algorithm/ppo.yaml").read_text()
    )["ppo"]
    assert ppo["cotrain_dataset_path"] is None  # opt-in


# ------------------------------------------------------- base model builders ----
class _Spec:
    """Stand-in for a TrainablePolicySpec."""

    def __init__(self, base_model, checkpoint="/ckpt", **options):
        self.base_model = base_model
        self.checkpoint = checkpoint
        self.options = options


def test_a_base_model_without_a_builder_is_rejected(dataset: Path):
    """Only a family that can encode demonstrations can score them.

    Returning a raw reader here would build a source the model cannot read,
    and the run would die later in the capability check instead of now.
    """
    from isaac_rl_posttraining.training.demonstrations import open_demonstration_source

    with pytest.raises(ValueError, match="no demonstration source builder"):
        open_demonstration_source(
            _cfg(cotrain_beta=0.1, cotrain_dataset_path=str(dataset)),
            action_horizon=8,
            trainable=_Spec("some_other_base_model"),
        )


def test_gr00t_n17_resolves_to_its_own_builder(dataset: Path, monkeypatch):
    """A policy family that encodes its own demonstrations must be used.

    Reimplementing a checkpoint's processor risks a silent normalization
    mismatch, so the base model's builder wins over the raw reader.
    """
    import isaac_rl_posttraining.training.demonstrations as module

    captured: dict = {}

    def fake_builder(path, **kwargs):
        captured.update(path=path, **kwargs)
        return "gr00t-source"

    monkeypatch.setattr(module, "_base_model_source_builder", lambda name: fake_builder)
    source = module.open_demonstration_source(
        _cfg(
            cotrain_beta=0.1,
            cotrain_dataset_path=str(dataset),
            cotrain_batch_size=8,
        ),
        action_horizon=16,
        trainable=_Spec(
            "gr00t_n17", checkpoint="/ckpt", embodiment_tag="new_embodiment"
        ),
    )
    assert source == "gr00t-source"
    assert captured["checkpoint"] == "/ckpt"
    assert captured["embodiment_tag"] == "new_embodiment"  # from spec.options
    assert captured["action_horizon"] == 16


def test_the_gr00t_builder_is_registered_for_n17():
    """The registry entry, without importing the model stack."""
    from isaac_rl_posttraining.training.demonstrations import _SOURCE_BUILDERS

    assert "gr00t_n17" in _SOURCE_BUILDERS
    assert _SOURCE_BUILDERS["gr00t_n17"].endswith(
        ":open_gr00t_n17_demonstration_source"
    )


def test_the_gr00t_builder_is_registered_for_n15():
    """Both generations are served from a buffer; only the source differs."""
    from isaac_rl_posttraining.training.demonstrations import _SOURCE_BUILDERS

    assert "gr00t_n15" in _SOURCE_BUILDERS
    assert _SOURCE_BUILDERS["gr00t_n15"].endswith(
        ":open_gr00t_n15_demonstration_source"
    )


def test_the_gr00t_builder_module_imports_without_the_model_stack():
    """Core must stay importable when gr00t is absent."""
    import isaac_rl_posttraining.policies.gr00t.demonstrations as builder

    assert callable(builder.open_gr00t_n17_demonstration_source)
