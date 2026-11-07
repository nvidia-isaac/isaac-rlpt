# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the N1.5 source hands the checkpoint's transform, and what binds it.

The transform normalizes, pads and masks; the source owns only the *layout*,
which is the part that can be wrong without raising -- a transposed group
still trains, against the wrong joints. So is which statistics are bound.

A recording transform stands in for the real one.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

CAMERA = "room_view"
EMBODIMENT = "new_embodiment"
HORIZON = 4
STATE_DIM = 5
ACTION_DIM = 3
INSTRUCTION = "do the thing"
CHECKPOINT_STATS = "checkpoint-statistics"
SAMPLES = 2


class _GR00TTransform:
    """The stage carrying the Eagle processor the batch is collated with."""

    def __init__(self) -> None:
        self.eagle_processor = "the-transform's-own-processor"


class _Transform:
    """Records the samples it is asked to encode."""

    def __init__(self) -> None:
        self.transforms = [_GR00TTransform()]
        self.seen: list[dict[str, Any]] = []
        self.metadata: Any = None
        self.training = False

    def set_metadata(self, metadata: Any) -> None:
        self.metadata = metadata

    def train(self) -> None:
        self.training = True

    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(sample)
        return sample


def _fake_gr00t(monkeypatch) -> None:
    """Install the ``gr00t`` pieces the source reaches for."""
    transforms = types.ModuleType("gr00t.model.transforms")
    transforms.GR00TTransform = _GR00TTransform
    transforms.collate = lambda features, processor: {
        "rows": len(features),
        "processor": processor,
    }
    schema = types.ModuleType("gr00t.data.schema")
    schema.DatasetMetadata = types.SimpleNamespace(
        model_validate=lambda block: CHECKPOINT_STATS
    )
    for name, module in {
        "gr00t": types.ModuleType("gr00t"),
        "gr00t.data": types.ModuleType("gr00t.data"),
        "gr00t.data.schema": schema,
        "gr00t.model": types.ModuleType("gr00t.model"),
        "gr00t.model.transforms": transforms,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset root carrying the modality layout the source reads."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {
                    "arm": {"start": 0, "end": 3},
                    "hand": {"start": 3, "end": 5},
                },
                "action": {"arm": {"start": 0, "end": 3}},
                "video": {CAMERA: {"original_key": f"observation.images.{CAMERA}"}},
            }
        )
    )
    (meta / "tasks.jsonl").write_text(json.dumps({"task": INSTRUCTION}))
    return tmp_path


def _source(dataset: Path) -> Any:
    from isaac_rl_posttraining.policies.gr00t.demonstrations import (
        Gr00tN15DemonstrationSource,
    )

    return Gr00tN15DemonstrationSource(
        _Transform(),
        modality_layout=json.loads((dataset / "meta" / "modality.json").read_text()),
        cameras=[CAMERA],
        instruction=INSTRUCTION,
    )


def _batch() -> dict[str, Any]:
    """A drawn batch in the shape the buffer's collate produces."""
    return {
        "obs": {
            "state": np.arange(SAMPLES * STATE_DIM, dtype=np.float32).reshape(
                SAMPLES, STATE_DIM
            ),
            CAMERA: np.zeros((SAMPLES, 8, 8, 3), np.uint8),
        },
        "action": np.arange(SAMPLES * HORIZON * ACTION_DIM, dtype=np.float32).reshape(
            SAMPLES, HORIZON, ACTION_DIM
        ),
    }


def test_each_window_reaches_the_transform_in_the_datasets_layout(dataset, monkeypatch):
    """A transposed group or a lost time axis would still train, on the wrong thing."""
    _fake_gr00t(monkeypatch)
    source = _source(dataset)
    batch = _batch()

    collated = source(batch)

    # A second Eagle processor could tokenize differently from the policy's.
    assert collated == {"rows": SAMPLES, "processor": "the-transform's-own-processor"}
    # Only training mode emits the action target and its mask.
    assert source.transform.training is True
    assert len(source.transform.seen) == SAMPLES
    sample = source.transform.seen[1]
    state = batch["obs"]["state"][1]
    assert sample["state.arm"].shape == (1, 3)
    np.testing.assert_array_equal(sample["state.arm"][0], state[0:3])
    np.testing.assert_array_equal(sample["state.hand"][0], state[3:5])
    assert sample["action.arm"].shape == (HORIZON, 3)
    np.testing.assert_array_equal(sample["action.arm"], batch["action"][1][:, 0:3])
    assert sample[f"video.{CAMERA}"].shape == (1, 8, 8, 3)
    # The image processor requires uint8; a float frame is not an image.
    assert sample[f"video.{CAMERA}"].dtype == np.uint8
    assert sample["annotation.human.task_description"] == [INSTRUCTION]


# ----------------------------------------------------------------------
# The builder: which statistics bind, and which windows are refused
# ----------------------------------------------------------------------
@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A checkpoint carrying statistics for the embodiment under test."""
    directory = tmp_path / "ckpt" / "experiment_cfg"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({EMBODIMENT: {"statistics": {}}})
    )
    return tmp_path / "ckpt"


def _fake_data_config(monkeypatch) -> _Transform:
    """Resolve the data config to one action window and a recording transform."""
    from isaac_rl_posttraining.policies.gr00t import n15

    transform = _Transform()
    config = types.SimpleNamespace(
        modality_config=lambda: {
            "action": types.SimpleNamespace(delta_indices=list(range(HORIZON))),
            "video": types.SimpleNamespace(modality_keys=[f"video.{CAMERA}"]),
        },
        transform=lambda: transform,
    )
    monkeypatch.setattr(n15, "resolve_data_config", lambda spec: config)
    monkeypatch.setattr(n15, "resolve_model_embodiment_tag", lambda tag: tag)
    return transform


def _from_checkpoint(dataset: Path, checkpoint: Path, **overrides: Any) -> Any:
    from isaac_rl_posttraining.policies.gr00t.demonstrations import (
        Gr00tN15DemonstrationSource,
    )

    options: dict[str, Any] = {
        "embodiment_tag": EMBODIMENT,
        "data_config": "fake:Config",
    }
    options.update(overrides)
    return Gr00tN15DemonstrationSource.from_checkpoint(checkpoint, dataset, **options)


def test_the_checkpoints_statistics_are_what_binds(dataset, checkpoint, monkeypatch):
    """The dataset's own statistics would renormalize the anchor."""
    _fake_gr00t(monkeypatch)
    transform = _fake_data_config(monkeypatch)

    _from_checkpoint(dataset, checkpoint)

    assert transform.metadata == CHECKPOINT_STATS


def test_a_checkpoint_without_the_embodiment_is_rejected(
    dataset, checkpoint, monkeypatch
):
    """Falling back to the dataset's statistics would be the silent failure."""
    _fake_gr00t(monkeypatch)
    _fake_data_config(monkeypatch)

    with pytest.raises(KeyError, match="no statistics for embodiment"):
        _from_checkpoint(dataset, checkpoint, embodiment_tag="absent_embodiment")


def test_a_window_the_policy_does_not_execute_is_rejected(
    dataset, checkpoint, monkeypatch
):
    """A target of another length teaches a geometry the model cannot emit."""
    from isaac_rl_posttraining.policies.gr00t.demonstrations import (
        open_gr00t_n15_demonstration_source,
    )

    _fake_gr00t(monkeypatch)
    _fake_data_config(monkeypatch)

    with pytest.raises(ValueError, match=f"span {HORIZON} steps but the policy"):
        open_gr00t_n15_demonstration_source(
            str(dataset),
            checkpoint=str(checkpoint),
            embodiment_tag=EMBODIMENT,
            data_config="fake:Config",
            action_horizon=HORIZON + 4,
        )
