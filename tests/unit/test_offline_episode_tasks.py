# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline training labels each frame with that episode's task, not one shared string."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np


def _two_task_dataset(tmp_path):
    import pandas as pd

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        )
    )
    (meta / "modality.json").write_text(
        json.dumps({"video": {}, "state": {}, "action": {}})
    )
    (meta / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "open the drawer"}),
                json.dumps({"task_index": 1, "task": "close the drawer"}),
            ]
        )
        + "\n"
    )
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    for episode, task_index in enumerate((0, 1)):
        rows = {
            "observation.state": [np.zeros(2, np.float32), np.ones(2, np.float32)],
            "action": [np.zeros(2, np.float32), np.ones(2, np.float32)],
            "task_index": [task_index, task_index],
        }
        pd.DataFrame(rows).to_parquet(data / f"episode_{episode:06d}.parquet")
    return tmp_path


class _Policy:
    video_keys = ("video.image",)
    state_keys = ("state.q",)
    action_window = 1

    def __init__(self, instruction=""):
        self.instruction = instruction
        self.prompts = []

    def build_inputs(self, frames, states, instruction):
        self.prompts.append(list(instruction))
        return {}

    def encode_actions(self, states, actions):
        return actions


def test_pi0_and_the_demonstration_buffer_keep_each_episodes_task(
    tmp_path, monkeypatch
):
    root = _two_task_dataset(tmp_path)
    from isaac_rl_posttraining.policies.pi0.offline import open_pi0_offline_source
    from isaac_rl_posttraining.training.demonstration_buffer import (
        load_demonstration_buffer,
    )

    policy = _Policy()
    source = open_pi0_offline_source(str(root), policy=policy)
    monkeypatch.setattr(source, "_frame", lambda *_args: np.zeros((2, 2, 3), np.uint8))
    # index 0 is episode 0, index 2 is the first frame of episode 1.
    monkeypatch.setattr(np.random, "randint", lambda *_a, **_k: np.array([0, 2]))
    source.sample(2)
    assert policy.prompts[0] == ["open the drawer", "close the drawer"]

    overridden = _Policy("put the bowl away")
    source = open_pi0_offline_source(str(root), policy=overridden)
    monkeypatch.setattr(source, "_frame", lambda *_args: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(np.random, "randint", lambda *_a, **_k: np.array([0, 2]))
    source.sample(2)
    assert overridden.prompts[0] == ["put the bowl away", "put the bowl away"]

    buffer = load_demonstration_buffer(
        root, action_horizon=1, cameras=(), pad_episode_end=True
    )
    info = buffer.collate(range(len(buffer)))["info"]
    assert [row["task"] for row in info] == [
        "open the drawer",
        "open the drawer",
        "close the drawer",
        "close the drawer",
    ]


def test_gr00t_source_uses_per_row_tasks_and_the_override_helper():
    from isaac_rl_posttraining.policies.gr00t.demonstrations import (
        Gr00tN17DemonstrationSource,
    )
    from isaac_rl_posttraining.policies.gr00t.offline import _row_tasks

    source = Gr00tN17DemonstrationSource(
        SimpleNamespace(max_action_horizon=1, max_action_dim=1),
        modality_layout={"state": {}, "action": {}},
        embodiment_tag="new_embodiment",
        cameras=(),
        instruction="first line only",
    )
    assert source._instructions({}, 2) == ["first line only", "first line only"]
    assert source._instructions(
        {"tasks": ["open the drawer", "close the drawer"]}, 2
    ) == [
        "open the drawer",
        "close the drawer",
    ]
    batch = {
        "reward": np.zeros(2),
        "info": [{"task": "open the drawer"}, {"task": "close the drawer"}],
    }
    assert _row_tasks(batch, "") == ["open the drawer", "close the drawer"]
    assert _row_tasks(batch, "put the bowl away") == [
        "put the bowl away",
        "put the bowl away",
    ]
