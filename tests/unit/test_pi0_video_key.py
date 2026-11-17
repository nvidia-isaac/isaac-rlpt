# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0 reads LeRobot v2's ``observation.images.<cam>`` video folders."""

from __future__ import annotations

import json
import os

import numpy as np


def test_video_path_uses_the_info_json_feature_not_the_slash_mapping(tmp_path):
    import pandas as pd

    from isaac_rl_posttraining.policies.pi0.offline import open_pi0_offline_source

    meta = tmp_path / "meta"
    meta.mkdir()
    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.wrist_image": {"dtype": "video"},
        "observation.state": {"dtype": "float32"},
    }
    (meta / "info.json").write_text(json.dumps({"features": features}))
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "lift"}\n')
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": [np.zeros(2, np.float32)],
            "action": [np.zeros(2, np.float32)],
        }
    ).to_parquet(data / "episode_000000.parquet")

    class _Policy:
        video_keys = ("video.image", "video.wrist_image")
        state_keys = ("state.q",)
        action_window = 1

        def __init__(self):
            self.observation_keys = {
                "video.image": "observation/image",
                "video.wrist_image": "observation/wrist_image",
            }

    source = open_pi0_offline_source(str(tmp_path), policy=_Policy())
    folders = {
        os.path.basename(os.path.dirname(path))
        for path in source.episodes[0]["videos"].values()
    }
    video_features = {key for key, spec in features.items() if spec["dtype"] == "video"}
    assert folders == video_features
    image = source.episodes[0]["videos"]["video.image"]
    assert image.endswith(
        "videos/chunk-000/observation.images.image/episode_000000.mp4"
    )
