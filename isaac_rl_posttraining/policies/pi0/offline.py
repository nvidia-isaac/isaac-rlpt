# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0 offline batches from a LeRobot dataset, encoded through the adapter's ``build_inputs``."""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.training.demonstrations import read_reward_done

__all__ = ["open_pi0_offline_source"]

#: Video decoders kept open at once; a LIBERO suite has thousands of (episode, camera) videos.
_MAX_OPEN_READERS = 64


def open_pi0_offline_source(
    dataset_dir: str,
    *,
    policy: Any,
    dataset_adapter: Any = None,
) -> _Pi0LeRobotSource:
    """Encode LIBERO/LeRobot demonstrations with the loaded pi0 adapter."""
    del dataset_adapter
    return _Pi0LeRobotSource(dataset_dir, policy)


class _Pi0LeRobotSource:
    """One-camera (or mapped) LeRobot episodes in the keys ``build_inputs`` reads."""

    def __init__(self, dataset_dir: str, policy: Any) -> None:
        self.policy = policy
        self.H = int(policy.action_window)
        parquet_files = sorted(
            glob.glob(os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet"))
        )
        if not parquet_files:
            raise ValueError(f"no LeRobot episode parquet files under {dataset_dir}")
        import pandas as pd

        override = str(getattr(policy, "instruction", "") or "")
        tasks = {} if override else _read_tasks(dataset_dir)
        self.episodes: list[dict[str, Any]] = []
        for parquet_file in parquet_files:
            episode_id = re.search(r"episode_(\d+)\.parquet", parquet_file).group(1)
            chunk = os.path.basename(os.path.dirname(parquet_file))
            df = pd.read_parquet(parquet_file)
            states = np.stack(df["observation.state"].to_numpy())
            actions = np.stack(df["action"].to_numpy())
            reward, done = read_reward_done(df, len(df))
            videos = {
                key: os.path.join(
                    dataset_dir,
                    "videos",
                    chunk,
                    _lerobot_video_dir(key, policy),
                    f"episode_{episode_id}.mp4",
                )
                for key in policy.video_keys
            }
            self.episodes.append(
                dict(
                    states=states.astype(np.float32),
                    actions=actions.astype(np.float32),
                    reward=reward,
                    done=done,
                    videos=videos,
                    length=len(states),
                    task=_episode_task(df, tasks, override),
                )
            )
        self.index = [
            (e, f) for e, episode in enumerate(self.episodes) for f in range(episode["length"])
        ]
        self._readers: dict[tuple[int, str], Any] = {}

    def sample(self, batch_size: int) -> dict[str, Any]:
        idxs = np.random.randint(0, len(self.index), size=batch_size)
        frames_cur = {key: [] for key in self.policy.video_keys}
        frames_nxt = {key: [] for key in self.policy.video_keys}
        cur_states, nxt_states, actions, rewards, dones, prompts = [], [], [], [], [], []
        for k in idxs:
            episode_idx, frame_idx = self.index[k]
            episode = self.episodes[episode_idx]
            length = episode["length"]
            for key in self.policy.video_keys:
                frames_cur[key].append(self._frame(episode_idx, key, frame_idx))
                frames_nxt[key].append(self._frame(episode_idx, key, min(frame_idx + 1, length - 1)))
            cur_states.append(episode["states"][frame_idx])
            nxt_states.append(episode["states"][min(frame_idx + 1, length - 1)])
            chunk_ids = [min(frame_idx + j, length - 1) for j in range(self.H)]
            actions.append(episode["actions"][chunk_ids])
            rewards.append(episode["reward"][frame_idx])
            dones.append(episode["done"][frame_idx])
            prompts.append(episode["task"])
        encoded = self.policy.encode_actions(np.stack(cur_states), np.stack(actions))
        state_key = self.policy.state_keys[0]
        cur = self.policy.build_inputs(
            {key: np.stack(value) for key, value in frames_cur.items()},
            {state_key: np.stack(cur_states)},
            prompts,
        )
        nxt = self.policy.build_inputs(
            {key: np.stack(value) for key, value in frames_nxt.items()},
            {state_key: np.stack(nxt_states)},
            prompts,
        )
        return {
            "inputs_cur": cur,
            "inputs_next": nxt,
            "action": torch.as_tensor(encoded, dtype=torch.float32),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
        }

    def _frame(self, episode_idx: int, key: str, frame_idx: int) -> np.ndarray:
        reader_key = (episode_idx, key)
        if reader_key not in self._readers:
            import decord

            if len(self._readers) >= _MAX_OPEN_READERS:
                self._readers.pop(next(iter(self._readers)))  # close the oldest decoder

            self._readers[reader_key] = decord.VideoReader(self.episodes[episode_idx]["videos"][key])
        frame = self._readers[reader_key][frame_idx].asnumpy()
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame.astype(np.uint8)


def _read_tasks(dataset_dir: str) -> dict[int, str]:
    path = os.path.join(dataset_dir, "meta", "tasks.jsonl")
    if not os.path.isfile(path):
        return {}
    tasks = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                tasks[int(record["task_index"])] = str(record["task"])
    return tasks


def _episode_task(frame: Any, tasks: dict[int, str], override: str) -> str:
    """One episode's language: the explicit override, else its ``task_index``."""
    if override:
        return override
    if not tasks:
        raise ValueError("no meta/tasks.jsonl; set base_model.task_instruction or add the dataset tasks")
    index = int(frame["task_index"].iloc[0]) if "task_index" in frame.columns else 0
    try:
        return tasks[index]
    except KeyError:
        raise KeyError(f"task_index {index} is not in meta/tasks.jsonl") from None


def _lerobot_video_dir(policy_key: str, policy: Any) -> str:
    """LeRobot v2 stores a camera at ``videos/chunk-*/observation.images.<cam>/``."""
    if str(policy_key).startswith("video."):
        return f"observation.images.{str(policy_key).split('.', 1)[1]}"
    mapping = getattr(policy, "observation_keys", {}) or {}
    return str(mapping.get(policy_key, policy_key))
