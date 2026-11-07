# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstrations stored in a replay buffer and shipped to the learner ranks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaac_rl_posttraining.buffers.samplers import UniformSampler
from isaac_rl_posttraining.training.demonstration_buffer import (
    build_demonstration_buffer,
    demonstration_slice,
    demonstration_transition,
    draw_demonstration_tensors,
)

STATE_DIM, HORIZON, ACTION_DIM = 28, 16, 28
HEIGHT, WIDTH = 8, 12


def _window(value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "state": np.full(STATE_DIM, value, dtype=np.float32),
        "room_view": np.full((HEIGHT, WIDTH, 3), int(value) % 256, dtype=np.uint8),
    }


def _windows(count: int):
    for index in range(count):
        yield (
            0,
            index,
            _window(float(index + 1)),
            np.full((HORIZON, ACTION_DIM), float(index), dtype=np.float32),
            index == count - 1,
        )


def _sampler(count: int, seed: int = 0) -> UniformSampler:
    return UniformSampler(
        build_demonstration_buffer(_windows(count)), rng=np.random.default_rng(seed)
    )


def test_a_demonstration_is_an_ordinary_transition():
    """The expert's chunk is the supervised target, not a side channel, and a
    BC target has no successor -- a copy rather than an alias would double the
    memory of every frame."""
    action = np.arange(HORIZON * ACTION_DIM, dtype=np.float32).reshape(
        HORIZON, ACTION_DIM
    )
    transition = demonstration_transition(0, 0, _window(), action, last=False)
    assert transition.observation is transition.next_observation
    assert transition.learner_action is not None
    np.testing.assert_array_equal(transition.learner_action.value, action)
    assert transition.behavior_source == "expert"


def test_an_empty_demonstration_set_is_rejected():
    with pytest.raises(ValueError, match="at least one window"):
        build_demonstration_buffer([])


def test_a_draw_flattens_into_what_the_transport_accepts():
    """LearnerBatch carries arrays, so the drawn batch is one flat mapping."""
    drawn = draw_demonstration_tensors(_sampler(16), 6)
    assert sorted(drawn) == ["action", "room_view", "state"]
    assert drawn["state"].shape == (6, STATE_DIM)
    assert drawn["action"].shape == (6, HORIZON, ACTION_DIM)
    assert drawn["room_view"].dtype == np.uint8  # encoded frames would be float32


def test_a_slice_reads_its_own_rows():
    """Slice n scores rows [n*size, (n+1)*size) of what the rank was shipped."""
    drawn = draw_demonstration_tensors(_sampler(64), 12)
    second = demonstration_slice(drawn, 4, 4)
    np.testing.assert_array_equal(second["action"], drawn["action"][4:8])
    np.testing.assert_array_equal(second["obs"]["state"], drawn["state"][4:8])
    assert sorted(second) == ["action", "obs"]  # the shape a source is written against


def test_a_rank_scoring_more_slices_than_it_was_shipped_fails_loudly():
    """Running off the end would silently rescore the first demonstrations."""
    from isaac_rl_posttraining.training.cotrain import CoTrainer

    cotrainer = CoTrainer(lambda batch: batch, beta=0.1, batch_size=4)
    drawn = draw_demonstration_tensors(_sampler(16), 8)  # two slices' worth
    cotrainer.next_batch(drawn, 0, torch.device("cpu"))
    cotrainer.next_batch(drawn, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="shipped"):
        cotrainer.next_batch(drawn, 2, torch.device("cpu"))


def _lerobot(tmp_path, lengths=(6, 5)):
    """A LeRobot root whose episodes end in a terminal reward, as the offline sets do."""
    import json

    import pandas as pd

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": len(lengths),
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (meta / "modality.json").write_text(
        json.dumps({"video": {"room_view": {"original_key": "room"}}})
    )
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "lift"}))
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    for episode, length in enumerate(lengths):
        done = [0.0] * (length - 1) + [1.0]
        pd.DataFrame(
            {
                "observation.state": [
                    np.full(STATE_DIM, step, np.float32) for step in range(length)
                ],
                "action": [
                    np.full(ACTION_DIM, step, np.float32) for step in range(length)
                ],
                "next.reward": done,
                "next.done": done,
            }
        ).to_parquet(tmp_path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    return tmp_path


def _load(root, monkeypatch, **options):
    from isaac_rl_posttraining.training.demonstration_buffer import (
        load_demonstration_buffer,
    )
    from isaac_rl_posttraining.training.demonstrations import LeRobotDemonstrationReader

    def decode(self, episode, camera):
        length = len(self.episode_arrays(episode)[0])
        return np.arange(length, dtype=np.uint8)[:, None, None, None] * np.ones(
            (1, HEIGHT, WIDTH, 3), np.uint8
        )

    monkeypatch.setattr(LeRobotDemonstrationReader, "decode_episode", decode)
    return load_demonstration_buffer(
        root, action_horizon=3, cameras=["room_view"], **options
    )


def test_the_default_buffer_keeps_its_co_training_windows(tmp_path, monkeypatch):
    """Co-training's windows are unchanged: length - horizon per episode, none padded."""
    buffer = _load(_lerobot(tmp_path), monkeypatch)
    assert len(buffer) == (6 - 3) + (5 - 3)
    last = buffer.transitions()[2]
    assert (
        last.truncated[-1] and not last.terminated[-1]
    )  # its episode goes on past the window


def test_padded_buffer_keeps_the_terminal_reward_and_real_successors(
    tmp_path, monkeypatch
):
    """IQL learns its only reward from the final step, so it must be in the buffer."""
    buffer = _load(_lerobot(tmp_path), monkeypatch, pad_episode_end=True)
    assert len(buffer) == 6 + 5
    batch = buffer.collate(range(6))
    np.testing.assert_array_equal(batch["reward"], [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(batch["done"], [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(batch["next_obs"]["state"][:, 0], [1, 2, 3, 4, 5, 5])
    np.testing.assert_array_equal(
        batch["next_obs"]["room_view"][:, 0, 0, 0], [1, 2, 3, 4, 5, 5]
    )
    np.testing.assert_array_equal(
        batch["action"][4, :, 0], [4, 5, 5]
    )  # last action repeated
    first, second = buffer.transitions()[:2]
    assert (
        first.next_observation is second.observation
    )  # shared, not a second copy of the frames
