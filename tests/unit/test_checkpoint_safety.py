# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint write that fails midway must leave the earlier checkpoints whole and kept."""

import os

import numpy as np
import pytest

from isaac_rl_posttraining.runtimes.base import PolicyState
from isaac_rl_posttraining.runtimes.learner import LearnerGroup
from isaac_rl_posttraining.training import checkpoints
from isaac_rl_posttraining.training.checkpoints import CheckpointStore


class _Learner(LearnerGroup):
    """Writes ``learner.pt``; ``fail`` makes it die after the first byte, like a full disk."""

    fail = False

    def save_checkpoint(self, directory: str) -> None:
        with open(os.path.join(directory, "learner.pt"), "wb") as stream:
            stream.write(b"x")
            if self.fail:
                raise OSError(122, "Disk quota exceeded")
            stream.write(b"rest")


def _save(store: CheckpointStore, learner: _Learner, tag: int) -> bool:
    state = PolicyState(version=tag, tensors={"x": np.zeros(2, dtype=np.float32)})
    return store.save(
        tag, learner_group=learner, policy_state=state, metadata={"env_steps": tag}
    )


def test_failed_save_keeps_the_last_good_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checkpoints.time, "sleep", lambda seconds: None)
    store = CheckpointStore(str(tmp_path), retention="best_last_final")
    learner = _Learner()
    assert _save(store, learner, 10) and _save(store, learner, 20)
    learner.fail = True
    assert not _save(store, learner, 30)

    # Best evaluation at 10, last good checkpoint 20: a half-written 30 would have pruned 20.
    store.apply_retention({10: 0.9, 20: 0.1})
    assert sorted(os.listdir(tmp_path)) == ["10", "20"]
    with open(tmp_path / "20" / "learner.pt", "rb") as stream:
        assert stream.read() == b"xrest"


def test_named_checkpoint_is_replaced_whole(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path))
    learner = _Learner()
    assert _save(store, learner, 10)
    state = PolicyState(version=3, tensors={"x": np.zeros(2, dtype=np.float32)})
    for _ in range(2):
        assert store.save(
            "final", learner_group=learner, policy_state=state, metadata={}
        )
    assert sorted(os.listdir(tmp_path)) == ["10", "final"]


def test_a_failed_rename_keeps_the_previous_named_checkpoint(
    tmp_path, monkeypatch
) -> None:
    store = CheckpointStore(str(tmp_path))
    state = PolicyState(version=3, tensors={"x": np.zeros(2, dtype=np.float32)})
    assert store.save(
        "final", learner_group=_Learner(), policy_state=state, metadata={}
    )
    replace = os.replace

    def fail_on_staging(source, target):
        if str(source).endswith("final.tmp"):
            raise OSError(5, "I/O error")
        replace(source, target)

    monkeypatch.setattr(checkpoints.os, "replace", fail_on_staging)
    with pytest.raises(OSError):
        store.save("final", learner_group=_Learner(), policy_state=state, metadata={})
    with open(tmp_path / "final" / "learner.pt", "rb") as stream:
        assert stream.read() == b"xrest"


def test_failed_buffer_save_keeps_the_previous_copy(tmp_path, monkeypatch) -> None:
    from conftest import make_transition

    from isaac_rl_posttraining.buffers import (
        ReplayBuffer,
        load_buffer,
        save_buffer,
        storage,
    )

    buffer = ReplayBuffer(8)
    buffer.add(make_transition(episode_id=0, done=True))
    save_buffer(buffer, tmp_path / "buffer_latest")
    buffer.add(make_transition(episode_id=1, done=True))

    def full_disk(*args, **kwargs):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(storage.np, "savez", full_disk)
    with pytest.raises(OSError):
        save_buffer(buffer, tmp_path / "buffer_latest")
    monkeypatch.undo()
    assert len(load_buffer(str(tmp_path / "buffer_latest"))) == 1


def test_a_failed_buffer_rename_keeps_the_previous_copy(tmp_path, monkeypatch) -> None:
    from conftest import make_transition

    from isaac_rl_posttraining.buffers import (
        ReplayBuffer,
        load_buffer,
        save_buffer,
        storage,
    )

    buffer = ReplayBuffer(8)
    buffer.add(make_transition(episode_id=0, done=True))
    save_buffer(buffer, tmp_path / "buffer_latest")
    buffer.add(make_transition(episode_id=1, done=True))
    replace = os.replace

    def fail_on_staging(source, target):
        if str(source).endswith(".buffer_latest.tmp"):
            raise OSError(5, "I/O error")
        replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail_on_staging)
    with pytest.raises(OSError):
        save_buffer(buffer, tmp_path / "buffer_latest")
    monkeypatch.undo()
    assert len(load_buffer(str(tmp_path / "buffer_latest"))) == 1
