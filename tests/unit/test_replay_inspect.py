# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved replay buffers stay readable: older files load with defaults, and ``inspect`` summarizes them."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from conftest import make_transition

from isaac_rl_posttraining.buffers import ReplayBuffer, load_buffer, save_buffer
from isaac_rl_posttraining.buffers.storage import save_buffer_part
from isaac_rl_posttraining.data.inspect import main, summary_lines
from isaac_rl_posttraining.training.recording import PersistSpec, persist_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _saved_buffer(path: Path) -> ReplayBuffer:
    buffer = ReplayBuffer(32, reward_channel="sparse_outcome")
    for episode_id, success in enumerate((True, False)):
        for step in range(2):
            buffer.add(
                make_transition(
                    episode_id=episode_id,
                    step_in_episode=step,
                    obs={
                        "state": np.full(3, step, np.float32),
                        "instruction": "push the button",
                    },
                    next_obs={
                        "state": np.full(3, step + 1, np.float32),
                        "instruction": "push the button",
                    },
                    action=np.zeros((2, 4), np.float32),
                    reward=0.25,
                    channels={
                        "sparse_outcome": np.asarray(
                            [0.0 if success or step == 0 else -1.0], np.float32
                        )
                    },
                    done=step == 1,
                    info={"is_success": success, "num_steps": step},
                    behavior_source="policy" if episode_id == 0 else "warmup",
                    log_prob=-0.5,
                )
            )
    buffer.add(
        make_transition(
            episode_id=7,
            obs={"state": np.zeros(3, np.float32)},
            next_obs={"state": np.zeros(3, np.float32)},
            action=np.zeros((2, 4)),
            channels={"sparse_outcome": np.zeros(1, np.float32)},
        )
    )
    save_buffer(buffer, path)
    return buffer


def test_a_file_written_by_an_older_schema_loads_with_defaults(tmp_path):
    """Fields a newer writer adds are optional on read: a file without them rebuilds with their defaults."""
    _saved_buffer(tmp_path / "buffer")
    metadata_path = tmp_path / "buffer" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for record in metadata["transitions"]:
        # An older writer knew neither the per-decision metadata beyond the step, the text nor the info.
        record["metadata"] = {"step_in_episode": record["metadata"]["step_in_episode"]}
        for key in ("text", "next_text", "extra", "intervention_phase"):
            record.pop(key, None)
    metadata_path.write_text(json.dumps(metadata))
    schema_path = tmp_path / "buffer" / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema.pop("drop_components")
    schema_path.write_text(json.dumps(schema))

    loaded = load_buffer(tmp_path / "buffer")

    assert len(loaded) == 5 and loaded.drop_components == ()
    first = loaded.transitions()[0]
    assert (
        first.metadata.decision_index is None
        and first.metadata.reset_seed_scope is None
    )
    assert first.metadata.extra == {} and first.observation.text() == {}
    assert first.observation.arrays()["state"].tolist() == [0.0, 0.0, 0.0]
    assert float(first.policy_outputs.log_prob) == -0.5


def test_inspect_summarizes_the_schema_episodes_and_returns(tmp_path, capsys):
    _saved_buffer(tmp_path / "buffer")

    lines = summary_lines(str(tmp_path / "buffer"))

    assert (
        lines[1]
        == "schema version 1; capacity 32; reward channel 'sparse_outcome' aggregated by 'last'"
    )
    assert "reward channels: env, sparse_outcome" in lines
    assert "  state: float32[3]" in lines and "  instruction: text" in lines
    assert "transitions: 5; episodes: 3" in lines
    assert "episodes by outcome: success 1, failure 1, timeout 0, ongoing 1" in lines
    assert "transitions by source: policy 3, warmup 2" in lines
    assert (
        "mean return on 'env' over 2 complete episodes: 0.5000 (min 0.5000, max 0.5000)"
        in lines
    )
    assert (
        "mean return on 'sparse_outcome' over 2 complete episodes: -0.5000 (min -1.0000, max 0.0000)"
        in lines
    )
    assert "  learner action: latent float32[2, 4]" in lines
    assert "  env action: none" in lines
    assert "  policy outputs: log_prob" in lines
    assert main([str(tmp_path / "buffer")]) == 0
    assert capsys.readouterr().out.splitlines() == lines


def test_inspect_runs_as_a_module(tmp_path):
    _saved_buffer(tmp_path / "buffer")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaac_rl_posttraining.data.inspect",
            str(tmp_path / "buffer"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == f"replay buffer {tmp_path / 'buffer'}"


def test_inspect_reads_a_folder_of_saved_parts(tmp_path):
    """The folder ``data.persist.path`` writes (part-NNNNN buffers) is summarized as ``load_buffer`` reads it."""
    first = _saved_buffer(tmp_path / "single")
    second = ReplayBuffer(32, reward_channel="sparse_outcome")
    for transition in first.transitions():
        second.add(
            dataclasses.replace(transition, episode_id=f"later:{transition.episode_id}")
        )
    for part in (first, second):
        save_buffer_part(part, tmp_path / "persisted")

    lines = summary_lines(str(tmp_path / "persisted"))

    assert lines[0] == f"replay buffer {tmp_path / 'persisted'} (2 parts)"
    assert lines[1] == "parts: part-00000 .. part-00001"
    assert lines[2].startswith("schema version 1; capacity 10;")
    assert "transitions: 10; episodes: 6" in lines


def test_persist_from_config_resolves_a_relative_path_under_the_run(tmp_path):
    assert persist_from_config({"path": None}, tmp_path) is None
    spec = persist_from_config(
        {"path": "episodes", "every_episodes": 5, "keep_frames": True}, tmp_path
    )
    assert spec == PersistSpec(
        str(tmp_path / "episodes"), every_episodes=5, keep_frames=True
    )
    assert persist_from_config({"path": "/data/x"}, tmp_path).path == "/data/x"
