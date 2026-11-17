# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The episode-organized replay buffer: storage, batches, eviction, relabeling, files.

Pins the contract every off-policy learner relies on: dict observations
round-trip through a batch, the batch keys and dtypes, chunked-action shapes,
whole-episode eviction at capacity, the done-mask on truncated transitions
(no bootstrap through episode caps), graceful oversampling, ``clear()``, the
by-outcome and by-source index, sample-time reward channels, and the saved
file format. The last tests are the parity harness: the batch a buffer of
typed transitions collates equals the one the legacy buffer built from the
same field dictionaries.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from conftest import make_transition

from isaac_rl_posttraining.buffers import (
    ReplayBuffer,
    UniformSampler,
    collate,
    load_buffer,
    save_buffer,
    to_tensors,
)
from isaac_rl_posttraining.buffers.storage import read_schema
from isaac_rl_posttraining.data import Transition

CHUNK, LATENT = 4, 3


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _transition(
    i: int,
    *,
    episode_id: int | None = None,
    step: int = 0,
    done=False,
    truncated=False,
    success=False,
    source="policy",
):
    """Decision ``i``: its own one-decision episode unless an episode and step are given."""
    return make_transition(
        episode_id=i if episode_id is None else episode_id,
        step_in_episode=step,
        obs={
            "state": np.full((5,), float(i), dtype=np.float32),
            "dino_embedding": np.full((6,), float(i) + 0.25, dtype=np.float32),
        },
        action=np.full((CHUNK, LATENT), float(i), dtype=np.float32),
        reward=float(i),
        next_obs={
            "state": np.full((5,), float(i) + 0.5, dtype=np.float32),
            "dino_embedding": np.full((6,), float(i) + 0.75, dtype=np.float32),
        },
        done=done,
        truncated=truncated,
        info={"is_success": success},
        behavior_source=source,
    )


def _add_episode(
    buffer: ReplayBuffer,
    episode_id: int,
    *,
    length: int,
    success: bool,
    source: str = "policy",
):
    """Add one complete episode whose transition ``k`` carries reward ``episode_id * 10 + k``."""
    for step in range(length):
        buffer.add(
            _transition(
                episode_id * 10 + step,
                episode_id=episode_id,
                step=step,
                done=step == length - 1,
                success=success,
                source=source,
            )
        )


def _batch(buffer: ReplayBuffer, size: int):
    return to_tensors(UniformSampler(buffer).sample(size))


def test_dict_obs_roundtrip():
    buffer = ReplayBuffer(8)
    buffer.add(_transition(7))
    batch = _batch(buffer, 1)
    assert set(batch["obs"].keys()) == {"state", "dino_embedding"}
    torch.testing.assert_close(batch["obs"]["state"], torch.full((1, 5), 7.0))
    torch.testing.assert_close(batch["obs"]["dino_embedding"], torch.full((1, 6), 7.25))
    torch.testing.assert_close(batch["next_obs"]["state"], torch.full((1, 5), 7.5))
    torch.testing.assert_close(batch["reward"], torch.tensor([7.0]))


def test_sample_keys_contract():
    buffer = ReplayBuffer(8)
    for i in range(4):
        buffer.add(_transition(i))
    batch = _batch(buffer, 4)
    required = {
        "obs",
        "action",
        "reward",
        "next_obs",
        "done",
        "truncated",
        "bootstrap_steps",
    }
    assert required <= set(batch.keys()), (
        f"missing keys: {required - set(batch.keys())}"
    )
    for key in ("action", "reward", "done", "truncated"):
        assert isinstance(batch[key], torch.Tensor), key
        assert batch[key].dtype == torch.float32, key
    assert batch["bootstrap_steps"].dtype == torch.int64
    for obs_key in ("obs", "next_obs"):
        for name, value in batch[obs_key].items():
            assert isinstance(value, torch.Tensor), f"{obs_key}.{name}"
            assert value.dtype == torch.float32, f"{obs_key}.{name}"
            assert value.shape[0] == 4, f"{obs_key}.{name}"
    assert batch["action"].shape == (4, CHUNK, LATENT)


def test_open_episode_is_sampleable_and_eviction_removes_whole_oldest_episodes():
    buffer = ReplayBuffer(5)
    _add_episode(buffer, 0, length=3, success=False)
    _add_episode(buffer, 1, length=3, success=True)
    # 6 > 5: episode 0 leaves as a whole, never a prefix of it.
    assert [
        t.decision_reward("env", aggregation="last") for t in buffer.transitions()
    ] == [10.0, 11.0, 12.0]
    _add_episode(buffer, 2, length=3, success=False)
    assert [t.episode_id for t in buffer.transitions()] == [2, 2, 2]
    assert len(buffer) == 3
    # An episode still running is sampleable; capacity evicts complete episodes first...
    for step in range(4):
        buffer.add(_transition(30 + step, episode_id=3, step=step))
    assert len(buffer) == 4 and [episode.id for episode in buffer.episodes()] == [3]
    # ...and never the running one, even when it alone exceeds the capacity.
    for step in range(4, 6):
        buffer.add(_transition(30 + step, episode_id=3, step=step))
    assert len(buffer) == 6 and buffer.episodes(complete=True) == ()
    assert buffer.get_statistics() == {
        "transitions": 6,
        "episodes": 1,
        "open_episodes": 1,
        "success_episodes": 0,
        "failure_episodes": 0,
        "timeout_episodes": 0,
        "expert_transitions": 0,
    }
    assert set(UniformSampler(buffer).sample(32)["reward"].tolist()) <= {
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        35.0,
    }


def test_done_is_terminated_even_when_truncated():
    """A time-limit cut alone is never terminal, but a terminal on the limit's last step is, as in PPO."""
    buffer = ReplayBuffer(8)
    buffer.add(_transition(0, episode_id=0, step=0, done=True, truncated=True))
    buffer.add(_transition(1, episode_id=1, step=0, done=True, truncated=False))
    buffer.add(_transition(2, episode_id=2, step=0, done=False, truncated=True))
    batch = _batch(buffer, 3)
    rows = sorted(
        zip(
            batch["reward"].tolist(),
            batch["done"].tolist(),
            batch["truncated"].tolist(),
        )
    )
    assert rows == [
        (0.0, 1.0, 1.0),  # terminated wins over truncated
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 1.0),
    ]


def test_oversample_small_buffer_and_clear():
    buffer = ReplayBuffer(16)
    for i in range(3):
        buffer.add(_transition(i))
    batch = _batch(buffer, 8)
    assert batch["reward"].shape == (8,)
    assert set(batch["reward"].tolist()) <= {0.0, 1.0, 2.0}
    assert not buffer.is_empty()
    buffer.clear()
    assert len(buffer) == 0 and buffer.is_empty() and buffer.episodes() == ()
    assert UniformSampler(buffer).sample(4) == {}
    buffer.add(_transition(9))
    assert len(buffer) == 1


def test_add_checks_episode_bookkeeping():
    buffer = ReplayBuffer(8)
    with pytest.raises(ValueError, match="starts at step 2"):
        buffer.add(_transition(0, step=2))
    buffer.add(_transition(0, step=0))
    with pytest.raises(ValueError, match="continues at step 5, expected 1"):
        buffer.add(_transition(1, episode_id=0, step=5))
    buffer.add(_transition(1, episode_id=0, step=1, done=True))
    with pytest.raises(ValueError, match="already complete"):
        buffer.add(_transition(2, episode_id=0, step=2))
    with pytest.raises(TypeError, match="Transition records"):
        buffer.add({"obs": 1})


def test_indices_select_by_outcome_completion_and_source():
    buffer = ReplayBuffer(64)
    _add_episode(buffer, 0, length=2, success=True)
    _add_episode(buffer, 1, length=3, success=False)
    _add_episode(buffer, 2, length=2, success=False, source="expert")
    buffer.add(_transition(30, episode_id=3, step=0))
    buffer.add(
        make_transition(
            episode_id=4, step_in_episode=0, action=np.zeros(1), truncated=True
        )
    )

    assert buffer.indices(outcome="success").tolist() == [0, 1]
    assert buffer.indices(complete=True, exclude_outcome="success").tolist() == [
        2,
        3,
        4,
        5,
        6,
        8,
    ]
    assert buffer.indices(outcome="timeout").tolist() == [8]
    assert buffer.indices(complete=False).tolist() == [7]
    assert buffer.indices(source="expert").tolist() == [5, 6]
    assert [episode.outcome for episode in buffer.episodes()] == [
        "success",
        "failure",
        "failure",
        "ongoing",
        "timeout",
    ]
    assert (
        buffer.transition(6).episode_id == 2
        and buffer.transition(6).behavior_source == "expert"
    )
    with pytest.raises(IndexError):
        buffer.transition(9)


def test_dropped_components_never_enter_the_buffer():
    buffer = ReplayBuffer(8, drop_components=("image",))
    buffer.add(
        make_transition(
            obs={
                "state": np.zeros(2, np.float32),
                "image": np.zeros((4, 4, 3), np.uint8),
                "instruction": "lift",
            },
            next_obs={
                "state": np.ones(2, np.float32),
                "image": np.zeros((4, 4, 3), np.uint8),
            },
            action=np.zeros(1),
        )
    )
    stored = buffer.transitions()[0]
    assert set(stored.observation.arrays()) == {
        "state"
    } and stored.observation.text() == {"instruction": "lift"}
    assert set(stored.next_observation.arrays()) == {"state"}


def test_relabel_adds_a_channel_and_leaves_env_alone():
    buffer = ReplayBuffer(64, reward_channel="reward_model/progress")
    _add_episode(buffer, 0, length=2, success=True)
    buffer.add(_transition(10, episode_id=1, step=0))  # still running: not scored

    def score(episode):
        return [
            np.full(t.steps, 0.5 * (index + 1), np.float32)
            for index, t in enumerate(episode.transitions)
        ]

    assert buffer.relabel("reward_model/progress", score) == 2
    complete = buffer.transitions()[:2]
    assert [t.reward("reward_model/progress").tolist() for t in complete] == [
        [0.5],
        [1.0],
    ]
    assert [t.reward("env").tolist() for t in complete] == [[0.0], [1.0]]
    assert "reward_model/progress" not in buffer.transitions()[2].rewards
    batch = buffer.collate([0, 1])
    assert batch["reward"].tolist() == [0.5, 1.0]
    with pytest.raises(ValueError, match="never rewritten"):
        buffer.relabel("env", score)
    with pytest.raises(ValueError, match="3 reward arrays for the 2 transitions"):
        buffer.relabel(
            "reward_model/other", lambda episode: [np.zeros(1, np.float32)] * 3
        )


def test_buffer_validates_its_settings():
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(0)
    with pytest.raises(ValueError, match="reward_aggregation"):
        ReplayBuffer(4, reward_aggregation="mean")
    with pytest.raises(ValueError, match="reward_channel"):
        ReplayBuffer(4, reward_channel="")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _assert_same_observations(actual: Transition, expected: Transition) -> None:
    assert actual.observation.text() == expected.observation.text()
    for side in ("observation", "next_observation"):
        left, right = getattr(actual, side).arrays(), getattr(expected, side).arrays()
        assert set(left) == set(right), side
        for name in left:
            np.testing.assert_array_equal(
                left[name], right[name], err_msg=f"{side}/{name}"
            )
            assert left[name].dtype == right[name].dtype, f"{side}/{name}"


def _assert_same_transition(actual: Transition, expected: Transition) -> None:
    assert actual.episode_id == expected.episode_id
    assert actual.behavior_source == expected.behavior_source
    _assert_same_observations(actual, expected)
    if expected.learner_action is None:
        assert actual.learner_action is None
    else:
        assert actual.learner_action.kind == expected.learner_action.kind
        np.testing.assert_array_equal(
            actual.learner_action.value, expected.learner_action.value
        )
    if expected.env_action is None:
        assert actual.env_action is None
    else:
        np.testing.assert_array_equal(
            actual.env_action.commands, expected.env_action.commands
        )
        assert (
            actual.env_action.execution_length == expected.env_action.execution_length
        )
    assert set(actual.rewards) == set(expected.rewards)
    for channel in expected.rewards:
        np.testing.assert_array_equal(
            actual.rewards[channel], expected.rewards[channel], err_msg=channel
        )
    for flag in ("terminated", "truncated", "success"):
        np.testing.assert_array_equal(
            getattr(actual, flag), getattr(expected, flag), err_msg=flag
        )
    for name in (
        "step_in_episode",
        "decision_index",
        "policy_version",
        "environment_index",
        "reset_seed",
        "reset_seed_scope",
        "wall_time",
        "max_steps_in_episode",
    ):
        assert getattr(actual.metadata, name) == getattr(expected.metadata, name), name
    for name in ("log_prob", "value", "next_value"):
        left, right = (
            getattr(actual.policy_outputs, name),
            getattr(expected.policy_outputs, name),
        )
        assert (left is None) == (right is None), name
        if right is not None:
            np.testing.assert_array_equal(left, right, err_msg=name)
    if expected.policy_outputs.features is None:
        assert actual.policy_outputs.features is None
    else:
        for key, value in expected.policy_outputs.features.items():
            np.testing.assert_array_equal(
                actual.policy_outputs.features[key], value, err_msg=key
            )
            assert actual.policy_outputs.features[key].dtype == value.dtype


def _rich_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(
        64,
        drop_components=("image",),
        reward_channel="sparse_outcome",
        reward_aggregation="sum",
    )
    for step in range(3):
        buffer.add(
            make_transition(
                episode_id="demo-0",
                step_in_episode=step * 2,
                obs={
                    "state": np.full(3, step, np.float32),
                    "image": np.zeros((2, 2, 3), np.uint8),
                    "instruction": "lift the drill",
                },
                next_obs={
                    "state": np.full(3, step + 1, np.float32),
                    "instruction": "lift the drill",
                },
                action=np.full((2, 4), step, np.float32),
                env_action=np.full((2, 5), step, np.float32),
                rewards=np.asarray([0.1, 0.2], np.float32),
                channels={
                    "sparse_outcome": np.asarray(
                        [0.0, -1.0 if step == 2 else 0.0], np.float32
                    )
                },
                done=step == 2,
                info={
                    "success": False,
                    "num_steps": step,
                    "hash": "abc",
                    "flags": np.asarray([1, 2]),
                },
                behavior_source="expert",
                log_prob=-0.5,
                value=1.5,
                features={"tokens_bf16": np.full((4, 2), step, np.int16)},
            )
        )
    state = np.ones(3, np.float32)
    sparse = {"sparse_outcome": np.zeros(1, np.float32)}
    action = np.zeros((2, 4), np.float32)
    buffer.add(
        make_transition(
            episode_id=7,
            step_in_episode=0,
            obs=state,
            next_obs=state,
            action=action,
            channels=sparse,
        )
    )
    buffer.add(
        make_transition(
            episode_id=7,
            step_in_episode=1,
            obs=state,
            next_obs=state,
            action=action,
            channels=sparse,
            truncated=True,
            info={"is_success": True},
        )
    )
    buffer.add(
        make_transition(
            episode_id=8,
            step_in_episode=0,
            obs=state,
            next_obs=state,
            action=action,
            channels=sparse,
        )
    )
    return buffer


def test_save_and_load_round_trip(tmp_path):
    buffer = _rich_buffer()
    save_buffer(buffer, tmp_path / "buffer")

    schema = read_schema(tmp_path / "buffer")
    assert (
        schema["schema_version"] == 1
        and schema["transitions"] == 6
        and schema["episodes"] == 3
    )
    assert schema["reward_channels"] == ["env", "sparse_outcome"]
    assert (
        schema["reward_channel"] == "sparse_outcome"
        and schema["reward_aggregation"] == "sum"
    )
    assert (
        schema["observation"]["text"] == ["instruction"]
        and "image" not in schema["observation"]["arrays"]
    )
    metadata = json.loads((tmp_path / "buffer" / "metadata.json").read_text())
    assert [episode["complete"] for episode in metadata["episodes"]] == [
        True,
        True,
        False,
    ]

    loaded = load_buffer(tmp_path / "buffer")
    assert (
        loaded.capacity == 64
        and loaded.reward_channel == "sparse_outcome"
        and loaded.reward_aggregation == "sum"
    )
    assert loaded.drop_components == ("image",)
    assert len(loaded) == 6 and [episode.id for episode in loaded.episodes()] == [
        "demo-0",
        7,
        8,
    ]
    assert [episode.complete for episode in loaded.episodes()] == [True, True, False]
    for actual, expected in zip(
        loaded.transitions(), buffer.transitions(), strict=True
    ):
        _assert_same_transition(actual, expected)
    # The untyped info comes back with arrays as arrays.
    np.testing.assert_array_equal(
        loaded.transitions()[0].metadata.extra["flags"], [1, 2]
    )
    assert loaded.transitions()[0].metadata.extra["hash"] == "abc"
    # A demonstration buffer is loaded with the learner's own read settings.
    demos = load_buffer(
        tmp_path / "buffer", capacity=8, reward_channel="env", reward_aggregation="last"
    )
    assert demos.capacity == 8 and demos.collate([0])[
        "reward"
    ].tolist() == pytest.approx([0.2])


def test_saved_files_are_checked_before_loading(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_buffer(ReplayBuffer(4), tmp_path / "empty")
    buffer = ReplayBuffer(4)
    buffer.add(_transition(0, done=True))
    save_buffer(buffer, tmp_path / "buffer")
    schema_path = tmp_path / "buffer" / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema_path.write_text(json.dumps({**schema, "schema_version": 99}))
    with pytest.raises(ValueError, match="schema version 99"):
        load_buffer(tmp_path / "buffer")
    schema_path.write_text(json.dumps({**schema, "format": "something-else"}))
    with pytest.raises(ValueError, match="not a saved replay buffer"):
        load_buffer(tmp_path / "buffer")


# ---------------------------------------------------------------------------
# Parity with the batches the legacy buffer built
# ---------------------------------------------------------------------------


def _legacy_fields(i: int, *, done: bool, truncated: bool, steps: int = 3) -> dict:
    """The field mapping the legacy buffer stored for one decision, as the workers pushed it."""
    sequence = np.asarray([0.0] * (steps - 1) + [float(i)], np.float32)
    return {
        "obs": {
            "state": np.full(5, float(i), np.float32),
            "dino_embedding": np.full(6, float(i) + 0.25, np.float32),
        },
        "action": np.full((CHUNK, LATENT), float(i), np.float32),
        "reward": float(sequence[-1]),
        "next_obs": {
            "state": np.full(5, float(i) + 0.5, np.float32),
            "dino_embedding": np.full(6, float(i) + 0.75, np.float32),
        },
        "done": done,
        "truncated": truncated,
        "episode_id": i,
        "step_in_episode": 0,
        "max_steps_in_episode": 100,
        "timestamp": 1.0,
        "language_instruction": None,
        "info": {"is_success": done and not truncated, "policy_version": 1},
        "reward_sequence": sequence,
        "bootstrap_steps": steps,
        "behavior_source": "policy",
        "intervention_phase": "none",
        "policy_tick_mask": None,
        "expert_tick_mask": None,
        "pre_intervention_tick_mask": None,
        "post_intervention_tick_mask": None,
        "cached_features": None,
    }


def _legacy_batch(fields: list[dict]) -> dict:
    """What the legacy buffer's sampled batch held for these transitions, in this order."""
    done = np.asarray([f["done"] for f in fields], np.float32)
    truncated = np.asarray([f["truncated"] for f in fields], np.float32)
    return {
        "obs": {
            key: np.stack([f["obs"][key] for f in fields]).astype(np.float32)
            for key in fields[0]["obs"]
        },
        "action": np.stack([f["action"] for f in fields]).astype(np.float32),
        "reward": np.asarray([f["reward"] for f in fields], np.float32),
        "next_obs": {
            key: np.stack([f["next_obs"][key] for f in fields]).astype(np.float32)
            for key in fields[0]["next_obs"]
        },
        "done": done * (1.0 - truncated),
        "truncated": truncated,
        "bootstrap_steps": np.asarray([f["bootstrap_steps"] for f in fields], np.int64),
    }


def test_collated_batches_match_the_legacy_buffer_field_for_field():
    fields = [
        _legacy_fields(0, done=False, truncated=False),
        _legacy_fields(1, done=True, truncated=False),
        _legacy_fields(2, done=False, truncated=True),
        _legacy_fields(3, done=True, truncated=True),
    ]
    typed = [Transition.from_legacy_fields(f, kind="latent") for f in fields]

    batch = collate(typed, reward_channel="env", reward_aggregation="last")
    expected = _legacy_batch(fields)

    for key in ("action", "reward", "done", "truncated", "bootstrap_steps"):
        np.testing.assert_array_equal(batch[key], expected[key], err_msg=key)
        assert batch[key].dtype == expected[key].dtype, key
    for side in ("obs", "next_obs"):
        assert set(batch[side]) == set(expected[side])
        for name in expected[side]:
            np.testing.assert_array_equal(
                batch[side][name], expected[side][name], err_msg=f"{side}/{name}"
            )
            assert batch[side][name].dtype == np.float32
    assert [row.tolist() for row in batch["reward_sequence"]] == [
        f["reward_sequence"].tolist() for f in fields
    ]
    assert batch["behavior_source"] == ["policy"] * 4
    assert [info["is_success"] for info in batch["info"]] == [False, True, False, False]
    # The buffer collates the same batch from stored transitions, in index order.
    buffer = ReplayBuffer(8)
    for transition in typed:
        buffer.add(transition)
    stored = buffer.collate([3, 1])
    np.testing.assert_array_equal(stored["reward"], expected["reward"][[3, 1]])
    np.testing.assert_array_equal(stored["done"], expected["done"][[3, 1]])
    tensors = to_tensors(stored, device="cpu")
    assert (
        tensors["bootstrap_steps"].dtype == torch.int64
        and tensors["obs"]["state"].dtype == torch.float32
    )


def test_a_camera_frame_survives_the_batch_as_bytes():
    """An observation declares frames uint8, so the batch must not float them.

    The cost is not only the four-fold growth of every sampled batch: the
    policies that read frames check for uint8 and reject anything else, so an
    upcast here turns a valid observation into one the run was not checked
    for. Values must arrive unchanged, not rescaled.
    """
    frames = [np.full((4, 4, 3), value, np.uint8) for value in (7, 11)]
    transitions = [
        make_transition(
            obs={"state": np.zeros(3, np.float32), "ego_view_image": frame},
            action=np.zeros(2),
        )
        for frame in frames
    ]

    batch = collate(transitions)

    assert batch["obs"]["ego_view_image"].dtype == np.uint8
    np.testing.assert_array_equal(batch["obs"]["ego_view_image"], np.stack(frames))
    assert to_tensors(batch, device="cpu")["obs"]["ego_view_image"].dtype == torch.uint8


def test_a_camera_frame_is_still_bytes_after_a_save_and_load(tmp_path):
    """Frames must cross the on-disk format unchanged, not just the collate."""
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    buffer = ReplayBuffer(4)
    buffer.add(
        make_transition(
            obs={"state": np.zeros(3, np.float32), "ego_view_image": frame},
            action=np.zeros(2),
        )
    )
    save_buffer(buffer, tmp_path / "buffer")

    restored = load_buffer(tmp_path / "buffer").collate([0])

    assert restored["obs"]["ego_view_image"].dtype == np.uint8
    np.testing.assert_array_equal(restored["obs"]["ego_view_image"][0], frame)


def test_float_components_are_still_narrowed_to_one_width():
    """Learners want a single float width, whatever the simulator emitted."""
    transitions = [
        make_transition(obs={"state": np.full(3, 0.5, np.float64)}, action=np.zeros(2))
        for _ in range(2)
    ]

    batch = collate(transitions)

    assert batch["obs"]["state"].dtype == np.float32
    np.testing.assert_allclose(batch["obs"]["state"], np.full((2, 3), 0.5, np.float32))


def test_collate_rejects_mixed_shapes_and_components():
    wide = make_transition(obs={"state": np.zeros(3, np.float32)}, action=np.zeros(2))
    narrow = make_transition(obs={"state": np.zeros(2, np.float32)}, action=np.zeros(2))
    with pytest.raises(ValueError, match="mixed shapes"):
        collate([wide, narrow])
    other = make_transition(obs={"pose": np.zeros(3, np.float32)}, action=np.zeros(2))
    with pytest.raises(ValueError, match="components differ"):
        collate([wide, other])
    assert collate([]) == {}


# ----------------------------------------------------------------------------
# The episode index: filters, draws and statistics against brute force
# ----------------------------------------------------------------------------


def _naive_indices(
    buffer: ReplayBuffer,
    *,
    outcome=None,
    exclude_outcome=None,
    complete=None,
    source=None,
):
    """The filter semantics spelled out over the stored episodes, independent of the buffer's index."""
    chosen, start = [], 0
    for episode in buffer.episodes():
        keep = (
            (complete is None or (episode.transitions[-1].ended) == complete)
            and (outcome is None or episode.outcome == outcome)
            and (exclude_outcome is None or episode.outcome != exclude_outcome)
        )
        for offset, transition in enumerate(episode.transitions):
            if keep and (source is None or transition.behavior_source == source):
                chosen.append(start + offset)
        start += len(episode.transitions)
    return chosen


_FILTERS = [
    {},
    {"outcome": "success"},
    {"outcome": "failure"},
    {"outcome": "timeout"},
    {"outcome": "ongoing"},
    {"complete": True},
    {"complete": False},
    {"complete": True, "exclude_outcome": "success"},
    {"exclude_outcome": "failure"},
    {"source": "expert"},
    {"outcome": "success", "source": "expert"},
    {"complete": False, "source": "policy"},
]


def _random_episode_plan(rng: np.random.Generator) -> dict:
    """A fresh episode's random length, outcome, ending and source."""
    return {
        "step": 0,
        "length": int(rng.integers(1, 6)),
        "success": bool(rng.random() < 0.4),
        "timeout": bool(rng.random() < 0.3),
        "source": "expert" if rng.random() < 0.3 else "policy",
        "early_success": bool(rng.random() < 0.2),
    }


def _random_buffer_history(seed: int, *, capacity: int, episodes: int):
    """Interleaved episodes of random length, ending, outcome and source; yields the buffer after every add."""
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity)
    open_episodes: dict[int, dict] = {}
    next_id = 0
    while next_id < episodes or open_episodes:
        if open_episodes and (next_id >= episodes or rng.random() < 0.7):
            episode_id = int(rng.choice(sorted(open_episodes)))
        else:
            episode_id = next_id
            next_id += 1
            open_episodes[episode_id] = _random_episode_plan(rng)
        plan = open_episodes[episode_id]
        step, last = plan["step"], plan["step"] == plan["length"] - 1
        buffer.add(
            make_transition(
                episode_id=episode_id,
                step_in_episode=step,
                obs={"state": np.full((3,), float(episode_id), dtype=np.float32)},
                action=np.full((2,), float(step), dtype=np.float32),
                reward=float(rng.normal()),
                next_obs={
                    "state": np.full((3,), float(episode_id) + 0.5, dtype=np.float32)
                },
                done=last and not plan["timeout"],
                truncated=last and plan["timeout"],
                info={
                    "is_success": plan["success"] and (last or plan["early_success"])
                },
                behavior_source=plan["source"],
            )
        )
        plan["step"] += 1
        if last:
            del open_episodes[episode_id]
        yield buffer


def test_index_filters_match_brute_force_through_adds_completions_and_evictions():
    for seed in range(4):
        for buffer in _random_buffer_history(seed, capacity=24, episodes=40):
            assert len(buffer) == len(buffer.transitions())
            for select in _FILTERS:
                expected = _naive_indices(buffer, **select)
                assert buffer.indices(**select).tolist() == expected, (seed, select)
                assert buffer.count(**select) == len(expected), (seed, select)
            assert [
                len(e.transitions) for e in buffer.episodes()
            ] == buffer.episode_lengths().tolist()


def test_draw_matches_a_choice_over_the_materialized_pool_stream_for_stream():
    buffer = ReplayBuffer(64)
    for step, done in ((0, False), (1, False), (2, True)):
        buffer.add(_transition(step, episode_id=0, step=step, done=done, success=True))
    _add_episode(buffer, 1, length=4, success=False)
    _add_episode(buffer, 2, length=2, success=True, source="expert")
    buffer.add(_transition(30, episode_id=3, step=0))
    for select in _FILTERS:
        for count in (1, 3, 5, 40):
            pool = buffer.indices(**select)
            if len(pool) == 0:
                assert buffer.draw(np.random.default_rng(1), count, **select).size == 0
                continue
            expected = np.random.default_rng(7).choice(
                pool, size=count, replace=len(pool) < count
            )
            picked = buffer.draw(np.random.default_rng(7), count, **select)
            np.testing.assert_array_equal(
                picked, expected, err_msg=f"{select} count={count}"
            )
            assert set(picked.tolist()) <= set(pool.tolist())


def test_uniform_sampler_draws_the_same_indices_as_before_the_index():
    """Same seed, same batch: the sampler's draw equals ``rng.choice`` over its materialized pool."""
    buffer = ReplayBuffer(64)
    _add_episode(buffer, 0, length=3, success=True)
    _add_episode(buffer, 1, length=5, success=False)
    sampler = UniformSampler(buffer, rng=np.random.default_rng(3), outcome="success")
    reference = np.random.default_rng(3).choice(sampler.pool(), size=2, replace=False)
    batch = sampler.sample(2)
    assert batch["reward"].tolist() == [
        float(buffer.transition(int(i)).decision_reward("env", aggregation="last"))
        for i in reference
    ]


def test_statistics_come_from_running_records_and_match_a_recount():
    for seed in range(3):
        for buffer in _random_buffer_history(seed, capacity=20, episodes=30):
            transitions = buffer.transitions()
            episodes = buffer.episodes()
            assert buffer.get_statistics() == {
                "transitions": len(transitions),
                "episodes": len(episodes),
                "open_episodes": sum(
                    1 for e in episodes if not e.transitions[-1].ended
                ),
                "success_episodes": sum(1 for e in episodes if e.outcome == "success"),
                "failure_episodes": sum(1 for e in episodes if e.outcome == "failure"),
                "timeout_episodes": sum(1 for e in episodes if e.outcome == "timeout"),
                "expert_transitions": sum(
                    1 for t in transitions if t.behavior_source == "expert"
                ),
            }
            if transitions:
                rewards = np.asarray(
                    [t.decision_reward("env", aggregation="last") for t in transitions]
                )
                record = buffer.reward_statistics()["env"]
                assert record["mean"] == pytest.approx(rewards.mean(), abs=1e-9)
                assert record["std"] == pytest.approx(rewards.std(), abs=1e-9)
                assert record["min"] == rewards.min() and record["max"] == rewards.max()
            else:
                assert buffer.reward_statistics() == {}


def test_relabel_updates_the_running_reward_records():
    buffer = ReplayBuffer(64)
    _add_episode(buffer, 0, length=3, success=True)
    _add_episode(buffer, 1, length=2, success=False)
    buffer.relabel(
        "shaped",
        lambda episode: [
            np.full((t.steps,), 2.0, dtype=np.float32) for t in episode.transitions
        ],
    )
    stats = buffer.reward_statistics()
    assert stats["shaped"] == {"mean": 2.0, "min": 2.0, "max": 2.0, "std": 0.0}
    buffer.relabel(
        "shaped",
        lambda episode: [
            np.full((t.steps,), float(t.metadata.step_in_episode), dtype=np.float32)
            for t in episode.transitions
        ],
    )
    rewards = np.asarray(
        [t.decision_reward("shaped", aggregation="last") for t in buffer.transitions()]
    )
    record = buffer.reward_statistics()["shaped"]
    assert (record["mean"], record["min"], record["max"]) == (
        rewards.mean(),
        rewards.min(),
        rewards.max(),
    )
    assert record["std"] == pytest.approx(rewards.std())


def test_sequence_sampler_starts_match_brute_force():
    from isaac_rl_posttraining.buffers import SequenceSampler

    for seed in range(2):
        for buffer in _random_buffer_history(seed, capacity=30, episodes=25):
            for length in (1, 2, 3):
                expected, start = [], 0
                for episode in buffer.episodes():
                    fits = len(episode.transitions) - length + 1
                    expected.extend(range(start, start + max(fits, 0)))
                    start += len(episode.transitions)
                assert SequenceSampler(buffer, length).starts().tolist() == expected


# ----------------------------------------------------------------------------
# Loading: the requested channel and the file layout are checked up front
# ----------------------------------------------------------------------------


def test_load_refuses_a_reward_channel_the_file_lacks_and_legacy_npz_files(tmp_path):
    from isaac_rl_posttraining.buffers.storage import load_buffer, save_buffer

    buffer = ReplayBuffer(16)
    _add_episode(buffer, 0, length=2, success=True)
    save_buffer(buffer, tmp_path / "demos")
    with pytest.raises(
        ValueError,
        match=r"no reward channel 'sparse_outcome' \(its channels: \['env'\]\)",
    ):
        load_buffer(tmp_path / "demos", reward_channel="sparse_outcome")
    legacy = tmp_path / "replay.npz"
    np.savez(legacy, obs=np.zeros(3))
    with pytest.raises(
        ValueError, match="legacy single-file .npz replay files are not supported"
    ):
        load_buffer(legacy)
    with pytest.raises(FileNotFoundError, match="holds no schema.json"):
        load_buffer(tmp_path)


def test_create_replay_mixes_saved_demonstrations_into_every_draw(tmp_path):
    from isaac_rl_posttraining.buffers.storage import save_buffer
    from isaac_rl_posttraining.utils.training_utils import create_replay

    demos = ReplayBuffer(16)
    _add_episode(demos, 0, length=4, success=True, source="expert")
    save_buffer(demos, tmp_path / "demos")
    replay = create_replay(
        capacity=32,
        drop_components=(),
        reward_channel="env",
        reward_aggregation="last",
        success_fraction=None,
        demonstrations=str(tmp_path / "demos"),
        demonstration_fraction=0.25,
    )
    assert (
        len(replay.demonstrations) == 4
        and replay.demonstrations.reward_channel == "env"
    )
    # Before the robot has any experience the demonstrations supply the whole batch...
    batch = replay.sampler.sample(8)
    assert batch["behavior_source"] == ["expert"] * 8
    # ...and once it does, the configured quarter of every batch is demonstrations.
    _add_episode(replay.buffer, 7, length=3, success=False)
    batch = replay.sampler.sample(8)
    assert batch["behavior_source"].count("expert") == 2 and len(batch["reward"]) == 8
    with pytest.raises(ValueError, match="no reward channel 'shaped'"):
        create_replay(
            capacity=32,
            drop_components=(),
            reward_channel="shaped",
            reward_aggregation="last",
            success_fraction=None,
            demonstrations=str(tmp_path / "demos"),
            demonstration_fraction=0.25,
        )
