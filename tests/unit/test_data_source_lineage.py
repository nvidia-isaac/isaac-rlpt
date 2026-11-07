# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data source lineage, episode-id namespacing, telemetry tables, persistence and the dashboard."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import make_transition

from isaac_rl_posttraining.buffers import (
    MixedSampler,
    ReplayBuffer,
    UniformSampler,
    load_buffer,
    save_buffer,
)
from isaac_rl_posttraining.rollouts import InProcessBufferSink
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes.base import (
    CollectedDecision,
    PolicyState,
    RolloutRequest,
    RolloutResult,
)
from isaac_rl_posttraining.runtimes.concurrency import RolloutGroupBase
from isaac_rl_posttraining.training import DataSource, PersistSpec, RuntimeDataSource
from isaac_rl_posttraining.training.recording import RunCounters
from isaac_rl_posttraining.training.telemetry import (
    SCHEMA_METADATA_KEY,
    SCHEMA_VERSION,
    TABLES,
    open_run_telemetry,
    read_telemetry,
)
from isaac_rl_posttraining.viz import write_dashboard

DECISIONS_PER_EPISODE = 2


class _Group(RolloutGroupBase):
    """Workers with ``envs_per_worker`` environments; episodes of two decisions, numbered per environment from 0."""

    def __init__(
        self, workers: int = 2, envs_per_worker: int = 2, *, frames: bool = False
    ) -> None:
        self.num_rollout_workers = workers
        self.envs_per_worker = envs_per_worker
        self.num_envs = workers * envs_per_worker
        self.frames = frames
        self.version = 0
        self.decision = [0] * self.num_envs
        self.episode = [0] * self.num_envs
        self.closed = False

    def collect(self, requests: list[RolloutRequest]) -> tuple[RolloutResult, ...]:
        results = []
        for worker, _ in enumerate(requests):
            decisions = []
            for local in range(self.envs_per_worker):
                env = worker * self.envs_per_worker + local
                index = self.decision[env]
                ended = index == DECISIONS_PER_EPISODE - 1
                obs = {"state": np.full(2, env, dtype=np.float32)}
                if self.frames:
                    obs["image"] = np.zeros((4, 4, 3), dtype=np.uint8)
                transition = make_transition(
                    episode_id=self.episode[env],
                    step_in_episode=index * 2,
                    obs=obs,
                    next_obs=obs,
                    action=np.zeros(2, dtype=np.float32),
                    rewards=[0.5, 1.0],
                    done=ended,
                    success=ended,
                    info={"is_success": ended} if ended else None,
                )
                transition = _with_metadata(transition, env, index, self.version)
                chunk = ActionChunk(
                    learner_action=np.zeros((1, 2), dtype=np.float32),
                    environment_actions=np.zeros((1, 2, 3), dtype=np.float32),
                    execution_length=2,
                    learner_action_kind="latent",
                    policy_version=self.version,
                )
                decisions.append(
                    CollectedDecision(
                        environment_index=env, transition=transition, action_chunk=chunk
                    )
                )
                self.decision[env] = 0 if ended else index + 1
                if ended:
                    self.episode[env] += 1
            results.append(
                RolloutResult(
                    worker_index=worker,
                    decisions=tuple(decisions),
                    env_steps=2 * len(decisions),
                    episodes_completed=0,
                    policy_version=self.version,
                    timings={
                        "action_inference_seconds": 0.2,
                        "environment_step_seconds": 0.4,
                        "policy_calls": 2.0,
                    },
                )
            )
        return tuple(results)

    def set_policy_state(self, state: PolicyState) -> tuple[int, ...]:
        self.version = state.version
        return (state.version,) * self.num_rollout_workers

    def close(self) -> None:
        self.closed = True


def _with_metadata(
    transition, environment_index: int, decision_index: int, version: int
):
    from dataclasses import replace

    metadata = replace(
        transition.metadata,
        environment_index=environment_index,
        decision_index=decision_index,
        policy_version=version,
        wall_time=1000.0 + decision_index,
    )
    return replace(transition, metadata=metadata)


def _source(group: _Group, buffer: ReplayBuffer, **kwargs) -> DataSource:
    return DataSource(group, InProcessBufferSink(buffer), **kwargs)


def test_old_name_is_an_alias() -> None:
    assert RuntimeDataSource is DataSource


def test_lineage_is_stamped_on_every_transition() -> None:
    buffer = ReplayBuffer(1000)
    group = _Group()
    source = _source(group, buffer)
    source.run(can_train=True)
    source.publish(PolicyState(version=1, tensors={}))
    source.run(can_train=True)

    transitions = buffer.transitions()
    assert len(transitions) == 8
    assert {t.metadata.source for t in transitions} == {"train"}
    assert {t.origin for t in transitions} == {"real"}
    assert {t.metadata.worker_index for t in transitions} == {0, 1}
    assert sorted({t.episode_id for t in transitions}) == [
        0,
        1,
        2,
        3,
    ]  # plain integers for "train"
    assert all(t.metadata.policy_lag == 0 for t in transitions)


def test_policy_lag_counts_versions_published_while_in_flight() -> None:
    buffer = ReplayBuffer(1000)
    source = _source(_Group(), buffer)
    pending = source.start(can_train=True)
    pending.call.result()  # the round ran on version 0
    source.publish(PolicyState(version=1, tensors={}))
    source.publish(PolicyState(version=2, tensors={}))
    source.finish(pending)
    assert {t.metadata.policy_lag for t in buffer.transitions()} == {2}
    assert source.recorder.last_policy_lag == (2.0, 2)


def test_two_sources_share_one_buffer_with_namespaced_episode_ids() -> None:
    buffer = ReplayBuffer(1000)
    counters = RunCounters()
    real = _source(_Group(), buffer, counters=counters)
    imagined = _source(
        _Group(), buffer, source="imagined", origin="model", counters=counters
    )
    for _ in range(2):
        real.run(can_train=True)
        imagined.run(can_train=True)

    ids = {t.episode_id for t in buffer.transitions()}
    assert ids == {0, 1, 2, 3, "imagined:0", "imagined:1", "imagined:2", "imagined:3"}
    assert len(buffer.indices(origin="model")) == 8
    assert len(buffer.indices(origin="real")) == 8
    assert buffer.count(origin="model", outcome="success") == 8
    batch = buffer.collate(buffer.indices(origin="model"))
    assert batch["origin"] == ["model"] * 8
    assert "origin" not in buffer.collate(buffer.indices(origin="real"))
    assert counters.env_steps == 32  # both sources advance the run's step count


def test_decisions_beyond_max_policy_lag_stay_out_of_the_buffer(tmp_path: Path) -> None:
    buffer = ReplayBuffer(1000)
    persisted = tmp_path / "persisted"
    sink = open_run_telemetry(tmp_path)
    source = _source(
        _Group(),
        buffer,
        max_policy_lag=1,
        telemetry=sink,
        persist=PersistSpec(str(persisted), every_episodes=1),
    )
    source.run(can_train=True)  # lag 0: kept
    pending = source.start(can_train=True)  # collects on version 0
    source.finish(pending, learner_version=1)  # lag 1: kept
    for _ in range(2):
        pending = source.start(can_train=True)
        source.finish(pending, learner_version=2)  # lag 2: dropped
    source.close()
    sink.close()

    assert len(buffer) == 8 and source.decisions_collected_last_run == 0
    assert (source.dropped_stale_last_run, source.dropped_stale_total) == (4, 8)
    assert {t.metadata.policy_lag for t in buffer.transitions()} == {0, 1}
    rounds = read_telemetry(tmp_path, "rounds").to_pylist()
    assert [row["dropped_stale"] for row in rounds] == [0, 0, 4, 4]
    # Episode metrics and persistence still see every decision.
    assert source.total_episodes == 8
    assert len(load_buffer(str(persisted))) == 16


def test_an_episode_that_lost_a_stale_decision_is_dropped_to_its_end() -> None:
    buffer = ReplayBuffer(1000)
    source = _source(_Group(), buffer, max_policy_lag=0)
    pending = source.start(can_train=True)  # first decisions, collected on version 0
    source.finish(pending, learner_version=1)  # lag 1: dropped
    source.publish(PolicyState(version=1, tensors={}))
    source.run(can_train=True)  # fresh, but the rest of the cut episodes: dropped
    assert len(buffer) == 0 and source.dropped_stale_total == 8
    source.run(can_train=True)  # new episodes start at step 0 and are kept
    assert len(buffer) == 4 and source.dropped_stale_last_run == 0


def test_a_cut_episode_leaves_no_open_head_in_the_buffer() -> None:
    buffer = ReplayBuffer(1000)
    source = _source(_Group(), buffer, max_policy_lag=0)
    source.run(can_train=True)  # the episodes' first decisions: kept
    assert len(buffer) == 4
    pending = source.start(can_train=True)
    source.finish(
        pending, learner_version=1
    )  # their ends, lag 1: dropped, so the heads go too
    assert len(buffer) == 0 and buffer.get_statistics()["open_episodes"] == 0


def test_same_source_name_twice_reports_the_collision() -> None:
    buffer = ReplayBuffer(1000)
    first, second = _source(_Group(), buffer), _source(_Group(), buffer)
    for _ in range(2):
        first.run(can_train=True)
    with pytest.raises(ValueError, match="distinct source names"):
        for _ in range(2):
            second.run(can_train=True)


def test_mixed_sampler_fills_origin_for_real_only_members() -> None:
    real, imagined = ReplayBuffer(100), ReplayBuffer(100)
    real_source = _source(_Group(), real)
    model_source = _source(_Group(), imagined, source="imagined", origin="model")
    for _ in range(2):
        real_source.run(can_train=True)
        model_source.run(can_train=True)
    sampler = MixedSampler(
        {
            "real": UniformSampler(real, rng=np.random.default_rng(0)),
            "model": UniformSampler(imagined, rng=np.random.default_rng(1)),
        },
        {"real": 0.5, "model": 0.5},
    )
    batch = sampler.sample(8)
    assert sorted(set(batch["origin"])) == ["model", "real"]
    assert len(batch["origin"]) == 8


def test_saved_buffer_keeps_lineage_and_origin(tmp_path: Path) -> None:
    buffer = ReplayBuffer(1000)
    source = _source(_Group(), buffer, source="imagined", origin="model")
    source.run(can_train=True)
    source.run(can_train=True)
    save_buffer(buffer, tmp_path / "saved")
    loaded = load_buffer(tmp_path / "saved")
    original, restored = buffer.transitions(), loaded.transitions()
    assert [t.episode_id for t in restored] == [t.episode_id for t in original]
    assert {t.origin for t in restored} == {"model"}
    assert [t.metadata.worker_index for t in restored] == [
        t.metadata.worker_index for t in original
    ]
    assert {t.metadata.source for t in restored} == {"imagined"}
    assert {t.metadata.policy_lag for t in restored} == {0}


def test_telemetry_tables_are_written_with_the_schema(tmp_path: Path) -> None:
    sink = open_run_telemetry(
        tmp_path, config={"algorithm": "dsrl_sac"}, placement={"runtime": "native"}
    )
    group = _Group()
    source = _source(group, ReplayBuffer(1000), telemetry=sink, run_id="run-1")
    eval_recorder = source.recorder.child("eval")
    source.run(can_train=False)
    source.run(can_train=True)
    source.log_learner(7, 8, {"critic_loss": 0.25, "note": "text"}, publish_version=0)  # type: ignore[dict-item]
    eval_recorder.ingest(
        group.collect([RolloutRequest(episodes_per_environment=1, mode="eval")] * 2),
        mode="eval",
        wall_start=0.0,
    )
    source.close()
    sink.close()

    telemetry = tmp_path / "telemetry"
    assert (telemetry / "config.yaml").read_text().startswith("algorithm: dsrl_sac")
    assert (telemetry / "placement.json").exists()
    for name in TABLES:
        part = next((telemetry / name).glob("part-*.parquet"))
        metadata = pq.read_schema(part).metadata[SCHEMA_METADATA_KEY]
        assert f'"schema_version": {SCHEMA_VERSION}'.encode() in metadata
        table = read_telemetry(tmp_path, name)
        assert table.column_names == list(TABLES[name].names)
        for row in table.to_pylist():
            TABLES[name].validate(
                {
                    key: dict(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )

    transitions = read_telemetry(tmp_path, "transitions").to_pylist()
    assert len(transitions) == 12
    assert {row["mode"] for row in transitions} == {"warmup", "train", "eval"}
    assert (
        transitions[-1]["source"] == "eval" and transitions[-1]["env_steps"] == 16
    )  # eval adds no steps
    assert transitions[0]["inference_seconds"] == pytest.approx(
        0.1
    )  # 0.2 s over two decisions
    episodes = read_telemetry(tmp_path, "episodes").to_pylist()
    train_episodes = [row for row in episodes if row["source"] == "train"]
    assert [row["length_decisions"] for row in train_episodes] == [2] * 4
    assert all(
        row["success"] and row["terminal_reason"] == "terminated"
        for row in train_episodes
    )
    assert train_episodes[0]["return"] == pytest.approx(
        2.0
    )  # "last" aggregation: 1.0 per decision
    rounds = read_telemetry(tmp_path, "rounds").to_pylist()
    assert [row["mode"] for row in rounds] == ["warmup", "train", "eval"]
    assert [row["round_index"] for row in rounds] == [0, 1, 2]
    (learner,) = read_telemetry(tmp_path, "learner").to_pylist()
    assert learner["learner_step"] == 7 and learner["env_steps"] == 8
    assert dict(learner["metrics"]) == {
        "critic_loss": 0.25
    }  # non-numeric values are left out


def test_a_round_ingested_late_records_when_the_workers_finished(
    tmp_path: Path,
) -> None:
    sink = open_run_telemetry(tmp_path, config={}, placement={})
    source = _source(_Group(), ReplayBuffer(1000), telemetry=sink, run_id="run-1")
    pending = source.start(can_train=True)
    pending.call.result()
    time.sleep(0.5)  # a learner burst between the round's end and its ingest
    source.finish(pending)
    source.close()
    sink.close()

    (row,) = read_telemetry(tmp_path, "rounds").to_pylist()
    assert row["wall_end"] - row["wall_start"] < 0.25


def test_telemetry_rolls_part_files(tmp_path: Path) -> None:
    sink = open_run_telemetry(tmp_path, flush_rows=4)
    source = _source(_Group(), ReplayBuffer(1000), telemetry=sink)
    for _ in range(3):
        source.run(can_train=True)  # four transitions per round: one part each
    parts = sorted((tmp_path / "telemetry" / "transitions").glob("part-*.parquet"))
    assert [part.name for part in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert read_telemetry(tmp_path, "transitions").num_rows == 12
    assert not list(
        (tmp_path / "telemetry" / "transitions").glob(".*")
    )  # no half-written parts left
    sink.close()


def test_persisted_episodes_round_trip(tmp_path: Path) -> None:
    buffer = ReplayBuffer(1000)
    spec = PersistSpec(str(tmp_path / "data"), every_episodes=3)
    source = _source(_Group(frames=True), buffer, persist=spec, source="robot")
    for _ in range(4):
        source.run(can_train=True)
    source.run(can_train=True)  # four episodes left running: not persisted
    source.close()

    parts = sorted(path.name for path in (tmp_path / "data").iterdir())
    assert parts == [
        "part-00000",
        "part-00001",
        "part-00002",
    ]  # 3 + 3 + the 2 left at close
    loaded = load_buffer(tmp_path / "data")
    complete = buffer.episodes(complete=True)
    assert [episode.id for episode in loaded.episodes()] == [
        episode.id for episode in complete
    ]
    for saved, original in zip(
        loaded.transitions(), [t for e in complete for t in e.transitions], strict=True
    ):
        assert saved.episode_id == original.episode_id
        assert saved.metadata.source == "robot"
        assert "image" not in saved.observation.arrays()  # frames dropped by default
        np.testing.assert_array_equal(
            saved.observation.arrays()["state"], original.observation.arrays()["state"]
        )
        np.testing.assert_array_equal(saved.rewards["env"], original.rewards["env"])


def test_persisted_frames_are_kept_on_request(tmp_path: Path) -> None:
    source = _source(
        _Group(frames=True),
        ReplayBuffer(100),
        persist=PersistSpec(str(tmp_path), every_episodes=1, keep_frames=True),
    )
    for _ in range(2):
        source.run(can_train=True)
    source.close()
    assert "image" in load_buffer(tmp_path).transitions()[0].observation.arrays()


def test_dashboard_is_one_standalone_file(tmp_path: Path) -> None:
    run_dirs = []
    for name in ("run_a", "run_b"):
        sink = open_run_telemetry(tmp_path / name)
        source = _source(_Group(), ReplayBuffer(1000), telemetry=sink, run_id=name)
        for step in range(4):
            source.run(can_train=step > 0)
            source.log_learner(
                step + 1,
                8 * (step + 1),
                {"critic_loss": 1.0 / (step + 1), "q_values_mean": float(step)},
            )
        source.recorder.child("eval").ingest(
            source.group.collect([RolloutRequest(episodes_per_environment=1)] * 2),
            mode="eval",
            wall_start=0.0,
        )
        sink.close()
        run_dirs.append(str(tmp_path / name))

    output = write_dashboard(run_dirs, str(tmp_path / "dashboard.html"))
    text = Path(output).read_text()
    assert text.startswith("<!DOCTYPE html>")
    assert "run_a" in text and "run_b" in text
    for heading in (
        "Success rate (evaluation)",
        "Collection throughput",
        "Policy lag",
        "critic_loss",
        "Rollout workers",
    ):
        assert heading in text
    # Self-contained: no external scripts, styles, images or fonts.
    assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", text)
    assert "<script src" not in text and "<link" not in text and "@import" not in text


def test_uniform_warmup_draws_from_the_seeded_numpy_stream() -> None:
    import gymnasium as gym

    from isaac_rl_posttraining.envs.steering_wrappers import LatentSpaceEnv

    space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, shape=(2,))})
    env = LatentSpaceEnv(
        space,
        gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        None,
        noise_dim=3,
        chunk_size=4,
        action_bound=0.5,
        use_vlm_features=False,
    )
    np.random.seed(123)
    first = env.sample_action()
    np.random.seed(123)
    second = env.sample_action()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 4, 3) and first.dtype == env.chunk_action_space.dtype
    assert np.all(np.abs(first) <= 0.5)
    assert not np.array_equal(first, env.sample_action())


def test_a_source_without_a_sink_hands_on_policy_rounds_back_seeded() -> None:
    group = _Group()
    requests: list[RolloutRequest] = []
    collect = group.collect
    group.collect = lambda batch: (requests.extend(batch), collect(batch))[1]  # type: ignore[method-assign]
    source = DataSource(group, None, count_by="episode", reward_aggregation="sum")
    source.run(can_train=True, reset_seed=7)
    assert requests == [RolloutRequest(episodes_per_environment=1, reset_seed=7)] * 2
    assert [result.worker_index for result in source.last_results] == [0, 1]
    assert (
        source.last_results[0].decisions[0].transition.metadata.source is None
    )  # untagged, as returned
    assert (
        source.decisions_collected_last_run == 4
        and source.last_transitions[0].metadata.source == "train"
    )
    with pytest.raises(ValueError, match="count_by='episode'"):
        DataSource(_Group(), None).run(can_train=True, reset_seed=7)


def test_replacing_the_workers_closes_the_old_first_and_resends_the_weights() -> None:
    old, new = _Group(), _Group()
    events: list[str] = []
    source = DataSource(old, None, count_by="episode")
    source.publish(PolicyState(version=3, tensors={}))
    old.close = lambda: events.append("closed")  # type: ignore[method-assign]
    source.replace_group(lambda: (events.append("opened"), new)[1])
    assert events == ["closed", "opened"]
    assert source.group is new and new.version == 3 and source.policy_version == 3
    with pytest.raises(RuntimeError, match="topology changed"):
        source.replace_group(lambda: _Group(workers=1))
