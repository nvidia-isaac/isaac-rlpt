# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint that shares a step tag with an eval holds the evaluated weights.

Regression for the save/eval offset: saves used to fire immediately when the
env-step counter crossed ``save_interval`` (mid-episode, before the episode's
training burst), while evals deferred to the episode boundary and ran AFTER
the burst — so "the checkpoint at the peak step" was systematically one
episode-burst older than the weights the eval measured. Saves now defer to
the same gate as evals and execute after them, with no training in between.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from conftest import (
    RecordingBuffer,
    ScriptedLearnerGroup,
    ScriptedRuntimeSource,
    single_learner_batch,
)

from isaac_rl_posttraining.training import (
    CheckpointStore,
    FingerprintSink,
    OffPolicyConfig,
    OffPolicyTrainer,
)
from isaac_rl_posttraining.utils.checkpoint_retention import (
    EVALUATION_SUMMARY_FILE,
)


class _WeightRecordingEvaluator:
    """Evaluator double recording the learner's update count per evaluation."""

    def __init__(self, record, source, learner):
        self._record = record
        self._source = source
        self._learner = learner

    def evaluate(self, policy_state, *, env_steps=0):
        self._record.append(
            {
                "cum_steps": self._source.cum_steps,
                "train_steps_done": len(self._learner.train_step_calls),
            }
        )
        return {
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "success_rate": 1.0,
            "avg_steps": 1.0,
        }


def _make_setup(tmp_path, script, action_exec_len, num_rollouts, freq, actor=None):
    eval_record = []
    save_dir = str(tmp_path / "ckpts")
    source = ScriptedRuntimeSource(script, action_exec_len)
    learner = ScriptedLearnerGroup(
        step_at_call=lambda: source.cum_steps, policy_module=actor
    )
    trainer = OffPolicyTrainer(
        train_source=source,
        learner_group=learner,
        buffer=RecordingBuffer(len_fn=lambda: source.cum_steps),
        draw_batch=single_learner_batch,
        evaluator=_WeightRecordingEvaluator(eval_record, source, learner),
        config=OffPolicyConfig(
            num_rollouts=num_rollouts,
            learning_starts=0,
            train_after_episode=True,
            eval_freq=freq,
            eval_at_episode_boundary=True,
            save_interval=freq,
        ),
        fingerprint_sink=FingerprintSink(save_dir),
        checkpoint_store=CheckpointStore(save_dir),
    )
    return trainer, source, learner, eval_record


def _saved_train_steps(save_dirs):
    """Map checkpoint tag -> train-step count recorded inside the payload."""
    out = {}
    for d in save_dirs:
        tag = os.path.basename(os.path.normpath(d))
        payload = torch.load(os.path.join(d, "algo_state.pt"), weights_only=False)
        out[tag] = payload["train_steps"]
    return out


def test_checkpoint_weights_match_coinciding_eval(tmp_path):
    """Save defers to the episode boundary and captures post-burst weights.

    Script: episodes of 30 env steps (3 iterations x 10); the 20-step
    save/eval interval crosses mid-episode. Both must execute at the episode
    boundary, after that episode's burst, with identical weight versions.
    """
    trainer, _, learner, eval_record = _make_setup(
        tmp_path,
        script=[{"steps": 10, "episodes": 1 if i % 3 == 2 else 0} for i in range(9)],
        action_exec_len=10,
        num_rollouts=90,
        freq=20,
    )
    trainer.run()

    saved = _saved_train_steps(learner.save_dirs)
    # Periodic checkpoints carry the eval-step tags (episode boundaries at
    # 30/60/90 env steps), not the mid-episode crossing steps (20/40/80).
    periodic_tags = [t for t in saved if t != "final"]
    assert sorted(int(t) for t in periodic_tags) == [30, 60, 90], (
        f"saves must defer to episode boundaries; got tags {sorted(saved)}"
    )

    # Weight-version identity: the train-step count inside each checkpoint
    # equals the count at the moment of the coinciding eval.
    evals_by_step = {e["cum_steps"]: e["train_steps_done"] for e in eval_record}
    for tag in periodic_tags:
        assert int(tag) in evals_by_step, f"no eval recorded at checkpoint step {tag}"
        assert saved[tag] == evals_by_step[int(tag)], (
            f"checkpoint-{tag} holds train_steps={saved[tag]} but the eval at that "
            f"step measured train_steps={evals_by_step[int(tag)]}"
        )


def test_checkpoint_metadata_records_eval_sr(tmp_path):
    """A checkpoint coinciding with an eval records that eval's success rate in its manifest."""
    trainer, _, learner, eval_record = _make_setup(
        tmp_path,
        script=[{"steps": 10, "episodes": 1 if i % 3 == 2 else 0} for i in range(9)],
        action_exec_len=10,
        num_rollouts=90,
        freq=20,
    )
    trainer.run()
    assert eval_record, "expected at least one eval"
    for d in learner.save_dirs:
        tag = os.path.basename(os.path.normpath(d))
        if tag == "final":
            continue
        with open(os.path.join(d, "manifest.json")) as stream:
            manifest = json.load(stream)
        assert manifest["eval_success_rate"] == pytest.approx(1.0), (
            f"checkpoint {tag} must record its coinciding eval"
        )


def test_evaluation_summary_json_records_every_evaluation(tmp_path):
    """The loop writes evaluation_summary.json next to the checkpoints after every evaluation."""
    trainer, _, _, eval_record = _make_setup(
        tmp_path,
        script=[{"steps": 10, "episodes": 1 if i % 3 == 2 else 0} for i in range(9)],
        action_exec_len=10,
        num_rollouts=90,
        freq=20,
    )
    trainer.run()

    with open(tmp_path / "ckpts" / EVALUATION_SUMMARY_FILE) as stream:
        summary = json.load(stream)
    assert (
        [record["step"] for record in summary["history"]]
        == [e["cum_steps"] for e in eval_record]
        == [30, 60, 90]
    )
    assert summary["latest"] == summary["history"][-1]
    assert summary["latest"]["success_rate"] == pytest.approx(1.0)
    assert summary["latest"]["avg_steps"] == pytest.approx(1.0)


def test_fingerprints_jsonl_matches_eval_and_save(tmp_path):
    """Every eval/save pair sharing a step shares the actor weight digest."""
    from torch import nn

    trainer, _, _, _ = _make_setup(
        tmp_path,
        script=[{"steps": 10, "episodes": 1 if i % 3 == 2 else 0} for i in range(9)],
        action_exec_len=10,
        num_rollouts=90,
        freq=20,
        actor=nn.Linear(4, 2),
    )
    trainer.run()

    fp_path = tmp_path / "ckpts" / "fingerprints.jsonl"
    assert fp_path.is_file(), "fingerprints.jsonl must be written next to checkpoints"
    records = [json.loads(line) for line in fp_path.read_text().splitlines()]
    evals = {r["step"]: r["actor_sha256"] for r in records if r["event"] == "eval"}
    saves = {r["step"]: r["actor_sha256"] for r in records if r["event"] == "save"}
    assert evals and saves
    shared_steps = set(evals) & set(saves)
    assert shared_steps, "expected eval and save records at shared steps"
    for step in shared_steps:
        assert evals[step] == saves[step], (
            f"fingerprint mismatch at step {step}: eval={evals[step][:16]} save={saves[step][:16]}"
        )
    for record in records:
        assert len(record["actor_sha256"]) == 64
