# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-trace tests that do not require RoboLab or model dependencies."""

from __future__ import annotations

import json

import numpy as np
import pytest

from simulators.robolab.runtime.trace import TraceRecorder
from simulators.robolab.runtime.types import PolicyProposal, RuntimeDecision


def test_trace_recorder_writes_a_replayable_jsonl_decision(tmp_path):
    """A trace contains action provenance and explicit arbiter authority."""
    path = tmp_path / "nested" / "trace.jsonl"
    proposal = PolicyProposal(
        actions=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        latent=None,
        predicted_future=None,
        predicted_value=None,
        base_model_name="test_base_model",
        inference_ms=4.5,
    )
    decision = RuntimeDecision(
        kind="execute",
        execute_prefix_len=2,
        reason="no_op",
        selected_base_model="test_base_model",
    )
    recorder = TraceRecorder(path)

    event = recorder.record(
        episode_id="env-0-episode-0",
        env_id=0,
        step_idx=7,
        proposal_boundary=True,
        proposal=proposal,
        decision=decision,
        action=proposal.actions[0],
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == event
    assert persisted["proposal_shape"] == [2, 2]
    assert persisted["action"] == [1.0, 2.0]
    assert persisted["decision"] == {
        "execute_prefix_len": 2,
        "kind": "execute",
        "reason": "no_op",
        "selected_base_model": "test_base_model",
        "steering_payload": None,
    }
    assert len(persisted["proposal_sha256"]) == 64
    assert len(persisted["action_sha256"]) == 64


def test_trace_digests_distinguish_same_values_with_different_shape():
    """Shape participates in the digest so replay comparisons cannot alias it."""
    proposal = PolicyProposal(
        actions=np.array([[1.0, 2.0]], dtype=np.float32),
        latent=None,
        predicted_future=None,
        predicted_value=None,
        base_model_name="test_base_model",
        inference_ms=0.0,
    )
    decision = RuntimeDecision("execute", 1, "no_op", "test_base_model")
    recorder = TraceRecorder()
    vector = recorder.record(
        episode_id="e",
        env_id=0,
        step_idx=0,
        proposal_boundary=True,
        proposal=proposal,
        decision=decision,
        action=np.array([1.0, 2.0], dtype=np.float32),
    )
    matrix = recorder.record(
        episode_id="e",
        env_id=0,
        step_idx=1,
        proposal_boundary=False,
        proposal=proposal,
        decision=decision,
        action=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    assert vector["action_sha256"] != matrix["action_sha256"]


def test_trace_records_latent_values_and_server_clipping_metadata():
    """DSRL traces retain the control actually proposed alongside its boundary."""
    proposal = PolicyProposal(
        actions=np.array([[1.0, 2.0]], dtype=np.float32),
        latent=np.array([4.0, -3.0], dtype=np.float32),
        predicted_future=None,
        predicted_value=None,
        base_model_name="gr00t_n1.7_droid_latent",
        inference_ms=1.0,
        control_metadata={"latent_clip": 3.0, "clipped_entries": 1, "additive": True},
    )
    decision = RuntimeDecision("steer", 1, "dsrl", proposal.base_model_name)
    event = TraceRecorder().record(
        episode_id="e",
        env_id=0,
        step_idx=0,
        proposal_boundary=True,
        proposal=proposal,
        decision=decision,
        action=proposal.actions[0],
    )
    assert event["latent"]["values"] == [4.0, -3.0]
    assert event["latent"]["metadata"]["clipped_entries"] == 1


def test_trace_records_latent_execution_outcome_and_learner_update(tmp_path):
    """A DSRL trace joins latent provenance to simulator and learner events."""
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path)
    outcome = recorder.record_steering_outcome(
        episode_id="robolab-BananaInBowl-episode-0",
        env_id=0,
        step_idx=8,
        task="BananaInBowl",
        requested_latent=np.array([0.25, -0.25], dtype=np.float32),
        clipping={"latent_clip": 3.0, "clipped_entries": 0, "additive": True},
        executed_action_count=8,
        reward=1.0,
        terminated=True,
        truncated=False,
        update_count=0,
    )
    update = recorder.record_training_update(
        task="BananaInBowl",
        update_count=1,
        metrics={"online/actor_loss": np.float32(0.5)},
    )

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [outcome, update]
    assert outcome["executed_action_count"] == 8
    assert outcome["requested_latent"]["shape"] == [2]
    assert outcome["clipping"]["clipped_entries"] == 0
    assert update["metrics"]["online/actor_loss"] == pytest.approx(0.5)
