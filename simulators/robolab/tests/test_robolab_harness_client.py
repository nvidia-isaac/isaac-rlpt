# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-op parity tests for the RoboLab policy-client harness.

These tests deliberately use the four-hook interface documented by RoboLab
instead of importing Isaac Sim.  The reference client below is a direct small
implementation of RoboLab's public ``InferenceClient`` chunking semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulators.robolab.policies.gr00t import Gr00tRoboLabAdapter
from simulators.robolab.policies.harness_client import HarnessInferenceClient
from simulators.robolab.runtime.executive import RuntimeExecutive
from simulators.robolab.runtime.types import RuntimeDecision


class _HookClient:
    """Deterministic stand-in for RoboLab's GR00T client hooks."""

    open_loop_horizon = 2

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reset_calls: list[int | None] = []
        self.closed = False

    def _extract_observation(
        self, raw_obs: dict[str, Any], *, env_id: int = 0
    ) -> dict[str, Any]:
        return {"env_id": env_id, "frame": int(raw_obs["frames"][env_id])}

    def _pack_request(
        self, extracted_obs: dict[str, Any], instruction: str
    ) -> dict[str, Any]:
        return {"instruction": instruction, **extracted_obs}

    def _query_server(self, request: dict[str, Any]) -> dict[str, np.ndarray]:
        self.requests.append(request)
        base = 100 * request["env_id"] + 10 * request["frame"]
        return {
            "chunk": np.array(
                [[base + 1.0, 0.2], [base + 2.0, 0.8], [base + 3.0, 0.2]],
                dtype=np.float32,
            )
        }

    def _unpack_response(self, response: dict[str, np.ndarray]) -> np.ndarray:
        return response["chunk"]

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        processed = chunk.copy()
        processed[:, -1] = (processed[:, -1] > 0.5).astype(processed.dtype)
        return processed

    def _build_visualization(self, extracted_obs: dict[str, Any]) -> np.ndarray:
        return np.array(
            [extracted_obs["env_id"], extracted_obs["frame"]], dtype=np.int64
        )

    def reset(self, *, env_id: int | None = None) -> None:
        self.reset_calls.append(env_id)

    def close(self) -> None:
        self.closed = True


class _ReferenceInferenceClient(_HookClient):
    """The documented upstream InferenceClient flow, isolated from RoboLab."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        extracted = self._extract_observation(obs, env_id=env_id)
        if (
            env_id not in self._chunks
            or self._counters[env_id] >= self.open_loop_horizon
        ):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            self._chunks[env_id] = self._postprocess_chunk(
                self._unpack_response(response)
            )
            self._counters[env_id] = 0
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return {"action": action, "viz": self._build_visualization(extracted)}

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)


def test_noop_harness_matches_documented_chunking_requests_and_outputs():
    """The wrapper preserves per-env cache cadence and exact postprocessed actions."""
    native = _ReferenceInferenceClient()
    wrapped_native = _HookClient()
    harness = HarnessInferenceClient(Gr00tRoboLabAdapter(wrapped_native))
    observation = {"frames": np.array([3, 7], dtype=np.int64)}

    for env_id in (0, 1, 0, 1, 0):
        expected = native.infer(observation, "place banana", env_id=env_id)
        actual = harness.infer(observation, "place banana", env_id=env_id)
        assert set(actual) == {"action", "viz"}
        np.testing.assert_array_equal(actual["action"], expected["action"])
        np.testing.assert_array_equal(actual["viz"], expected["viz"])

    assert wrapped_native.requests == native.requests
    assert [event["proposal_boundary"] for event in harness.trace_recorder.events] == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert harness.capabilities.action_horizon == 3
    assert harness.capabilities.action_dim == 2
    assert harness.capabilities.latent_steering is False


def test_noop_harness_reset_is_scoped_to_one_environment_and_forwards_to_client():
    """Resetting one environment cannot discard another environment's chunk."""
    native = _ReferenceInferenceClient()
    wrapped_native = _HookClient()
    harness = HarnessInferenceClient(Gr00tRoboLabAdapter(wrapped_native))
    observation = {"frames": np.array([1, 2], dtype=np.int64)}

    for env_id in (0, 1):
        native.infer(observation, "move", env_id=env_id)
        harness.infer(observation, "move", env_id=env_id)
    native.reset(env_id=0)
    harness.reset(env_id=0)

    # Env 0 must re-query after reset. Env 1 still consumes the second cached action.
    for env_id in (0, 1):
        expected = native.infer(observation, "move", env_id=env_id)
        actual = harness.infer(observation, "move", env_id=env_id)
        np.testing.assert_array_equal(actual["action"], expected["action"])

    assert wrapped_native.requests == native.requests
    assert wrapped_native.reset_calls == [0]
    assert harness.trace_recorder.events[-2]["episode_id"] == "env-0-episode-1"


class _ShortPrefixArbiter:
    """A future-monitor-like arbiter used only to test the bounded seam."""

    def decide(self, observation, proposal) -> RuntimeDecision:
        del observation
        return RuntimeDecision(
            kind="execute",
            execute_prefix_len=1,
            reason="test_short_prefix",
            selected_base_model=proposal.base_model_name,
        )


def test_non_noop_arbiter_can_force_early_reobservation_without_action_rewriting():
    """The future monitor seam changes cache length, not GR00T action values."""
    wrapped_native = _HookClient()
    harness = HarnessInferenceClient(
        Gr00tRoboLabAdapter(wrapped_native),
        executive=RuntimeExecutive(_ShortPrefixArbiter()),
    )
    observation = {"frames": np.array([4], dtype=np.int64)}

    first = harness.infer(observation, "move", env_id=0)
    second = harness.infer(observation, "move", env_id=0)

    # One-step approved prefixes require a fresh proposal for every call.
    assert len(wrapped_native.requests) == 2
    np.testing.assert_array_equal(
        first["action"], np.array([41.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        second["action"], np.array([41.0, 0.0], dtype=np.float32)
    )
    assert all(
        event["decision"]["execute_prefix_len"] == 1
        for event in harness.trace_recorder.events
    )
