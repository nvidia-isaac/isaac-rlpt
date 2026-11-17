# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A behavior-preserving RoboLab policy-client wrapper.

The wrapper composes a normal RoboLab client rather than forking RoboLab's
evaluation loop.  It reproduces ``InferenceClient``'s documented chunking
logic using the wrapped client's four hooks, which lets the harness inspect a
full proposal before caching a bounded prefix.
"""

from __future__ import annotations

from time import time
from typing import Any

import numpy as np

from simulators.robolab.policies.adapter import RoboLabPolicyAdapter
from simulators.robolab.runtime.arbiter import NoOpArbiter
from simulators.robolab.runtime.executive import RuntimeExecutive
from simulators.robolab.runtime.trace import TraceRecorder
from simulators.robolab.runtime.types import RuntimeObservation


class HarnessInferenceClient:
    """RoboLab-compatible client with an explicit proposal/decision boundary.

    No-op construction preserves the upstream client's exact observation
    extraction, request packing, transport, action unpacking, postprocessing,
    refresh horizon, and per-environment chunk semantics.  The underlying
    client remains the owner of transport and any server-specific reset state.
    """

    def __init__(
        self,
        adapter: RoboLabPolicyAdapter,
        *,
        executive: RuntimeExecutive | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.adapter = adapter
        self.executive = executive or RuntimeExecutive(NoOpArbiter())
        self.trace_recorder = trace_recorder or TraceRecorder()
        # These match the state held by RoboLab's InferenceClient base class.
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}
        self._refresh_lengths: dict[int, int] = {}
        self._episode_indices: dict[int, int] = {}
        self._step_indices: dict[int, int] = {}
        self._active_proposals: dict[int, Any] = {}
        self._active_decisions: dict[int, Any] = {}

    @property
    def open_loop_horizon(self) -> int:
        """Expose the native client's exact refresh horizon to RoboLab."""
        return self.adapter.open_loop_horizon

    @property
    def capabilities(self):
        """Expose the adapter's explicit policy capability handshake."""
        return self.adapter.capabilities

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        """Return the same ``{action, viz}`` shape as RoboLab clients.

        In no-op mode this is behaviorally equivalent to the upstream
        ``InferenceClient.infer`` implementation.  The only added work is a
        proposal/decision record written beside RoboLab's native evidence.
        """
        extracted = self.adapter.extract_observation(obs, env_id=env_id)
        proposal_boundary = self._needs_refresh(env_id)
        if proposal_boundary:
            proposal = self.adapter.propose(extracted, instruction)
            runtime_obs = RuntimeObservation(
                obs=extracted,
                timestamp=time(),
                episode_id=self._episode_id(env_id),
                step_idx=self._step_indices.get(env_id, 0),
                simulator_state=None,
            )
            decision = self.executive.decide(runtime_obs, proposal)
            # Future arbiters may deliberately select a shorter prefix.  The
            # no-op arbiter selects the full proposal, exactly like RoboLab.
            self._set_chunk(env_id, proposal.actions[: decision.execute_prefix_len])
            self._active_proposals[env_id] = proposal
            self._active_decisions[env_id] = decision

        action = self._next_action(env_id)
        proposal = self._active_proposals[env_id]
        decision = self._active_decisions[env_id]
        self.trace_recorder.record(
            episode_id=self._episode_id(env_id),
            env_id=env_id,
            step_idx=self._step_indices.get(env_id, 0),
            proposal_boundary=proposal_boundary,
            proposal=proposal,
            decision=decision,
            action=action,
        )
        self._step_indices[env_id] = self._step_indices.get(env_id, 0) + 1
        return {"action": action, "viz": self.adapter.visualization(extracted)}

    def infer_batch(
        self, obs: Any, instruction: str, *, env_ids: list[int]
    ) -> dict[int, dict[str, Any]]:
        """Match RoboLab's default serial batch behavior exactly."""
        return {env_id: self.infer(obs, instruction, env_id=env_id) for env_id in env_ids}

    def reset(self, *, env_id: int | None = None) -> None:
        """Reset only the requested environment's harness and native state."""
        self.adapter.reset(env_id=env_id)
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
            self._refresh_lengths.clear()
            self._active_proposals.clear()
            self._active_decisions.clear()
            self._step_indices.clear()
            for known_env in self._episode_indices:
                self._episode_indices[known_env] += 1
            return
        self._chunks.pop(env_id, None)
        self._counters.pop(env_id, None)
        self._refresh_lengths.pop(env_id, None)
        self._active_proposals.pop(env_id, None)
        self._active_decisions.pop(env_id, None)
        self._step_indices[env_id] = 0
        self._episode_indices[env_id] = self._episode_indices.get(env_id, 0) + 1

    def close(self) -> None:
        """Close the original client transport."""
        self.adapter.close()

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        """Match RoboLab's visualization convenience method."""
        return self.adapter.visualization(self.adapter.extract_observation(obs, env_id=env_id))

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self._refresh_lengths[env_id]

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = np.asarray(chunk)
        self._counters[env_id] = 0
        # The upstream client normally refreshes at open_loop_horizon.  A
        # future arbiter may approve a shorter prefix, which must reobserve
        # before the cached prefix is exhausted.
        self._refresh_lengths[env_id] = min(self.open_loop_horizon, len(self._chunks[env_id]))

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action

    def _episode_id(self, env_id: int) -> str:
        return f"env-{env_id}-episode-{self._episode_indices.get(env_id, 0)}"
