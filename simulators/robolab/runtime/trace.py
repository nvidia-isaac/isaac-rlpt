# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small JSONL trace recorder for simulator-neutral runtime decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulators.robolab.runtime.types import PolicyProposal, RuntimeDecision


class TraceRecorder:
    """Append deterministic, compact runtime-decision records to JSONL."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        episode_id: str,
        env_id: int,
        step_idx: int,
        proposal_boundary: bool,
        proposal: PolicyProposal,
        decision: RuntimeDecision,
        action: np.ndarray,
    ) -> dict[str, Any]:
        """Record one returned policy action and its approving decision."""
        action_arr = np.ascontiguousarray(np.asarray(action))
        proposal_arr = np.ascontiguousarray(np.asarray(proposal.actions))
        event = {
            "episode_id": episode_id,
            "env_id": int(env_id),
            "step_idx": int(step_idx),
            "proposal_boundary": bool(proposal_boundary),
            "base_model_name": proposal.base_model_name,
            "inference_ms": float(proposal.inference_ms),
            "proposal_shape": list(proposal_arr.shape),
            "proposal_sha256": _array_digest(proposal_arr),
            "latent": _latent_trace(proposal),
            "decision": {
                "kind": decision.kind,
                "execute_prefix_len": int(decision.execute_prefix_len),
                "reason": decision.reason,
                "selected_base_model": decision.selected_base_model,
                "steering_payload": decision.steering_payload,
            },
            "action": action_arr.tolist(),
            "action_sha256": _array_digest(action_arr),
        }
        self._append(event)
        return event

    def record_steering_outcome(
        self,
        *,
        episode_id: str,
        env_id: int,
        step_idx: int,
        task: str,
        requested_latent: np.ndarray,
        clipping: dict[str, Any],
        executed_action_count: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        update_count: int,
    ) -> dict[str, Any]:
        """Record the environment result of one latent-guided action chunk.

        A proposal becomes useful training evidence only after the simulator
        reports what executing its approved prefix actually did.  This record
        joins the server-authoritative clipping metadata with that outcome.
        """
        latent = np.ascontiguousarray(np.asarray(requested_latent))
        if executed_action_count < 1:
            raise ValueError("executed_action_count must be positive")
        event = {
            "event_type": "steering_outcome",
            "episode_id": episode_id,
            "env_id": int(env_id),
            "step_idx": int(step_idx),
            "task": str(task),
            "requested_latent": {
                "values": latent.tolist(),
                "shape": list(latent.shape),
                "sha256": _array_digest(latent),
            },
            "clipping": clipping,
            "executed_action_count": int(executed_action_count),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "update_count": int(update_count),
        }
        self._append(event)
        return event

    def record_training_update(
        self, *, task: str, update_count: int, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Record one learner update beside the steering outcomes it consumed."""
        event = {
            "event_type": "training_update",
            "task": str(task),
            "update_count": int(update_count),
            "metrics": _json_scalars(metrics),
        }
        self._append(event)
        return event

    def _append(self, event: dict[str, Any]) -> None:
        """Keep the in-memory and optional JSONL views in lockstep."""
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
                stream.write("\n")


def _array_digest(array: np.ndarray) -> str:
    """Hash dtype, shape, and contents so traces can detect action drift."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _latent_trace(proposal: PolicyProposal) -> dict[str, Any] | None:
    """Serialize an injected latent and its clipping metadata when present."""
    if proposal.latent is None:
        return None
    latent = np.ascontiguousarray(np.asarray(proposal.latent))
    return {
        "values": latent.tolist(),
        "shape": list(latent.shape),
        "sha256": _array_digest(latent),
        "metadata": proposal.control_metadata,
    }


def _json_scalars(values: dict[str, Any]) -> dict[str, Any]:
    """Convert scalar metric values without widening the trace data contract."""
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            result[str(key)] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result
