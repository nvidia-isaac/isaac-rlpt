# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint retention shared by the DSRL and PPO trainers.

``checkpoint_retention`` has one meaning in both: ``"all"`` keeps every
checkpoint; ``"best_last_final"`` keeps the checkpoint nearest the best
evaluation, the newest one, and ``final``. The best-evaluated weights are
never deleted, and with no evaluation on record nothing is pruned. The
evaluations retention decides on are the ones the trainers record in
:data:`EVALUATION_SUMMARY_FILE` next to the checkpoints.
"""

from __future__ import annotations

__all__ = [
    "EVALUATION_SUMMARY_FILE",
    "RETENTION_MODES",
    "select_protected_checkpoints",
    "validate_checkpoint_retention",
]

#: Latest evaluation and full history, written next to the checkpoints after
#: every evaluation; launchers read the result from it instead of log lines.
EVALUATION_SUMMARY_FILE = "evaluation_summary.json"

RETENTION_MODES = ("all", "best_last_final")


def validate_checkpoint_retention(mode: str) -> str:
    """Return ``mode`` if it is a known retention mode, else raise."""
    if mode not in RETENTION_MODES:
        raise ValueError(f"checkpoint_retention must be one of {list(RETENTION_MODES)}, got {mode!r}")
    return mode


def select_protected_checkpoints(saved_tags: list[int], eval_history: dict[int, float]) -> set[int]:
    """Pick the numeric checkpoint tags a retention pass must keep.

    Protects the last checkpoint and the one nearest the best-evaluating
    step. With no recorded evaluations every tag is protected (retention
    must never guess).

    Args:
        saved_tags: Numeric tags of the checkpoints currently on disk.
        eval_history: Mapping of step -> success rate for completed
            in-training evaluations.

    Returns:
        The subset of ``saved_tags`` to keep ("final" is handled by the
        caller and is always kept).
    """
    if not saved_tags:
        return set()
    if not eval_history:
        return set(saved_tags)
    best_step = max(eval_history, key=lambda k: (eval_history[k], k))
    nearest_to_best = min(saved_tags, key=lambda t: (abs(t - best_step), -t))
    return {nearest_to_best, max(saved_tags)}
