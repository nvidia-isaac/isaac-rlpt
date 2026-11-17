# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints and weight fingerprints shared by the training loops.

A checkpoint is the learner group's own files plus a ``policy_state.pt``
snapshot of the published weights and a small ``manifest.json``. The store
also keeps ``evaluation_summary.json`` current and prunes checkpoints under
the configured retention rule, so both loops write the same layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from loguru import logger

from isaac_rl_posttraining.loggers.base import Logger
from isaac_rl_posttraining.runtimes.base import ActorFailedError, PolicyState
from isaac_rl_posttraining.runtimes.learner import LearnerGroup
from isaac_rl_posttraining.runtimes.policy_snapshot import (
    SNAPSHOT_FILE_NAME,
    policy_state_tensors,
    save_policy_snapshot,
)
from isaac_rl_posttraining.utils.checkpoint_retention import (
    EVALUATION_SUMMARY_FILE,
    select_protected_checkpoints,
    validate_checkpoint_retention,
)
from isaac_rl_posttraining.utils.compat import add_note

__all__ = [
    "ABORTED_CHECKPOINT_TAG",
    "CheckpointStore",
    "FingerprintSink",
    "policy_state_fingerprint",
    "save_aborted_checkpoint",
]

#: The checkpoint a run writes when it ends with an error.
ABORTED_CHECKPOINT_TAG = "aborted"

#: Seconds before a failed checkpoint write is retried once.
_CHECKPOINT_RETRY_SECONDS = 60.0


def policy_state_fingerprint(state: PolicyState) -> str:
    """SHA-256 over the published tensors, the digest ``actor_fingerprint`` computes for the actor."""
    digest = hashlib.sha256()
    tensors = policy_state_tensors(state)
    for key in sorted(tensors):
        digest.update(key.encode())
        digest.update(tensors[key].to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


class FingerprintSink:
    """Append policy digests to ``<save_dir>/fingerprints.jsonl``.

    One line per eval/save event; equal digests across an evaluation, the
    checkpoint saved at that step, and a later re-evaluation prove they used
    identical weights.
    """

    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir

    def record(self, event: str, *, step: int, state: PolicyState, **extra: Any) -> None:
        fingerprint = policy_state_fingerprint(state)
        record: dict[str, Any] = {"event": event, "step": step, "actor_sha256": fingerprint}
        record.update({key: value for key, value in extra.items() if value is not None})
        logger.info(f"[fingerprint] {event}@{step} actor={fingerprint[:16]}")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "fingerprints.jsonl"), "a") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning(f"[fingerprint] append failed (non-fatal): {exc}")


class CheckpointStore:
    """Write checkpoints under ``<save_dir>/<tag_prefix><tag>`` and prune them.

    Numeric tags are the loop's own progress unit (environment steps for the
    off-policy loop, iterations for the on-policy loop); ``tag_prefix`` names
    the directories (``iteration_7``) and stays off the retention arithmetic.
    A transient filesystem outage costs one checkpoint, not the run: a failed
    write is retried once after a backoff, then skipped. Each checkpoint is
    written to ``<dir>.tmp`` and renamed into place, so a failed write never
    leaves a partial checkpoint under a real name. ``algorithm`` (the registry
    name, such as ``ppo`` or ``dsrl_sac``) goes into every manifest, so an
    evaluator can tell which trainer wrote a checkpoint.
    """

    def __init__(
        self,
        save_dir: str,
        *,
        retention: str = "all",
        tag_prefix: str = "",
        artifact_logger: Logger | None = None,
        algorithm: str | None = None,
    ) -> None:
        self.save_dir = os.path.abspath(save_dir)
        self.algorithm = algorithm
        self.retention = validate_checkpoint_retention(retention)
        self.tag_prefix = tag_prefix
        self.artifact_logger = artifact_logger
        os.makedirs(self.save_dir, exist_ok=True)

    def path(self, tag: int | str) -> str:
        """The directory a checkpoint with ``tag`` is written to.

        Numeric tags take the prefix (``iteration_40``); named checkpoints such
        as ``final`` or ``interrupted`` keep their name.
        """
        name = f"{self.tag_prefix}{tag}" if isinstance(tag, int) else str(tag)
        return os.path.join(self.save_dir, name)

    def _write(
        self,
        directory: str,
        tag: str,
        learner_group: LearnerGroup | None,
        policy_state: PolicyState,
        metadata: Mapping[str, Any],
    ) -> None:
        shutil.rmtree(directory, ignore_errors=True)  # a failed attempt's leftovers
        os.makedirs(directory)
        if learner_group is not None:
            learner_group.save_checkpoint(directory)
        save_policy_snapshot(policy_state, os.path.join(directory, SNAPSHOT_FILE_NAME))
        # The learner group may have written its own manifest.json (the distributed learner
        # records schema, checkpoint id, world size and step there and refuses to resume
        # without it); add the run's fields to it instead of replacing the file.
        manifest_path = os.path.join(directory, "manifest.json")
        manifest: dict[str, Any] = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
        manifest.update({"tag": tag, "policy_version": policy_state.version, **metadata})
        if self.algorithm is not None:
            manifest["algorithm"] = self.algorithm
        temporary = manifest_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        os.replace(temporary, manifest_path)

    def save(
        self,
        tag: int | str,
        *,
        learner_group: LearnerGroup | None,
        policy_state: PolicyState,
        metadata: Mapping[str, Any],
    ) -> bool:
        """Write one checkpoint; returns whether it landed on disk.

        ``learner_group=None`` writes the policy alone (the offline loop keeps no learner state).
        """
        directory = self.path(tag)
        staging = directory + ".tmp"
        try:
            self._write(staging, str(tag), learner_group, policy_state, metadata)
        except ActorFailedError:
            raise  # a dead learner process does not come back after a pause
        except (OSError, RuntimeError) as exc:
            logger.warning(
                f"checkpoint save failed at tag {tag} ({exc}); retrying in "
                f"{_CHECKPOINT_RETRY_SECONDS:.0f}s"
            )
            time.sleep(_CHECKPOINT_RETRY_SECONDS)
            try:
                self._write(staging, str(tag), learner_group, policy_state, metadata)
            except (OSError, RuntimeError) as retry_exc:
                logger.error(f"checkpoint save failed again ({retry_exc}); skipping tag {tag}")
                shutil.rmtree(staging, ignore_errors=True)
                return False
        # A named tag (final, interrupted) written before stays whole until the new one is in place.
        previous = directory + ".old"
        if os.path.isdir(directory):
            shutil.rmtree(previous, ignore_errors=True)
            os.replace(directory, previous)
        try:
            os.replace(staging, directory)
        except OSError:
            if os.path.isdir(previous):
                os.replace(previous, directory)
            raise
        shutil.rmtree(previous, ignore_errors=True)
        if self.artifact_logger is not None:
            aliases = ["latest", f"rollout-{tag}" if isinstance(tag, int) else str(tag)]
            try:
                # The logger names the artifact <run>-<tag> from the directory.
                self.artifact_logger.log_artifact(
                    path=directory,
                    type="checkpoint",
                    metadata=dict(metadata),
                    aliases=aliases,
                )
            # An upload failure must not end the run.
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"checkpoint artifact upload failed for tag {tag}: {exc}")
        return True

    def saved_tags(self) -> list[int]:
        """Numeric checkpoint tags currently on disk, ascending."""
        tags = []
        for entry in os.listdir(self.save_dir):
            digits = entry[len(self.tag_prefix) :]
            if entry.startswith(self.tag_prefix) and digits.isdigit():
                tags.append(int(digits))
        return sorted(tags)

    def write_evaluation_summary(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Write ``evaluation_summary.json``: the latest evaluation and the full history.

        Launchers and reports read the result from this file instead of
        parsing log lines.
        """
        summary = {"latest": records[-1], "history": list(records)}
        path = os.path.join(self.save_dir, EVALUATION_SUMMARY_FILE)
        with open(path + ".tmp", "w") as stream:
            json.dump(summary, stream, indent=2)
        os.replace(path + ".tmp", path)

    def apply_retention(self, eval_history: Mapping[int, float]) -> None:
        """Prune saved checkpoints to {nearest-to-best-eval, last, final} when configured.

        The peak-selection methodology depends on the best-evaluating weights
        surviving, so the default keeps everything (``utils/checkpoint_retention.py``).
        """
        if self.retention != "best_last_final":
            return
        numeric = self.saved_tags()
        protected = select_protected_checkpoints(numeric, dict(eval_history))
        for tag in numeric:
            if tag in protected:
                continue
            path = self.path(tag)
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"checkpoint retention: pruned {path}")


_aborted_lock = threading.Lock()


def save_aborted_checkpoint(trainer: Any, error: BaseException) -> None:
    """Write the trainer's learner and last published weights as ``aborted`` after ``error``; best effort.

    A failure to save (for example a dead learner process) becomes a note on
    ``error`` so the original error stays the one raised. A trainer's aborted
    checkpoint is written once (a frozen rollout may reach here twice).
    """
    with _aborted_lock:
        if getattr(trainer, "_aborted_checkpoint_written", False):
            return
        trainer._aborted_checkpoint_written = True
    try:
        trainer.save_checkpoint(ABORTED_CHECKPOINT_TAG)
    except Exception as exc:  # noqa: BLE001 - the run's own error is the one to raise
        add_note(error, f"the {ABORTED_CHECKPOINT_TAG!r} checkpoint could not be written: {exc!r}")
    else:
        logger.warning(f"the run failed; wrote the {ABORTED_CHECKPOINT_TAG!r} checkpoint")
