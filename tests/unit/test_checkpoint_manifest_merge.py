# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint store must not clobber the learner group's manifest.json.

The distributed learner writes ``manifest.json`` (schema version, checkpoint id, world
size, module and optimizer names, learner step) into the checkpoint directory and its
``load_checkpoint`` preflight rejects a directory whose manifest lacks those fields. The
store adds the run's own fields (tag, policy version, metadata) to that file.
"""

import json
import os

import numpy as np

from isaac_rl_posttraining.runtimes.base import PolicyState
from isaac_rl_posttraining.runtimes.learner import LearnerGroup
from isaac_rl_posttraining.training.checkpoints import CheckpointStore

LEARNER_MANIFEST = {
    "schema_version": 1,
    "checkpoint_id": "deadbeef",
    "world_size": 4,
    "module_names": ["actor_critic"],
    "optimizer_names": ["actor_critic"],
    "learner_step": 1260,
}


class _LearnerGroupWritingAManifest(LearnerGroup):
    def save_checkpoint(self, directory: str) -> None:
        with open(
            os.path.join(directory, "manifest.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(LEARNER_MANIFEST, stream)


class _SilentLearnerGroup(LearnerGroup):
    def save_checkpoint(self, directory: str) -> None:
        # Writes nothing, so the store must write the manifest itself.
        pass


def _state() -> PolicyState:
    return PolicyState(version=421, tensors={"x": np.zeros(2, dtype=np.float32)})


def test_store_adds_its_fields_to_the_learner_manifest(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path))
    store.save(
        420,
        learner_group=_LearnerGroupWritingAManifest(),
        policy_state=_state(),
        metadata={"iteration": 420},
    )
    with open(
        os.path.join(store.path(420), "manifest.json"), encoding="utf-8"
    ) as stream:
        manifest = json.load(stream)
    assert {key: manifest[key] for key in LEARNER_MANIFEST} == LEARNER_MANIFEST
    assert (manifest["tag"], manifest["policy_version"], manifest["iteration"]) == (
        "420",
        421,
        420,
    )
    assert not os.path.exists(os.path.join(store.path(420), "manifest.json.tmp"))


def test_store_writes_its_manifest_when_the_learner_wrote_none(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path))
    store.save(
        "final",
        learner_group=_SilentLearnerGroup(),
        policy_state=_state(),
        metadata={"iteration": 500},
    )
    with open(tmp_path / "final" / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest == {"tag": "final", "policy_version": 421, "iteration": 500}


def test_store_records_the_algorithm_that_wrote_the_checkpoint(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), algorithm="ppo")
    store.save(
        "final",
        learner_group=_LearnerGroupWritingAManifest(),
        policy_state=_state(),
        metadata={"iteration": 500},
    )
    with open(tmp_path / "final" / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["algorithm"] == "ppo"
    assert manifest["learner_step"] == LEARNER_MANIFEST["learner_step"]
