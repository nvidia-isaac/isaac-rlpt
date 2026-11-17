# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print which trainer wrote a checkpoint directory, so eval.sh scores it the right way.

The algorithm's registry name (``ppo``, ``dsrl_ppo``, ``dsrl_sac``, ``sft``, ...) when the
checkpoint records it: in its ``manifest.json``, or, for checkpoints from before the
manifest carried it, in the on-policy run's ``<run>/ppo_config.yaml`` two levels above
``<run>/checkpoints/<tag>``. Otherwise a guess from the layout: ``ppo`` for an on-policy
checkpoint copied out of its run (learner state and an ``iteration`` in its manifest; G1's
on-policy recipe), ``offline`` for a policy without learner state (SFT, IQL), and
``other`` for anything else, scored as a DSRL checkpoint.

    python checkpoint_kind.py <checkpoint dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def checkpoint_kind(directory: Path) -> str:
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if "algorithm" in manifest:
        return str(manifest["algorithm"])
    config = directory.resolve().parent.parent / "ppo_config.yaml"
    if config.is_file():
        return str(
            yaml.safe_load(config.read_text(encoding="utf-8"))["algorithm"]["name"]
        )
    has_learner = (directory / "learner.pt").is_file()
    if has_learner and "iteration" in manifest:
        return "ppo"
    if (directory / "policy_state.pt").is_file() and not has_learner:
        return "offline"
    return "other"


if __name__ == "__main__":
    print(checkpoint_kind(Path(sys.argv[1])))
