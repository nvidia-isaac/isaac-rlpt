# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run that never evaluates builds no eval environment.

``eval.eval_freq: 0`` with ``eval.eval_on_first_step: false`` is the
collect-only configuration (e.g. a throughput measurement or a collector in
a fan-out); it must not require an eval simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from isaac_rl_posttraining.configs import evaluation_enabled
from isaac_rl_posttraining.runners.off_policy_launch import (
    validate_evaluation_for_paradigm,
)
from isaac_rl_posttraining.utils import env_utils

CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ((), True),
        (("eval.eval_freq=0",), True),  # the canonical config still evaluates at step 0
        (("eval.eval_freq=0", "eval.eval_on_first_step=false"), False),
        (("eval.eval_freq=4000", "eval.eval_on_first_step=false"), True),
    ],
)
def test_evaluation_enabled(overrides, expected):
    assert evaluation_enabled(_compose(*overrides)) is expected


def test_make_env_without_eval_env_builds_only_the_train_env(monkeypatch):
    calls: list[dict] = []

    def fake_make_env(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(env_utils, "_make_env", fake_make_env)
    _train_env, eval_env = env_utils.make_env(
        env_name="dummy", num_envs=4, with_eval_env=False
    )

    assert eval_env is None
    assert len(calls) == 1 and calls[0]["num_envs"] == 4, (
        "only the vectorized train env may be built"
    )


def test_direct_action_paradigm_rejects_a_run_without_evaluation():
    collect_only = ("eval.eval_freq=0", "eval.eval_on_first_step=false")
    # The direct-action paradigm wraps the eval env for its reference actions.
    cfg = _compose("algorithm=sac_edit", *collect_only)
    with pytest.raises(ValueError, match="direct-action"):
        validate_evaluation_for_paradigm(cfg)
    # Latent steering and the default (evaluating) direct-action config pass.
    validate_evaluation_for_paradigm(_compose(*collect_only))
    validate_evaluation_for_paradigm(_compose("algorithm=sac_edit"))
