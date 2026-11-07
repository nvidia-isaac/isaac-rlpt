# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra schemas are matched explicitly (base_<name> entries), never by the deprecated same-name path."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from isaac_rl_posttraining.configs import register_configs

CONFIG_DIR = Path(__file__).resolve().parents[2] / "isaac_rl_posttraining" / "configs"
SAME_NAME_WARNING = "validated against ConfigStore schema with the same name"


def _compose(*overrides: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("base_model=mock", "algorithm=rfs", "runtime=rlinf"),
        ("algorithm=ppo", "base_model.ckpt=/ckpt/x"),
        ("base_model=mock", "algorithm=dsrl_ppo"),
        ("algorithm=iql", "base_model=gr00t_n17"),
        ("algorithm=sft", "base_model=pi0"),
        ("base_model=dreamzero_local",),
        ("base_model=dreamzero_server",),
    ],
)
def test_composition_emits_no_same_name_schema_warning(overrides):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _compose(*overrides)
    offending = [str(w.message) for w in caught if SAME_NAME_WARNING in str(w.message)]
    assert not offending, offending
    assert cfg.base_model.name and cfg.runtime.name


def test_group_schemas_still_validate_keys():
    """The explicit schema match keeps struct validation: a typo in a group override fails at compose."""
    with pytest.raises(ConfigCompositionException):
        _compose("base_model.no_such_field=1")
    with pytest.raises(ConfigCompositionException):
        _compose("runtime.rlinf.no_such_field=1")
    with pytest.raises(ConfigCompositionException):
        _compose("algorithm=ppo", "base_model.ckpt=/ckpt/x", "ppo.no_such_field=1")
    with pytest.raises(ConfigCompositionException):
        _compose("algorithm=ppo", "base_model.ckpt=/ckpt/x", "ppo.iterations=many")
    with pytest.raises(ConfigCompositionException):
        _compose("algorithm=dsrl_ppo", "algorithm.no_such_field=1")
