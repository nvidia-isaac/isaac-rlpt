# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The algorithm config group is the one selector: each option names itself; the registry declares the rest."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaac_rl_posttraining.algorithms.registry import (
    algorithm_spec,
    available_algorithms,
)
from isaac_rl_posttraining.configs import register_configs, selected_algorithm_name
from isaac_rl_posttraining.runners.build import algorithm_family

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
GROUP_OPTIONS = sorted(p.stem for p in (CONFIG_DIR / "algorithm").glob("*.yaml"))


def _compose(*overrides: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=["base_model=mock", *overrides])


@pytest.mark.parametrize("algorithm", GROUP_OPTIONS)
def test_group_file_names_itself_and_the_registry_declares_its_family(algorithm):
    cfg = _compose(f"algorithm={algorithm}")
    assert cfg.algorithm.name == algorithm, (
        "configs/algorithm/<name>.yaml must declare name: <name>"
    )
    assert selected_algorithm_name(cfg) == algorithm
    assert algorithm_family(cfg) == algorithm_spec(algorithm).family
    lines = (CONFIG_DIR / "algorithm" / f"{algorithm}.yaml").read_text().splitlines()
    assert not any(line.strip().startswith("family:") for line in lines), (
        "the registry declares the family; the yaml leaves it out"
    )


def test_every_option_is_registered_and_every_registered_algorithm_has_an_option():
    assert set(GROUP_OPTIONS) == set(available_algorithms())


def test_group_option_values_are_struct_checked():
    with pytest.raises(Exception, match="algorithm.no_such_field"):
        _compose("algorithm=rfs", "algorithm.no_such_field=1")


def test_missing_algorithm_group_fails_loud():
    cfg = _compose("~algorithm")
    assert OmegaConf.select(cfg, "algorithm") is None
    with pytest.raises(ValueError, match="algorithm.name is not set"):
        selected_algorithm_name(cfg)
    with pytest.raises(ValueError, match="algorithm.name is not set"):
        algorithm_family(cfg)


def test_sac_edit_needs_no_chunk_override():
    """SAC-edit edits whole decoded chunks, so its option clears the steering chunk itself."""
    assert _compose("algorithm=sac_edit").training.chunk_size is None
    assert _compose("algorithm=dsrl_sac").training.chunk_size == 40
