# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve what a composed config selects: the algorithm, the base model, whether the run evaluates."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

__all__ = ["evaluation_enabled", "selected_algorithm_name", "selected_base_model_name"]


def selected_algorithm_name(cfg: DictConfig) -> str:
    """Return the composed algorithm's registry name.

    The ``algorithm`` config group is the single selector: each
    ``configs/algorithm/<name>.yaml`` declares its ``name`` (and ``family``),
    and the entry point dispatches on them.
    """
    name = OmegaConf.select(cfg, "algorithm.name")
    if not name:
        raise ValueError(
            "algorithm.name is not set; compose the algorithm with algorithm=<name> "
            "(each configs/algorithm/<name>.yaml declares its name)"
        )
    return str(name)


def selected_base_model_name(cfg: DictConfig) -> str:
    """Return the base model name the ``base_model`` config group selects."""
    name = OmegaConf.select(cfg, "base_model.name")
    if not name:
        raise ValueError("base_model.name is not set; compose the 'base model' config group")
    return str(name)


def evaluation_enabled(cfg: DictConfig) -> bool:
    """Whether the run evaluates at all.

    ``eval.eval_freq: 0`` disables periodic evaluation; with no first-step
    evaluation either, the run opens no evaluation workers.
    """
    return cfg.eval.eval_freq != 0 or bool(OmegaConf.select(cfg.eval, "eval_on_first_step", default=False))
