# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint load must restore module weights IN PLACE.

Regression for the random-actor evaluation bug: ``BaseAlgorithm.load`` used
to replace module components via ``setattr``, so any reference captured
before the load — the trainer's eval actor, rollout collectors, judge
re-eval harnesses — kept pointing at the freshly initialized module. Every
``training.load_dir`` evaluation therefore scored a random-init actor
instead of the checkpoint.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from isaac_rl_posttraining.algorithms.base import BaseAlgorithm, BaseAlgorithmConfig


class _TinyAlgorithm(BaseAlgorithm):
    def __init__(self, config: BaseAlgorithmConfig) -> None:
        super().__init__(config)
        self.actor = nn.Linear(4, 3)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-3, weight_decay=0
        )
        self.component_names = ["actor", "actor_optimizer"]

    def train_step(self, rollout_step=None):
        return {}


def _save_per_component(algorithm: BaseAlgorithm, save_dir: str) -> None:
    """Write the per-component layout older runs saved (``<name>.pt`` + ``training_state.json``)."""
    for component in algorithm.component_names:
        torch.save(
            getattr(algorithm, component), os.path.join(save_dir, f"{component}.pt")
        )
    state = {"step": algorithm.step_counter}
    for name in algorithm.training_state_tensors:
        state[name] = float(getattr(algorithm, name))
    with open(os.path.join(save_dir, "training_state.json"), "w") as stream:
        json.dump(state, stream)


def _make_algorithm() -> _TinyAlgorithm:
    config = BaseAlgorithmConfig()
    config.logger = None
    return _TinyAlgorithm(config)


def test_load_restores_weights_through_preexisting_references(tmp_path):
    torch.manual_seed(0)
    source = _make_algorithm()
    with torch.no_grad():
        source.actor.weight.fill_(1.25)
        source.actor.bias.fill_(-0.5)
    _save_per_component(source, str(tmp_path))

    torch.manual_seed(1)
    fresh = _make_algorithm()
    # The trainer captures this reference BEFORE load (as train_dsrl does).
    eval_actor = fresh.actor
    assert not torch.allclose(eval_actor.weight, source.actor.weight)

    fresh.load(str(tmp_path))

    assert fresh.actor is eval_actor, "load must not swap the module object"
    assert torch.allclose(eval_actor.weight, torch.full((3, 4), 1.25))
    assert torch.allclose(eval_actor.bias, torch.full((3,), -0.5))


def test_load_restores_optimizer_state_in_place(tmp_path):
    torch.manual_seed(0)
    source = _make_algorithm()
    loss = source.actor(torch.ones(2, 4)).sum()
    loss.backward()
    source.actor_optimizer.step()
    _save_per_component(source, str(tmp_path))
    saved_state = source.actor_optimizer.state_dict()

    torch.manual_seed(1)
    fresh = _make_algorithm()
    optimizer_ref = fresh.actor_optimizer
    fresh.load(str(tmp_path))

    assert fresh.actor_optimizer is optimizer_ref
    restored = optimizer_ref.state_dict()
    assert restored["param_groups"] == saved_state["param_groups"]
    # Adam moment buffers restored onto the optimizer that owns the live params.
    saved_step = next(iter(saved_state["state"].values()))["step"]
    restored_step = next(iter(restored["state"].values()))["step"]
    assert torch.equal(torch.as_tensor(restored_step), torch.as_tensor(saved_step))


def test_load_restores_plain_tensor_in_place(tmp_path):
    """log_ent_coef-style tensor components restore into the live object.

    Replacing the tensor would orphan it from the optimizer that references
    it, silently freezing temperature tuning on resumed runs.
    """

    class _EntAlgorithm(BaseAlgorithm):
        def __init__(self, config):
            super().__init__(config)
            self.log_ent_coef = torch.zeros(1, requires_grad=True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=1e-2, weight_decay=0
            )
            self.component_names = ["log_ent_coef"]

        def train_step(self, rollout_step=None):
            return {}

    config = BaseAlgorithmConfig()
    config.logger = None

    source = _EntAlgorithm(config)
    with torch.no_grad():
        source.log_ent_coef.fill_(-1.7)
    _save_per_component(source, str(tmp_path))

    fresh = _EntAlgorithm(config)
    tensor_ref = fresh.log_ent_coef
    optimizer_param = fresh.ent_coef_optimizer.param_groups[0]["params"][0]
    fresh.load(str(tmp_path))

    assert fresh.log_ent_coef is tensor_ref, "tensor object must not be replaced"
    assert optimizer_param is fresh.log_ent_coef, (
        "optimizer must still own the live tensor"
    )
    assert torch.allclose(tensor_ref, torch.tensor([-1.7]))
