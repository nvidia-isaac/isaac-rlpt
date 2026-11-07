# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T's supervised action loss, reduced over a padded action grid.

The action head emits a fixed ``[B, max_horizon, max_action_dim]`` grid and
reports the region an embodiment occupies in ``action_mask``. Reducing over the
whole grid divides by the padding too, so the loss reads low by the padding
ratio -- and because the error is a clean scale factor, a coefficient applied
to this loss is quietly shrunk by the same amount with nothing to show for it.

These pin the reduction without the real 3B model: ``actor_bc_loss`` only
reaches through ``self.base_model``, so a stub head standing in for it is
enough to state what the arithmetic must be.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isaac_rl_posttraining.policies.gr00t.model import Gr00tModel

# A 28-DoF embodiment with a 16-step chunk inside GR00T's 40x132 grid: the
# real trocar geometry, and the case the dilution was measured on.
HORIZON, ACTION_DIM = 40, 132
VALID_HORIZON, VALID_ACTION_DIM = 16, 28


class _StubBaseModel:
    """Stands in for the GR00T base model, returning a known loss and mask."""

    def __init__(self, action_loss: torch.Tensor, action_mask: torch.Tensor) -> None:
        self._out = {"action_loss": action_loss, "action_mask": action_mask}
        self.action_head = self
        self.backbone = lambda _inputs: None

    def prepare_input(self, inputs):
        return inputs, inputs

    def forward(self, backbone_output, action_input):
        return self._out


def _model(action_loss: torch.Tensor, action_mask: torch.Tensor) -> Gr00tModel:
    """A Gr00tModel shell whose only live part is the stubbed base model."""
    model = Gr00tModel.__new__(Gr00tModel)
    model.base_model = _StubBaseModel(action_loss, action_mask)
    return model


def _padded_mask(samples: int) -> torch.Tensor:
    mask = torch.zeros(samples, HORIZON, ACTION_DIM)
    mask[:, :VALID_HORIZON, :VALID_ACTION_DIM] = 1.0
    return mask


def test_the_loss_averages_only_over_the_region_the_embodiment_uses():
    """A constant loss on the valid region must reduce to that constant.

    Averaging the full grid would return it scaled by the padding ratio
    instead, which is the whole bug: the value still looks plausible.
    """
    mask = _padded_mask(2)
    action_loss = torch.full((2, HORIZON, ACTION_DIM), 7.0) * mask

    result = _model(action_loss, mask).actor_bc_loss({})

    torch.testing.assert_close(result, torch.full((2,), 7.0))


def test_each_sample_is_reduced_independently():
    """Per-sample output keeps batch weighting honest for the caller."""
    mask = _padded_mask(2)
    action_loss = torch.zeros(2, HORIZON, ACTION_DIM)
    action_loss[0, :VALID_HORIZON, :VALID_ACTION_DIM] = 2.0
    action_loss[1, :VALID_HORIZON, :VALID_ACTION_DIM] = 5.0

    result = _model(action_loss, mask).actor_bc_loss({})

    torch.testing.assert_close(result, torch.tensor([2.0, 5.0]))


def test_a_fully_masked_sample_does_not_divide_by_zero():
    """An empty mask must yield a finite zero rather than a NaN."""
    mask = torch.zeros(1, HORIZON, ACTION_DIM)
    action_loss = torch.full((1, HORIZON, ACTION_DIM), 3.0)

    result = _model(action_loss, mask).actor_bc_loss({})

    assert torch.isfinite(result).all() and float(result) == pytest.approx(0.0)


def test_a_boolean_mask_reduces_the_same_as_a_float_one():
    """The head may report the mask as bool; the arithmetic must not change."""
    mask = _padded_mask(1)
    action_loss = torch.full((1, HORIZON, ACTION_DIM), 4.0) * mask

    as_float = _model(action_loss, mask).actor_bc_loss({})
    as_bool = _model(action_loss, mask.bool()).actor_bc_loss({})

    torch.testing.assert_close(as_float, as_bool)
