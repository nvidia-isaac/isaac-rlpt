# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.5's supervised loss for co-training, reduced per sample.

The head's own training forward computes this loss and then reduces it to a
single scalar over the whole batch. Co-training needs ``[B]`` instead, so a
gradient slice is weighted by the demonstrations it actually scored, and so a
slice of a different size is not silently weighted the same. That difference
is the reason the adapter reimplements the recipe rather than calling the
head's forward, and it is what these pin.

The real 3B model is not needed: the loss reaches through the head's
submodules, so stubs standing in for them state what the arithmetic must be.
Capturing the noised trajectory the encoder receives lets the test recover the
noise the loss sampled, and so predict the expected value exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from isaac_rl_posttraining.policies.gr00t import flow_sde as n15_flow_sde
from isaac_rl_posttraining.policies.gr00t.flow_sde import Gr00tFlowSDEActorCritic

# A padded grid whose embodiment occupies part of it, as the real heads emit.
BATCH, HORIZON, ACTION_DIM = 3, 4, 6
VALID_ACTION_DIM = 2
TIME = 0.25
BUCKETS = 1000


class _Head:
    """The action-head submodules the supervised loss reaches through."""

    def __init__(self, prediction: torch.Tensor) -> None:
        self.num_timestep_buckets = BUCKETS
        self.config = SimpleNamespace(add_pos_embed=False, noise_s=0.999)
        self.beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.bfloat16),
            torch.tensor(1.0, dtype=torch.bfloat16),
        )
        self.future_tokens = SimpleNamespace(weight=torch.zeros(1, 2))
        self._prediction = prediction
        self.noisy: torch.Tensor | None = None

    def process_backbone_output(self, raw):
        return raw

    def state_encoder(self, state, embodiment_id):
        return torch.zeros(state.shape[0], 1, 2)

    def action_encoder(self, noisy, buckets, embodiment_id):
        self.noisy = noisy.clone()
        return torch.zeros(noisy.shape[0], noisy.shape[1], 2)

    def model(
        self, *, hidden_states, encoder_hidden_states, encoder_attention_mask, timestep
    ):
        return hidden_states

    def action_decoder(self, hidden, embodiment_id):
        return self._prediction


def _fixed_time(monkeypatch) -> None:
    """Pin the sampled flow time so the expected loss is computable."""
    monkeypatch.setattr(
        n15_flow_sde,
        "_sample_flow_time",
        lambda head, samples, device: torch.full((samples,), TIME, device=device),
    )


def _policy(prediction: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor):
    """The adapter with only what :meth:`supervised_loss` touches."""
    policy = object.__new__(Gr00tFlowSDEActorCritic)
    head = _Head(prediction)
    policy.model = SimpleNamespace(action_head=head)
    action_inputs = SimpleNamespace(
        embodiment_id=torch.zeros(BATCH, dtype=torch.long),
        state=torch.zeros(BATCH, 1, 2),
        action=actions,
        action_mask=action_mask,
    )
    raw = {
        "backbone_features": torch.zeros(BATCH, 1, 2),
        "backbone_attention_mask": torch.ones(BATCH, 1, dtype=torch.bool),
    }
    policy._backbone = lambda batch: (raw, action_inputs)
    return policy, head


def _expected(head, actions, prediction, mask):
    """The masked per-sample mean, recomputed from the noise the loss drew."""
    noise = (head.noisy - TIME * actions) / (1 - TIME)
    velocity = actions - noise
    error = (prediction - velocity).square() * mask
    return error.reshape(BATCH, -1).sum(dim=1) / mask.reshape(BATCH, -1).sum(
        dim=1
    ).clamp_min(1.0)


def _mask(valid_dim: int = VALID_ACTION_DIM) -> torch.Tensor:
    mask = torch.zeros(BATCH, HORIZON, ACTION_DIM)
    mask[:, :, :valid_dim] = 1.0
    return mask


def test_the_loss_is_one_value_per_demonstration(monkeypatch):
    """CoTrainer.reduce rejects a scalar; the head's own forward returns one."""
    _fixed_time(monkeypatch)
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
    prediction = torch.randn(BATCH, HORIZON, ACTION_DIM)
    policy, head = _policy(prediction, _mask(), actions)

    loss = policy.supervised_loss({})

    assert loss.shape == (BATCH,)
    torch.testing.assert_close(loss, _expected(head, actions, prediction, _mask()))


def test_the_loss_averages_only_over_the_region_the_embodiment_uses(monkeypatch):
    """Averaging the padding too would shrink the co-training weight silently.

    The same prediction and target scored against a wider mask must give a
    different number; if the reduction ignored the mask the two would agree.
    """
    _fixed_time(monkeypatch)
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
    prediction = torch.randn(BATCH, HORIZON, ACTION_DIM)

    torch.manual_seed(0)
    narrow, _ = _policy(prediction, _mask(VALID_ACTION_DIM), actions)
    narrow_loss = narrow.supervised_loss({})
    torch.manual_seed(0)
    wide, _ = _policy(prediction, _mask(ACTION_DIM), actions)
    wide_loss = wide.supervised_loss({})

    assert not torch.allclose(narrow_loss, wide_loss)


def test_a_fully_masked_demonstration_does_not_divide_by_zero(monkeypatch):
    """An embodiment occupying nothing must not make the whole slice NaN."""
    _fixed_time(monkeypatch)
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
    prediction = torch.randn(BATCH, HORIZON, ACTION_DIM)
    mask = _mask()
    mask[1] = 0.0
    policy, _ = _policy(prediction, mask, actions)

    loss = policy.supervised_loss({})

    assert torch.isfinite(loss).all() and float(loss[1]) == pytest.approx(0.0)


def test_the_family_declares_that_it_scores_demonstrations():
    """The capability gate reads this flag and nothing else."""
    assert Gr00tFlowSDEActorCritic.scores_demonstrations is True


def test_the_flow_time_is_drawn_even_from_a_bfloat16_head():
    """A bf16 checkpoint's Beta cannot be sampled: dirichlet has no bf16 kernel.

    Nothing else in the adapter samples a time -- the flow-SDE path derives its
    timestep from the Euler step index -- so this path is the only one that
    meets the checkpoint's own distribution, and it must not inherit its dtype.
    """
    head = _Head(torch.zeros(BATCH, HORIZON, ACTION_DIM))
    assert head.beta_dist.concentration1.dtype == torch.bfloat16

    time = n15_flow_sde._sample_flow_time(head, BATCH, torch.device("cpu"))

    assert time.shape == (BATCH,) and time.dtype == torch.float32
    assert torch.isfinite(time).all()
