# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstration co-training: source contract, capability gate, and the RL update.

The behaviours pinned here are the ones a silent failure would hide: that the
supervised term actually reaches the gradients, that it rides the RL backward
rather than taking its own, and that enabling it on a model which cannot score
demonstrations fails loudly instead of training without the anchor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.training.cotrain import (
    CoTrainer,
    CoTrainingUnsupported,
    supports_cotraining,
)


class _ScoringModel(nn.Module):
    """Follows the contract: flag set, supervised batch in the call, loss 4th."""

    scores_demonstrations = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.bc_calls = 0

    def forward(  # NOSONAR - mirrors the co-training contract (4th output)
        self, observations=None, actions=None, *, supervised_batch=None
    ):
        logp = self.weight.sum().expand(1)
        if supervised_batch is None:
            return logp, logp, logp
        self.bc_calls += 1
        target = supervised_batch["actions"]
        per_sample = (self.weight.unsqueeze(0) - target).square().mean(dim=1)
        return logp, logp, logp, per_sample


class _MethodOnlyModel(nn.Module):
    """Scores demonstrations only through a separate method -- the wrong route.

    FSDP all-gathers parameters around the module's forward, so this shape
    would run against a rank's flat shard. It must be rejected, which it is by
    never claiming the capability: the flag is the only way to claim it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))

    def forward(self, observations=None, actions=None):
        return (self.weight.sum().expand(1),) * 3

    def actor_bc_loss(self, inputs) -> torch.Tensor:
        return (self.weight.unsqueeze(0) - inputs["actions"]).square().mean(dim=1)


class _UnscoreableModel(nn.Module):
    """A steering-style actor-critic with no supervised path at all."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))

    def forward(self, observations=None, actions=None):
        return (self.weight.sum().expand(1),) * 3


def _passthrough(batch):
    inputs = dict(batch["obs"])
    inputs["action"] = batch["action"]
    return inputs


def _shipment(rows: int, width: int = 4) -> dict:
    return {
        "action": torch.arange(rows * width, dtype=torch.float32).reshape(rows, width),
        "actions": torch.full((rows, width), 1.0),
    }


def _sampler(rows: int = 8):
    class _Sampler:
        def sample(self, size):
            return {
                "obs": {"state": np.zeros((size, 4), np.float32)},
                "action": np.zeros((size, 2, 4), np.float32),
            }

        def can_sample(self, size):
            return True

    return _Sampler()


# ------------------------------------------------------------ capability ----
def test_supports_cotraining_reads_the_declared_flag():
    """Opting in is explicit; nothing is inferred from the model's shape.

    A signature test would answer for the base class that declares
    ``supervised_batch``, and under a sharded learner it would answer for
    ``FullyShardedDataParallel`` instead of the policy at all.
    """
    assert supports_cotraining(_ScoringModel())
    assert not supports_cotraining(_UnscoreableModel())


def test_a_model_scoring_only_through_a_method_is_rejected():
    """The FSDP trap: a separate method never sees all-gathered parameters."""
    assert not supports_cotraining(_MethodOnlyModel())
    cotrainer = CoTrainer(_passthrough, beta=0.1)
    with pytest.raises(CoTrainingUnsupported, match="scores_demonstrations"):
        cotrainer.check_supported(_MethodOnlyModel())


def test_enabling_cotraining_on_an_unscoreable_model_raises():
    cotrainer = CoTrainer(_passthrough, beta=0.1)
    with pytest.raises(CoTrainingUnsupported, match="scores_demonstrations"):
        cotrainer.check_supported(_UnscoreableModel())


def test_disabled_cotraining_tolerates_an_unscoreable_model():
    """beta=0 must cost nothing, including the capability requirement."""
    cotrainer = CoTrainer(_passthrough, beta=0.0)
    cotrainer.check_supported(_UnscoreableModel())  # must not raise
    assert not cotrainer.enabled


def test_negative_beta_is_rejected():
    with pytest.raises(ValueError, match="beta must be >= 0"):
        CoTrainer(_passthrough, beta=-0.1)


# ------------------------------------------------------------------ loss ----
def test_reduce_averages_the_models_per_sample_loss():
    cotrainer = CoTrainer(_passthrough, beta=0.25)
    assert float(cotrainer.reduce(torch.tensor([1.0, 3.0]))) == pytest.approx(2.0)


def test_a_scalar_supervised_loss_is_rejected():
    """Per-sample losses keep batch-size weighting honest across slices."""
    cotrainer = CoTrainer(_passthrough, beta=0.1)
    with pytest.raises(ValueError, match="per-sample"):
        cotrainer.reduce(torch.tensor(1.0))


def test_the_supervised_term_reaches_the_gradients_through_the_model_call():
    """The anchor must move parameters, and must arrive via __call__."""
    model = _ScoringModel()
    cotrainer = CoTrainer(_passthrough, beta=1.0)
    batch = cotrainer.next_batch(_shipment(rows=64), 0, torch.device("cpu"))
    *_, per_sample = model(None, None, supervised_batch=batch)
    (cotrainer.beta * cotrainer.reduce(per_sample)).backward()
    assert model.weight.grad is not None
    # d/dw mean((w - 1)^2) at w=0 is -0.5 per element.
    torch.testing.assert_close(model.weight.grad, torch.full((4,), -0.5))
    assert model.bc_calls == 1


def test_one_backward_covers_both_terms():
    """RL and supervised gradients accumulate from a single backward.

    Two backwards would double a sharded learner's gradient reduction and let
    the terms see different parameter versions.
    """
    model = _ScoringModel()
    cotrainer = CoTrainer(_passthrough, beta=1.0)
    batch = cotrainer.next_batch(_shipment(rows=64), 0, torch.device("cpu"))
    *_, per_sample = model(None, None, supervised_batch=batch)

    rl_loss = (model.weight * 2.0).sum()  # gradient of 2 per weight
    (rl_loss + cotrainer.beta * cotrainer.reduce(per_sample)).backward()

    torch.testing.assert_close(model.weight.grad, torch.full((4,), 2.0 - 0.5))
    assert model.bc_calls == 1  # one scoring pass per slice


def test_a_model_called_without_a_supervised_batch_returns_three_outputs():
    """beta=0 leaves the model's own call arity untouched."""
    model = _ScoringModel()
    assert len(model(None, None)) == 3
    assert model.bc_calls == 0  # demonstrations untouched


def test_batches_consumed_tracks_slices():
    cotrainer = CoTrainer(_passthrough, beta=0.1)
    shipment = _shipment(rows=5 * cotrainer.batch_size)
    for index in range(5):
        cotrainer.next_batch(shipment, index, torch.device("cpu"))
    assert cotrainer.batches_consumed == 5


# --------------------------------------------------- learner-factory gate ----
def _ppo(*, demonstrations=None, **overrides):
    """A PPO algorithm with the given config overrides."""
    from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig
    from isaac_rl_posttraining.algorithms.ppo.ppo import PPO

    return PPO(PPOConfig(**overrides), demonstrations=demonstrations)


def test_enabling_beta_without_demonstrations_is_rejected():
    """Co-training must not silently do nothing for want of a dataset."""
    with pytest.raises(ValueError, match="no demonstration source was supplied"):
        _ppo(cotrain_beta=0.1, demonstrations=_sampler()).learner(_ScoringModel)


def test_demonstrations_without_beta_are_rejected():
    """A source that would never be read is a configuration mistake."""
    with pytest.raises(ValueError, match="would never be read"):
        _ppo(cotrain_beta=0.0, demonstrations=_sampler()).learner(
            _ScoringModel, demonstration_source=_passthrough
        )


def test_learner_factory_carries_the_cotrainer_when_enabled():
    factory = _ppo(
        cotrain_beta=0.25, cotrain_batch_size=8, demonstrations=_sampler()
    ).learner(_ScoringModel, demonstration_source=_passthrough)
    cotrainer = factory.keywords["cotrainer"]
    assert (
        cotrainer.enabled
        and cotrainer.beta == pytest.approx(0.25)
        and cotrainer.batch_size == 8
    )


def test_learner_factory_carries_no_cotrainer_by_default():
    assert _ppo().learner(_ScoringModel).keywords["cotrainer"] is None


def test_both_ppo_paths_share_the_cotraining_surface():
    """`ppo` and `dsrl_ppo` compose one `ppo.*` config, so beta reaches both."""
    from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig

    assert hasattr(PPOConfig(), "cotrain_beta")
    factory = _ppo(cotrain_beta=0.5, demonstrations=_sampler()).learner(
        _ScoringModel, demonstration_source=_passthrough
    )
    assert factory.keywords["cotrainer"].beta == pytest.approx(0.5)


def test_ppo_config_validates_cotraining_fields():
    """Bad YAML should fail at config construction, not mid-run."""
    from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig

    with pytest.raises(ValueError, match="cotrain_beta"):
        PPOConfig(cotrain_beta=-1.0)
    with pytest.raises(ValueError, match="cotrain_batch_size"):
        PPOConfig(cotrain_batch_size=0)


def test_reduce_does_not_second_guess_the_models_masking():
    """The model owns padding; reduce only averages across the batch.

    Embodiment-agnostic heads emit a padded [B, H, A] grid and mask the part
    an embodiment uses. A model that reports the masked mean and one that
    reports the diluted mean differ by the padding ratio -- silently scaling
    beta -- so the contract puts that decision in the model, and reduce must
    pass the per-sample values through untouched.
    """
    masked = torch.tensor([7.43e-4, 5.30e-4])  # measured on a real N1.7 batch
    diluted = masked * (448 / 5280)  # same batch, averaged over padding
    cotrainer = CoTrainer(_passthrough, beta=1.0)
    assert float(cotrainer.reduce(masked)) == pytest.approx(float(masked.mean()))
    assert float(cotrainer.reduce(diluted)) == pytest.approx(float(diluted.mean()))
    # The two differ by the padding ratio -- the reason the model must mask.
    assert float(cotrainer.reduce(masked)) / float(
        cotrainer.reduce(diluted)
    ) == pytest.approx(5280 / 448, rel=1e-3)


# ------------------------------------------------- launcher translation ----
def test_cotraining_settings_survive_the_config_translation():
    """The launcher rebuilds PPOConfig field by field, so a field it forgets is
    silently lost: the source factory and the learner gate both read the same
    zeroed value, agree co-training is off, and neither guard fires. A run then
    validates, starts, and trains with no anchor at all.
    """
    import dataclasses

    from isaac_rl_posttraining.configs.schema import PPOTrainingConfig
    from isaac_rl_posttraining.runners.ppo_launch import ppo_config_from

    training = PPOTrainingConfig(
        cotrain_beta=0.25,
        cotrain_batch_size=8,
        cotrain_dataset_path="/data/demos",
    )
    config = ppo_config_from(training, num_learner_ranks=1, seed=0)

    assert config.cotrain_beta == pytest.approx(0.25)
    assert config.cotrain_batch_size == 8
    assert config.cotrain_dataset_path == "/data/demos"

    # Every co-training field above was set away from its default, so one left
    # at its default is one the translation dropped -- including a field added
    # later, which is how this was missed the first time.
    dropped = [
        field.name
        for field in dataclasses.fields(config)
        if field.name.startswith("cotrain_")
        and getattr(config, field.name) == field.default
    ]
    assert not dropped, f"ppo_config_from does not carry {dropped}"


def test_a_rank_trains_on_the_batch_the_controller_ships():
    """The demonstrations field rides in the batch but is not an RL field.

    It has its own row count, and the rank encodes it rather than the learner
    moving it to the device with the rest.
    """
    from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig
    from isaac_rl_posttraining.algorithms.ppo.learner import PPOLearner
    from isaac_rl_posttraining.runtimes.learner import LearnerBatch, LearnerContext
    from isaac_rl_posttraining.training.demonstration_buffer import DEMONSTRATIONS_KEY

    class _Modules:
        def prepare(self, model, **kwargs):
            return model

        def register_optimizer(self, *args, **kwargs):
            return None

    learner = PPOLearner(
        LearnerContext(learner_rank=0, world_size=1, device="cpu"),
        _Modules(),
        _ScoringModel,
        PPOConfig(
            minibatch_size=1,
            micro_batch_size=1,
            epochs=1,
            cotrain_beta=1.0,
            cotrain_batch_size=2,
        ),
        # The rank encodes the rows it is shipped, as a policy family would.
        cotrainer=CoTrainer(
            lambda drawn: {"actions": torch.as_tensor(drawn["action"])},
            beta=1.0,
            batch_size=2,
        ),
    )
    batch = LearnerBatch(
        tensors={
            "observations": torch.zeros(1, 4),
            "actions": torch.zeros(1, 4),
            "old_log_probs": torch.zeros(1),
            "advantages": torch.zeros(1),
            "returns": torch.zeros(1),
            "old_values": torch.zeros(1),
            # Two demonstration rows for one RL sample.
            DEMONSTRATIONS_KEY: {"action": np.ones((2, 4), np.float32)},
        },
        samples=1,
    )

    metrics = learner.train_step(batch)

    assert learner.model.bc_calls == 1
    assert metrics["bc_loss"] > 0
