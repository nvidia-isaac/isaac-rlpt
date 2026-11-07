# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstration co-training: a supervised anchor inside the RL update.

Online RL in simulation drifts. A policy that raises its simulated return by
exploiting simulator-specific behaviour has moved off the distribution its
demonstrations came from, and that is exactly the distribution a real robot
was taught on -- so simulated reward can climb while transfer degrades.

Co-training bounds that drift by adding a supervised term over the *original
demonstrations* to every RL update:

    L = L_PPO + beta * L_BC

The reference distribution is external and fixed. That is the distinction
from self-imitation, which draws its supervised batch from the agent's own
successful episodes: if a policy drifts into a simulator-specific strategy
that succeeds in simulation, those successes enter the buffer and the
supervised term reinforces the drift rather than opposing it.

The supervised loss belongs to the model, not to this module -- a policy
family knows how its own actions are scored (flow matching, diffusion,
Gaussian likelihood). A model opts in by accepting a ``supervised_batch``
keyword in its ``__call__``, returning the per-sample supervised loss as a
fourth output, and setting ``scores_demonstrations``;
:func:`supports_cotraining` reads that flag.

It must be the model *call*, not a separate method. FSDP installs its
parameter all-gather on the module's forward, so a method invoked directly on
a sharded module runs against that rank's flat shard -- silently wrong, or a
size mismatch. Routing through ``__call__`` also lets a model concatenate the
RL and supervised batches into one backbone pass instead of two.

The per-sample loss must already account for padding. Embodiment-agnostic
heads emit a fixed ``[B, max_horizon, max_action_dim]`` grid and mask the
part a given embodiment actually uses, so averaging over the whole grid
divides by the padding rather than by the real elements. Measured on a GR00T
N1.7 checkpoint whose embodiment occupies 448 of 5280 slots, the unmasked
mean is 11.8x smaller than the masked one -- which would silently scale
``beta`` down by the same factor, with nothing in the metrics to show it. A
model reports the masked mean; :meth:`CoTrainer.reduce` averages across the
batch and nothing more.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

SUPERVISED_BATCH_KEY = "supervised_batch"
"""Keyword by which a demonstration batch enters the model call."""

__all__ = [
    "SUPERVISED_BATCH_KEY",
    "CoTrainer",
    "CoTrainingUnsupported",
    "supports_cotraining",
]


class CoTrainingUnsupported(TypeError):
    """Raised when co-training is enabled for a model that cannot score actions."""


def supports_cotraining(model: Any) -> bool:
    """Whether ``model`` declares that it can score demonstrations.

    The contract is ``model(observations, actions, supervised_batch=...)``
    returning ``(log_prob, entropy, values, per_sample_supervised_loss)``, and
    a family opts in by setting ``scores_demonstrations``. The flag is the
    whole test rather than a fallback for models that lack it: the keyword
    lives on a shared base class, so reading the call signature would advertise
    the capability for every subclass, including the ones with no supervised
    loss to give. It would also read whatever the learner wrapped the policy in
    -- ``FullyShardedDataParallel.forward`` is ``forward(*args, **kwargs)``,
    which accepts every keyword on behalf of a model that may accept none.
    """
    return bool(getattr(model, "scores_demonstrations", False))


def _require_cotrainable(model: Any) -> None:
    """Raise unless ``model`` can score demonstrations in its forward call."""
    if not supports_cotraining(model):
        raise CoTrainingUnsupported(
            f"{type(model).__name__} does not set 'scores_demonstrations', so "
            "it cannot be co-trained on demonstrations. A policy family opts in "
            f"by setting that flag, accepting {SUPERVISED_BATCH_KEY!r} in its "
            "forward and returning its per-sample supervised loss as a fourth "
            "output; a separate method cannot be used, because FSDP only "
            "all-gathers parameters around the module's forward."
        )


class CoTrainer:
    """Produces the weighted supervised term added to each RL update.

    Held by the learner and consulted once per gradient slice, so the
    supervised gradients enter the same backward -- and, under a sharded
    learner, the same gradient reduction -- as the RL gradients. Two passes
    would double the collective traffic and let the two terms see different
    parameter versions.

    The method is that of Shi et al., "Beyond Imitation: Reinforcement
    Learning-Based Sim-Real Co-Training for VLA Models", arXiv:2602.12628:
    warm-start on real and simulated demonstrations, then improve by RL in
    simulation while a supervised term over the real data holds the policy
    near the behaviour that deploys.

    Args:
        source: Turns a drawn batch of demonstrations into model inputs.
        beta: Weight on the supervised term; ``0`` disables co-training.
        batch_size: Demonstrations scored per call.
    """

    def __init__(
        self,
        source: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        beta: float,
        batch_size: int = 32,
    ) -> None:
        if beta < 0.0:
            raise ValueError(f"co-training beta must be >= 0, got {beta}")
        if batch_size <= 0:
            raise ValueError(f"co-training batch_size must be > 0, got {batch_size}")
        self.source = source
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.batches_consumed = 0

    @property
    def enabled(self) -> bool:
        """Whether this co-trainer contributes anything to the loss."""
        return self.beta > 0.0

    def check_supported(self, model: Any) -> None:
        """Fail at construction rather than silently training without the anchor.

        Raises:
            CoTrainingUnsupported: If ``model`` cannot score demonstrations.
        """
        if self.enabled:
            _require_cotrainable(model)

    def next_batch(
        self,
        demonstrations: Mapping[str, Any],
        batch_index: int,
        device: torch.device,
    ) -> Mapping[str, Any]:
        """The ``batch_index``-th co-training batch of the step, as model inputs.

        The controller shipped this rank's whole step in ``demonstrations``, so
        a batch indexes into it rather than reading the dataset, and encoding
        happens here, on the rank that scores it.
        """
        from isaac_rl_posttraining.training.demonstration_buffer import (
            demonstration_slice,
        )

        start = batch_index * self.batch_size
        available = len(demonstrations["action"])
        if start + self.batch_size > available:
            raise ValueError(
                f"co-training batch {batch_index} needs demonstrations "
                f"[{start}, {start + self.batch_size}) but the controller shipped "
                f"{available}; the rank is drawing more batches than it was sent"
            )
        drawn = demonstration_slice(demonstrations, start, self.batch_size)
        encoded = self.source(drawn)
        self.batches_consumed += 1
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }

    @staticmethod
    def reduce(per_sample: torch.Tensor) -> torch.Tensor:
        """Reduce a model's per-sample supervised loss to one scalar.

        Raises:
            ValueError: If the model returned a scalar; per-sample losses keep
                batch-size weighting honest when batches differ in size.
        """
        if per_sample.ndim == 0:
            raise ValueError(
                "the supervised loss must be per-sample, shaped [B]; got a scalar"
            )
        return per_sample.float().mean()
