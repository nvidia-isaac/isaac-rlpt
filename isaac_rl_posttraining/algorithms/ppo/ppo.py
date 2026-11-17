# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO as one algorithm object for the on-policy loop.

The controller half turns collected trajectories into learner batches
(semi-Markov GAE, shuffling, one shard per rank); the learner half is the
clipped-update worker every learner rank hosts. Both read the same
:class:`PPOConfig`, so the rank minibatch the batches are cut to and the one
the update consumes cannot drift apart.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import nn

from isaac_rl_posttraining.algorithms.ppo.batches import build_rank_batches
from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig
from isaac_rl_posttraining.algorithms.ppo.learner import PPOLearner
from isaac_rl_posttraining.runtimes.learner import LearnerBatch, LearnerWorkerSpec
from isaac_rl_posttraining.training.cotrain import CoTrainer
from isaac_rl_posttraining.training.trajectories import ChunkTrajectory, DiscountPer

__all__ = ["PPO"]


@dataclass(frozen=True)
class PPO:
    """Proximal policy optimization over action chunks (the ``OnPolicyAlgorithm`` of the loop)."""

    config: PPOConfig = field(default_factory=PPOConfig)
    demonstrations: Any = None

    @property
    def gamma(self) -> float:
        return self.config.gamma

    @property
    def discount_per(self) -> DiscountPer:
        return self.config.discount_per

    def build_batches(
        self,
        trajectories: Sequence[ChunkTrajectory],
        *,
        num_learner_ranks: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[LearnerBatch, ...], Mapping[str, float]]:
        batches, stats = build_rank_batches(
            trajectories,
            num_learner_ranks=num_learner_ranks,
            minibatch_size=self.config.minibatch_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            rng=rng,
            discount_per=self.config.discount_per,
            bootstrap_at_time_limit=self.config.bootstrap_at_time_limit,
        )
        if self.demonstrations is None:
            return batches, stats
        return self._with_demonstrations(batches), stats

    def _with_demonstrations(
        self, batches: Sequence[LearnerBatch]
    ) -> tuple[LearnerBatch, ...]:
        """Add each rank's co-training anchor to its batch.

        Ranks draw different demonstrations, as they do different samples of
        the RL batch, so the all-reduced supervised gradient covers
        ``cotrain_batch_size`` per rank.
        """
        from isaac_rl_posttraining.training.demonstration_buffer import (
            DEMONSTRATIONS_KEY,
            draw_demonstration_tensors,
        )

        carried = []
        for batch in batches:
            count = (
                self.cotrain_batches_per_step(batch.samples)
                * self.config.cotrain_batch_size
            )
            tensors = dict(batch.tensors)
            tensors[DEMONSTRATIONS_KEY] = draw_demonstration_tensors(
                self.demonstrations, count
            )
            carried.append(LearnerBatch(tensors=tensors, samples=batch.samples))
        return tuple(carried)

    def cotrain_batches_per_step(self, samples: int) -> int:
        """Co-training batches one ``train_step`` over ``samples`` draws.

        One per forward/backward pass, hence the loop geometry below. Early
        stopping can cut a step short, so it is an upper bound.
        """
        micro = self.config.micro_batch_size or self.config.minibatch_size
        minibatches = math.ceil(samples / self.config.minibatch_size)
        return (
            self.config.epochs
            * minibatches
            * math.ceil(self.config.minibatch_size / micro)
        )

    def learner(
        self,
        build_model: Callable[[], nn.Module],
        *,
        frozen_submodules: Sequence[str] = (),
        demonstration_source: Any = None,
    ) -> LearnerWorkerSpec:
        """The learner worker each rank builds: the model from ``build_model``, trained by this PPO.

        ``frozen_submodules`` names submodules (dotted paths such as
        ``"model.backbone"``) that never train; the runtime keeps them whole on
        every rank and leaves them out of published policy states and checkpoints.

        ``demonstration_source`` supplies the co-training anchor and is required
        exactly when ``ppo.cotrain_beta > 0``.

        Raises:
            ValueError: If co-training is enabled with no demonstrations, or
                demonstrations are supplied with co-training disabled.
        """
        cotrainer = None
        beta = self.config.cotrain_beta
        if beta > 0.0:
            if demonstration_source is None:
                raise ValueError(
                    f"ppo.cotrain_beta={beta} enables demonstration co-training, but no "
                    "demonstration source was supplied; pass demonstration_source= or "
                    "set ppo.cotrain_beta=0"
                )
            if self.demonstrations is None:
                raise ValueError(
                    f"ppo.cotrain_beta={beta} enables demonstration co-training, but "
                    "this PPO has no demonstrations to draw from; pass "
                    "demonstrations= or set ppo.cotrain_beta=0"
                )
            cotrainer = CoTrainer(
                demonstration_source,
                beta=beta,
                batch_size=self.config.cotrain_batch_size,
            )
        elif demonstration_source is not None or self.demonstrations is not None:
            raise ValueError(
                "demonstrations were supplied but ppo.cotrain_beta is 0, so they "
                "would never be read; set a positive beta or drop them"
            )
        return functools.partial(
            PPOLearner,
            build_model=build_model,
            config=self.config,
            frozen_submodules=tuple(frozen_submodules),
            cotrainer=cotrainer,
        )
