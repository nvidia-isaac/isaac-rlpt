# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Residual Flow Steering."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from isaac_rl_posttraining.algorithms.sac import SACConfig


__all__ = ["RFSConfig", "RFSModulationConfig"]


@dataclass
class RFSConfig(SACConfig):
    """SAC configuration whose concrete learner is identified as RFS."""

    @property
    def algorithm_class(self) -> type:
        from isaac_rl_posttraining.algorithms.rfs.rfs import RFS

        return RFS


@dataclass
class RFSModulationConfig:
    """Geometry of the additive correction applied after frozen-policy decode.

    Attributes:
        residual_action_bound: Symmetric raw-action bound for each residual
            dimension.
            The actor emits normalized values internally, which are mapped to
            this range before the residual is added to the decoded action.
        initial_residual_log_std: Initial log standard deviation of the
            normalized residual branch.
        residual_exploration_std: Fixed normalized standard deviation used
            to perturb residuals during stochastic action selection.
            Evaluation continues to use the deterministic residual mean.
        proposal_conditioned_residual: Whether the residual also consumes the
            current sampled flow proposal through a zero-initialized linear
            conditioner.
        reference_action_conditioning: Whether actor and critic observations
            include GR00T's native decoded action chunk for the current image.
            This is deployable policy output, not simulator state.
        reference_action_key: Observation key used for the decoded chunk.
    """

    residual_action_bound: float = field(
        default=0.01,
        metadata={"help": "Symmetric bound for raw additive action corrections"},
    )
    initial_residual_log_std: float = field(
        default=-5.0,
        metadata={"help": "Initial normalized residual log standard deviation"},
    )
    residual_exploration_std: float = field(
        default=0.0,
        metadata={"help": "Fixed normalized residual exploration standard deviation"},
    )
    proposal_conditioned_residual: bool = field(
        default=False,
        metadata={"help": "Condition residual corrections on sampled flow proposals"},
    )
    reference_action_conditioning: bool = field(
        default=False,
        metadata={"help": "Condition recurrent RFS on GR00T's decoded action proposal"},
    )
    reference_action_key: str = field(
        default="reference_action",
        metadata={"help": "Observation key for GR00T's decoded action proposal"},
    )

    def __post_init__(self) -> None:
        if self.residual_action_bound <= 0:
            raise ValueError(
                f"rfs.residual_action_bound must be positive, got {self.residual_action_bound}"
            )
        if not math.isfinite(self.initial_residual_log_std):
            raise ValueError(
                f"rfs.initial_residual_log_std must be finite, got {self.initial_residual_log_std}"
            )
        if (
            not math.isfinite(self.residual_exploration_std)
            or self.residual_exploration_std < 0.0
        ):
            raise ValueError(
                "rfs.residual_exploration_std must be finite and non-negative, "
                f"got {self.residual_exploration_std}"
            )
        if not self.reference_action_key:
            raise ValueError("rfs.reference_action_key must not be empty")
