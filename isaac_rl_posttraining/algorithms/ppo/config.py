# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for provider-neutral clipped PPO updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from isaac_rl_posttraining.training.trajectories import DiscountPer

__all__ = ["PPOConfig"]


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters of PPO over action chunks.

    The advantage settings (``gamma``, ``gae_lambda``, ``discount_per``,
    ``bootstrap_at_time_limit``) are applied by the controller when it builds
    the batches; the rest drives the update on the learner ranks. Batch sizes
    are rank-local. Distributed callers must give every learner rank the same
    number of samples so FSDP gradient averaging represents the global batch
    without rank-dependent weighting.

    The ``cotrain_*`` settings add a supervised term over the demonstrations
    the policy was taught on, inside the same update as the RL loss, which
    anchors it to a distribution simulated reward alone does not preserve. See
    :mod:`isaac_rl_posttraining.training.cotrain`.

    Attributes:
        gamma: Discount factor; see ``discount_per`` for what one factor spans.
        gae_lambda: GAE trace decay, spanning the same unit as ``gamma``.
        discount_per: ``"simulator_step"`` discounts by the number of simulator
            steps a chunk executed; ``"decision"`` treats every chunk as one
            step (the convention of RLinf's flow-matching recipes).
        bootstrap_at_time_limit: Bootstrap a truncated episode from the value
            of its last observation; off, the time limit ends the episode with
            a zero successor value (RLinf's convention).
        learning_rate: Optimizer step size for every trainable parameter, or
            for the policy parameters when ``value_learning_rate`` is set.
        value_learning_rate: Separate step size for the value parameters
            (those whose name contains ``value_parameter_pattern``). ``None``
            trains everything at ``learning_rate``.
        value_parameter_pattern: Substring that marks value-function parameters
            when ``value_learning_rate`` is set.
        optimizer: ``"adam"`` or ``"adamw"`` (decoupled weight decay).
        adam_betas: Adam moment decay rates.
        optimizer_eps: Adam denominator epsilon.
        weight_decay: Weight decay of the optimizer (``0`` disables it).
        clip_ratio: PPO probability-ratio clip.
        value_clip: Clip on the value change per update, ``None`` for unclipped.
        value_loss: ``"mse"`` (half squared error) or ``"huber"`` (quadratic
            below ``huber_delta``, linear above; robust to return outliers).
        huber_delta: Transition point of the Huber loss.
        value_loss_coef: Weight of the value loss in the total loss.
        entropy_coef: Weight of the entropy bonus.
        epochs: Passes over the batch per update.
        minibatch_size: Rank-local samples per optimizer step.
        micro_batch_size: Samples per forward/backward slice inside a minibatch;
            gradients accumulate across slices so a minibatch can exceed what
            fits in memory. ``None`` runs the minibatch in one slice.
        max_grad_norm: Global gradient-norm clip, ``None`` to disable.
        target_kl: Stop the update early once the mean approximate KL exceeds
            this value, ``None`` to always finish every epoch.
        normalize_advantages: Standardize advantages within each rank-local
            batch (across ranks when distributed).
        seed: Seed of the minibatch shuffling and of the demonstration draw.
        cotrain_beta: Weight of the supervised term over the demonstrations in
            every RL update, ``L = L_PPO + beta * L_BC``. ``0`` disables
            co-training and costs nothing.
        cotrain_batch_size: Demonstrations scored per gradient slice.
        cotrain_dataset_path: LeRobot dataset the demonstrations are read from.
            Required exactly when ``cotrain_beta > 0``. Which cameras and
            how long a chunk spans come from the checkpoint and the run.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    discount_per: DiscountPer = "simulator_step"
    bootstrap_at_time_limit: bool = True
    learning_rate: float = 3e-4
    value_learning_rate: float | None = None
    value_parameter_pattern: str = "value_head"
    optimizer: Literal["adam", "adamw"] = "adam"
    adam_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_loss: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    epochs: int = 10
    minibatch_size: int = 64
    micro_batch_size: int | None = None
    max_grad_norm: float | None = 0.5
    target_kl: float | None = None
    normalize_advantages: bool = True
    seed: int = 0
    cotrain_beta: float = 0.0
    cotrain_batch_size: int = 32
    cotrain_dataset_path: str | None = None

    def __post_init__(self) -> None:
        self._validate_advantage_settings()
        self._validate_positive_settings()
        self._validate_cotrain_and_optimizer_settings()
        self._validate_loss_settings()
        self._validate_batch_settings()

    def _validate_advantage_settings(self) -> None:
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if self.discount_per not in ("simulator_step", "decision"):
            raise ValueError("discount_per must be 'simulator_step' or 'decision'")

    def _validate_positive_settings(self) -> None:
        positive_floats = {
            "learning_rate": self.learning_rate,
            "optimizer_eps": self.optimizer_eps,
            "clip_ratio": self.clip_ratio,
            "value_loss_coef": self.value_loss_coef,
            "huber_delta": self.huber_delta,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        optional_positive = {
            "value_learning_rate": self.value_learning_rate,
            "value_clip": self.value_clip,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
        }
        for name, optional_value in optional_positive.items():
            if optional_value is None:
                continue
            if not math.isfinite(optional_value) or optional_value <= 0.0:
                raise ValueError(f"{name} must be finite and positive or None")
        if self.value_learning_rate is not None and not self.value_parameter_pattern:
            raise ValueError(
                "value_parameter_pattern must be non-empty when value_learning_rate is set"
            )

    def _validate_cotrain_and_optimizer_settings(self) -> None:
        if not math.isfinite(self.cotrain_beta) or self.cotrain_beta < 0.0:
            raise ValueError(
                "cotrain_beta must be finite and >= 0 (0 disables co-training)"
            )
        if self.cotrain_batch_size <= 0:
            raise ValueError("cotrain_batch_size must be positive")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if len(self.adam_betas) != 2 or not all(
            math.isfinite(beta) and 0.0 <= beta < 1.0 for beta in self.adam_betas
        ):
            raise ValueError("adam_betas must be two values in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")

    def _validate_loss_settings(self) -> None:
        if self.value_loss not in ("mse", "huber"):
            raise ValueError("value_loss must be 'mse' or 'huber'")
        if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be finite and non-negative")

    def _validate_batch_settings(self) -> None:
        for name in ("epochs", "minibatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.micro_batch_size is not None:
            micro = self.micro_batch_size
            if isinstance(micro, bool) or not isinstance(micro, int) or micro < 1:
                raise ValueError("micro_batch_size must be a positive integer or None")
            if self.minibatch_size % micro:
                raise ValueError(
                    "minibatch_size must be a multiple of micro_batch_size, got "
                    f"{self.minibatch_size} and {micro}"
                )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
