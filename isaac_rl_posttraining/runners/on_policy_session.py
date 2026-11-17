# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An open on-policy run: its data source, learner and trainer, and step-wise access to them.

The flow-SDE PPO and DSRL-PPO launchers open one and run
:meth:`OnPolicyTrainer.run` on it; a Python :class:`~isaac_rl_posttraining.run.Run`
drives its pieces step by step (collect, train, publish, evaluate, save).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaac_rl_posttraining.runners.ppo_launch import close_resources
from isaac_rl_posttraining.runtimes import PolicyState
from isaac_rl_posttraining.training import DataSource, OnPolicyTrainer

if TYPE_CHECKING:
    from isaac_rl_posttraining.buffers import ReplayBuffer
    from isaac_rl_posttraining.training.telemetry import ParquetTelemetrySink

__all__ = ["OnPolicySession"]


@dataclass
class OnPolicySession:
    """The pieces of an open on-policy run and step-wise access to them.

    One step is one iteration of the loop: :meth:`collect` plays the next
    iteration's episodes and :meth:`train` updates on them and publishes.
    ``resources`` (the learner group and the runtime) close after the data
    source, which closes the rollout workers, and before ``telemetry``.
    """

    trainer: OnPolicyTrainer
    run_dir: str
    resources: tuple[Any, ...]
    wandb_logger: Any
    telemetry: ParquetTelemetrySink | None = None

    @property
    def train_source(self) -> DataSource:
        return self.trainer.train_source

    @property
    def buffer(self) -> ReplayBuffer:
        raise RuntimeError(
            "an on-policy run keeps no replay buffer; its episodes are in run.telemetry, "
            "and data.persist.path saves them as a loadable buffer"
        )

    def collect(self, rounds: int = 1) -> Mapping[str, float]:
        """Collect the next iteration's episodes with the newest weights; returns their metrics."""
        if rounds != 1:
            raise ValueError(
                "an on-policy algorithm trains on each collected iteration once: call collect() and then "
                f"train() for every iteration (got rounds={rounds})"
            )
        return self.trainer.collect()

    def train(self, steps: int) -> Mapping[str, float]:
        """Run the algorithm's update on the collected episodes, then publish; returns the iteration's metrics."""
        if steps != 1:
            raise ValueError(
                "an on-policy algorithm updates once per collected iteration (its update epochs are "
                f"ppo.update_epochs): call collect() and then train() for every iteration (got steps={steps})"
            )
        return self.trainer.train()

    def evaluate(self) -> Mapping[str, float]:
        """Score the newest weights on the fixed evaluation episodes (``ppo.eval_*``)."""
        metrics = self.trainer.evaluate()
        return {name.removeprefix("eval/"): value for name, value in metrics.items()}

    def publish(self) -> PolicyState:
        """Send the learner's current weights to the rollout workers (a new policy version)."""
        return self.trainer.publish()

    def save(self, directory: str) -> None:
        """Write the learner's checkpoint to ``directory``."""
        self._learner("save").save_checkpoint(os.path.abspath(directory))

    def load(self, directory: str) -> None:
        """Load a learner checkpoint from ``directory`` and publish it."""
        self._learner("load").load_checkpoint(os.path.abspath(directory))
        self.trainer.publish()

    def close(self, primary_error: BaseException | None = None) -> None:
        """Finish W&B, then close the data source (and its rollout workers) and the other resources."""
        close_resources((self.train_source, *self.resources, self.telemetry), self.wandb_logger, primary_error)

    def _learner(self, method: str) -> Any:
        if self.trainer.learner_group is None:
            raise RuntimeError(f"{method} needs a learner; this run evaluates without one (placement.learner.ranks=0)")
        return self.trainer.learner_group
