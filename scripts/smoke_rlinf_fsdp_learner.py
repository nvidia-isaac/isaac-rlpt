#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a small, restart-checked RLInf FSDP learner job."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from isaac_rl_posttraining.runtimes import (
    LearnerBatch,
    LearnerContext,
    LearnerModuleManager,
    RLInfRuntime,
    RLInfRuntimeConfig,
    open_distributed_learner_group,
)


class LinearLearner:
    """Minimal algorithm delegate used to validate the distributed seam."""

    def __init__(
        self,
        context: LearnerContext,
        modules: LearnerModuleManager,
    ) -> None:
        self.device = torch.device(context.device)
        self.model = modules.prepare(
            torch.nn.Linear(4, 1),
            name="actor",
            publish=True,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-2, weight_decay=0.0
        )
        modules.register_optimizer("actor", self.optimizer, module=self.model)
        self.updates = 0

    def train_step(self, batch: LearnerBatch) -> dict[str, float]:
        inputs = torch.tensor(
            batch.tensors["inputs"],
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.tensor(
            batch.tensors["targets"],
            dtype=torch.float32,
            device=self.device,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(self.model(inputs), targets)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        return {"loss": float(loss.detach())}

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"updates": np.array(self.updates, dtype=np.int64)}

    def load_checkpoint_state(self, state: dict[str, np.ndarray]) -> None:
        self.updates = int(state["updates"].item())

    def close(self) -> None:
        """The linear learner holds no resources."""


def make_linear_learner(
    context: LearnerContext,
    modules: LearnerModuleManager,
) -> LinearLearner:
    return LinearLearner(context, modules)


def _batch(rank: int, step: int, samples: int) -> LearnerBatch:
    rng = np.random.default_rng(10_000 + 1_000 * step + rank)
    inputs = rng.normal(size=(samples, 4)).astype(np.float32)
    targets = inputs.sum(axis=1, keepdims=True)
    return LearnerBatch(
        tensors={"inputs": inputs, "targets": targets},
        samples=samples,
    )


def _changed(left, right) -> bool:
    return left.tensors.keys() == right.tensors.keys() and any(
        not np.array_equal(left.tensors[name], right.tensors[name])
        for name in left.tensors
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-learner-ranks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--samples-per-rank", type=int, default=16)
    parser.add_argument("--first-accelerator", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "bf16"),
        default="none",
    )
    args = parser.parse_args()

    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            learner_first_accelerator=args.first_accelerator,
            learner_mixed_precision=args.mixed_precision,
            learner_seed=0,
            shutdown_ray_on_close=True,
        )
    )
    try:
        group = open_distributed_learner_group(
            runtime,
            make_linear_learner,
            num_learner_ranks=args.num_learner_ranks,
            name="fsdp_learner_smoke",
        )
        initial = group.policy_state(version=0)
        losses = []
        for step in range(args.steps):
            results = group.train_step(
                tuple(
                    _batch(rank, step, args.samples_per_rank)
                    for rank in range(args.num_learner_ranks)
                )
            )
            if tuple(result.learner_rank for result in results) != tuple(
                range(args.num_learner_ranks)
            ) or any(result.step != step + 1 for result in results):
                raise RuntimeError("FSDP learner ranks or steps are inconsistent")
            mean_loss = np.mean([result.metrics["loss"] for result in results])
            if not np.isfinite(mean_loss):
                raise RuntimeError("FSDP learner produced a non-finite loss")
            losses.append(float(mean_loss))
            print(f"step={step + 1} mean_rank_loss={mean_loss:.6f}", flush=True)

        before = group.policy_state(version=1)
        if not _changed(initial, before):
            raise RuntimeError("FSDP training did not change the published policy")
        if len(losses) > 1 and len(set(losses)) == 1:
            raise RuntimeError("FSDP learner loss did not change across updates")
        with tempfile.TemporaryDirectory(prefix="rlpt-fsdp-smoke-") as directory:
            checkpoint = str(Path(directory).resolve())
            group.save_checkpoint(checkpoint)
            group.train_step(
                tuple(
                    _batch(rank, args.steps, args.samples_per_rank)
                    for rank in range(args.num_learner_ranks)
                )
            )
            perturbed = group.policy_state(version=2)
            if not _changed(before, perturbed):
                raise RuntimeError("FSDP perturbation step did not change the policy")
            group.load_checkpoint(checkpoint)
        restored = group.policy_state(version=3)
        if before.tensors.keys() != restored.tensors.keys() or any(
            not np.array_equal(before.tensors[name], restored.tensors[name])
            for name in before.tensors
        ):
            raise RuntimeError("FSDP checkpoint did not restore the published policy")
        resumed = group.train_step(
            tuple(
                _batch(rank, args.steps + 1, args.samples_per_rank)
                for rank in range(args.num_learner_ranks)
            )
        )
        if any(result.step != args.steps + 1 for result in resumed):
            raise RuntimeError("FSDP checkpoint did not restore the learner step")
        print(
            f"FSDP_SMOKE_OK num_learner_ranks={args.num_learner_ranks} steps={args.steps} "
            f"samples_per_rank={args.samples_per_rank}",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
