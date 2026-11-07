#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark identical PPO updates under direct and RLInf-managed FSDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from isaac_rl_posttraining.algorithms.ppo import PPO, PPOConfig, PPOLearner
from isaac_rl_posttraining.runtimes import (
    LearnerBatch,
    LearnerContext,
    RLInfRuntime,
    RLInfRuntimeConfig,
    open_distributed_learner_group,
)

RESULT_PREFIX = "RLPT_PPO_FSDP_RESULT="


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


REPO_ROOT = Path(__file__).resolve().parents[1]
#: Names the code revision when the script runs outside a git checkout (a source tarball).
REVISION_ENV = "ISAAC_RL_POSTTRAINING_REVISION"


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _revision(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The code revision, never failing: git, else ``$ISAAC_RL_POSTTRAINING_REVISION``, else a ``REVISION`` file."""
    try:
        commit = _git_output("rev-parse", "HEAD")
        return {
            "git_commit": commit,
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "revision_source": "git",
        }
    except (OSError, subprocess.CalledProcessError):
        pass
    revision = os.environ.get(REVISION_ENV, "").strip()
    source = REVISION_ENV
    if not revision and (root / "REVISION").is_file():
        revision = (root / "REVISION").read_text(encoding="utf-8").strip()
        source = "REVISION file"
    if not revision:
        return {"git_commit": None, "git_dirty": None, "revision_source": None}
    return {"git_commit": revision, "git_dirty": None, "revision_source": source}


@dataclass(frozen=True)
class TransformerModelSpec:
    observation_dim: int
    sequence_length: int
    action_dim: int
    width: int
    layers: int
    heads: int

    def __call__(self) -> nn.Module:
        return TransformerActorCritic(
            observation_dim=self.observation_dim,
            sequence_length=self.sequence_length,
            action_dim=self.action_dim,
            width=self.width,
            layers=self.layers,
            heads=self.heads,
        )


class TransformerActorCritic(nn.Module):
    """Transformer workload with analytically known initial policy heads."""

    def __init__(
        self,
        *,
        observation_dim: int,
        sequence_length: int,
        action_dim: int,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(observation_dim, width)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)
        self.policy_mean = nn.Linear(width, action_dim)
        self.value_head = nn.Linear(width, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.policy_mean.weight)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations, actions):
        features = self.input_projection(observations) + self.position
        features = self.output_norm(self.transformer(features).mean(dim=1))
        mean = self.policy_mean(features)
        distribution = torch.distributions.Normal(
            mean,
            self.log_std.clamp(-20.0, 2.0).exp().expand_as(mean),
        )
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        log_prob = (
            distribution.log_prob(raw) - torch.log(1.0 - bounded.square() + 1e-6)
        ).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.value_head(features).squeeze(-1)
        return log_prob, entropy, values


class _DirectFSDPModules:
    def __init__(self, mixed_precision: str, device: torch.device) -> None:
        self.mixed_precision = mixed_precision
        self.device = device
        self.module = None
        self.optimizer = None
        self.parameter_count = 0

    def prepare(
        self,
        module: nn.Module,
        *,
        name: str,
        publish: bool = False,
        frozen_submodules: Sequence[str] = (),
        shard: bool = True,
    ) -> nn.Module:
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision

        if not shard:
            raise ValueError("direct PPO benchmark shards its model")
        if frozen_submodules:
            raise ValueError("direct PPO benchmark requires a fully trainable model")
        if name != "actor_critic" or not publish or self.module is not None:
            raise ValueError("direct PPO benchmark expects one published actor_critic")
        self.parameter_count = sum(
            parameter.numel() for parameter in module.parameters()
        )
        precision = None
        if self.mixed_precision == "bf16":
            precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        self.module = FullyShardedDataParallel(
            module,
            device_id=self.device,
            mixed_precision=precision,
            sync_module_states=True,
            use_orig_params=True,
        )
        return self.module

    def register_optimizer(self, name, optimizer, *, module):
        if name != "actor_critic" or module is not self.module:
            raise ValueError("optimizer must belong to the prepared actor_critic")
        self.optimizer = optimizer


def _initial_log_prob(raw_actions: np.ndarray) -> np.ndarray:
    std = math.exp(-0.5)
    gaussian = (
        -0.5 * (raw_actions / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    )
    actions = np.tanh(raw_actions)
    correction = np.log(1.0 - actions**2 + 1e-6)
    return (gaussian - correction).sum(axis=-1).astype(np.float32)


def _global_batch_arrays(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.data_seed)
    observations = rng.normal(
        size=(args.global_batch, args.sequence_length, args.observation_dim)
    ).astype(np.float32)
    raw_actions = rng.normal(
        loc=0.0,
        scale=math.exp(-0.5),
        size=(args.global_batch, args.action_dim),
    ).astype(np.float32)
    advantages = rng.normal(size=args.global_batch).astype(np.float32)
    returns = rng.normal(size=args.global_batch).astype(np.float32)
    return observations, raw_actions, advantages, returns


def _batch(args: argparse.Namespace, learner_rank: int) -> LearnerBatch:
    local_samples = args.global_batch // args.num_learner_ranks
    start = learner_rank * local_samples
    stop = start + local_samples
    observations, raw_actions, advantages, returns = _global_batch_arrays(args)
    local_raw_actions = raw_actions[start:stop]
    return LearnerBatch(
        tensors={
            "observations": observations[start:stop],
            "actions": np.tanh(local_raw_actions).astype(np.float32),
            "old_log_probs": _initial_log_prob(local_raw_actions),
            "advantages": advantages[start:stop],
            "returns": returns[start:stop],
            "old_values": np.zeros(local_samples, dtype=np.float32),
        },
        samples=local_samples,
    )


def _global_batch_sha256(args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    for array in _global_batch_arrays(args):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def _aggregate(
    args: argparse.Namespace,
    *,
    launcher: str,
    startup_seconds: float,
    step_seconds: list[float],
    rank_metrics: list[list[dict[str, float]]],
    parameter_count: int,
) -> dict[str, Any]:
    visits_per_update = args.global_batch * args.epochs
    total_seconds = sum(step_seconds)
    peak_allocated = max(
        metric.get("peak_allocated_bytes", 0.0)
        for step in rank_metrics
        for metric in step
    )
    peak_reserved = max(
        metric.get("peak_reserved_bytes", 0.0)
        for step in rank_metrics
        for metric in step
    )
    optimizer_seconds = [
        max(metric["optimizer_seconds"] for metric in step) for step in rank_metrics
    ]
    controller_overhead = [
        max(0.0, wall - optimizer)
        for wall, optimizer in zip(step_seconds, optimizer_seconds, strict=True)
    ]
    return {
        "schema_version": 1,
        "benchmark": "ppo_fsdp_scaling",
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "launcher": launcher,
        "world_size": args.num_learner_ranks,
        "global_batch": args.global_batch,
        "learner_rank_local_batch": args.global_batch // args.num_learner_ranks,
        "global_minibatch": args.global_minibatch,
        "learner_rank_local_minibatch": (
            args.global_minibatch // args.num_learner_ranks
        ),
        "epochs": args.epochs,
        "warmup_updates": args.warmup_updates,
        "timed_updates": args.timed_updates,
        "parameter_count": parameter_count,
        "mixed_precision": args.mixed_precision,
        "global_batch_sha256": _global_batch_sha256(args),
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "shuffle_seed": args.shuffle_seed,
        "startup_seconds": startup_seconds,
        "step_seconds": _summary(step_seconds),
        "optimizer_seconds": _summary(optimizer_seconds),
        "controller_overhead_seconds": _summary(controller_overhead),
        "input_samples_per_second": args.global_batch
        * len(step_seconds)
        / total_seconds,
        "sample_visits_per_second": visits_per_update
        * len(step_seconds)
        / total_seconds,
        "peak_allocated_bytes_per_rank_max": peak_allocated,
        "peak_reserved_bytes_per_rank_max": peak_reserved,
        "model": {
            "observation_dim": args.observation_dim,
            "sequence_length": args.sequence_length,
            "action_dim": args.action_dim,
            "width": args.width,
            "layers": args.layers,
            "heads": args.heads,
        },
        "packages": {
            name: _package_version(name)
            for name in ("isaac-rl-posttraining", "rlinf", "ray", "torch", "numpy")
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            **_revision(),
        },
        "python": platform.python_version(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        minibatch_size=args.global_minibatch // args.num_learner_ranks,
        target_kl=None,
        seed=args.shuffle_seed,
    )


def _model_spec(args: argparse.Namespace) -> TransformerModelSpec:
    return TransformerModelSpec(
        observation_dim=args.observation_dim,
        sequence_length=args.sequence_length,
        action_dim=args.action_dim,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
    )


def _run_rlinf(args: argparse.Namespace) -> dict[str, Any]:
    build_model = _model_spec(args)
    parameter_count = sum(parameter.numel() for parameter in build_model().parameters())
    learner_spec = PPO(_config(args)).learner(build_model)
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            learner_first_accelerator=0,
            learner_mixed_precision=args.mixed_precision,
            learner_seed=args.model_seed,
            shutdown_ray_on_close=True,
        )
    )
    started = time.perf_counter()
    group = None
    try:
        group = open_distributed_learner_group(
            runtime,
            learner_spec,
            num_learner_ranks=args.num_learner_ranks,
            name=f"ppo_fsdp_{args.num_learner_ranks}",
        )
        startup_seconds = time.perf_counter() - started
        batches = tuple(
            _batch(args, learner_rank) for learner_rank in range(args.num_learner_ranks)
        )
        for _ in range(args.warmup_updates):
            group.train_step(batches)
        step_seconds: list[float] = []
        rank_metrics: list[list[dict[str, float]]] = []
        for _ in range(args.timed_updates):
            started = time.perf_counter()
            results = group.train_step(batches)
            step_seconds.append(time.perf_counter() - started)
            rank_metrics.append([dict(result.metrics) for result in results])
        return _aggregate(
            args,
            launcher="rlinf",
            startup_seconds=startup_seconds,
            step_seconds=step_seconds,
            rank_metrics=rank_metrics,
            parameter_count=parameter_count,
        )
    finally:
        runtime.close()


def _run_torchrun(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    learner_rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != args.num_learner_ranks:
        raise ValueError("--num-learner-ranks must match torchrun WORLD_SIZE")
    torch.cuda.set_device(local_rank)
    started = time.perf_counter()
    dist.init_process_group("nccl")
    torch.manual_seed(args.model_seed)
    device = torch.device("cuda", local_rank)
    modules = _DirectFSDPModules(args.mixed_precision, device)
    build_model = _model_spec(args)
    learner = PPOLearner(
        LearnerContext(
            learner_rank=learner_rank,
            world_size=args.num_learner_ranks,
            device=str(device),
        ),
        modules,
        build_model,
        _config(args),
    )
    startup = torch.tensor(time.perf_counter() - started, device=device)
    dist.all_reduce(startup, op=dist.ReduceOp.MAX)
    startup_seconds = float(startup.item())
    parameter_count = modules.parameter_count
    batch = _batch(args, learner_rank)
    autocast = {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": args.mixed_precision == "bf16",
    }
    for _ in range(args.warmup_updates):
        with torch.autocast(**autocast):
            learner.train_step(batch)
    step_seconds: list[float] = []
    rank_metrics: list[list[dict[str, float]]] = []
    for _ in range(args.timed_updates):
        dist.barrier()
        begin = time.perf_counter()
        with torch.autocast(**autocast):
            metrics = learner.train_step(batch)
        dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - begin, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        gathered: list[dict[str, float] | None] = [None] * args.num_learner_ranks
        dist.all_gather_object(gathered, metrics)
        if learner_rank == 0:
            step_seconds.append(float(elapsed.item()))
            rank_metrics.append([item for item in gathered if item is not None])
    result = None
    if learner_rank == 0:
        result = _aggregate(
            args,
            launcher="torchrun",
            startup_seconds=startup_seconds,
            step_seconds=step_seconds,
            rank_metrics=rank_metrics,
            parameter_count=parameter_count,
        )
    learner.close()
    dist.destroy_process_group()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", choices=("rlinf", "torchrun"), required=True)
    parser.add_argument("--num-learner-ranks", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--global-minibatch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--timed-updates", type=int, default=5)
    parser.add_argument("--observation-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=35)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mixed-precision", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=20260911)
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    positive = (
        "num_learner_ranks",
        "global_batch",
        "global_minibatch",
        "epochs",
        "warmup_updates",
        "timed_updates",
        "observation_dim",
        "sequence_length",
        "action_dim",
        "width",
        "layers",
        "heads",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("all size and count arguments must be positive")
    if (
        args.global_batch % args.num_learner_ranks
        or args.global_minibatch % args.num_learner_ranks
    ):
        raise ValueError("global batch sizes must be divisible by num_learner_ranks")
    if args.global_batch % args.global_minibatch:
        raise ValueError("global batch must be divisible by global minibatch")
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads")

    result = _run_rlinf(args) if args.launcher == "rlinf" else _run_torchrun(args)
    if result is not None:
        # The measurement is written first, so nothing after it can lose it.
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
