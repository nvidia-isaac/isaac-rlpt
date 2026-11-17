# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark RLInf shared policy-service scaling on fixed LIBERO inputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry


def _combine(observations: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for key in observations[0]:
        values = [observation[key] for observation in observations]
        if all(isinstance(value, np.ndarray) for value in values):
            combined[key] = np.stack(values)
        elif all(value == values[0] for value in values[1:]):
            combined[key] = values[0]
        else:
            combined[key] = values
    return combined


def _hash_value(hasher: Any, value: Any) -> int:
    if isinstance(value, dict):
        size = 0
        for key in sorted(value):
            hasher.update(key.encode())
            size += _hash_value(hasher, value[key])
        return size
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
        return int(array.nbytes)
    encoded = str(value).encode()
    hasher.update(encoded)
    return len(encoded)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class _TraceEnvironments(SimulatorEntry):
    """Each worker replays one recorded observation forever."""

    name = "trace"

    def __init__(self, observations: tuple[dict[str, Any], ...]) -> None:
        super().__init__()
        self.observations = observations

    def make_environment(self, spec, phase, context):
        del spec, phase
        return _TraceEnvironment(self.observations[context.worker_index])


class _TraceEnvironment:
    def __init__(self, observation: dict[str, Any]) -> None:
        self.observation = observation

    def reset(self):
        return self.observation, {}

    def step(self, action):
        del action
        return self.observation, 0.0, False, False, {}

    def close(self) -> None:
        """A replayed observation holds no resources."""


@dataclass(frozen=True)
class _Gr00tPolicySpec:
    checkpoint: str

    def __call__(self, context):
        return _Gr00tPolicy(self.checkpoint, context.device)


class _Gr00tPolicy:
    def __init__(self, checkpoint: str, device: str) -> None:
        import torch

        from isaac_rl_posttraining.policies.gr00t.model import LatentInjectionConfig
        from isaac_rl_posttraining.policies.gr00t.wrapper import SteeredGr00tWrapper

        self.device = device
        self.wrapper = SteeredGr00tWrapper(
            base_ckpt=checkpoint,
            embodiment_tag="libero_sim",
            device=device,
            noise_dim=7,
            latent_injection=LatentInjectionConfig(fresh_base=False),
        )
        self.parameter_dtypes = sorted(
            {
                str(parameter.dtype)
                for parameter in self.wrapper.policy.model.parameters()
            }
        )
        torch.cuda.reset_peak_memory_stats()

    def action_chunks(self, observations, *, mode, session_ids):
        del mode, session_ids
        from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk

        batch_size = len(observations)
        latents = np.zeros((batch_size, 7), dtype=np.float32)
        actions = self.wrapper.infer(_combine(tuple(observations)), latents)["actions"]
        return tuple(
            ActionChunk.from_arrays(
                learner_action=latents[index : index + 1],
                environment_actions=actions[index : index + 1],
                execution_length=1,
                source="policy",
            )
            for index in range(batch_size)
        )

    def reset_episode(self, session_id=0) -> None:
        del session_id

    def set_policy_state(self, state) -> int:
        return int(state.version)

    def benchmark_metadata(self) -> dict[str, Any]:
        return {"parameter_dtypes": self.parameter_dtypes}


class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[list[dict[str, float | int]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                sample = []
                for line in output.splitlines():
                    index, utilization, memory = (
                        item.strip() for item in line.split(",")
                    )
                    sample.append(
                        {
                            "index": int(index),
                            "utilization_percent": float(utilization),
                            "memory_used_mib": float(memory),
                        }
                    )
                self.samples.append(sample)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def _gpu_summary(samples, service_count: int) -> list[dict[str, Any]]:
    result = []
    for index in range(service_count):
        rows = [row for sample in samples for row in sample if row["index"] == index]
        if not rows:
            continue
        utilizations = [float(row["utilization_percent"]) for row in rows]
        memories = [float(row["memory_used_mib"]) for row in rows]
        result.append(
            {
                "index": index,
                "samples": len(rows),
                "utilization_mean_percent": statistics.fmean(utilizations),
                "utilization_max_percent": max(utilizations),
                "memory_used_mean_mib": statistics.fmean(memories),
                "memory_used_max_mib": max(memories),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--service-count", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--warmup-decisions", type=int, default=2)
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", default="8")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < args.service_count or args.decisions < 1:
        parser.error("num-envs must cover services and decisions must be positive")

    import torch

    from isaac_rl_posttraining.runtimes import RLInfRuntime, RLInfRuntimeConfig
    from isaac_rl_posttraining.runtimes.base import RolloutRequest
    from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
    from simulators.libero.policies.gr00t import make_libero_gr00t_env

    np.random.seed(args.seed)
    task_kwargs = (
        {"task_id": int(args.task)} if args.task.isdigit() else {"task_name": args.task}
    )
    trace_started = time.perf_counter()
    source_env = make_libero_gr00t_env(
        suite=args.suite,
        max_episode_steps=720,
        use_init_states=True,
        seed=args.seed,
        **task_kwargs,
    )
    observations = []
    try:
        for index in range(args.num_envs):
            observation, _ = source_env.reset(seed=args.seed + index)
            observations.append(observation)
    finally:
        source_env.close()
    trace_seconds = time.perf_counter() - trace_started
    input_hasher = hashlib.sha256()
    observation_bytes = sum(
        _hash_value(input_hasher, observation) for observation in observations
    )
    latent_bytes = args.num_envs * args.decisions * 7 * np.dtype(np.float32).itemsize
    input_hasher.update(np.zeros(latent_bytes, dtype=np.uint8).tobytes())

    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="gpu",
            first_accelerator=0,
            policy_mode="served",
            num_policy_servers=args.service_count,
            collect_policy_server_metrics=True,
            policy_batch_size=32,
            policy_batch_wait_seconds=0.005,
            shutdown_ray_on_close=True,
        )
    )
    spec = GymWorkerSpec(
        entry=_TraceEnvironments(tuple(observations)),
        environment=EnvironmentSpec(
            env_name="trace/libero",
            num_envs=args.num_envs,
            max_episode_steps=1,
            seed=None,
        ),
        phase="eval",
        policy=_Gr00tPolicySpec(str(Path(args.checkpoint).resolve())),
        supports_batched_policy=True,
    )
    group_started = time.perf_counter()
    group = runtime.open_rollout_group(
        spec,
        num_rollout_workers=args.num_envs,
        num_envs=args.num_envs,
        name="gr00t_scale",
    )
    group_startup_seconds = time.perf_counter() - group_started
    try:
        if args.warmup_decisions:
            warmup = RolloutRequest(
                decisions_per_environment=args.warmup_decisions,
                mode="eval",
            )
            group.collect([warmup] * args.num_envs)
        group.reset_policy_server_stats()
        request = RolloutRequest(
            decisions_per_environment=args.decisions,
            mode="eval",
        )
        with _GpuSampler() as sampler:
            started = time.perf_counter()
            results = group.collect([request] * args.num_envs)
            elapsed = time.perf_counter() - started
        service_stats = group.policy_server_stats()
    finally:
        group.close()
        runtime.close()

    action_outputs = np.concatenate(
        [
            decision.action_chunk.environment_actions.astype(np.float32, copy=False)
            for result in results
            for decision in result.decisions
        ],
        axis=0,
    )
    compressed_actions = base64.b64encode(
        zlib.compress(action_outputs.tobytes())
    ).decode()
    latencies = [
        float(latency)
        for stats in service_stats
        for latency in stats["request_latencies_seconds"]
    ]
    batch_sizes = [
        int(batch_size)
        for stats in service_stats
        for batch_size in stats["batch_sizes"]
    ]
    requests = args.num_envs * args.decisions
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    result = {
        "schema_version": 2,
        "benchmark": "gr00t_n1.7_rlinf_policy_service_scaling",
        "service_count": args.service_count,
        "num_envs": args.num_envs,
        "num_rollout_workers": args.num_envs,
        "warmup_decisions_per_worker": args.warmup_decisions,
        "measured_decisions_per_worker": args.decisions,
        "measured_requests": requests,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "batch_limit": 32,
        "batch_wait_seconds": 0.005,
        "observation_bytes_total": observation_bytes,
        "observation_bytes_per_request": observation_bytes / args.num_envs,
        "latent_bytes_per_request": 7 * np.dtype(np.float32).itemsize,
        "matched_input_sha256": input_hasher.hexdigest(),
        "trace_creation_seconds": trace_seconds,
        "group_startup_seconds": group_startup_seconds,
        "measured_seconds": elapsed,
        "total_requests_per_second": requests / elapsed,
        "requests_per_second_per_service": requests / elapsed / args.service_count,
        "service_request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "realized_batch_sizes": {
            "count": len(batch_sizes),
            "mean": statistics.fmean(batch_sizes),
            "min": min(batch_sizes),
            "max": max(batch_sizes),
            "histogram": {
                str(size): batch_sizes.count(size) for size in sorted(set(batch_sizes))
            },
        },
        "service_stats": service_stats,
        "gpu_samples": _gpu_summary(sampler.samples, args.service_count),
        "action_outputs": {
            "shape": action_outputs.shape,
            "dtype": str(action_outputs.dtype),
            "sha256": hashlib.sha256(action_outputs.tobytes()).hexdigest(),
            "zlib_base64": compressed_actions,
        },
        "hardware": {
            "allocated_gpus": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": git_commit,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary_keys = (
        "service_count",
        "measured_requests",
        "total_requests_per_second",
        "requests_per_second_per_service",
        "service_request_latency_seconds",
        "realized_batch_sizes",
    )
    summary = {key: result[key] for key in summary_keys}
    print(f"BENCHMARK_SUMMARY={json.dumps(summary, sort_keys=True)}")
    print(f"BENCHMARK_OUTPUT={args.output}")


if __name__ == "__main__":
    main()
