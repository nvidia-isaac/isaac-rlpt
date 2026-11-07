#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure benchmarks: transports, replay buffers, rollout scaling.

Measures the three infrastructure planes where this repository has both a
native implementation and an alternative backend, using contract-shaped
payloads (the G1 Factory DSRL geometry: 480x640x3 uint8 ego frames, 43-float
proprioceptive state, [40, 35] float32 action/latent chunks):

1. ``serialization`` -- codec cost alone: pickle (native transport frames)
   vs msgpack+numpy (ZeroMQ transport frames), so transport overhead can be
   separated from encoding cost.
2. ``transport`` -- request/response round-trips through in-process
   loopback servers: native TCP vs ZeroMQ over TCP vs ZeroMQ over ipc://,
   with and without the observation in the response (``need_obs``).
3. ``buffer`` -- add and sample throughput of the replay buffer at DSRL
   shapes, sampling 256-transition batches as 4x64 chunks like the SAC
   update loop.
4. ``rollout`` -- steps/sec collecting from mock environments with a fixed
   ~1 ms per-step cost: the native serial collector vs the RLinf-scheduler
   ``ParallelRolloutAdapter`` at 1/2/4/8 environments (skipped without
   ``[rlinf]``).

Extended planes (``--plane all-extended`` runs 5-10; results in
``docs/infrastructure.md``):

5. ``rollout-scaling`` -- the rollout plane pushed to 1..32 environments
   on this 48-core box, with per-env and fixed-total step budgets, speedup
   and parallel efficiency (speedup / envs).
6. ``rollout-stepcost`` -- parallel efficiency vs per-step weight:
   {1, 4, 8} envs x {1, 10, 50} ms/step (models LIBERO farms and
   CPU-physics arenas), native serial baseline at every step cost.
7. ``transport-sweep`` -- round-trip cost vs response payload size
   ({10 KiB, 100 KiB, 900 KiB contract, 4 MiB} synthetic frames) for
   native TCP / zmq TCP / zmq ipc; locates the end-to-end crossover
   between pickle's and msgpack's frame encodes.
8. ``transport-tails`` -- per-call latency tails (p50/p90/p99/max over
   2000 calls at the 900 KiB contract payload), solo and under a CPU
   antagonist (8 busy-loop spinner processes sharing a 4-core cpuset
   with the benchmark), for the jitter comparison.
9. ``buffer-trace`` -- trace-driven replay of the real trainer cadence
   (per decision step: 1 contract-shaped ``add`` then 10 sample bursts of
   256 as 4x64) for 2000 decision steps: wall time per decision step and
   RSS deltas, measured in a fresh subprocess so memory accounting is clean.
10. ``ray-footprint`` -- RSS/PSS of the Ray daemon fleet and ``ray::``
    workers at 1/8/32 environments (summed from ``/proc``), plus Ray boot
    and per-worker-group launch times.

Every mode degrades gracefully: rows or planes whose dependencies
(``pyzmq``/``msgpack``, ``rlinf``) are absent are skipped with a printed
message, so ``--plane all-extended`` runs in both venvs.

Results print as markdown tables (medians over ``--trials`` runs, with
p10/p90 dispersion where it matters; tail plane reports per-call
percentiles) and can be dumped to JSON. CPU-only; no GPU is touched. Run
it in both environments to get the full picture: the learner venv covers
the native+zmq planes, the rlinf venv all of them.

Usage::

    python scripts/bench_infra.py --plane all [--quick] [--json out.json]
    python scripts/bench_infra.py --plane all-extended [--smoke] [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from functools import partial

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ray refuses object creation near-full disks; benchmarks move tiny objects,
# so raise the threshold at this process boundary (before ray boots).
os.environ.setdefault("RAY_local_fs_capacity_threshold", "0.999")

EGO_HW = (480, 640)
STATE_DIM = 43
CHUNK, ACTION_DIM = 40, 35
DSRL_OBS = {"state": (STATE_DIM,), "dino_embedding": (768,)}

ZMQ_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("zmq", "msgpack", "msgpack_numpy")
)
RLINF_AVAILABLE = importlib.util.find_spec("rlinf") is not None

RESULTS: dict[str, object] = {}

P10_P90_HEADER = "p10 / p90 ms"
BUFTRACE_PREFIX = "BUFTRACE_JSON: "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
    }


def _table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    sys.stdout.flush()


def make_step_response(with_obs: bool, rng: np.random.Generator) -> dict:
    """A STEP response payload shaped like the Arena env server's."""
    payload = {
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "success": False,
        "num_steps": 8,
        "info": {},
    }
    if with_obs:
        payload.update(
            ego_view=rng.integers(0, 256, size=(*EGO_HW, 3), dtype=np.uint8),
            state=rng.standard_normal(STATE_DIM).astype(np.float32),
            prompt="pick up the drill",
        )
    return payload


def make_step_request(rng: np.random.Generator) -> dict:
    """A STEP request payload as ``ArenaEnv`` sends it (nested-list chunk)."""
    action = rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32)
    return {"v": 1, "type": "STEP", "action": action.tolist(), "need_obs": True}


# ---------------------------------------------------------------------------
# Plane 1: serialization
# ---------------------------------------------------------------------------


def _mean_call_s(call: Callable[[], object], reps: int) -> float:
    """Mean wall time of ``reps`` back-to-back calls."""
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    return (time.perf_counter() - t0) / reps


def bench_serialization(trials: int, quick: bool) -> None:
    rng = np.random.default_rng(0)
    reps = 30 if quick else 100
    payloads = {
        "STEP response (with 480x640 obs)": make_step_response(True, rng),
        "STEP response (need_obs=False)": make_step_response(False, rng),
        "STEP request ([40,35] chunk)": make_step_request(rng),
    }

    codecs = {"pickle (native frames)": (
        lambda p: pickle.dumps(p, protocol=4), pickle.loads)}
    if ZMQ_AVAILABLE:
        from isaac_rl_posttraining.envs.zmq_transport import pack_frame, unpack_frame

        codecs["msgpack+numpy (zmq frames)"] = (pack_frame, unpack_frame)

    rows = []
    plane: dict[str, dict] = {}
    for payload_name, payload in payloads.items():
        for codec_name, (enc, dec) in codecs.items():
            data = enc(payload)
            enc_times, dec_times = [], []
            for _ in range(trials):
                enc_times.append(_mean_call_s(partial(enc, payload), reps))
                dec_times.append(_mean_call_s(partial(dec, data), reps))
            enc_s, dec_s = _stats(enc_times), _stats(dec_times)
            rows.append(
                [
                    payload_name,
                    codec_name,
                    f"{len(data) / 1024:.1f}",
                    f"{enc_s['median'] * 1e6:.1f}",
                    f"{dec_s['median'] * 1e6:.1f}",
                ]
            )
            plane[f"{payload_name} / {codec_name}"] = {
                "bytes": len(data),
                "encode_us": enc_s,
                "decode_us": dec_s,
            }
    RESULTS["serialization"] = plane
    _table(
        "Serialization only (codec cost, no sockets)",
        ["payload", "codec", "KiB", "encode µs (med)", "decode µs (med)"],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 2: transport round-trips
# ---------------------------------------------------------------------------


def _serve_native(handler):
    from isaac_rl_posttraining.envs.transport import NativeTcpServer

    server = NativeTcpServer(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, args=(handler,), daemon=True)
    thread.start()
    url = f"tcp://127.0.0.1:{server.address[1]}"
    return server, thread, url


def _serve_zmq(handler, bind: str, url: str | None = None):
    from isaac_rl_posttraining.envs.zmq_transport import ZmqServer

    server = ZmqServer(bind)
    thread = threading.Thread(target=server.serve_forever, args=(handler,), daemon=True)
    thread.start()
    return server, thread, url or server.endpoint


def _step_server(step_response: Callable[[dict], dict]):
    """Loopback server handler that answers STEP with ``step_response(message)``."""

    def handler(message: dict):
        if message.get("type") == "STEP":
            return step_response(message)
        if message.get("type") == "CLOSE":
            return None
        return {"protocol": 1, "device": "cpu", "spaces": {}}

    return handler


def _time_steps(
    transport, action, *, need_obs: bool, warmup: int, trials: int, calls: int
) -> list[float]:
    """Per-call STEP seconds, averaged over ``calls`` calls, for each of ``trials`` trials."""
    step = partial(transport.step, action, need_obs=need_obs)
    for _ in range(warmup):
        step()
    return [_mean_call_s(step, calls) for _ in range(trials)]


def bench_transport(trials: int, quick: bool) -> None:
    rng = np.random.default_rng(1)
    calls = 40 if quick else 150
    action = rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32).tolist()
    responses = {
        True: make_step_response(True, rng),
        False: make_step_response(False, rng),
    }
    handler = _step_server(
        lambda message: responses[bool(message.get("need_obs", True))]
    )

    rows = []
    plane: dict[str, dict] = {}
    for kind, transport_id in _transport_kinds():
        transport, close = _open_transport(transport_id, handler, timeout=30.0)
        try:
            for need_obs in (True, False):
                resp_bytes = len(pickle.dumps(responses[need_obs], protocol=4))
                per_call = _time_steps(
                    transport,
                    action,
                    need_obs=need_obs,
                    warmup=10,
                    trials=trials,
                    calls=calls,
                )
                stats = _stats(per_call)
                calls_per_s = 1.0 / stats["median"]
                mb_per_s = calls_per_s * resp_bytes / (1024 * 1024)
                label = "full obs" if need_obs else "no obs"
                rows.append(
                    [
                        kind,
                        label,
                        f"{stats['median'] * 1e3:.2f}",
                        f"{stats['p10'] * 1e3:.2f} / {stats['p90'] * 1e3:.2f}",
                        f"{calls_per_s:.0f}",
                        f"{mb_per_s:.0f}" if need_obs else "-",
                    ]
                )
                plane[f"{kind} / {label}"] = {
                    "latency_ms": {k: v * 1e3 for k, v in stats.items()},
                    "calls_per_s": calls_per_s,
                    "resp_payload_bytes": resp_bytes,
                }
        finally:
            close()
    RESULTS["transport"] = plane
    _table(
        f"Transport round-trip (STEP with [40,35] chunk; median of {trials}x{calls} calls)",
        [
            "transport",
            "response",
            "ms/call (med)",
            P10_P90_HEADER,
            "calls/s",
            "obs MB/s",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 3: replay buffers
# ---------------------------------------------------------------------------


def _dsrl_transition(i: int, rng: np.random.Generator):
    """Decision ``i`` of a stream of 50-decision episodes, at the DSRL geometry."""
    from isaac_rl_posttraining.data import (
        DynamicObservation,
        LearnerAction,
        Transition,
        TransitionMetadata,
    )

    obs = {k: rng.standard_normal(shape).astype(np.float32) for k, shape in DSRL_OBS.items()}
    next_obs = {k: rng.standard_normal(shape).astype(np.float32) for k, shape in DSRL_OBS.items()}
    ended = (i % 50) == 49
    return Transition(
        episode_id=i // 50,
        observation=DynamicObservation.from_mapping(obs),
        next_observation=DynamicObservation.from_mapping(next_obs),
        learner_action=LearnerAction("latent", rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32)),
        env_action=None,
        rewards={"env": np.full(8, float(i % 7), dtype=np.float32)},
        terminated=np.asarray([False] * 7 + [ended]),
        truncated=np.zeros(8, dtype=bool),
        success=np.asarray([False] * 7 + [ended and i % 100 == 49]),
        behavior_source="policy",
        metadata=TransitionMetadata(step_in_episode=(i % 50) * 8, extra={"is_success": i % 100 == 49}),
    )


def _bench_one_buffer(buffer, sampler, size: int, sample_iters: int) -> dict:
    import dataclasses

    rng = np.random.default_rng(2)
    # Pre-generate a pool of whole episodes and cycle it: keeps the measurement
    # on the buffer, not on numpy sampling. Each cycle restamps the episode ids,
    # because the buffer refuses decisions for an episode that already ended.
    pool = [_dsrl_transition(i, rng) for i in range(250)]
    episodes_per_pool = len(pool) // 50
    t0 = time.perf_counter()
    for i in range(size):
        cycle, index = divmod(i, len(pool))
        transition = pool[index]
        buffer.add(
            dataclasses.replace(
                transition, episode_id=transition.episode_id + cycle * episodes_per_pool
            )
        )
    add_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(sample_iters):
        for _ in range(4):  # the SAC loop draws 256 as 4x64 chunks
            sampler.sample(64)
    sample_seconds = time.perf_counter() - t0
    return {
        "adds_per_s": size / add_seconds,
        "batches256_per_s": sample_iters / sample_seconds,
        "sample_ms_per_batch256": 1e3 * sample_seconds / sample_iters,
    }


def bench_buffer(trials: int, quick: bool) -> None:
    del trials  # fills are single-shot; per-batch dispersion is tiny vs fill cost
    from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler

    sizes = [10_000] if quick else [10_000, 100_000]
    sample_iters = 20 if quick else 50

    rows = []
    plane: dict[str, dict] = {}
    for size in sizes:
        name = "ReplayBuffer"
        buffer = ReplayBuffer(size)
        result = _bench_one_buffer(buffer, UniformSampler(buffer), size, sample_iters)
        rows.append(
            [
                name,
                f"{size:,}",
                f"{result['adds_per_s']:,.0f}",
                f"{result['batches256_per_s']:.1f}",
                f"{result['sample_ms_per_batch256']:.1f}",
            ]
        )
        plane[f"{name} @ {size}"] = result
        del buffer
    RESULTS["buffer"] = plane
    _table(
        "Replay buffer at DSRL shapes (obs 43+768 f32, action [40,35] f32; sample = 4x64)",
        ["buffer", "capacity=fill", "adds/s", "batches(256)/s", "ms/batch(256)"],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 4: rollout scaling
# ---------------------------------------------------------------------------

STEP_COST_S = 0.001


class CostedEnv:
    """Unbatched mock env with a fixed ~1 ms per-step cost (adapter side)."""

    def __init__(self, rank: int):
        self.rank = rank
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return {"state": np.zeros(STATE_DIM, dtype=np.float32)}, {}

    def close(self):
        """A mock env holds no resources."""

    def step(self, _action):
        time.sleep(STEP_COST_S)
        self.t += 1
        done = self.t >= 50
        if done:
            self.t = 0
        return (
            {"state": np.zeros(STATE_DIM, dtype=np.float32)},
            0.0,
            done,
            False,
            {"success": False, "is_success": False},
        )


class NullBuffer:
    def add(self, transition):
        del transition


def make_costed_env(rank: int) -> CostedEnv:
    return CostedEnv(rank)


def make_costed_policy(rank: int):
    del rank
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    return lambda obs: action


class _ZeroChunkPolicy:
    """One zero action per decision over the raw observation."""

    def __init__(self):
        self._policy_version = 0

    def action_chunk(self, observation, *, mode):
        from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk

        del observation, mode
        return ActionChunk.from_arrays(
            learner_action=None,
            environment_actions=np.zeros((1, 1, ACTION_DIM), dtype=np.float32),
            execution_length=1,
            source="native",
            policy_version=self._policy_version,
        )

    def set_policy_state(self, state):
        self._policy_version = state.version
        return self._policy_version


def _native_steps_per_s(num_envs: int, total_steps: int, make_env) -> float:
    """Steps/s of the native runtime: one in-process worker per env, stepped in turn."""
    from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
    from isaac_rl_posttraining.runtimes import (
        GymWorkerSpec,
        NativeRuntime,
        NativeRuntimeConfig,
        RolloutRequest,
    )

    class _CostedEnvironments(SimulatorEntry):
        """Builds the benchmark's costed environment for each worker."""

        name = "bench"

        def make_environment(self, spec, phase, context):
            del spec, phase
            return make_env(context)

    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    group = runtime.open_rollout_group(
        GymWorkerSpec(
            entry=_CostedEnvironments(),
            environment=EnvironmentSpec(
                env_name="bench/step_cost", num_envs=num_envs, max_episode_steps=total_steps, seed=0
            ),
            phase="eval",
            policy=lambda context: _ZeroChunkPolicy(),
        ),
        num_rollout_workers=num_envs,
        num_envs=num_envs,
        name="bench_rollout",
    )
    request = RolloutRequest(decisions_per_environment=math.ceil(total_steps / num_envs), mode="eval")
    t0 = time.perf_counter()
    results = group.collect([request] * num_envs)
    elapsed = time.perf_counter() - t0
    runtime.close()
    return float(sum(result.env_steps for result in results)) / elapsed


def _native_serial_steps_per_s(num_envs: int, total_steps: int) -> float:
    return _native_steps_per_s(
        num_envs, total_steps, lambda context: CostedEnv(context.environment_indices[0])
    )


def bench_rollout(trials: int, quick: bool) -> None:
    del trials  # one timed collection per env count; the 1 ms cost dominates
    env_counts = [1, 2, 4] if quick else [1, 2, 4, 8]
    total_steps = 200 if quick else 400

    rows = []
    plane: dict[str, dict] = {}
    adapters = []
    for n in env_counts:
        native_sps = _native_serial_steps_per_s(n, total_steps)
        row = [str(n), f"{native_sps:.0f}"]
        entry: dict[str, float] = {"native_serial_steps_per_s": native_sps}
        if RLINF_AVAILABLE:
            from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

            adapter = ParallelRolloutAdapter(
                make_costed_env, make_costed_policy, NullBuffer(), num_envs=n,
                num_rollouts=total_steps, count_by="step",
            )
            adapters.append(adapter)
            adapter.collect(n * 20)  # warmup: actors and code paths hot
            t0 = time.perf_counter()
            adapter.collect(total_steps)
            elapsed = time.perf_counter() - t0
            adapter_sps = total_steps / elapsed
            row += [f"{adapter_sps:.0f}", f"{adapter_sps / native_sps:.2f}x", f"{adapter.startup_seconds:.1f}"]
            entry.update(
                rlinf_parallel_steps_per_s=adapter_sps,
                speedup=adapter_sps / native_sps,
                adapter_startup_s=adapter.startup_seconds,
            )
        else:
            row += ["-", "-", "-"]
        rows.append(row)
        plane[f"envs={n}"] = entry
    for adapter in adapters:
        adapter.close()
    if not RLINF_AVAILABLE:
        print("\n(rlinf rollout skipped: rlinf not installed)")
    RESULTS["rollout"] = plane
    _table(
        f"Rollout collection, ~{STEP_COST_S * 1e3:.0f} ms/step mock envs, {total_steps} steps",
        ["envs", "native serial steps/s", "rlinf parallel steps/s", "speedup", "adapter startup s"],
        rows,
    )


# ---------------------------------------------------------------------------
# Extended-plane helpers: sized payloads, tail stats, transport sessions,
# parameterized costed envs, /proc memory accounting, CPU antagonists
# ---------------------------------------------------------------------------

SMOKE_N = 5
"""Calls/steps per measurement under ``--smoke`` (trivial-size CI runs)."""


def _tail_stats(samples_s: list[float]) -> dict[str, float]:
    """Per-call percentile stats (ms) over individually timed samples."""
    ordered = sorted(samples_s)
    n = len(ordered)

    def _pct(p: float) -> float:
        return ordered[min(n - 1, int(round(p * (n - 1))))]

    return {
        "p50_ms": _pct(0.50) * 1e3,
        "p90_ms": _pct(0.90) * 1e3,
        "p99_ms": _pct(0.99) * 1e3,
        "max_ms": ordered[-1] * 1e3,
        "mean_ms": 1e3 * sum(ordered) / n,
        "n": n,
    }


def make_sized_response(target_bytes: int | None, rng: np.random.Generator) -> dict:
    """A STEP response whose synthetic square frame is ~``target_bytes``.

    ``None`` keeps the contract geometry (480x640x3, ~900 KiB). The 43-float
    state and prompt ride along in every size so the payload shape matches
    the Arena schema, not just its byte count.
    """
    payload = {
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "success": False,
        "num_steps": 8,
        "info": {},
        "state": rng.standard_normal(STATE_DIM).astype(np.float32),
        "prompt": "pick up the drill",
    }
    if target_bytes is None:
        hw = EGO_HW
    else:
        side = max(8, int(round((target_bytes / 3) ** 0.5)))
        hw = (side, side)
    payload["ego_view"] = rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
    return payload


def _transport_kinds() -> list[tuple[str, str]]:
    """Transport (label, id) pairs available in this venv; zmq rows gated."""
    kinds = [("native tcp (pickle)", "native-tcp")]
    if ZMQ_AVAILABLE:
        kinds += [("zmq tcp (msgpack)", "zmq-tcp"), ("zmq ipc (msgpack)", "zmq-ipc")]
    else:
        print("\n(zmq transports skipped: pyzmq/msgpack not installed)")
    return kinds


def _open_transport(kind: str, handler, timeout: float = 60.0):
    """Start a loopback server of ``kind`` and return a connected transport.

    Returns:
        ``(transport, close_fn)``; ``close_fn`` tears down transport, server,
        and server thread.
    """
    from isaac_rl_posttraining.envs.transport import transport_from_uri

    if kind == "native-tcp":
        server, thread, url = _serve_native(handler)
    elif kind == "zmq-tcp":
        server, thread, url = _serve_zmq(handler, "tcp://127.0.0.1:*")
        url = url.replace("tcp://", "zmq+tcp://")
    elif kind == "zmq-ipc":
        socket_dir = tempfile.mkdtemp(prefix="isaacrl_bench_")
        endpoint = f"ipc://{os.path.join(socket_dir, 'bench.sock')}"
        server, thread, url = _serve_zmq(handler, endpoint, endpoint)
    else:
        raise ValueError(f"unknown transport kind {kind!r}")
    transport = transport_from_uri(
        url, request_timeout=timeout, connect_timeout=timeout, retry_interval=0.2
    )
    transport.connect()

    def _close() -> None:
        transport.close()
        server.close()
        thread.join(timeout=10)
        if kind == "zmq-ipc":
            shutil.rmtree(socket_dir, ignore_errors=True)

    return transport, _close


def _wire_bytes(kind: str, payload: dict) -> int:
    """Serialized size of ``payload`` under the codec transport ``kind`` uses."""
    if kind == "native-tcp":
        return len(pickle.dumps(payload, protocol=4))
    from isaac_rl_posttraining.envs.zmq_transport import pack_frame

    return len(pack_frame(payload))


class ParamCostedEnv:
    """Unbatched mock env with a configurable fixed per-step cost."""

    def __init__(self, rank: int, step_cost_s: float):
        self.rank = rank
        self.step_cost_s = float(step_cost_s)
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return {"state": np.zeros(STATE_DIM, dtype=np.float32)}, {}

    def close(self):
        """A mock env holds no resources."""

    def step(self, _action):
        time.sleep(self.step_cost_s)
        self.t += 1
        done = self.t >= 50
        if done:
            self.t = 0
        return (
            {"state": np.zeros(STATE_DIM, dtype=np.float32)},
            0.0,
            done,
            False,
            {"success": False, "is_success": False},
        )


class CostedEnvSpec:
    """Picklable ``rank -> env`` recipe with a configurable step cost."""

    def __init__(self, step_cost_s: float):
        self.step_cost_s = float(step_cost_s)

    def __call__(self, rank: int) -> ParamCostedEnv:
        return ParamCostedEnv(rank, self.step_cost_s)


def _native_serial_sps(num_envs: int, total_steps: int, step_cost_s: float) -> float:
    """Native runtime steps/s at a parameterized per-step cost."""
    return _native_steps_per_s(
        num_envs,
        total_steps,
        lambda context: ParamCostedEnv(context.environment_indices[0], step_cost_s),
    )


def _rlinf_parallel_sps(
    num_envs: int, budgets: list[int], step_cost_s: float, warmup_steps: int
) -> tuple[list[float], float]:
    """Adapter steps/s for each step budget in ``budgets`` (one worker group).

    Returns:
        ``(steps_per_s_per_budget, startup_seconds)``.
    """
    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

    adapter = ParallelRolloutAdapter(
        CostedEnvSpec(step_cost_s),
        make_costed_policy,
        NullBuffer(),
        num_envs=num_envs,
        num_rollouts=max(budgets),
        count_by="step",
    )
    try:
        adapter.collect(warmup_steps)  # actors and code paths hot
        rates = []
        for budget in budgets:
            t0 = time.perf_counter()
            adapter.collect(budget)
            rates.append(budget / (time.perf_counter() - t0))
        return rates, adapter.startup_seconds
    finally:
        adapter.close()


def _spin_forever(cpus: list[int], ready) -> None:
    """Busy-loop antagonist body: pin to ``cpus``, signal readiness, spin."""
    try:
        if cpus:
            os.sched_setaffinity(0, set(cpus))
    except OSError:
        pass
    ready.set()
    while True:
        pass  # NOSONAR - deliberate busy loop that occupies the CPU


def _start_spinners(count: int, cpus: list[int]) -> list:
    """Spawn ``count`` busy-loop processes pinned to ``cpus``; wait until live."""
    ctx = multiprocessing.get_context("spawn")
    started = []
    for _ in range(count):
        ready = ctx.Event()
        proc = ctx.Process(target=_spin_forever, args=(list(cpus), ready), daemon=True)
        proc.start()
        started.append((proc, ready))
    for _, ready in started:
        ready.wait(timeout=60)
    return [proc for proc, _ in started]


def _stop_spinners(spinners: list) -> None:
    for proc in spinners:
        proc.terminate()
    for proc in spinners:
        proc.join(timeout=10)


def _pin_all_threads(cpus: set[int]) -> list[tuple[int, set[int]]]:
    """Pin every thread of this process to ``cpus``; return state to restore.

    ``sched_setaffinity`` is per-thread on Linux, so the main thread alone is
    not enough -- server threads already running must be pinned too. Threads
    created afterwards inherit their creator's mask.
    """
    saved = []
    for tid_str in os.listdir("/proc/self/task"):
        tid = int(tid_str)
        try:
            saved.append((tid, os.sched_getaffinity(tid)))
            os.sched_setaffinity(tid, cpus)
        except OSError:
            pass  # thread exited between listing and pinning
    return saved


def _restore_affinity(saved: list[tuple[int, set[int]]]) -> None:
    for tid, cpus in saved:
        try:
            os.sched_setaffinity(tid, cpus)
        except OSError:
            pass


def _self_mem_kb() -> dict[str, int]:
    """Current and peak RSS of this process from ``/proc/self/status`` (KiB)."""
    out = {"VmRSS": 0, "VmHWM": 0}
    with open("/proc/self/status", encoding="ascii", errors="replace") as fh:
        for line in fh:
            key = line.split(":", 1)[0]
            if key in out:
                out[key] = int(line.split()[1])
    return out


def _proc_mem_kb(pid: int) -> tuple[int, int] | None:
    """``(rss_kb, pss_kb)`` for ``pid``; PSS falls back to RSS if unreadable."""
    try:
        rss = 0
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                    break
    except OSError:
        return None
    pss = rss
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    pss = int(line.split()[1])
                    break
    except OSError:
        pass
    return rss, pss


def _proc_name(pid: int) -> str:
    """Process name: full cmdline when available (Ray retitles it), else comm."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if cmdline:
            return cmdline
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def _descendant_pids(root_pid: int) -> list[int]:
    """All live descendants of ``root_pid`` from a single ``/proc`` scan."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                stat = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        # comm (field 2) may contain spaces/parens; parse after its closing paren.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 1:
            children.setdefault(int(fields[1]), []).append(int(entry))
    out: list[int] = []
    stack = [root_pid]
    while stack:
        kids = children.get(stack.pop(), [])
        out.extend(kids)
        stack.extend(kids)
    return out


def _ray_fleet_snapshot() -> dict[str, object]:
    """RSS/PSS of this process's Ray fleet, split ``ray::`` workers vs daemons."""
    groups = {
        "workers": {"count": 0, "rss_mb": 0.0, "pss_mb": 0.0},
        "daemons": {"count": 0, "rss_mb": 0.0, "pss_mb": 0.0},
    }
    procs: list[tuple[float, str]] = []
    for pid in _descendant_pids(os.getpid()):
        mem = _proc_mem_kb(pid)
        if mem is None:
            continue  # exited during the scan
        rss_kb, pss_kb = mem
        name = _proc_name(pid)
        group = groups["workers"] if "ray::" in name else groups["daemons"]
        group["count"] += 1
        group["rss_mb"] += rss_kb / 1024
        group["pss_mb"] += pss_kb / 1024
        procs.append((rss_kb / 1024, name[:80]))
    procs.sort(reverse=True)
    return {
        **groups,
        "driver_rss_mb": _self_mem_kb()["VmRSS"] / 1024,
        "top_procs": [{"rss_mb": round(m, 1), "name": n} for m, n in procs[:6]],
    }


# ---------------------------------------------------------------------------
# Plane 5: rollout scaling to 32 envs
# ---------------------------------------------------------------------------


def bench_rollout_scaling(trials: int, smoke: bool) -> None:
    del trials  # one timed collection per (envs, budget); the step cost dominates
    env_counts = [1, 2] if smoke else [1, 2, 4, 8, 16, 32]
    steps_per_env = SMOKE_N if smoke else 400
    fixed_total = SMOKE_N if smoke else 400
    step_cost_s = 0.001

    if not RLINF_AVAILABLE:
        print("\n(rlinf rollout-scaling columns skipped: rlinf not installed)")

    rows = []
    plane: dict[str, dict] = {}
    for n in env_counts:
        total = steps_per_env * n
        native_sps = _native_serial_sps(n, total, step_cost_s)
        entry: dict[str, float] = {
            "native_serial_steps_per_s": native_sps,
            "steps_per_env": steps_per_env,
        }
        row = [str(n), f"{native_sps:.0f}"]
        if RLINF_AVAILABLE:
            (sps_scaled, sps_fixed), startup = _rlinf_parallel_sps(
                n, [total, fixed_total], step_cost_s, warmup_steps=n * (2 if smoke else 20)
            )
            speedup = sps_scaled / native_sps
            efficiency = speedup / n
            row += [
                f"{sps_scaled:.0f}",
                f"{speedup:.2f}x",
                f"{efficiency * 100:.0f}%",
                f"{sps_fixed:.0f}",
                f"{startup:.1f}",
            ]
            entry.update(
                rlinf_parallel_steps_per_s=sps_scaled,
                rlinf_fixed_total_steps_per_s=sps_fixed,
                speedup=speedup,
                efficiency=efficiency,
                adapter_startup_s=startup,
            )
        else:
            row += ["-", "-", "-", "-", "-"]
        rows.append(row)
        plane[f"envs={n}"] = entry
    RESULTS["rollout_scaling"] = plane
    _table(
        f"Rollout scaling, ~1 ms/step mock envs, {steps_per_env} steps/env "
        f"(fixed-total {fixed_total}-step column shows dispatch amortization)",
        [
            "envs",
            "native serial steps/s",
            f"rlinf steps/s ({steps_per_env}/env)",
            "speedup",
            "efficiency",
            f"rlinf steps/s ({fixed_total} total)",
            "startup s",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 6: parallel efficiency vs per-step cost
# ---------------------------------------------------------------------------


def bench_rollout_stepcost(trials: int, smoke: bool) -> None:
    del trials  # one timed collection per combo; the step cost dominates
    if smoke:
        step_costs, env_counts = [0.001, 0.010], [1]
        steps_per_env = {0.001: SMOKE_N, 0.010: SMOKE_N}
    else:
        step_costs, env_counts = [0.001, 0.010, 0.050], [1, 4, 8]
        # Heavier steps need fewer of them for a stable rate estimate.
        steps_per_env = {0.001: 400, 0.010: 100, 0.050: 40}

    if not RLINF_AVAILABLE:
        print("\n(rlinf rollout-stepcost columns skipped: rlinf not installed)")

    rows = []
    plane: dict[str, dict] = {}
    for step_cost_s in step_costs:
        for n in env_counts:
            total = steps_per_env[step_cost_s] * n
            native_sps = _native_serial_sps(n, total, step_cost_s)
            entry: dict[str, float] = {
                "native_serial_steps_per_s": native_sps,
                "steps_per_env": steps_per_env[step_cost_s],
            }
            row = [f"{step_cost_s * 1e3:.0f}", str(n), str(steps_per_env[step_cost_s]),
                   f"{native_sps:.1f}"]
            if RLINF_AVAILABLE:
                (sps,), startup = _rlinf_parallel_sps(
                    n, [total], step_cost_s, warmup_steps=n * (2 if smoke else 5)
                )
                speedup = sps / native_sps
                efficiency = speedup / n
                row += [f"{sps:.1f}", f"{speedup:.2f}x", f"{efficiency * 100:.0f}%"]
                entry.update(
                    rlinf_parallel_steps_per_s=sps,
                    speedup=speedup,
                    efficiency=efficiency,
                    adapter_startup_s=startup,
                )
            else:
                row += ["-", "-", "-"]
            rows.append(row)
            plane[f"cost={step_cost_s * 1e3:.0f}ms/envs={n}"] = entry
    RESULTS["rollout_stepcost"] = plane
    _table(
        "Parallel efficiency vs per-step cost (native serial baseline per cell)",
        ["step ms", "envs", "steps/env", "native steps/s", "rlinf steps/s",
         "speedup", "efficiency"],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 7: transport round-trip vs payload size
# ---------------------------------------------------------------------------

SWEEP_SIZES: list[tuple[str, int | None]] = [
    ("10 KiB", 10 * 1024),
    ("100 KiB", 100 * 1024),
    ("900 KiB (contract)", None),
    ("4 MiB", 4 * 1024 * 1024),
]


def bench_transport_sweep(trials: int, smoke: bool) -> None:
    rng = np.random.default_rng(3)
    calls = SMOKE_N if smoke else 150
    trials = 1 if smoke else trials
    action = rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32).tolist()
    kinds = _transport_kinds()

    rows = []
    plane: dict[str, dict] = {}
    for size_label, target_bytes in SWEEP_SIZES:
        response = make_sized_response(target_bytes, rng)
        handler = _step_server(lambda message, _resp=response: _resp)
        for label, kind in kinds:
            wire = _wire_bytes(kind, response)
            transport, close = _open_transport(kind, handler)
            try:
                per_call = _time_steps(
                    transport,
                    action,
                    need_obs=True,
                    warmup=3 if smoke else 10,
                    trials=trials,
                    calls=calls,
                )
            finally:
                close()
            stats = _stats(per_call)
            mb_per_s = wire / stats["median"] / (1024 * 1024)
            rows.append(
                [
                    size_label,
                    label,
                    f"{wire / 1024:.0f}",
                    f"{stats['median'] * 1e3:.3f}",
                    f"{stats['p10'] * 1e3:.3f} / {stats['p90'] * 1e3:.3f}",
                    f"{1.0 / stats['median']:.0f}",
                    f"{mb_per_s:.0f}",
                ]
            )
            plane[f"{size_label} / {label}"] = {
                "wire_bytes": wire,
                "latency_ms": {k: v * 1e3 for k, v in stats.items()},
                "calls_per_s": 1.0 / stats["median"],
                "mb_per_s": mb_per_s,
            }
    RESULTS["transport_sweep"] = plane
    _table(
        f"Transport round-trip vs payload size (STEP with [40,35] chunk; "
        f"median of {trials}x{calls} calls)",
        [
            "payload",
            "transport",
            "wire KiB",
            "ms/call (med)",
            P10_P90_HEADER,
            "calls/s",
            "MB/s",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 8: latency tails, solo and under CPU contention
# ---------------------------------------------------------------------------

ANTAGONIST_SPINNERS = 8
ANTAGONIST_CPUSET_SIZE = 4


def bench_transport_tails(trials: int, smoke: bool) -> None:
    del trials  # tails are per-call percentiles over one continuous stream
    rng = np.random.default_rng(4)
    calls = SMOKE_N if smoke else 2000
    warmup = 3 if smoke else 50
    response = make_step_response(True, rng)  # the 900 KiB contract payload
    action = rng.standard_normal((CHUNK, ACTION_DIM)).astype(np.float32).tolist()

    handler = _step_server(lambda message: response)
    kinds = _transport_kinds()
    # 8 floating spinners cannot contend on a 48-core box, so the contended
    # condition shares one small cpuset between the benchmark threads (client
    # + server) and the spinners -- a deliberate, reproducible squeeze.
    cpuset = set(sorted(os.sched_getaffinity(0))[:ANTAGONIST_CPUSET_SIZE])
    conditions = [("solo", 0), ("contended", ANTAGONIST_SPINNERS)]

    rows = []
    plane: dict[str, dict] = {}
    for cond_label, spinner_count in conditions:
        saved_affinity = None
        spinners: list = []
        if spinner_count:
            saved_affinity = _pin_all_threads(cpuset)
            spinners = _start_spinners(spinner_count, sorted(cpuset))
            time.sleep(0.2 if smoke else 1.0)  # let the squeeze settle
        try:
            for label, kind in kinds:
                transport, close = _open_transport(kind, handler)
                try:
                    # one sample per call: the tails need individually timed calls
                    samples = _time_steps(
                        transport,
                        action,
                        need_obs=True,
                        warmup=warmup,
                        trials=calls,
                        calls=1,
                    )
                finally:
                    close()
                stats = _tail_stats(samples)
                rows.append(
                    [
                        label,
                        cond_label,
                        f"{stats['p50_ms']:.2f}",
                        f"{stats['p90_ms']:.2f}",
                        f"{stats['p99_ms']:.2f}",
                        f"{stats['max_ms']:.2f}",
                    ]
                )
                plane[f"{label} / {cond_label}"] = stats
        finally:
            _stop_spinners(spinners)
            if saved_affinity is not None:
                _restore_affinity(saved_affinity)
    RESULTS["transport_tails"] = plane
    _table(
        f"Transport latency tails, 900 KiB contract payload, {calls} calls "
        f"(contended = {ANTAGONIST_SPINNERS} busy-loop processes sharing a "
        f"{ANTAGONIST_CPUSET_SIZE}-core cpuset with the benchmark)",
        ["transport", "condition", "p50 ms", "p90 ms", "p99 ms", "max ms"],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 9: trace-driven buffer replay (real trainer cadence)
# ---------------------------------------------------------------------------

TRACE_SAMPLE_BURSTS = 10
"""Sample bursts of 256 (as 4x64) per decision step -- the SAC update cadence."""

REALISTIC_DECISION_STEP_S = 2.5
"""Wall clock of one real decision step (chunk rollout + updates) for the tax %."""


def _buffer_trace_child_main(args: argparse.Namespace) -> int:
    """Child process body: run the trace against one buffer implementation.

    Runs in a fresh interpreter so RSS deltas are attributable to the buffer
    under test, and prints one ``BUFTRACE_JSON: {...}`` line for the parent.
    """
    from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler

    buffer = ReplayBuffer(args.child_capacity)
    sampler = UniformSampler(buffer)
    rng = np.random.default_rng(2)

    # Every add gets unique arrays, as real collection produces them -- the
    # recycled 256-transition pool of the throughput plane would understate
    # memory for by-reference storage (the buffer would retain 256 payloads
    # instead of one per add). Generation stays outside the timed window: it
    # stands in for the env/collector work of the decision step.
    mem_pre = _self_mem_kb()
    for i in range(args.child_prefill):
        buffer.add(_dsrl_transition(i, rng))
    per_step: list[float] = []
    for d in range(args.child_decision_steps):
        transition = _dsrl_transition(args.child_prefill + d, rng)
        t0 = time.perf_counter()
        buffer.add(transition)
        for _ in range(TRACE_SAMPLE_BURSTS):
            for _ in range(4):  # 256 drawn as 4x64, like the SAC loop
                sampler.sample(64)
        per_step.append(time.perf_counter() - t0)
    mem_end = _self_mem_kb()

    result = {
        "impl": args.buffer_trace_child,
        "decision_steps": args.child_decision_steps,
        "prefill": args.child_prefill,
        "capacity": args.child_capacity,
        "step_stats": _tail_stats(per_step),
        "step_median_ms": _stats(per_step)["median"] * 1e3,
        "step_p10_ms": _stats(per_step)["p10"] * 1e3,
        "step_p90_ms": _stats(per_step)["p90"] * 1e3,
        "total_s": sum(per_step),
        "rss_delta_mb": (mem_end["VmRSS"] - mem_pre["VmRSS"]) / 1024,
        "peak_rss_delta_mb": (mem_end["VmHWM"] - mem_pre["VmHWM"]) / 1024,
        "rss_end_mb": mem_end["VmRSS"] / 1024,
    }
    print(BUFTRACE_PREFIX + json.dumps(result))
    return 0


def _run_buffer_trace_child(impl: str, decision_steps: int, prefill: int,
                            capacity: int) -> dict | None:
    """Launch one buffer-trace child and parse its result line."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--buffer-trace-child", impl,
        "--child-decision-steps", str(decision_steps),
        "--child-prefill", str(prefill),
        "--child-capacity", str(capacity),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=_REPO_ROOT)
    for line in proc.stdout.splitlines():
        if line.startswith(BUFTRACE_PREFIX):
            return json.loads(line[len(BUFTRACE_PREFIX) :])
    print(
        f"\n(buffer-trace {impl} child failed, rc={proc.returncode}: "
        f"{(proc.stderr or proc.stdout)[-500:]})"
    )
    return None


def bench_buffer_trace(trials: int, smoke: bool) -> None:
    del trials  # dispersion comes from the per-decision-step distribution
    decision_steps = SMOKE_N if smoke else 2000
    prefill = 70 if smoke else 5000
    capacity = 256 if smoke else 100_000

    rows = []
    plane: dict[str, dict] = {}
    for impl in ("native",):
        result = _run_buffer_trace_child(impl, decision_steps, prefill, capacity)
        if result is None:
            continue
        tax_pct = result["step_median_ms"] / (REALISTIC_DECISION_STEP_S * 1e3) * 100
        result["tax_pct_of_decision_step"] = tax_pct
        rows.append(
            [
                "ReplayBuffer",
                f"{result['step_median_ms']:.1f}",
                f"{result['step_p10_ms']:.1f} / {result['step_p90_ms']:.1f}",
                f"{result['step_stats']['p99_ms']:.1f}",
                f"{tax_pct:.2f}%",
                f"{result['rss_delta_mb']:.0f}",
                f"{result['peak_rss_delta_mb']:.0f}",
            ]
        )
        plane[impl] = result
    RESULTS["buffer_trace"] = plane
    _table(
        f"Trace-driven buffer replay: per decision step 1 add + "
        f"{TRACE_SAMPLE_BURSTS}x(4x64) samples; {decision_steps} decision steps, "
        f"prefill {prefill}, capacity {capacity:,} (fresh subprocess per impl)",
        [
            "buffer",
            "ms/decision-step (med)",
            P10_P90_HEADER,
            "p99 ms",
            f"% of {REALISTIC_DECISION_STEP_S:.1f}s step",
            "ΔRSS MiB",
            "Δpeak RSS MiB",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Plane 10: Ray fleet memory footprint
# ---------------------------------------------------------------------------


def bench_ray_footprint(trials: int, smoke: bool) -> None:
    del trials  # memory snapshots, not latency distributions
    if not RLINF_AVAILABLE:
        print("\n(ray-footprint skipped: rlinf not installed)")
        return
    if not os.path.isdir("/proc/self"):
        print("\n(ray-footprint skipped: needs Linux /proc)")
        return

    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter
    from isaac_rl_posttraining.compat.rlinf.rollout import ensure_cluster
    from rlinf.scheduler import Cluster

    already_booted = Cluster.has_initialized()
    t0 = time.monotonic()
    ensure_cluster()
    boot_s = None if already_booted else time.monotonic() - t0
    settle = 0.5 if smoke else 2.0
    time.sleep(settle)
    idle = _ray_fleet_snapshot()

    plane: dict[str, object] = {"boot_s": boot_s, "idle": idle}
    rows = [
        [
            "0 (idle fleet)",
            "-" if boot_s is None else f"{boot_s:.1f} (boot)",
            str(idle["workers"]["count"]),
            f"{idle['workers']['rss_mb']:.0f}",
            f"{idle['workers']['pss_mb']:.0f}",
            str(idle["daemons"]["count"]),
            f"{idle['daemons']['rss_mb']:.0f}",
            f"{idle['daemons']['pss_mb']:.0f}",
        ]
    ]
    env_counts = [1] if smoke else [1, 8, 32]
    for n in env_counts:
        adapter = ParallelRolloutAdapter(
            CostedEnvSpec(0.001),
            make_costed_policy,
            NullBuffer(),
            num_envs=n,
            num_rollouts=n * SMOKE_N,
            count_by="step",
        )
        try:
            adapter.collect(n * SMOKE_N)  # workers demonstrably doing work
            time.sleep(settle)
            snap = _ray_fleet_snapshot()
            snap["launch_s"] = adapter.startup_seconds
            plane[f"envs={n}"] = snap
            rows.append(
                [
                    str(n),
                    f"{adapter.startup_seconds:.1f}",
                    str(snap["workers"]["count"]),
                    f"{snap['workers']['rss_mb']:.0f}",
                    f"{snap['workers']['pss_mb']:.0f}",
                    str(snap["daemons"]["count"]),
                    f"{snap['daemons']['rss_mb']:.0f}",
                    f"{snap['daemons']['pss_mb']:.0f}",
                ]
            )
        finally:
            adapter.close()
        time.sleep(settle)  # let the raylet reap the group before the next launch
    RESULTS["ray_footprint"] = plane
    boot_note = "already booted by an earlier plane" if boot_s is None else f"{boot_s:.1f}s"
    print(f"\nRay runtime boot: {boot_note}; driver RSS "
          f"{idle['driver_rss_mb']:.0f} MiB at idle snapshot")
    _table(
        "Ray fleet memory (sum over this process's descendants; PSS from "
        "/proc/<pid>/smaps_rollup)",
        ["envs", "launch s", "worker procs", "workers RSS MiB", "workers PSS MiB",
         "daemon procs", "daemons RSS MiB", "daemons PSS MiB"],
        rows,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Extended planes in ``--plane all-extended`` execution order. Transport
#: planes run before anything imports torch/ray; ray-footprint precedes the
#: rollout planes so it observes the Ray runtime boot.
EXTENDED_PLANES = (
    "transport-sweep",
    "transport-tails",
    "buffer-trace",
    "ray-footprint",
    "rollout-scaling",
    "rollout-stepcost",
)

_EXTENDED_DISPATCH = {
    "transport-sweep": bench_transport_sweep,
    "transport-tails": bench_transport_tails,
    "buffer-trace": bench_buffer_trace,
    "ray-footprint": bench_ray_footprint,
    "rollout-scaling": bench_rollout_scaling,
    "rollout-stepcost": bench_rollout_stepcost,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--plane",
        choices=("serialization", "transport", "buffer", "rollout", "all")
        + EXTENDED_PLANES
        + ("all-extended",),
        default="all",
    )
    parser.add_argument("--trials", type=int, default=7, help="repeats per measurement")
    parser.add_argument("--quick", action="store_true", help="smaller sizes/counts")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"trivial sizes ({SMOKE_N} calls/steps) so every extended mode "
        "parses and runs in seconds (CI smoke)",
    )
    parser.add_argument("--json", default=None, help="write raw results to this path")
    # Internal: buffer-trace child process mode (fresh interpreter per
    # implementation so RSS deltas are attributable; see bench_buffer_trace).
    parser.add_argument("--buffer-trace-child", choices=("native",),
                        help=argparse.SUPPRESS)
    parser.add_argument("--child-decision-steps", type=int, default=2000,
                        help=argparse.SUPPRESS)
    parser.add_argument("--child-prefill", type=int, default=5000,
                        help=argparse.SUPPRESS)
    parser.add_argument("--child-capacity", type=int, default=100_000,
                        help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.buffer_trace_child:
        return _buffer_trace_child_main(args)

    print(f"# Infrastructure benchmarks (python {sys.version.split()[0]}, "
          f"zmq={'yes' if ZMQ_AVAILABLE else 'no'}, rlinf={'yes' if RLINF_AVAILABLE else 'no'})")
    if args.plane in ("serialization", "all"):
        bench_serialization(args.trials, args.quick)
    if args.plane in ("transport", "all"):
        bench_transport(args.trials, args.quick)
    if args.plane in ("buffer", "all"):
        bench_buffer(args.trials, args.quick)
    if args.plane in ("rollout", "all"):
        bench_rollout(args.trials, args.quick)
    for plane in EXTENDED_PLANES:
        if args.plane in (plane, "all-extended"):
            _EXTENDED_DISPATCH[plane](args.trials, args.smoke)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, indent=2, default=float)
        print(f"\nraw results written to {args.json}")
    return 0


if __name__ == "__main__":
    # Ray workers must not need to import this script as a module.
    if RLINF_AVAILABLE:
        try:
            from ray import cloudpickle

            cloudpickle.register_pickle_by_value(sys.modules["__main__"])
        except Exception:
            pass
    sys.exit(main())
