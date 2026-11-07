# Infrastructure

The training loop runs on native, dependency-free components by default.
Two optional stacks can replace parts of it: a ZeroMQ transport for the link to a simulator server (`uv sync --extra zmq`) and [RLInf](https://github.com/RLinf/RLinf) for distributed rollout execution (`uv sync --extra rlinf`, [runtimes.md](runtimes.md)).
The measurements below were taken on a 48-core workstation with `scripts/bench_infra.py`; every mode has a `--smoke` variant and re-runs on your hardware:

```bash
uv run python scripts/bench_infra.py --help
```

Run the Ray modes (`rollout`, `ray-footprint`, `rollout-scaling`, `rollout-stepcost`) as `uv run --extra rlinf python scripts/bench_infra.py ...`; under plain `uv run` the Ray workers fail with `No module named ray` ([runtimes.md](runtimes.md#setup)).

## Transport between the learner and a simulator server

`--transport {native,zmq}` on `simulators/g1_arena/env_server.py`; the launchers use the native default.

| | native TCP (pickle) | ZeroMQ (msgpack) |
|---|---|---|
| frame encode, 900 KiB observation | ~350 µs | ~35 µs |
| round trip, small payloads (up to 100 KiB) | 0.07 to 0.11 ms | 0.12 to 0.21 ms |
| round trip, 900 KiB observation | 0.23 to 1.41 ms (depends on thread placement) | 0.28 to 0.63 ms |
| round trip, 4 MiB | 1.4 to 1.6 GB/s | 3.3 to 3.9 GB/s |
| wire format | Python pickle | msgpack (language-neutral, no code execution on decode) |

Keep the native default: transport is well under 0.1% of episode wall time for single-camera control at these payload sizes.
Switch to ZeroMQ when payloads grow past about 1 MiB per step (several cameras, higher resolutions), when you want predictable latency without controlling thread placement, or when a non-Python endpoint or a safer wire format matters.

## Replay buffer

The replay buffer is native (`isaac_rl_posttraining/buffers/replay.py`, [data_model.md](data_model.md)).
An adapter over RLInf's replay buffer measured about 65 ms per decision at the training cadence (one insert plus ten 4x64 samples per decision) against about 8.5 ms native, the gap being per-transition conversion.

## Parallel rollout collection

Rollout workers run through the runtime layer ([runtimes.md](runtimes.md)); do not select the compatibility adapters under `isaac_rl_posttraining/compat/rlinf/` for new work.
Their measured scaling on simulated environments, N workers under Ray streaming transitions to one learner:

| environments | speedup | efficiency |
|---|---|---|
| 2 | 2.0x | 99% |
| 4 | 3.9x | 96% |
| 8 | 7.4x | 92 to 96% |
| 16 | 14.8x | 92% |
| 32 | 23.6x | 74% |

Efficiency rises with heavier environment steps (96% to about 100% moving from 1 ms to 50 ms steps), so collection with real simulators scales close to linearly; the bend at 32 is serial ingest on the driver side.
Ray uses about 4 GiB resident at idle, about 11 GiB at 32 environments, plus a few seconds of cluster boot per run.
Single-simulator runs gain nothing from Ray.
Measurements of the runtime layer itself are in [runtimes.md](runtimes.md#measured-performance).

## Version notes

- The RLInf runtime and the compatibility adapters are validated against `rlinf==0.3.0` with `ray[default]==2.58.0`, the pins of the `rlinf` extra. Install them in a dedicated environment.
- RLInf builds NCCL groups from detected hardware and ignores `CUDA_VISIBLE_DEVICES`; the adapters move data through Ray's object store to avoid silent hangs on machines where only some GPUs are visible.
- RLInf's buffer constructor seeds NumPy globally and its sampler draws with replacement only; the adapter handles both.
