#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-contact smoke test for a batched g1_arena env server.

Drives a running ``env_server.py --num-envs N`` through one full protocol
cycle — SPEC handshake, full reset, stepped batches with the images-off fast
path, a masked per-env reset — printing payload shapes and per-step timings.
This is the first thing to run on a new machine: it
proves the batched wire protocol, the WBC action assembly, and — because the
whole-body controller runs every sim step — that the homie stand/walk ONNX
policies accept a batch dimension greater than one.

Run from the learner environment (needs only numpy and this repository):

    # terminal 1 (sim environment):
    $SIM_PYTHON simulators/g1_arena/env_server.py --task LMDrillLift \
        --num-envs 2 --port 7300 --image-resolution 0
    # terminal 2 (learner environment):
    python example_workflows/g1_arena_posttraining/scripts/smoke_batched.py \
        --url tcp://127.0.0.1:7300 --num-envs 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaac_rl_posttraining.envs.transport import transport_from_uri

ACTION_DIM = 35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="tcp://127.0.0.1:7300", help="env-server URL")
    parser.add_argument(
        "--num-envs", type=int, required=True, help="must match the server's --num-envs"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=3,
        help="chunk cycles to run; chunk 0 is treated as warm-up in the steady-state summary",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=8,
        help="sim steps per chunk (images only on the last)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-request transport timeout in seconds",
    )
    return parser.parse_args()


def check(condition: bool, label: str) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}", flush=True)
    if not condition:
        raise SystemExit(f"[smoke_batched] {label}")


def check_step(payload: dict, n: int, chunk: int, last: bool) -> tuple[bool, int]:
    """Check one step's batched results; returns whether it carried images and how many episodes ended."""
    done = np.asarray(payload["done"], dtype=bool)
    trunc = np.asarray(payload["truncated"], dtype=bool)
    success = np.asarray(payload["success"], dtype=bool)
    check(
        done.shape == (n,) and success.shape == (n,),
        f"step results carry the [N] axis (chunk {chunk})",
    )
    has_images = "ego_view" in payload
    episode_end = bool(done.any() or trunc.any())
    if last or episode_end:
        check(has_images, "images present on chunk end / episode end")
    else:
        check(
            not has_images,
            "images skipped on intermediate steps (need_obs off)",
        )
    if not episode_end:
        return has_images, 0
    print(
        f"  episode end: done={done.tolist()} truncated={trunc.tolist()} success={success.tolist()}"
    )
    return has_images, int(done.sum() + trunc.sum())


def print_steady_state(
    n: int, steady_plain: list[float], steady_image: list[float]
) -> None:
    # Client-side round trips: server step + transport. The server log's
    # own "step timing" lines give the sim-only part; the image steps add
    # the N-image payload.
    steady_all = steady_plain + steady_image
    mean_all = float(np.mean(steady_all))
    plain_ms = f"{1000 * float(np.mean(steady_plain)):.0f}" if steady_plain else "n/a"
    image_ms = f"{1000 * float(np.mean(steady_image)):.0f}" if steady_image else "n/a"
    print(
        f"[smoke_batched] steady state N={n}: {1000 * mean_all:.0f} ms/step round trip over {len(steady_all)} steps "
        f"({n / mean_all:.1f} env-steps/s); steps without images {plain_ms} ms, with images {image_ms} ms"
    )


def main() -> int:
    args = parse_args()
    n = args.num_envs
    transport = transport_from_uri(args.url, request_timeout=args.timeout)
    transport.connect()

    spec = transport.spec()
    print(
        f"spec: task={spec.get('task')} embodiment={spec.get('embodiment')} device={spec.get('device')}"
    )
    check(int(spec["num_envs"]) == n, f"server hosts num_envs={n}")

    t0 = time.monotonic()
    payload = transport.reset()
    print(f"full reset: {time.monotonic() - t0:.1f}s")
    state = np.asarray(payload["state"])
    image = np.asarray(payload["ego_view"])
    print(f"reset obs: state{state.shape} ego_view{image.shape}")
    check(state.shape[0] == n and state.ndim == 2, "reset state carries the [N] axis")
    check(image.shape[0] == n and image.ndim == 4, "reset images carry the [N] axis")

    # Distinct per-env actions so a broadcast/aliasing bug would be visible in
    # the sim; small magnitudes keep the robots near their default pose.
    rng = np.random.default_rng(0)
    dones_seen = 0
    steady_plain: list[float] = []  # per-step round trips after warm-up, no images
    steady_image: list[float] = []  # ... and the decision steps that carry images
    for chunk in range(args.chunks):
        step_times = []
        for i in range(args.chunk_len):
            actions = (0.02 * rng.standard_normal((n, ACTION_DIM))).astype(np.float32)
            last = i == args.chunk_len - 1
            t0 = time.monotonic()
            payload = transport.step(actions.tolist(), need_obs=last)
            step_times.append(time.monotonic() - t0)
            has_images, ended = check_step(payload, n, chunk, last)
            if chunk > 0:
                (steady_image if has_images else steady_plain).append(step_times[-1])
            dones_seen += ended
        mean_ms = 1000 * float(np.mean(step_times))
        print(
            f"chunk {chunk}: {args.chunk_len} steps, {mean_ms:.0f} ms/step mean "
            f"({n / (float(np.mean(step_times)) or 1.0):.1f} env-steps/s)"
        )
    if steady_plain or steady_image:
        print_steady_state(n, steady_plain, steady_image)

    # Masked reset of env 0 alone: the rolling per-env restart the collector
    # relies on (also the path client-side truncations take).
    t0 = time.monotonic()
    payload = transport.reset(env_ids=[0])
    print(f"masked reset (env 0): {time.monotonic() - t0:.1f}s")
    check(
        np.asarray(payload["state"]).shape[0] == n,
        "masked reset returns the full batch",
    )

    payload = transport.step(
        (0.02 * rng.standard_normal((n, ACTION_DIM))).astype(np.float32).tolist()
    )
    check(
        np.asarray(payload["done"], dtype=bool).shape == (n,),
        "stepping continues after the masked reset",
    )

    transport.close()
    print(
        f"[smoke_batched] PASS — batched protocol healthy at N={n} (episode ends observed: {dones_seen})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
