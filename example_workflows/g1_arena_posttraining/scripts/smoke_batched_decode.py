#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched GR00T decode smoke on real weights — no simulator involved.

Loads the exact frozen steering wrapper training builds (``build_base_model``
on the composed DSRL config), then checks the batched decode against the
per-sample path on the real checkpoint:

* **row-0 equivalence** — under one fixed seed, the per-sample decode of env 0
  and row 0 of the B-batched decode must agree: both draw their base flow
  noise from the same RNG position (``randn(1, H, D)`` equals row 0 of
  ``randn(B, H, D)``), and the backbone processes each sequence independently,
  so any difference beyond float tolerance is a batching bug;
* **independence** — B different latents on identical observations must yield
  B different action chunks (a broadcast bug would collapse them onto one);
* **speedup** — wall-clock of B sequential decodes vs. one batched decode;
* **sweep** (``--sweep 8,64,128,256``) — steady-state time and peak GPU memory
  of one batched decode at each N, i.e. the learner's per-decision inference
  cost as a function of the number of envs (the scaling table's decode row).

Run in the learner environment (needs the GR00T stack and the checkpoint):

    python example_workflows/g1_arena_posttraining/scripts/smoke_batched_decode.py \
        --ckpt $POSTTRAIN_BASE_CKPT \
        --dataset $POSTTRAIN_DATA_ROOT/LMDrillLift/lerobot --num-envs 8 --sweep 8,64,128,256
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The G1 ego camera's native frame and the policy's proprioceptive width
# (simulators/g1_arena/env.py); only shapes matter here, content is random.
NATIVE_IMAGE_HW = (480, 640)
STATE_DIM = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", required=True, help="GR00T checkpoint dir (POSTTRAIN_BASE_CKPT)"
    )
    parser.add_argument("--dataset", required=True, help="<task>/lerobot dataset dir")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="row-0 equivalence tolerance"
    )
    parser.add_argument(
        "--sweep",
        type=lambda text: [int(n) for n in text.split(",")],
        default=[],
        help="comma-separated batch sizes to time one batched decode at (after the correctness checks)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed decodes per sweep point (after one warm-up)",
    )
    return parser.parse_args()


def check(condition: bool, label: str) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}", flush=True)
    if not condition:
        raise SystemExit(f"[smoke_batched_decode] {label}")


def main() -> int:
    args = parse_args()
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.runners.off_policy_launch import build_base_model

    register_configs()
    config_dir = os.path.join(_REPO_ROOT, "isaac_rl_posttraining", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[
                "base_model=gr00t",
                f"base_model.ckpt={args.ckpt}",
                f"base_model.bc_ckpt={args.ckpt}",
                f"base_model.dataset_dir={args.dataset}",
            ],
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.monotonic()
    wrapper = build_base_model(cfg, device)
    print(f"wrapper loaded in {time.monotonic() - t0:.1f}s on {device}", flush=True)
    check(
        getattr(wrapper, "supports_batched_infer", False),
        "wrapper advertises supports_batched_infer",
    )

    n = args.num_envs
    rng = np.random.default_rng(args.seed)
    # Identical observations for every env so only the latent differs, which
    # isolates the steering path in the independence check.
    image = rng.integers(0, 255, size=(*NATIVE_IMAGE_HW, 3), dtype=np.uint8)
    state = (0.1 * rng.standard_normal(STATE_DIM)).astype(np.float32)
    latents = rng.uniform(-1.0, 1.0, size=(n, int(wrapper.latent_dim))).astype(
        np.float32
    )
    obs_single = {"ego_view_image": image, "state": state}
    obs_batched = {
        "ego_view_image": np.repeat(image[None], n, axis=0),
        "state": np.repeat(state[None], n, axis=0),
    }

    # Warm-up (kernel selection / caches) so the timings below are steady-state.
    wrapper.infer(obs_single, latents[0:1])
    wrapper.infer_batch(obs_batched, latents)

    # (1) Per-sample decode of env 0 under a fixed seed.
    torch.manual_seed(args.seed)
    single0 = np.asarray(
        wrapper.infer(obs_single, latents[0:1])["actions"][0]
    )  # [H, A]
    # The old sequential path: N per-sample decodes.
    t0 = time.monotonic()
    for b in range(n):
        wrapper.infer(obs_single, latents[b : b + 1])
    t_sequential = time.monotonic() - t0

    # (2) One batched decode under the same seed: row 0 must reproduce single0.
    torch.manual_seed(args.seed)
    t0 = time.monotonic()
    batched = np.asarray(
        wrapper.infer_batch(obs_batched, latents)["actions"]
    )  # [N, H, A]
    t_batched = time.monotonic() - t0

    check(
        batched.shape == (n, *single0.shape),
        f"batched shape {batched.shape} == [N, H, A]",
    )
    check(bool(np.isfinite(batched).all()), "batched actions are finite")
    max_delta = float(np.abs(batched[0] - single0).max())
    check(
        max_delta < args.atol,
        f"batched[0] == per-sample env 0 (max|delta|={max_delta:.2e} < {args.atol})",
    )
    spread = np.abs(batched - batched[0:1]).max(
        axis=(1, 2)
    )  # per-env max |delta| vs env 0
    check(
        bool((spread[1:] > 1e-4).all()),
        f"envs steered independently (min spread vs env0={spread[1:].min():.2e})",
    )

    speedup = t_sequential / max(t_batched, 1e-9)
    print(
        f"timing: {n} sequential decodes {t_sequential:.2f}s ({t_sequential / n * 1000:.0f} ms each) "
        f"vs one batched decode {t_batched:.2f}s -> {speedup:.1f}x",
        flush=True,
    )
    print(
        f"[smoke_batched_decode] PASS — batched GR00T decode correct at N={n}, {speedup:.1f}x faster"
    )

    for sweep_n in args.sweep:
        time_batched_decode(wrapper, sweep_n, args.repeats, rng, device)
    return 0


def time_batched_decode(
    wrapper: Any, n: int, repeats: int, rng: np.random.Generator, device: torch.device
) -> None:
    """Time one N-wide batched decode (steady state) and its peak GPU memory.

    Distinct random images per env so the vision backbone sees the batch it
    would in training (identical images could be cached or deduplicated
    nowhere in this stack today, but the measurement should not depend on
    that).
    """
    obs = {
        "ego_view_image": rng.integers(
            0, 255, size=(n, *NATIVE_IMAGE_HW, 3), dtype=np.uint8
        ),
        "state": (0.1 * rng.standard_normal((n, STATE_DIM))).astype(np.float32),
    }
    latents = rng.uniform(-1.0, 1.0, size=(n, int(wrapper.latent_dim))).astype(
        np.float32
    )
    wrapper.infer_batch(obs, latents)  # warm-up: allocator growth, kernel selection
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        t0 = time.monotonic()
        wrapper.infer_batch(obs, latents)  # returns numpy, so this waits for the GPU
        times.append(time.monotonic() - t0)
    mean_s = float(np.mean(times))
    peak = (
        f", peak GPU alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
        if device.type == "cuda"
        else ""
    )
    print(
        f"[decode-sweep] N={n}: one batched decode {mean_s:.2f} s mean of {repeats} "
        f"(min {min(times):.2f}, max {max(times):.2f}; {1000 * mean_s / n:.1f} ms per env){peak}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
