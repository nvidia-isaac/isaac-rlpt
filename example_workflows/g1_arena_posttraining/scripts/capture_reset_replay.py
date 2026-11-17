#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture two identical-seed resets from a running G1 Factory server."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from isaac_rl_posttraining.envs.transport import transport_from_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="tcp://127.0.0.1:7300")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def _validate_payload(payload: dict[str, Any], *, num_envs: int) -> None:
    for name, dimensions in (("state", 2), ("ego_view", 4)):
        if name not in payload:
            raise RuntimeError(f"reset response has no {name!r} array")
        array = np.asarray(payload[name])
        if array.ndim != dimensions or array.shape[0] != num_envs:
            raise RuntimeError(
                f"reset {name!r} must have {dimensions} dimensions and a leading "
                f"num_envs={num_envs} axis, got {array.shape}"
            )


def main() -> int:
    args = parse_args()
    transport = transport_from_uri(args.url, request_timeout=args.timeout)
    transport.connect()
    try:
        spec = transport.spec()
        actual_num_envs = int(spec.get("num_envs", 0))
        if actual_num_envs != args.num_envs:
            raise RuntimeError(
                f"server reports num_envs={actual_num_envs}, expected {args.num_envs}"
            )
        for reset_index in range(2):
            started = time.monotonic()
            payload = transport.reset(seed=args.seed)
            _validate_payload(payload, num_envs=args.num_envs)
            elapsed = time.monotonic() - started
            print(
                f"reset {reset_index}: seed={args.seed} num_envs={args.num_envs} "
                f"elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
