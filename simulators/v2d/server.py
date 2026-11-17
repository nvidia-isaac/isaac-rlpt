# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic V2D/Isaac application and transport-server lifecycle.

The workflow supplies a capability factory. This package starts Isaac Sim,
hosts the TCP protocol, and owns shutdown without importing any particular
task or evaluator.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from simulators.v2d.contract import Payload


class V2DServerCapability(Protocol):
    """A workflow's task behavior behind the reusable V2D server."""

    def spec(self) -> dict[str, object]: ...

    def reset(self, seed: int | None, env_ids: list[int] | None) -> Payload: ...

    def step(self, action: object, *, need_obs: bool) -> Payload: ...

    def close(self) -> None: ...


CapabilityFactory = Callable[[Path, str, int, int], V2DServerCapability]


def load_capability_factory(target: str) -> CapabilityFactory:
    """Resolve an explicit ``module:function`` capability factory."""
    if ":" not in target:
        raise ValueError("capability factory must be written as module:function")
    module_name, attribute = target.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"V2D capability factory {target!r} is not callable")
    return cast(CapabilityFactory, factory)


def _reset_arguments(message: dict[str, object]) -> tuple[int | None, list[int] | None]:
    """The checked ``seed`` and ``env_ids`` of a reset message."""
    seed_value = message["seed"]
    if seed_value is not None and not isinstance(seed_value, int):
        raise TypeError("V2D reset seed must be an integer or null")
    env_ids_value = message.get("env_ids")
    if env_ids_value is not None and not isinstance(env_ids_value, list):
        raise TypeError("V2D env_ids must be a list or null")
    return seed_value, env_ids_value


def _message_handler(
    capability: V2DServerCapability,
) -> Callable[[dict[str, object]], dict[str, object] | None]:
    from isaac_rl_posttraining.envs.transport import (
        MSG_CLOSE,
        MSG_RESET,
        MSG_SPEC,
        MSG_STEP,
    )

    def handle(message: dict[str, object]) -> dict[str, object] | None:
        kind = message["type"]
        if kind == MSG_SPEC:
            return capability.spec()
        if kind == MSG_RESET:
            return capability.reset(*_reset_arguments(message))
        if kind == MSG_STEP:
            return capability.step(
                message["action"], need_obs=bool(message["need_obs"])
            )
        if kind == MSG_CLOSE:
            return None
        raise ValueError(f"Unsupported V2D message {kind!r}")

    return handle


def main() -> None:
    """Start Isaac Sim, construct one workflow capability, and serve it."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--capability-config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind; 0.0.0.0 serves other hosts, so only on a trusted network (frames are pickled)",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--startup-timeout-seconds", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        raise ValueError("V2D num_envs must be positive")
    if args.startup_timeout_seconds <= 60:
        raise ValueError("startup timeout must exceed the 60-second diagnostic lead")
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.startup_timeout_seconds - 60)
    args.enable_cameras = True
    app = AppLauncher(args).app
    capability = None
    try:
        factory = load_capability_factory(args.capability)
        capability = factory(
            args.capability_config.resolve(),
            str(args.device),
            args.seed,
            args.num_envs,
        )
        endpoint_spec = capability.spec()
        receipt = {"capability": args.capability, "endpoint": endpoint_spec}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        from isaac_rl_posttraining.envs.transport import NativeTcpServer

        server = NativeTcpServer(args.host, args.port)
        print(f"[v2d] listening on {server.address}", flush=True)
        faulthandler.cancel_dump_traceback_later()
        server.serve_forever(_message_handler(capability))
    except Exception:
        # Kit shutdown may terminate Python before an unwinding exception is printed.
        traceback.print_exc()
        raise
    finally:
        if capability is not None:
            capability.close()
        app.close()


if __name__ == "__main__":
    main()
