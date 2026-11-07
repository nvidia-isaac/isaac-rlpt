#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve GR00T N1.7 with an explicit, bounded flow-latent control option.

The public GR00T RoboLab server accepts decoded-action requests only.
This development server keeps its normal endpoint and flat simulation wrapper,
but recognizes an optional ``options[\"harness_latent\"]`` tensor.  It clips
that tensor once at the server boundary, substitutes it for the flow head's
initial Gaussian noise, and returns clipping provenance in the policy info.

This is a steering smoke-test endpoint, not a claim about task performance.
The stock server remains the default and the normal no-op harness never
imports this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from simulators.robolab.runtime.steering import bound_latent


@dataclass
class ServerConfig:
    """Configuration for a bounded GR00T latent-control endpoint."""

    model_path: str = "nvidia/GR00T-N1.7-DROID"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 5555
    latent_clip: float = 3.0
    additive_latent: bool = True


class LatentGr00tPolicy:
    """Factory wrapper that adds latent injection to an external Gr00TPolicy.

    The implementation subclasses the external policy at runtime so importing
    this source repository still does not require the GR00T package.
    """

    @staticmethod
    def create(config: ServerConfig):
        """Construct a policy whose ``options`` can carry ``harness_latent``."""
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        make_latent_config, build_base_noise, randn_override = _flow_helpers()

        latent_config = make_latent_config(
            latent_clip=config.latent_clip,
            additive=config.additive_latent,
        )

        class _Policy(Gr00tPolicy):
            def _get_action(
                self, observation: dict[str, Any], options: dict[str, Any] | None = None
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                options = options or {}
                seed_context = _seeded_rng(options.get("harness_seed"), self.model.device)
                raw_latent = options.get("harness_latent")
                if raw_latent is None:
                    with seed_context:
                        actions, info = super()._get_action(observation, options)
                    return actions, {**info, "harness_latent": None}

                bounded = bound_latent(raw_latent, clip=latent_config.latent_clip)
                batch_size = _batch_size(observation)
                if bounded.effective.ndim > 1 and bounded.effective.shape[0] != batch_size:
                    raise ValueError(
                        "latent batch dimension must match observation batch; "
                        f"got {bounded.effective.shape[0]} and {batch_size}"
                    )
                action_head = self.model.action_head
                if bounded.effective.shape[-1] > action_head.action_dim:
                    raise ValueError(
                        "latent width exceeds GR00T flow action width; "
                        f"got {bounded.effective.shape[-1]} and {action_head.action_dim}"
                    )
                latent = torch.as_tensor(
                    bounded.effective,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
                with seed_context:
                    noise = build_base_noise(
                        latent,
                        horizon=action_head.action_horizon,
                        action_dim=action_head.action_dim,
                        config=latent_config,
                    )
                    with randn_override(noise, tuple(noise.shape)):
                        actions, info = super()._get_action(observation, options)
                return actions, {
                    **info,
                    "harness_latent": bounded.metadata(additive=latent_config.additive),
                }

        return _Policy(
            embodiment_tag=EmbodimentTag.resolve(config.embodiment_tag),
            model_path=config.model_path,
            device=config.device,
            strict=True,
        )


def _batch_size(observation: Mapping[str, Any]) -> int:
    """Read batch size from either the public flat or wrapped nested observation.

    ``Gr00tSimPolicyWrapper`` converts the public flat request into nested
    ``video``/``state``/``language`` dictionaries before it calls the inner
    ``Gr00tPolicy``.  The steering hook lives on that inner policy, so it must
    support both shapes without deriving batch size from language strings.
    """
    for value in _leaves(observation):
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("could not infer batch size from the policy observation")


def _leaves(value: Any):
    """Yield array-bearing leaves while preserving the request's key order."""
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _leaves(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _leaves(nested)
    else:
        yield value


def _flow_helpers():
    """Load the in-repo GR00T flow helper without importing its heavy package init.

    The deployment launcher can ship this one helper file alongside the
    server script.  It remains the source-owned implementation rather than a
    second copy of latent-noise semantics in the deployment artifact.
    """
    module_path = (
        Path(__file__).resolve().parents[3]
        / "isaac_rl_posttraining"
        / "policies"
        / "gr00t"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location("harness_gr00t_flow_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load GR00T flow helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LatentInjectionConfig, module.build_base_noise, module._randn_override


def _seeded_rng(seed: object, device: torch.device):
    """Provide deterministic smoke comparisons only when a seed is requested."""
    if seed is None:
        return nullcontext()
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("harness_seed must be an integer when provided")
    device_index = device.index if device.type == "cuda" and device.index is not None else 0
    context = torch.random.fork_rng(devices=[device_index], enabled=device.type == "cuda")

    class _SeededContext:
        def __enter__(self):
            context.__enter__()
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        def __exit__(self, *args):
            return context.__exit__(*args)

    return _SeededContext()


def main(config: ServerConfig) -> None:
    """Start a standard GR00T PolicyServer with the latent-aware policy."""
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = LatentGr00tPolicy.create(config)
    sim_policy = Gr00tSimPolicyWrapper(policy)
    with PolicyServer(policy=sim_policy, host=config.host, port=config.port) as server:
        server.run()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ServerConfig))
