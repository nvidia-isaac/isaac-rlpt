# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, per-environment rejection sampling without stepping physics."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch

from example_workflows.v2d_box_lift_posttraining.fixed_reset import repeat_reset_draws
from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)


@dataclass(frozen=True)
class Penetration:
    first: str
    second: str
    depth_m: float


class ResetSampler:
    def __init__(
        self,
        num_envs: int,
        *,
        seed: int,
        fixed: bool,
        device: torch.device,
        config: ResetValidationConfig,
    ) -> None:
        if not config.enabled:
            raise ValueError("ResetSampler requires enabled collision validation")
        self.seeds = [seed] * num_envs
        self.fixed = fixed
        self.device = device
        self.config = config
        self.next_candidates = [0] * num_envs
        self.accepted_seeds: list[int | None] = [None] * num_envs
        self.attempts = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self.rejections = torch.zeros_like(self.attempts)
        self.max_penetration = torch.zeros(num_envs, device=device)

    def candidate_seed(self, env_index: int) -> int:
        candidate = self.next_candidates[env_index]
        self.next_candidates[env_index] += 1
        if candidate == 0:
            return (self.seeds[env_index] + env_index) % 2**63
        key = f"{self.seeds[env_index]}:{env_index}:{candidate}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % 2**63

    def reseed(self, env_ids: torch.Tensor | None, seed: int) -> None:
        """Honor explicit random-reset seeds; fixed resets retain their server seed."""
        if self.fixed:
            return
        indices = range(len(self.seeds)) if env_ids is None else env_ids.tolist()
        for env_index in indices:
            self.seeds[env_index] = seed
            self.next_candidates[env_index] = 0
            self.accepted_seeds[env_index] = None

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        sample: Callable[[torch.Tensor], None],
        check: Callable[[int], list[Penetration]],
    ) -> None:
        for env_index in env_ids.tolist():
            self.attempts[env_index] = 0
            self.rejections[env_index] = 0
            self.max_penetration[env_index] = 0
            draw_seed = self._draw_valid_reset(env_index, sample, check)
            if self.rejections[env_index]:
                self._log_rejections(env_index, draw_seed)

    def _draw_valid_reset(
        self,
        env_index: int,
        sample: Callable[[torch.Tensor], None],
        check: Callable[[int], list[Penetration]],
    ) -> int:
        """Draw resets for one environment until one is collision-free; return its seed."""
        accepted_seed = self.accepted_seeds[env_index]
        cached = self.fixed and accepted_seed is not None
        limit = 1 if cached else self.config.max_attempts
        contacts: list[Penetration] = []
        for _ in range(limit):
            if cached:
                assert accepted_seed is not None
                draw_seed = accepted_seed
            else:
                draw_seed = self.candidate_seed(env_index)
            ids = torch.tensor([env_index], device=self.device, dtype=torch.int64)
            # repeat_reset_draws adds the environment ID to its seed.
            repeat_reset_draws(
                sample, ids, seed=draw_seed - env_index, device=self.device
            )
            self.attempts[env_index] += 1
            contacts = check(env_index)
            deepest = max((contact.depth_m for contact in contacts), default=0.0)
            self.max_penetration[env_index] = max(
                float(self.max_penetration[env_index]), deepest
            )
            if not contacts:
                self.accepted_seeds[env_index] = draw_seed
                return draw_seed
            self.rejections[env_index] += 1
        raise RuntimeError(
            f"No valid reset for environment {env_index} "
            f"after {limit} attempts "
            f"(fixed_cached={cached}): {contacts}"
        )

    def _log_rejections(self, env_index: int, draw_seed: int) -> None:
        print(
            "RESET_VALIDATION "
            + json.dumps(
                {
                    "environment_index": env_index,
                    "accepted_seed": draw_seed,
                    "attempts": int(self.attempts[env_index]),
                    "rejections": int(self.rejections[env_index]),
                    "max_rejected_penetration_m": float(
                        self.max_penetration[env_index]
                    ),
                }
            ),
            flush=True,
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "reset_attempts": self.attempts.clone(),
            "reset_rejections": self.rejections.clone(),
            "reset_rejection_fraction": self.rejections / self.attempts.clamp_min(1),
            "reset_max_rejected_penetration_m": self.max_penetration.clone(),
        }
