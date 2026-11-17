# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coalesce full-scene texture writes while preserving every random draw."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp import randomize_visual_texture_material
    from isaaclab.managers import SceneEntityCfg
    from torch import Tensor


@dataclass
class BatchedTextureReset:
    event: randomize_visual_texture_material
    apply_materials: bool = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if self.apply_materials:
            self.event(
                env, env_ids, event_name, asset_cfg, texture_paths, texture_rotation
            )
            return
        # This Isaac event ignores env_ids and replaces the entire scene's textures.
        # Retain its independent RNG stream even when intermediate writes are omitted.
        count = len(self.event.material_prims)
        generator = self.event.texture_rng.generator
        generator.choice(texture_paths, size=count)
        generator.uniform(
            math.degrees(texture_rotation[0]),
            math.degrees(texture_rotation[1]),
            size=count,
        )


def coalesce_texture_resets(env: ManagerBasedRLEnv) -> tuple[BatchedTextureReset, ...]:
    from isaaclab.envs.mdp import randomize_visual_texture_material

    events = []
    for name in env.event_manager.active_terms["reset"]:
        config = env.event_manager.get_term_cfg(name)
        if isinstance(config.func, randomize_visual_texture_material):
            if not config.params["texture_paths"]:
                raise ValueError(
                    f"Texture reset {name} requires explicit nonempty texture_paths"
                )
            event = BatchedTextureReset(config.func)
            env.event_manager.set_term_cfg(name, replace(config, func=event))
            events.append(event)
    return tuple(events)
