# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve the texture RNG stream while batching global material writes."""

import math
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from example_workflows.v2d_box_lift_posttraining.reset_events import (
    BatchedTextureReset,
    coalesce_texture_resets,
)


class TextureEvent:
    def __init__(self, generator):
        self.material_prims = list(range(7552))
        self.texture_rng = SimpleNamespace(generator=generator)
        self.writes = []

    def __call__(
        self, env, env_ids, event_name, asset_cfg, texture_paths, texture_rotation
    ):
        textures = self.texture_rng.generator.choice(
            texture_paths, size=len(self.material_prims)
        )
        rotations = self.texture_rng.generator.uniform(
            math.degrees(texture_rotation[0]),
            math.degrees(texture_rotation[1]),
            size=len(self.material_prims),
        )
        self.writes.append((textures, rotations))


@pytest.mark.parametrize("count", [1, 2, 128])
def test_coalescing_preserves_final_materials_and_subsequent_rng(count):
    reference_rng = np.random.default_rng(2)
    batched_rng = np.random.default_rng(2)
    reference_events = [TextureEvent(reference_rng), TextureEvent(reference_rng)]
    batched_events = [BatchedTextureReset(TextureEvent(batched_rng)) for _ in range(2)]
    for index in range(count):
        for reference, batched in zip(reference_events, batched_events, strict=True):
            batched.apply_materials = index == count - 1
            arguments = (
                None,
                [index],
                "texture",
                None,
                ["a.png", "b.png"],
                (-0.1, 0.2),
            )
            reference(*arguments)
            batched(*arguments)
        np.testing.assert_array_equal(reference_rng.random(7), batched_rng.random(7))

    for reference, batched in zip(reference_events, batched_events, strict=True):
        assert len(reference.writes) == count
        assert len(batched.event.writes) == 1
        for expected, actual in zip(
            reference.writes[-1], batched.event.writes[-1], strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    assert reference_rng.bit_generator.state == batched_rng.bit_generator.state


@dataclass
class EventConfig:
    func: TextureEvent
    params: dict[str, list[str]]
    mode: str = "reset"


def test_install_uses_public_manager_api_and_preserves_other_events(monkeypatch):
    module = ModuleType("isaaclab.envs.mdp")
    module.randomize_visual_texture_material = TextureEvent
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", module)
    original = TextureEvent(np.random.default_rng(2))
    config = EventConfig(original, {"texture_paths": ["a.png"]})
    unrelated = SimpleNamespace(func=lambda: None)
    terms = {"texture": config, "unrelated": unrelated}
    manager = SimpleNamespace(
        active_terms={"reset": list(terms)},
        get_term_cfg=terms.__getitem__,
        set_term_cfg=terms.__setitem__,
    )

    events = coalesce_texture_resets(SimpleNamespace(event_manager=manager))

    assert len(events) == 1
    assert events[0].event is original
    assert terms["texture"].func is events[0]
    assert terms["texture"].params == config.params
    assert terms["texture"].mode == "reset"
    assert config.func is original
    assert terms["unrelated"] is unrelated
