# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared steering decision step: one implementation for both workers.

Pins that collection and evaluation delegate their observation projection,
latent selection, and frozen decode to the same `SteeringStep` class — the
guard against train/eval semantic drift.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaac_rl_posttraining.rollouts.steering_step import SteeringStep


class _RecordingBaseModel:
    def __init__(self):
        self.calls = []

    def infer(self, observations, noise):
        self.calls.append(noise)
        return {"actions": np.zeros((1, 4, 3), dtype=np.float32)}


class _ConstantActor:
    def act(self, obs, deterministic=False, actor_state=None):
        return torch.full((1, 2, 3), 0.5), actor_state


def _step(base_model=None):
    return SteeringStep(
        base_model=base_model or _RecordingBaseModel(),
        device=torch.device("cpu"),
        steering_key_mapping={"state": "state", "language": None, "image": ["image"]},
    )


def _obs():
    return {
        "state": np.ones((1, 5), dtype=np.float32),
        "dino_embedding": np.full((1, 7), 2.0, dtype=np.float32),
    }


def test_process_obs_collector_and_evaluator_modes_agree():
    """The two workers' projections differ only by the squeeze flag."""
    step = _step()
    collector_view = step.process_obs(_obs(), squeeze=False, vlm_features="off")
    evaluator_view = step.process_obs(_obs(), squeeze=True, vlm_features="probe")
    assert torch.equal(collector_view["state"].squeeze(0), evaluator_view["state"])
    assert torch.equal(
        collector_view["dino_embedding"].squeeze(0), evaluator_view["dino_embedding"]
    )
    assert set(collector_view) == set(
        evaluator_view
    )  # probe without get_features adds nothing


def test_process_obs_carries_critic_image():
    """Pixel-critic frames must reach the actor/critic input dict."""
    step = _step()
    obs = dict(_obs(), critic_image=np.full((1, 8, 8, 3), 9, dtype=np.uint8))
    view = step.process_obs(obs, squeeze=False, vlm_features="off")
    assert view["critic_image"].shape == (1, 8, 8, 3)
    squeezed = step.process_obs(obs, squeeze=True, vlm_features="off")
    assert squeezed["critic_image"].shape == (8, 8, 3)
    assert "critic_image" not in step.process_obs(_obs(), vlm_features="off")


def test_decode_passes_latents_through_unchanged():
    base_model = _RecordingBaseModel()
    step = _step(base_model)
    latents = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    actions = step.decode(_obs(), latents)
    assert actions.shape == (1, 4, 3)
    assert np.array_equal(base_model.calls[0], latents)
    step.decode(_obs(), None)
    assert base_model.calls[1] is None  # native mode reaches the base model as None


def test_select_latent_returns_numpy_and_state():
    step = _step()
    obs = step.process_obs(_obs())
    latents, state = step.select_latent(
        _ConstantActor(), obs, deterministic=True, actor_state="s0"
    )
    assert isinstance(latents, np.ndarray) and latents.shape == (1, 2, 3)
    assert state == "s0"
