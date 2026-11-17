# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeding determinism: actor init, first sampled latent, buffer sampling.

Reproducibility is the debugging lifeline for long RL runs: under a fixed
seed, (a) freshly built actors must be parameter-identical, (b) the first
stochastic latent an actor samples must be identical, and (c) replay-buffer
sampling must draw the same batch. Different seeds must actually diversify.
"""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from conftest import make_transition

from isaac_rl_posttraining.buffers import ReplayBuffer, UniformSampler
from isaac_rl_posttraining.modules import (
    TransformerActor,
    TransformerActorConfig,
)

STATE_DIM = 8
EMBED_DIM = 16
ACTION_DIM = 35
CHUNK_SIZE = 4


def _spaces():
    obs_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                -np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32
            ),
            "dino_embedding": gym.spaces.Box(
                -np.inf, np.inf, shape=(EMBED_DIM,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.Box(-2.0, 2.0, shape=(ACTION_DIM,), dtype=np.float32)
    return obs_space, action_space


def _build_actor():
    """Construct the ported transformer actor with a minimal offline-friendly config."""
    config_cls = TransformerActorConfig
    actor_cls = TransformerActor
    obs_space, action_space = _spaces()
    wanted = {
        "observation_space": obs_space,
        "action_space": action_space,
        "chunk_size": CHUNK_SIZE,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 1,
        # Production configs always give the extractor parameters via the
        # pre-transformer feature MLP (transformer.yaml: feature_hidden_dims).
        "feature_hidden_dims": [32],
        # Keep the build hermetic: no language encoder download, no image tower.
        "use_language_embeddings": False,
        "image_encoder_type": None,
        "featurizer": None,
    }
    try:
        fields = inspect.signature(config_cls).parameters
    except (TypeError, ValueError):
        fields = {}
    kwargs = {k: v for k, v in wanted.items() if not fields or k in fields}
    for required in ("observation_space", "action_space"):
        if fields and required not in fields:
            pytest.fail(
                f"API mismatch: {config_cls.__name__} lost its {required} field; "
                f"signature: {sorted(fields)}",
                pytrace=False,
            )
    config = config_cls(**kwargs)
    return actor_cls(config)


def _seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _obs_batch():
    return {
        "state": torch.zeros(1, STATE_DIM),
        "dino_embedding": torch.zeros(1, EMBED_DIM),
    }


def test_actor_init_reproducible_under_seed():
    """Two actors built under the same torch seed are parameter-identical."""
    _seed_all(123)
    actor_a = _build_actor()
    _seed_all(123)
    actor_b = _build_actor()

    params_a = dict(actor_a.named_parameters())
    params_b = dict(actor_b.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name, value in params_a.items():
        assert torch.equal(value, params_b[name]), (
            f"parameter {name} differs under equal seeds"
        )

    _seed_all(321)
    actor_c = _build_actor()
    assert any(
        not torch.equal(params_a[name], p) for name, p in actor_c.named_parameters()
    ), "different seeds must produce different actor initializations"


def test_first_sampled_latent_reproducible():
    """Under a fixed seed, the first stochastic latent sample is bit-reproducible."""

    def first_latent(seed):
        _seed_all(seed)
        actor = _build_actor()
        actor.eval()
        with torch.no_grad():
            latent, _ = actor.act(_obs_batch(), deterministic=False)
        return latent.detach().cpu().numpy()

    latent_1 = first_latent(7)
    latent_2 = first_latent(7)
    np.testing.assert_array_equal(latent_1, latent_2)
    assert np.all(np.isfinite(latent_1))

    latent_other = first_latent(8)
    assert not np.array_equal(latent_1, latent_other), (
        "different seeds must draw different first latents (the actor is stochastic)"
    )


def test_buffer_sampling_reproducible():
    """Replay-buffer sampling is reproducible under a fixed RNG seed."""
    buffer = ReplayBuffer(capacity=128)
    for i in range(40):
        buffer.add(
            make_transition(
                episode_id=i // 10,
                step_in_episode=i % 10,
                obs={"state": np.full((STATE_DIM,), float(i), dtype=np.float32)},
                action=np.full((3,), float(i), dtype=np.float32),
                reward=float(i),
                next_obs={
                    "state": np.full((STATE_DIM,), float(i + 1), dtype=np.float32)
                },
                done=(i % 10 == 9),
                info={"success": False},
            )
        )
    assert len(buffer) == 40

    def draw(seed):
        # A sampler seeds its own stream from the global one, so seeding
        # before building it reproduces the draw.
        _seed_all(seed)
        return UniformSampler(buffer).sample(8)["reward"].tolist()

    draw_a = draw(5)
    draw_b = draw(5)
    assert draw_a == draw_b, "same seed must reproduce the same sampled batch"

    draw_c = draw(6)
    assert draw_a != draw_c, (
        "different seeds must (overwhelmingly) sample different batches"
    )


def test_seed_everything_makes_torch_deterministic():
    """training.seed must actually seed the RNGs (it was cosmetic once)."""
    from isaac_rl_posttraining.utils.training_utils import seed_everything

    seed_everything(123)
    first = torch.randn(4)
    seed_everything(123)
    assert torch.equal(first, torch.randn(4))
