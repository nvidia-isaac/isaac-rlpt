# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-input critic pieces: critic_image emission and random-shift aug.

Pins the reference-parity mechanics (learned pixel encoders regularized by
DrQ-style augmentation): the DINO wrapper's downsized ``critic_image``
entry, its observation-space declaration, and the augmentation's
shape/dtype/range contract.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")


def test_spaces_with_embedding_adds_critic_image():
    from isaac_rl_posttraining.envs.dino_wrapper import _spaces_with_embedding

    base = gym.spaces.Dict(
        {"state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)}
    )
    space = _spaces_with_embedding(base, 384, critic_image_size=64)
    assert space["critic_image"].shape == (64, 64, 3)
    assert space["critic_image"].dtype == np.uint8
    assert "critic_image" not in _spaces_with_embedding(base, 384).spaces


def test_resize_uint8_downsizes_any_layout():
    from isaac_rl_posttraining.envs.dino_wrapper import _resize_uint8

    out = _resize_uint8(np.full((224, 224, 3), 200, dtype=np.uint8), 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert int(out.mean()) == 200  # constant image survives resizing
    chw = _resize_uint8(np.zeros((3, 128, 128), dtype=np.uint8), 64)
    assert chw.shape == (64, 64, 3)


def test_vector_wrapper_emits_critic_image():
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper

    class _Inner(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "observation/image": gym.spaces.Box(
                    low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    class _FakeDino(torch.nn.Module):
        class config:
            hidden_size = 16

        def __init__(self):
            super().__init__()
            # embedding_utils reads the device off the first parameter.
            self._device_anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values=None, **kwargs):
            out = type("O", (), {})()
            out.pooler_output = torch.zeros((pixel_values.shape[0], 16))
            out.last_hidden_state = torch.zeros((pixel_values.shape[0], 4, 16))
            return out

    class _Batch(dict):
        def to(self, *args, **kwargs):  # mimic transformers BatchFeature
            return self

    class _FakeProcessor:
        def __call__(self, images=None, return_tensors=None):
            arr = (
                images
                if torch.is_tensor(images)
                else torch.as_tensor(np.asarray(images, dtype=np.float32))
            )
            if arr.dim() == 3:
                arr = arr[None]
            if arr.shape[-1] == 3:  # HWC -> CHW
                arr = arr.permute(0, 3, 1, 2)
            return _Batch(pixel_values=arr.float())

    venv = gym.vector.SyncVectorEnv([_Inner for _ in range(2)])
    wrapper = VectorDinoEmbeddingWrapper(
        venv,
        _FakeDino(),
        _FakeProcessor(),
        device="cpu",
        image_keys=["observation/image"],
        critic_image_size=64,
    )
    assert wrapper.single_observation_space["critic_image"].shape == (64, 64, 3)
    obs, _ = wrapper.reset()
    assert obs["critic_image"].shape == (2, 64, 64, 3)
    assert obs["critic_image"].dtype == np.uint8
    assert obs["dino_embedding"].shape == (2, 16)


def test_mlp_actor_critic_forward_with_impala_pixels():
    """The pixel-parity stack end to end: MLP actor/critic with trainable
    Impala encoders over ``critic_image``, alongside vector obs keys."""
    from isaac_rl_posttraining.modules.mlp.actor import MLPActor, MLPActorConfig
    from isaac_rl_posttraining.modules.mlp.critic import MLPCritic, MLPCriticConfig

    obs_space = gym.spaces.Dict(
        {
            "critic_image": gym.spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            ),
            "dino_embedding": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32
            ),
            "state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
    common = {
        "observation_space": obs_space,
        "action_space": action_space,
        "featurizer": {key: [64] for key in obs_space.spaces},
        "image_encoder_type": "impala",
        "impala_use_smaller": True,
        "impala_output_dim": 64,
        "hidden_dims": (64, 64),
    }
    batch = {
        "critic_image": torch.randint(
            0, 256, (3, 64, 64, 3), dtype=torch.uint8
        ).float(),
        "dino_embedding": torch.zeros(3, 384),
        "state": torch.zeros(3, 8),
    }

    actor = MLPActor(MLPActorConfig(**common))
    mean, log_std, _ = actor.get_action_dist_params(batch)
    assert mean.shape == (3, 32) and log_std.shape == (3, 32)
    assert torch.isfinite(mean).all()

    critic = MLPCritic(MLPCriticConfig(use_action=True, **common))
    q = critic(batch, torch.zeros(3, 32))
    q_tensor = q[0] if isinstance(q, (tuple, list)) else q
    assert torch.isfinite(torch.as_tensor(q_tensor)).all()

    # Encoder parameters must be trainable (the whole point of parity).
    image_params = [
        p for n, p in actor.named_parameters() if "image" in n or "impala" in n.lower()
    ]
    assert any(p.requires_grad for p in image_params)


def test_random_shift_images_contract():
    from isaac_rl_posttraining.modules.augmentations import random_shift_images

    images = torch.randint(0, 256, (5, 64, 64, 3), dtype=torch.uint8)
    out = random_shift_images(images, pad=4)
    assert out.shape == (5, 64, 64, 3)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 255.0

    # A constant image is invariant under replicate-padded shifts.
    const = torch.full((2, 64, 64, 3), 111.0)
    np.testing.assert_allclose(random_shift_images(const).numpy(), 111.0, atol=1e-3)

    with pytest.raises(ValueError):
        random_shift_images(torch.zeros(4, 64, 3))  # missing batch dim
    with pytest.raises(ValueError):
        random_shift_images(torch.zeros(1, 64, 32, 3))  # non-square


def test_video_key_detection_skips_critic_image():
    """critic_image is a learner input, never a video stream."""
    from isaac_rl_posttraining.rollouts.video import detect_image_keys

    obs = {
        "observation/image": np.zeros((224, 224, 3)),
        "critic_image": np.zeros((64, 64, 3)),
        "state": np.zeros(8),
    }
    assert detect_image_keys(list(obs)) == ["observation/image"]


def test_impala_bounded_head_bounds_features():
    """The official-style encoder head (Linear+LayerNorm+tanh) keeps
    features in [-1, 1]; the default head is unbounded."""
    from isaac_rl_posttraining.modules.cnn import SmallerImpalaEncoder

    x = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float()
    bounded = SmallerImpalaEncoder(
        input_shape=(3, 64, 64), output_dim=50, bounded_head=True
    )
    out = bounded(x)
    assert out.shape == (2, 50)
    assert out.abs().max() <= 1.0

    plain = SmallerImpalaEncoder(input_shape=(3, 64, 64), output_dim=50)
    assert plain(x).shape == (2, 50)


def test_color_jitter_contract():
    from isaac_rl_posttraining.modules.augmentations import color_jitter_images

    torch.manual_seed(0)
    images = torch.randint(0, 256, (6, 32, 32, 3), dtype=torch.uint8)
    out = color_jitter_images(images)
    assert out.shape == images.shape and out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 255.0
    assert not torch.equal(out, images.float())  # something jittered

    # apply_prob=0 must be an exact no-op (float passthrough).
    same = color_jitter_images(images, apply_prob=0.0)
    torch.testing.assert_close(same, images.float())

    with pytest.raises(ValueError):
        color_jitter_images(torch.zeros(32, 32, 3))


def test_squashed_log_prob_forward_form_matches_and_fixes_saturation():
    from isaac_rl_posttraining.modules.base.distributions import (
        SquashedDiagGaussianDistribution,
    )

    d = SquashedDiagGaussianDistribution(3)
    torch.manual_seed(1)
    # Moderate regime: forward form equals the eps form to high precision.
    mean, log_std = torch.randn(4, 3) * 0.5, torch.full((4, 3), -0.5)
    actions, lp = d.log_prob_from_params(mean, log_std)
    ga = d.gaussian_actions
    lp_eps = d.distribution.log_prob(ga).sum(-1) - torch.log(1 - actions**2 + 1e-6).sum(
        -1
    )
    torch.testing.assert_close(lp, lp_eps, atol=0.05, rtol=0.01)

    # Saturated regime: the eps form saturates at ~ -log(eps) per dim while
    # the forward form keeps growing with |u|.
    mean = torch.tensor([[25.0, -25.0, 25.0]])
    actions, lp = d.log_prob_from_params(mean, log_std[:1])
    ga = d.gaussian_actions
    lp_eps = d.distribution.log_prob(ga).sum(-1) - torch.log(1 - actions**2 + 1e-6).sum(
        -1
    )
    assert lp.item() > lp_eps.item() + 50  # eps form under-reads by 50+ nats


def test_small_conv_encoder_matches_reference_spec():
    from isaac_rl_posttraining.modules.cnn import SmallConvEncoder

    enc = SmallConvEncoder((64, 64, 3), bottleneck_dim=50)
    x = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float()
    out = enc(x)
    assert enc.output_dim == 50 and out.shape == (2, 50)
    assert out.abs().max() <= 1.0  # LayerNorm + tanh bottleneck
    convs = [m for m in enc.convs if isinstance(m, torch.nn.Conv2d)]
    assert [c.stride[0] for c in convs] == [2, 1, 1, 1]
    assert all(c.out_channels == 32 and c.padding == (0, 0) for c in convs)


def test_official_policy_init_starts_at_prior():
    from isaac_rl_posttraining.modules.mlp.actor import MLPActor, MLPActorConfig

    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
    actor = MLPActor(
        MLPActorConfig(
            observation_space=obs_space,
            action_space=action_space,
            featurizer={"state": [64]},
            hidden_dims=(128, 128, 128),
            official_policy_init=True,
            log_std_init=-2.0,
        )
    )
    batch = {"state": torch.randn(64, 8)}
    mean, log_std, _ = actor.get_action_dist_params(batch)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(log_std, torch.full_like(log_std, -2.0))


def test_transformer_stack_counts_flattened_keys_and_bounds_features():
    """Chunked actor/critic width accounting includes non-image keys.

    The extractor seeds ``featurizer_cfg`` with only the image encoder, so
    the static width must still count flattened keys (``state``) exactly as
    ``forward`` concatenates them -- otherwise ``obs_projection`` is built
    too narrow and the first batch crashes.
    """
    from isaac_rl_posttraining.modules.transformer.actor import (
        TransformerActor,
        TransformerActorConfig,
    )
    from isaac_rl_posttraining.modules.transformer.critic import (
        TransformerCritic,
        TransformerCriticConfig,
    )

    obs_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
            "critic_image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
        }
    )
    act_space = gym.spaces.Box(-1, 1, (32,), np.float32)
    kwargs = {
        "observation_space": obs_space,
        "action_space": act_space,
        "chunk_size": 50,
        "image_encoder_type": "impala",
        "impala_use_smaller": True,
        "impala_bounded_head": True,
        "impala_output_dim": 256,
        "use_language_embeddings": False,
    }
    actor = TransformerActor(TransformerActorConfig(**kwargs))
    critic = TransformerCritic(TransformerCriticConfig(**kwargs))

    extractor = actor.feature_extractor
    assert extractor.obs_dim == 256 + 8

    obs = {"state": torch.zeros(2, 8), "critic_image": torch.zeros(2, 64, 64, 3)}
    actions, log_prob = actor.action_log_prob(obs)
    assert actions.shape == (2, 50, 32)
    assert log_prob.shape == (2, 50)
    assert critic(obs, actions).shape == (2, 1)

    encoder = extractor.image_encoders["critic_image"]
    features = encoder(obs["critic_image"] + 200.0)
    assert float(features.detach().abs().max()) <= 1.0


def test_transformer_extractor_supports_small_encoder():
    """The chunked stack accepts the reference 'small' pixel encoder."""
    from isaac_rl_posttraining.modules.cnn import SmallConvEncoder
    from isaac_rl_posttraining.modules.transformer.feature_extractor import (
        TransformerFeatureExtractor,
    )

    obs_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
            "critic_image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
        }
    )
    extractor = TransformerFeatureExtractor(
        observation_space=obs_space,
        featurizer_cfg=None,
        feature_hidden_dims=[],
        image_encoder_type="small",
        use_language_embeddings=False,
    )
    assert isinstance(extractor.image_encoders["critic_image"], SmallConvEncoder)
    encoder_dim = extractor.image_encoders["critic_image"].output_dim
    assert extractor.obs_dim == encoder_dim + 8

    obs = {"state": torch.zeros(2, 8), "critic_image": torch.zeros(2, 64, 64, 3)}
    assert extractor(obs).shape == (2, encoder_dim + 8)


def test_vector_wrapper_pixel_only_mode():
    """Without a DINO model the wrapper serves only the downsized critic_image."""
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper

    class _Inner(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "main_image": gym.spaces.Box(
                    low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    vec = gym.vector.SyncVectorEnv([_Inner for _ in range(2)])
    wrapped = VectorDinoEmbeddingWrapper(
        vec,
        dinov2_model=None,
        dinov2_processor=None,
        image_keys=["main_image"],
        critic_image_size=64,
    )
    assert "dino_embedding" not in wrapped.single_observation_space.spaces
    assert wrapped.single_observation_space["critic_image"].shape == (64, 64, 3)
    obs, _ = wrapped.reset()
    assert "dino_embedding" not in obs
    assert obs["critic_image"].shape == (2, 64, 64, 3)
    assert obs["critic_image"].dtype == np.uint8


def test_vector_wrapper_pixel_only_requires_critic_image():
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper

    class _Inner(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "main_image": gym.spaces.Box(
                    low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
                )
            }
        )
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    vec = gym.vector.SyncVectorEnv([_Inner])
    with pytest.raises(ValueError, match="pixel-only"):
        VectorDinoEmbeddingWrapper(vec, dinov2_model=None, dinov2_processor=None)


def test_critic_ensemble_preserves_small_encoder_init():
    """Ensemble copies re-run the encoder's bespoke init, not PyTorch defaults."""
    from isaac_rl_posttraining.algorithms.sac import SAC
    from isaac_rl_posttraining.modules.cnn import SmallConvEncoder

    def _assert_orthogonal_rows(module: torch.nn.Conv2d, label: str) -> None:
        w = module.weight.detach()
        flat = w.reshape(w.shape[0], -1)
        assert flat.shape[0] <= flat.shape[1], "check needs rows <= cols"
        gram = flat @ flat.T
        # Orthogonal init with gain sqrt(2): W W^T = 2 I on the row space.
        assert torch.allclose(gram, 2.0 * torch.eye(w.shape[0]), atol=1e-4), (
            f"{label} lost the orthogonal conv init"
        )

    proto = SmallConvEncoder((64, 64, 3), bottleneck_dim=64)
    # convs[2] is the second Conv2d (32 -> 32): 32 rows x 288 cols.
    _assert_orthogonal_rows(proto.convs[2], "prototype")
    ensemble = SAC._create_critic_ensemble(proto, num_critics=2, share_backbone=False)
    for member in ensemble.critics:
        _assert_orthogonal_rows(member.convs[2], "ensemble member")
