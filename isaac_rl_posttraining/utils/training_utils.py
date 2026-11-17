# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training assembly helpers: model construction, checkpoints, buffers."""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
from omegaconf import OmegaConf

from isaac_rl_posttraining.algorithms.base import BaseAlgorithm
from isaac_rl_posttraining.algorithms.replay_learner import load_runtime_checkpoint
from isaac_rl_posttraining.buffers import (
    MixedSampler,
    ReplayBuffer,
    RewardAggregation,
    Sampler,
    SuccessFractionSampler,
    UniformSampler,
    load_buffer,
)
from isaac_rl_posttraining.modules.mlp import MLPActor, MLPActorConfig, MLPCritic, MLPCriticConfig
from isaac_rl_posttraining.modules.rnn import RNNActor, RNNActorConfig, RNNCritic, RNNCriticConfig
from isaac_rl_posttraining.modules.transformer import (
    TransformerActor,
    TransformerActorConfig,
    TransformerCritic,
    TransformerCriticConfig,
)
from isaac_rl_posttraining.runtimes.learner import LEARNER_CHECKPOINT_FILE
from isaac_rl_posttraining.utils.logging_utils import get_logger

logger = get_logger()

__all__ = ["load_checkpoint", "build_actor_critic_models", "Replay", "create_replay"]

#: Width of the per-key MLP featurizers feeding the actor/critic trunks.
FEATURIZER_DIM = 384


def load_checkpoint(algorithm: BaseAlgorithm, load_dir: str) -> int:
    """Load an algorithm checkpoint and return its step counter.

    Accepts both layouts: the per-component files :meth:`BaseAlgorithm.save`
    writes and the runtime loop's learner checkpoint directory.
    """
    if os.path.isfile(os.path.join(load_dir, LEARNER_CHECKPOINT_FILE)):
        load_runtime_checkpoint(algorithm, load_dir)
    else:
        algorithm.load(load_dir)
    return algorithm.step_counter


def _impala_kwargs(cfg: Any) -> dict[str, Any]:
    """Trainable-image-encoder kwargs from the model config (empty unless enabled)."""
    encoder_type = OmegaConf.select(cfg, "model.image_encoder_type", default=None)
    if encoder_type == "small":
        return {
            "image_encoder_type": "small",
            "small_bottleneck_dim": int(
                OmegaConf.select(cfg, "model.small_bottleneck_dim", default=50)
            ),
        }
    if encoder_type != "impala":
        return {}
    return {
        "image_encoder_type": "impala",
        "impala_nn_scale": OmegaConf.select(cfg, "model.impala_nn_scale", default=1),
        "impala_num_blocks_per_stack": OmegaConf.select(cfg, "model.impala_num_blocks_per_stack", default=2),
        "impala_use_smaller": OmegaConf.select(cfg, "model.impala_use_smaller", default=False),
        "impala_bounded_head": OmegaConf.select(cfg, "model.impala_bounded_head", default=False),
        "impala_output_dim": FEATURIZER_DIM,
    }


def build_actor_critic_models(
    observation_space: gym.Space,
    action_space: gym.Space,
    cfg: Any,
    device: Any,
    remove_obs_keys: Optional[list[str]] = None,
) -> tuple[Any, Any, Any]:
    """Build the actor, Q-critic, and V-network for the configured family.

    Architecture selection: MLP when ``training.chunk_size`` is null,
    otherwise RNN (``training.use_rnn=true``) or transformer (the default
    chunked architecture).

    Args:
        observation_space: Dict observation space of the (latent) env.
        action_space: (Latent) action space.
        cfg: Composed Hydra config with ``training``/``policy``/
            ``value_function``/``model`` sections.
        device: Device the models are moved to.
        remove_obs_keys: Observation keys the networks must ignore.

    Returns:
        Tuple ``(actor, critic, v_net)``.
    """
    new_observation_space = copy.deepcopy(observation_space)
    if remove_obs_keys is not None:
        for key in remove_obs_keys:
            new_observation_space.spaces.pop(key, None)
    logger.info(f"Actor/critic observation space (after removing {remove_obs_keys}): {new_observation_space}")

    featurizer_dim = OmegaConf.select(cfg, "model.featurizer_dim", default=None)
    featurizer_dim = FEATURIZER_DIM if featurizer_dim is None else int(featurizer_dim)
    featurizer = {key: [featurizer_dim] for key in new_observation_space.spaces}
    impala_config = _impala_kwargs(cfg)

    policy_cfg = cfg.policy
    value_cfg = cfg.value_function

    if cfg.training.chunk_size is None:
        policy_kwargs = OmegaConf.to_container(policy_cfg.mlp, resolve=True)
        policy_kwargs["hidden_dims"] = tuple(policy_kwargs["hidden_dims"])
        policy_kwargs.update(
            {
                "use_tanh_output": policy_cfg.use_tanh_output,
                "deterministic": policy_cfg.deterministic,
                "log_std_init": policy_cfg.log_std_init,
                "log_std_min": policy_cfg.log_std_min,
                "log_std_max": policy_cfg.log_std_max,
                "official_policy_init": policy_cfg.official_policy_init,
            }
        )
        actor = MLPActor(
            MLPActorConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                preprocess_obs_transform=None,
                featurizer=featurizer,
                remove_obs_keys=remove_obs_keys,
                **policy_kwargs,
                **impala_config,
            )
        )

        value_kwargs = OmegaConf.to_container(value_cfg.mlp, resolve=True)
        value_kwargs["hidden_dims"] = tuple(value_kwargs["hidden_dims"])
        critic_config = MLPCriticConfig(
            observation_space=new_observation_space,
            action_space=action_space,
            preprocess_obs_transform=None,
            featurizer=featurizer,
            use_action=True,
            remove_obs_keys=remove_obs_keys,
            **value_kwargs,
            **impala_config,
        )
        critic = MLPCritic(critic_config)
        v_net_config = dataclasses.replace(critic_config, use_action=False, use_layer_norm=False)
        v_net = MLPCritic(v_net_config)
    elif cfg.training.use_rnn:
        rnn_policy_kwargs = OmegaConf.to_container(policy_cfg.rnn, resolve=True)
        rnn_policy_kwargs.update(
            {
                "log_std_init": policy_cfg.log_std_init,
                "log_std_min": policy_cfg.log_std_min,
                "log_std_max": policy_cfg.log_std_max,
            }
        )
        actor = RNNActor(
            RNNActorConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                chunk_size=cfg.training.chunk_size,
                remove_obs_keys=remove_obs_keys,
                featurizer=featurizer,
                **rnn_policy_kwargs,
                **impala_config,
            )
        )
        rnn_value_kwargs = OmegaConf.to_container(value_cfg.rnn, resolve=True)
        critic_config = RNNCriticConfig(
            observation_space=new_observation_space,
            action_space=action_space,
            chunk_size=cfg.training.chunk_size,
            remove_obs_keys=remove_obs_keys,
            featurizer=featurizer,
            **rnn_value_kwargs,
            **impala_config,
        )
        critic = RNNCritic(critic_config)
        v_net_config = dataclasses.replace(critic_config, use_action=False, use_layer_norm=False)
        v_net = RNNCritic(v_net_config)
    else:
        transformer_policy_kwargs = OmegaConf.to_container(policy_cfg.transformer, resolve=True)
        transformer_policy_kwargs.update(
            {
                "log_std_init": policy_cfg.log_std_init,
                "log_std_min": policy_cfg.log_std_min,
                "log_std_max": policy_cfg.log_std_max,
                "use_tanh_output": policy_cfg.use_tanh_output,
                "official_policy_init": policy_cfg.official_policy_init,
            }
        )
        # The actor has no pooled output; the pooling knob is critic-only.
        transformer_policy_kwargs.pop("pooling_strategy", None)
        actor = TransformerActor(
            TransformerActorConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                featurizer=featurizer,
                chunk_size=cfg.training.chunk_size,
                dinov2_model=None,
                dinov2_processor=None,
                use_language_embeddings=False,
                remove_obs_keys=remove_obs_keys,
                **transformer_policy_kwargs,
                **impala_config,
            )
        )
        transformer_value_kwargs = OmegaConf.to_container(value_cfg.transformer, resolve=True)
        critic_config = TransformerCriticConfig(
            observation_space=new_observation_space,
            action_space=action_space,
            featurizer=featurizer,
            chunk_size=cfg.training.chunk_size,
            dinov2_model=None,
            dinov2_processor=None,
            use_language_embeddings=False,
            remove_obs_keys=remove_obs_keys,
            use_action=True,
            **transformer_value_kwargs,
            **impala_config,
        )
        critic = TransformerCritic(critic_config)
        v_net_config = dataclasses.replace(critic_config, use_action=False, use_layer_norm=False)
        v_net = TransformerCritic(v_net_config)

    actor = actor.to(device)
    critic = critic.to(device)
    v_net = v_net.to(device)

    logger.info(f"Actor: {actor.__class__.__name__}")
    logger.info(f"Critic: {critic.__class__.__name__}")
    logger.info(f"V-Net: {v_net.__class__.__name__}")
    return actor, critic, v_net


@dataclass(frozen=True)
class Replay:
    """What a run trains from: the robot's own episodes, optional demonstrations, and the sampler of the main batch."""

    buffer: ReplayBuffer
    demonstrations: ReplayBuffer | None
    sampler: Sampler


def create_replay(
    *,
    capacity: int,
    drop_components: Sequence[str],
    reward_channel: str,
    reward_aggregation: RewardAggregation,
    success_fraction: float | None,
    demonstrations: str | None,
    demonstration_fraction: float | None,
) -> Replay:
    """Build the replay buffer and the sampler of the learner's main batch from the ``buffer`` config.

    The online buffer is sampled uniformly, or with a fixed share of every
    batch from successful episodes (``success_fraction``). A saved buffer of
    demonstrations (``demonstrations``, a directory ``save_buffer`` wrote) is
    loaded with the learner's own reward settings and mixed into every batch
    at ``demonstration_fraction``; the algorithm's expert-only terms draw
    from it directly.
    """
    buffer = ReplayBuffer(
        capacity,
        drop_components=drop_components,
        reward_channel=reward_channel,
        reward_aggregation=reward_aggregation,
    )
    if success_fraction is None:
        online: Sampler = UniformSampler(buffer)
        logger.info(f"Replay: {capacity} transitions, uniform sampling, reward {reward_channel!r}/{reward_aggregation}")
    else:
        online = SuccessFractionSampler(buffer, float(success_fraction))
        logger.info(
            f"Replay: {capacity} transitions, {float(success_fraction):.0%} of every batch from successful "
            f"episodes, reward {reward_channel!r}/{reward_aggregation}"
        )
    if demonstrations is None:
        if demonstration_fraction is not None:
            raise ValueError("buffer.demonstration_fraction needs buffer.demonstrations")
        return Replay(buffer=buffer, demonstrations=None, sampler=online)
    if demonstration_fraction is None:
        raise ValueError("buffer.demonstrations needs buffer.demonstration_fraction, its share of every batch")
    fraction = float(demonstration_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"buffer.demonstration_fraction must be in (0, 1), got {fraction}")
    demonstration_buffer = load_buffer(
        demonstrations, reward_channel=reward_channel, reward_aggregation=reward_aggregation
    )
    logger.info(
        f"Demonstrations: {len(demonstration_buffer)} transitions from {demonstrations}, "
        f"{fraction:.0%} of every batch"
    )
    sampler = MixedSampler(
        {"online": online, "demonstrations": UniformSampler(demonstration_buffer)},
        {"online": 1.0 - fraction, "demonstrations": fraction},
    )
    return Replay(buffer=buffer, demonstrations=demonstration_buffer, sampler=sampler)


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (incl. CUDA) RNGs for reproducible runs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeded python/numpy/torch RNGs with seed={seed}")


def enable_deterministic_torch() -> None:
    """Require deterministic torch kernels in an isolated policy process."""
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("GR00T_DIT_SDPA_MODE", "math")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for backend, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
        ("enable_math_sdp", True),
    ):
        configure = getattr(torch.backends.cuda, backend, None)
        if callable(configure):
            configure(enabled)


def actor_fingerprint(module: Any) -> Optional[str]:
    """SHA-256 over a module's state_dict tensors (keys sorted).

    The same weights always produce the same digest, so an in-training
    eval, the checkpoint saved at that step, and a later re-eval load can
    be proven to be the same policy by comparing one string. Returns None
    for objects without a ``state_dict`` (e.g. test doubles).
    """
    if not hasattr(module, "state_dict"):
        return None
    import hashlib

    import torch

    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode())
        digest.update(value.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
