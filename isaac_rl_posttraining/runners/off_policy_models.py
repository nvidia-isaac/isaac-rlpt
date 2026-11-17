# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The learner's environment stand-in and networks of each off-policy paradigm.

The steering and edit families act on a frozen policy in one of three ways.
Latent steering (``dsrl_sac``, ``dsrl_ppo``) acts only in the frozen policy's
flow-noise space; residual flow steering (RFS) adds a bounded residual after
the decode; direct action (SAC-edit) decodes a reference chunk and executes a
bounded edit of it in the raw action space. Each paradigm has its own
environment stand-in (the spaces the learner trains against) and its own
network layout; :func:`paradigm_models` builds both for the selected
algorithm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import algorithm_spec
from isaac_rl_posttraining.algorithms.rfs import (
    RFSActionModulator,
    configure_rfs_actor_distribution,
    initialize_rfs_residual_head,
)
from isaac_rl_posttraining.algorithms.sac_edit import (
    ReferenceEditActor,
    ReferenceEditActorConfig,
)
from isaac_rl_posttraining.envs.reference_action import ReferenceActionEnv
from isaac_rl_posttraining.runtimes.policy_families import PolicyFamily
from isaac_rl_posttraining.utils.training_utils import (
    build_actor_critic_models,
    seed_everything,
)

__all__ = [
    "ParadigmModels",
    "paradigm_models",
    "policy_family",
    "uses_reference_actions",
]

#: The model builders per registered action space: ``latent`` acts only in the
#: frozen flow-noise space (dsrl_ppo builds its actor here too);
#: ``latent_residual`` adds a bounded residual after decode; ``edit`` decodes a
#: reference and executes a bounded edit in the raw action space.
_ROUTED_ACTION_SPACES = ("latent", "latent_residual", "edit")


def _action_space(algorithm_name: str) -> str:
    spec = algorithm_spec(algorithm_name)
    if spec.action_space not in _ROUTED_ACTION_SPACES:
        raise ValueError(
            f"{algorithm_name!r} acts in the {spec.action_space!r} action space; the steering and edit "
            f"models cover {_ROUTED_ACTION_SPACES}"
        )
    return spec.action_space


def policy_family(algorithm_name: str) -> PolicyFamily:
    """The rollout policy family (steering or edit) an off-policy algorithm collects with."""
    return "edit" if _action_space(algorithm_name) == "edit" else "steering"


def uses_reference_actions(algorithm_name: str) -> bool:
    """Whether the actor edits the frozen policy's decoded reference chunk (the ``edit`` action space)."""
    return _action_space(algorithm_name) == "edit"


@dataclass(frozen=True)
class ParadigmModels:
    """What one paradigm builds for the learner: its environment stand-in and networks.

    ``env`` is the controller's local environment (wrapped for reference
    actions under direct action), ``latent_env`` the space the actor acts in,
    and ``remove_obs_keys`` the observation keys the networks drop.
    """

    env: Any
    remove_obs_keys: list[str]
    latent_env: Any
    action_modulator: RFSActionModulator | None
    actor: torch.nn.Module
    critic: torch.nn.Module
    v_net: torch.nn.Module | None


def paradigm_models(
    cfg: DictConfig,
    algorithm_name: str,
    *,
    base_model: Any,
    env: Any,
    remove_obs_keys: list[str],
    algorithm_config: Any,
    device: torch.device,
) -> ParadigmModels:
    """Build the environment stand-in and the networks of ``algorithm_name``'s paradigm.

    ``algorithm_config`` is the algorithm's config built from the composed
    group; the paradigms read the fields that shape their models from it (the
    reference-action key, the latent prior std). The networks are initialized
    from ``training.seed`` regardless of what loading the frozen policy and
    the environment drew from the generators before.
    """
    model_seed = int(OmegaConf.select(cfg, "training.seed", default=0))
    logger.info(f"Reseeding actor/value initialization with seed={model_seed}")
    seed_everything(model_seed)
    action_space = _action_space(algorithm_name)
    if action_space == "edit":
        return _direct_action_models(cfg, base_model, env, algorithm_config, device)
    if action_space == "latent_residual":
        return _residual_flow_steer_models(cfg, base_model, env, remove_obs_keys, device)
    return _latent_steer_models(cfg, base_model, env, remove_obs_keys, device)


def _learner_spaces(env: Any) -> tuple[Any, Any]:
    """The per-environment observation and action spaces of the sizing environment."""
    observation_space = getattr(env, "single_observation_space", None) or env.observation_space
    action_space = getattr(env, "single_action_space", None) or env.action_space
    return observation_space, action_space


def _latent_steer_models(
    cfg: DictConfig,
    base_model: Any,
    env: Any,
    remove_obs_keys: list[str],
    device: torch.device,
) -> ParadigmModels:
    """Build the flow-noise env stand-in and the steering models.

    DSRL acts only in the frozen policy's flow-noise space; execution chunking
    (``dsrl.action_exec_len``) is separate and handled by the rollout workers.
    """
    from isaac_rl_posttraining.envs import LatentSpaceEnv

    observation_space, action_space = _learner_spaces(env)
    latent_env = LatentSpaceEnv(
        observation_space,
        action_space,
        base_model,
        noise_dim=cfg.dsrl.noise_dim,
        chunk_size=cfg.training.chunk_size,
        action_bound=cfg.dsrl.noise_action_bound,
        warmup_exploration=cfg.dsrl.get("warmup_exploration", "uniform"),
        use_vlm_features=cfg.dsrl.get("use_vlm_features", False),
    )
    actor, critic, v_net = build_actor_critic_models(
        latent_env.observation_space, latent_env.action_space, cfg, device, remove_obs_keys
    )
    return ParadigmModels(
        env=env,
        remove_obs_keys=remove_obs_keys,
        latent_env=latent_env,
        action_modulator=None,
        actor=actor,
        critic=critic,
        v_net=v_net,
    )


def _residual_flow_steer_models(
    cfg: DictConfig,
    base_model: Any,
    env: Any,
    remove_obs_keys: list[str],
    device: torch.device,
) -> ParadigmModels:
    """Build the RFS env stand-in, action modulator, and joint steering models.

    RFS acts in a joint ``[flow-noise, residual]`` space: the latent steers the
    frozen sampler and a bounded residual is added after decode, so the
    residual width must match the frozen policy's decoded action. The actor's
    residual branch starts at zero so RFS begins from the same behavior
    distribution as DSRL.
    """
    from isaac_rl_posttraining.envs import ResidualFlowSpaceEnv

    observation_space, action_space = _learner_spaces(env)
    residual_action_dim = int(base_model.action_dim)
    if getattr(action_space, "shape", None) is None:
        raise ValueError(
            f"RFS requires a continuous Box-like environment action space, got {action_space}"
        )
    env_action_dim = int(action_space.shape[-1])
    if residual_action_dim != env_action_dim:
        raise ValueError(
            f"RFS decoded action width {residual_action_dim} does not match "
            f"environment action width {env_action_dim}"
        )
    latent_env = ResidualFlowSpaceEnv(
        observation_space=observation_space,
        action_space=action_space,
        base_model=base_model,
        noise_dim=int(cfg.dsrl.noise_dim),
        residual_action_dim=residual_action_dim,
        chunk_size=cfg.training.chunk_size,
        noise_action_bound=float(cfg.dsrl.noise_action_bound),
        residual_action_bound=float(cfg.rfs.residual_action_bound),
        warmup_exploration=str(cfg.dsrl.get("warmup_exploration", "uniform")),
        use_vlm_features=bool(cfg.dsrl.get("use_vlm_features", False)),
        include_reference_action=bool(cfg.rfs.reference_action_conditioning),
        reference_action_key=str(cfg.rfs.reference_action_key),
    )
    action_modulator = RFSActionModulator(
        latent_dim=int(cfg.dsrl.noise_dim),
        residual_action_dim=residual_action_dim,
        residual_action_bound=float(cfg.rfs.residual_action_bound),
    )
    actor, critic, v_net = build_actor_critic_models(
        latent_env.observation_space, latent_env.action_space, cfg, device, remove_obs_keys
    )
    logger.info(
        f"RFS joint policy action: {cfg.dsrl.noise_dim} flow-noise + {residual_action_dim} "
        f"residual dimensions; residual bound={cfg.rfs.residual_action_bound}"
    )
    # Install the joint policy: sample the flow latent, take the residual
    # mean deterministically, and start the residual branch at zero.
    initialize_rfs_residual_head(
        actor,
        latent_dim=int(cfg.dsrl.noise_dim),
        initial_residual_log_std=float(cfg.rfs.initial_residual_log_std),
    )
    configure_rfs_actor_distribution(
        actor,
        latent_dim=int(cfg.dsrl.noise_dim),
        residual_action_bound=float(cfg.rfs.residual_action_bound),
        residual_exploration_std=float(cfg.rfs.residual_exploration_std),
        proposal_conditioned_residual=bool(cfg.rfs.proposal_conditioned_residual),
    )
    return ParadigmModels(
        env=env,
        remove_obs_keys=remove_obs_keys,
        latent_env=latent_env,
        action_modulator=action_modulator,
        actor=actor,
        critic=critic,
        v_net=v_net,
    )


def _reference_edit_actor_observation_space(
    observation_space: Any,
    remove_obs_keys: list[str],
    reference_action_key: str,
) -> Any:
    """Build the direct-action actor's view, keeping its reference action.

    The critic drops the reference (it evaluates state + executed action); the
    actor keeps it as conditioning, so the actor's observation space differs
    from the critic's and is derived here.
    """
    actor_observation_space = copy.deepcopy(observation_space)
    for key in remove_obs_keys:
        actor_observation_space.spaces.pop(key, None)
    actor_observation_space.spaces.pop(ReferenceActionEnv.ACTION_MASK_KEY, None)
    if reference_action_key not in actor_observation_space.spaces:
        raise ValueError(
            "SAC-edit actor observation space is missing required reference action "
            f"key {reference_action_key!r}"
        )
    return actor_observation_space


def _validate_direct_action_input_normalizer_spaces(
    actor: Any,
    critic: Any,
    *,
    normalize_input: bool,
    input_normalizer_key: str,
) -> None:
    """Fail before algorithm setup when a normalized observation key is absent."""
    if not normalize_input:
        return
    receivers = [("actor", actor)]
    receivers.extend((f"critic[{index}]", module) for index, module in enumerate(critic.critics))
    missing = [
        name
        for name, module in receivers
        if input_normalizer_key not in module.config.observation_space.spaces
    ]
    if missing:
        raise ValueError(
            f"SAC-edit input normalization key {input_normalizer_key!r} is missing from: "
            f"{', '.join(missing)}. Set algorithm.normalize_input=false for no-state "
            "runs, or include the configured key in every normalized model observation space."
        )


def _direct_action_models(
    cfg: DictConfig,
    base_model: Any,
    env: Any,
    algorithm_config: Any,
    device: torch.device,
) -> ParadigmModels:
    """Wrap the env for direct reference-action edits and build the models.

    The frozen policy decodes a reference action chunk; the actor emits a
    bounded edit executed in the raw action space (no steering). The critic
    evaluates the executed action and drops the reference; the actor keeps the
    reference as conditioning. The wrapped ``env`` is the learner's
    environment stand-in and declares the keys the learner drops.
    """
    algorithm_name = str(cfg.algorithm.name).lower()
    # Chunk geometry defaults to the frozen policy's own decoded horizon and action width.
    horizon = int(cfg.direct_action.action_horizon or base_model.action_horizon)
    action_dim = int(cfg.direct_action.action_dim or base_model.action_dim)
    env = ReferenceActionEnv(
        env,
        base_model,
        action_horizon=horizon,
        action_dim=action_dim,
        action_bound=float(cfg.direct_action.action_bound),
        action_indices=cfg.direct_action.action_indices,
        environment_action_defaults=cfg.direct_action.environment_action_defaults,
        include_sim_state=bool(cfg.direct_action.include_sim_state),
        state_key=str(cfg.direct_action.state_key),
        reference_action_transform=str(cfg.direct_action.reference_action_transform),
        relative_action_scale=float(cfg.direct_action.relative_action_scale),
        reference_image_transform_mode=str(cfg.direct_action.reference_image_transform_mode),
    )
    remove_obs_keys = list(env.learner_remove_obs_keys)
    reference_action_key = str(algorithm_config.reference_action_key)

    critic_remove_obs_keys = list(remove_obs_keys)
    critic_remove_obs_keys.append(reference_action_key)
    _, critic, v_net = build_actor_critic_models(
        env.single_observation_space,
        env.single_action_space,
        cfg,
        device,
        critic_remove_obs_keys,
    )

    actor_observation_space = _reference_edit_actor_observation_space(
        env.single_observation_space, remove_obs_keys, reference_action_key
    )
    policy_kwargs = OmegaConf.to_container(cfg.policy.mlp, resolve=True)
    policy_kwargs["hidden_dims"] = tuple(policy_kwargs["hidden_dims"])
    policy_kwargs.update(
        {
            "use_tanh_output": True,
            "deterministic": False,
            "log_std_init": cfg.policy.log_std_init,
            "log_std_min": cfg.policy.log_std_min,
            "log_std_max": cfg.policy.log_std_max,
        }
    )
    actor = ReferenceEditActor(
        ReferenceEditActorConfig(
            observation_space=actor_observation_space,
            action_space=env.single_action_space,
            remove_obs_keys=remove_obs_keys,
            simnorm_dim=int(cfg.direct_action.simnorm_dim),
            simnorm_groups=int(cfg.direct_action.simnorm_groups),
            reference_action_key=reference_action_key,
            edit_scale=(
                None
                if cfg.direct_action.actor_edit_scale is None
                else float(cfg.direct_action.actor_edit_scale)
            ),
            reference_centered_full_support=bool(cfg.direct_action.actor_reference_centered_full_support),
            **policy_kwargs,
        )
    ).to(device)
    if algorithm_name == "sac_edit":
        _validate_direct_action_input_normalizer_spaces(
            actor,
            critic,
            normalize_input=bool(algorithm_config.normalize_input),
            input_normalizer_key=str(algorithm_config.input_normalizer_key),
        )
    logger.info(
        f"{algorithm_name} direct-action contract: chunk={horizon}x{action_dim}, "
        f"include_sim_state={cfg.direct_action.include_sim_state}, "
        f"reference_transform={cfg.direct_action.reference_action_transform}"
    )
    return ParadigmModels(
        env=env,
        remove_obs_keys=remove_obs_keys,
        latent_env=env,
        action_modulator=None,
        actor=actor,
        critic=critic,
        v_net=v_net,
    )
