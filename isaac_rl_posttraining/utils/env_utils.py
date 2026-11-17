# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment construction with the observation/embedding wrapper stack.

Builds the IsaacLab-Arena remote environment (or a generic Gymnasium env for
local smoke tests) wrapped with the language, DINO-embedding, and
action-chunking wrappers from :mod:`isaac_rl_posttraining.envs`.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

__all__ = ["make_env", "create_dummy_vectorized_env"]


def create_dummy_vectorized_env(
    env_name: str = "CartPole-v1",
    num_envs: int = 4,
    render_mode: str = "rgb_array",
    use_full_state: bool = False,
    kwargs: Optional[dict] = None,
) -> gym.vector.VectorEnv:
    """Create a simple vectorized Gymnasium environment for testing.

    Args:
        env_name: Gym environment id.
        num_envs: Number of parallel copies.
        render_mode: Render mode passed to ``gym.make``.
        use_full_state: Expose the flat state as a dict key (``True``) or
            the rendered image (``False``).
        kwargs: Extra ``gym.make`` keyword arguments.

    Returns:
        A ``SyncVectorEnv`` of dict-observation environments.
    """
    from isaac_rl_posttraining.envs.obs_wrappers import FlatToDictObsWrapper, ImageDictObsWrapper

    make_kwargs = kwargs or {}

    def _make_one() -> gym.Env:
        env = gym.make(env_name, render_mode=render_mode, **make_kwargs)
        if use_full_state:
            return FlatToDictObsWrapper(env)
        return ImageDictObsWrapper(env)

    return gym.vector.SyncVectorEnv([_make_one for _ in range(num_envs)])


def _vector_gym_env(
    env_name: str,
    *,
    num_envs: int,
    chunk_size: int | None,
    use_full_state: bool,
    dinov2_model: Any,
    dinov2_processor: Any,
    device: str,
    render_mode: str,
    image_keys: list[str] | None,
) -> gym.Env:
    """``num_envs`` copies of a generic Gymnasium env with the vector wrappers (local smoke path)."""
    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper
    from isaac_rl_posttraining.envs.obs_wrappers import FlatToDictObsWrapper, ImageDictObsWrapper

    def _make_single() -> gym.Env:
        single_env = gym.make(env_name, render_mode=render_mode)
        if use_full_state:
            return FlatToDictObsWrapper(single_env)
        return ImageDictObsWrapper(single_env)

    env = gym.vector.SyncVectorEnv([_make_single for _ in range(num_envs)])
    if dinov2_model is not None:
        single_space = getattr(env, "single_observation_space", env.observation_space)
        if isinstance(single_space, gym.spaces.Dict) and "image" in single_space.spaces:
            env = VectorDinoEmbeddingWrapper(
                env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys
            )
    if chunk_size is not None:
        env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)
    return env


def _single_gym_env(
    env_name: str,
    *,
    chunk_size: int | None,
    use_full_state: bool,
    dinov2_model: Any,
    dinov2_processor: Any,
    device: str,
    render_mode: str,
    image_keys: list[str] | None,
) -> gym.Env:
    """One generic Gymnasium env with the single-env wrappers (local smoke path)."""
    from isaac_rl_posttraining.envs import ActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import DinoEmbeddingWrapper
    from isaac_rl_posttraining.envs.obs_wrappers import FlatToDictObsWrapper, ImageDictObsWrapper

    env = gym.make(env_name, render_mode=render_mode)
    env = FlatToDictObsWrapper(env) if use_full_state else ImageDictObsWrapper(env)
    if dinov2_model is not None:
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces:
            env = DinoEmbeddingWrapper(
                env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys
            )
    if chunk_size is not None:
        env = ActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)
    return env


def _make_env(
    env_name: str,
    vectorized: bool = False,
    num_envs: int = 4,
    max_episode_steps: int = 400,
    chunk_size: Optional[int] = None,
    use_full_state: bool = False,
    dinov2_model: Any = None,
    dinov2_processor: Any = None,
    device: Optional[str] = None,
    sentence_model: Any = None,
    render_mode: str = "rgb_array",
    terminate_on_success: bool = True,
    image_keys: Optional[list[str]] = None,
    seed: Optional[int] = None,
    server_url: Optional[str] = None,
    action_dim: int = 35,
    image_resolution: int = 224,
    critic_image_size: Optional[int] = None,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    vector_backend: str = "sync",
) -> gym.Env:
    """Create one wrapped (vectorized) environment.

    Args:
        env_name: ``"arena/<task>"`` for the remote Arena env,
            ``"libero/<suite>/<task>"`` for an in-process LIBERO task in
            the GR00T checkpoint conventions (``<task>`` is a task name or
            index), or a Gym id.
        vectorized: Build a vectorized env (training path).
        num_envs: Number of parallel copies when vectorized.
        max_episode_steps: Client-side episode step cap.
        chunk_size: Action-chunking wrapper size (``None`` disables).
        dinov2_model: Loaded DINOv2 model for the embedding wrapper.
        dinov2_processor: Matching image processor.
        device: Device for the embedding models (auto-detected when None).
        sentence_model: Loaded sentence encoder for language wrappers.
        render_mode: Render mode for Gym environments.
        terminate_on_success: Reserved (Arena terminates server-side).
        image_keys: Image keys the DINO wrapper embeds.
        seed: Reserved seed hook.
        server_url: Arena env-server address (``None`` = stub mode).
        action_dim: Arena action width (embodiment-dependent; 35 = G1
            whole-body joint space).
        image_resolution: Client image resolution — must match the server's
            ``--image-resolution`` (it defines the obs-space shape and how
            the server's image bytes are reshaped).
        vector_backend: How the ``num_envs`` sims are stepped behind the shared
            ``gym.vector`` interface (see
            :func:`isaac_rl_posttraining.envs.build_vector_env`): ``"sync"``
            (sequential ``SyncVectorEnv``; the default), ``"async"`` (N sims in
            N worker processes), or ``"batched"`` (a natively-batched sim). Only
            the in-process LIBERO-pi0 path honors this today; others stay ``sync``.

    Returns:
        The wrapped environment.
    """
    # Imported lazily so this module (and isaac_rl_posttraining.utils) stays
    # importable without the envs extras.
    from isaac_rl_posttraining.envs.providers import get_env_builder

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    builder = get_env_builder(env_name)
    if builder is not None:
        # Simulator-specific construction lives in the simulator package,
        # resolved by env_name prefix through the provider registry -- core
        # never imports a simulator.
        return builder(
            env_name,
            num_envs=num_envs,
            vectorized=vectorized,
            max_episode_steps=max_episode_steps,
            chunk_size=chunk_size,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            device=device,
            sentence_model=sentence_model,
            image_keys=image_keys,
            seed=seed,
            server_url=server_url,
            action_dim=action_dim,
            image_resolution=image_resolution,
            critic_image_size=critic_image_size,
            use_init_states=use_init_states,
            fixed_init_state_id=fixed_init_state_id,
            vector_backend=vector_backend,
        )
    # Generic Gymnasium environments (local smoke-testing path).
    gym_kwargs: dict[str, Any] = {
        "chunk_size": chunk_size,
        "use_full_state": use_full_state,
        "dinov2_model": dinov2_model,
        "dinov2_processor": dinov2_processor,
        "device": device,
        "render_mode": render_mode,
        "image_keys": image_keys,
    }
    if vectorized:
        return _vector_gym_env(env_name, num_envs=num_envs, **gym_kwargs)
    return _single_gym_env(env_name, **gym_kwargs)


def make_env(
    env_name: str,
    num_envs: int = 4,
    max_episode_steps: int = 400,
    chunk_size: Optional[int] = None,
    use_full_state: bool = False,
    dinov2_model: Any = None,
    dinov2_processor: Any = None,
    device: Optional[str] = None,
    sentence_model: Any = None,
    render_mode: str = "rgb_array",
    terminate_on_success: bool = True,
    image_keys: Optional[list[str]] = None,
    seed: Optional[int] = None,
    server_url: Optional[str] = None,
    eval_server_url: Optional[str] = None,
    action_dim: int = 35,
    image_resolution: int = 224,
    critic_image_size: Optional[int] = None,
    use_init_states: bool = True,
    fixed_init_state_id: int | None = None,
    vector_backend: str = "sync",
    with_eval_env: bool = True,
    eval_num_envs: int = 1,
) -> tuple[gym.Env, gym.Env | None]:
    """Create the (train, eval) environment pair with matching wrappers.

    See :func:`_make_env` for parameter semantics. The Arena env-server is
    single-client, so train and eval must target SEPARATE server instances:
    the eval env uses ``eval_server_url`` (falling back to ``server_url``).

    ``vector_backend`` selects how the ``num_envs`` training sims are stepped;
    the eval env uses ``sync`` with one env unless ``eval_num_envs > 1``, which
    builds it behind the ``batched`` backend so the evaluator scores that many
    episodes in parallel (in-process simulators hosting several scenes).
    ``with_eval_env=False`` builds only the train env (a run that never
    evaluates needs no eval simulator) and returns ``None`` in its place.

    Returns:
        Tuple ``(train_env, eval_env)``.
    """
    train_env = _make_env(
        env_name=env_name,
        vectorized=True,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        chunk_size=chunk_size,
        use_full_state=use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode=render_mode,
        terminate_on_success=terminate_on_success,
        image_keys=image_keys,
        seed=seed,
        server_url=server_url,
        action_dim=action_dim,
        image_resolution=image_resolution,
        critic_image_size=critic_image_size,
        use_init_states=use_init_states,
        fixed_init_state_id=fixed_init_state_id,
        vector_backend=vector_backend,
    )
    if not with_eval_env:
        return train_env, None
    eval_env = _make_env(
        env_name=env_name,
        vectorized=True,
        num_envs=max(1, int(eval_num_envs)),
        max_episode_steps=max_episode_steps,
        chunk_size=chunk_size,
        use_full_state=use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode=render_mode,
        terminate_on_success=terminate_on_success,
        image_keys=image_keys,
        # A distinct stream: the eval env must not replay the train env's
        # initial-state sequence.
        seed=None if seed is None else seed + 1000,
        server_url=eval_server_url or server_url,
        action_dim=action_dim,
        image_resolution=image_resolution,
        critic_image_size=critic_image_size,
        use_init_states=use_init_states,
        fixed_init_state_id=fixed_init_state_id,
            vector_backend="batched" if int(eval_num_envs) > 1 else "sync",
    )
    return train_env, eval_env
