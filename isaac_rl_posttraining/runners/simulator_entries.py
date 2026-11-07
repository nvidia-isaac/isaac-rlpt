# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the simulator a composed config runs on.

A run collects and evaluates through the simulator entry registered for its
``env.env_name`` (:mod:`isaac_rl_posttraining.envs.providers`); a config may
also name an out-of-tree entry as ``simulator.entry="module:Class"``. These
helpers find that entry, check it against the selected base model and the run's
worker layout, and describe one phase's environments and server address the
way the entry builds them.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.configs import evaluation_enabled, selected_base_model_name
from isaac_rl_posttraining.data.spec import ArraySpec
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
    requested_layout,
)

__all__ = [
    "declared_environment",
    "describe_evaluation",
    "describe_launch_inputs",
    "environment_spec",
    "exported_server_url",
    "phase_server_url",
    "require_base_model_support",
    "require_evaluation_shape",
    "require_server_addresses",
    "require_simulator_entry",
    "required_phases",
    "resolve_simulator_entry",
    "selected_entry",
    "shares_training_workers",
    "sizing_environment",
    "steering_remove_obs_keys",
]


def environment_spec(cfg: DictConfig, *, num_envs: int, server_url: str | None) -> EnvironmentSpec:
    """The environment description a simulator entry builds one phase's workers from."""
    # Client-side geometry of remote embodiments; in-process simulators declare
    # none, and an entry that needs it rejects the missing value.
    action_dim = OmegaConf.select(cfg, "simulator.action_dim")
    image_resolution = OmegaConf.select(cfg, "simulator.image_resolution")
    return EnvironmentSpec(
        env_name=str(cfg.env.env_name),
        num_envs=num_envs,
        max_episode_steps=int(cfg.env.max_episode_steps),
        seed=int(OmegaConf.select(cfg, "training.seed", default=0)),
        server_url=server_url,
        action_dim=None if action_dim is None else int(action_dim),
        image_resolution=None if image_resolution is None else int(image_resolution),
        use_init_states=bool(OmegaConf.select(cfg, "env.use_init_states", default=True)),
        fixed_init_state_id=OmegaConf.select(cfg, "env.fixed_init_state_id", default=None),
    )


def exported_server_url(entry: SimulatorEntry, phase: Phase) -> str | None:
    """The launcher-exported server address of one phase; ``None`` when unset or for in-process simulators."""
    if entry.server_url_variables is None:
        return None
    variable = entry.server_url_variables[0 if phase == "train" else 1]
    # The launcher exports the simulator addresses under the names the entry declares.
    return os.environ.copy().get(variable) or None


def phase_server_url(entry: SimulatorEntry, phase: Phase) -> str | None:
    """The launcher-exported server address of one phase; ``None`` for in-process simulators.

    Raises when a served simulator's address was not exported.
    """
    url = exported_server_url(entry, phase)
    if entry.server_url_variables is not None and url is None:
        variable = entry.server_url_variables[0 if phase == "train" else 1]
        raise ValueError(f"{entry.name} needs {variable} for its {phase} environments")
    return url


def required_phases(cfg: DictConfig) -> tuple[Phase, ...]:
    """The phases whose environments the run opens: training unless it only evaluates or trains offline, evaluation when enabled."""
    phases: list[Phase] = []
    evaluation_only = bool(OmegaConf.select(cfg, "ppo.eval_only", default=False))
    offline = OmegaConf.select(cfg, "algorithm.family") == "offline"
    if int(cfg.training.num_rollouts) > 0 and not evaluation_only and not offline:
        phases.append("train")
    if evaluation_enabled(cfg):
        phases.append("eval")
    return tuple(phases)


def require_server_addresses(cfg: DictConfig, entry: SimulatorEntry) -> None:
    """Fail before any policy loads or worker starts when a served simulator's address is missing."""
    for phase in required_phases(cfg):
        phase_server_url(entry, phase)


def describe_launch_inputs(cfg: DictConfig, entry: SimulatorEntry) -> str:
    """The server addresses the run needs at launch and whether each is exported here, for the plan."""
    if entry.server_url_variables is None:
        return "none (the simulator runs in the rollout workers)"
    parts = []
    for phase in required_phases(cfg):
        variable = entry.server_url_variables[0 if phase == "train" else 1]
        url = exported_server_url(entry, phase)
        parts.append(f"{variable} ({phase}) " + (f"= {url}" if url else "needs to be exported at launch"))
    return "; ".join(parts) or "none (no phase opens environments)"


def shares_training_workers(cfg: DictConfig, entry: SimulatorEntry) -> bool:
    """Whether evaluation runs on the training workers: a shared entry whose training group has the evaluation's size.

    A ``shared`` entry evaluates on the training environments at episode
    boundaries when there are ``eval.eval_num_envs`` of them; otherwise (an
    evaluation-only run, or a different evaluation count) it evaluates on
    workers of its own.
    """
    return (
        entry.evaluation == "shared"
        and "train" in required_phases(cfg)
        and int(cfg.training.num_envs) == int(cfg.eval.eval_num_envs)
    )


def describe_evaluation(cfg: DictConfig, entry: SimulatorEntry) -> str:
    """Where the evaluation episodes run and in how many rounds, for the plan."""
    if not evaluation_enabled(cfg):
        return "disabled"
    environments = int(cfg.eval.eval_num_envs)
    episodes = int(cfg.eval.eval_num_episodes)
    if shares_training_workers(cfg, entry):
        where = "on the training workers"
    elif entry.evaluation == "fresh_workers":
        where = "on workers rebuilt for every evaluation"
    else:
        where = "on workers of their own"
    rounds = f"{episodes // environments} round(s)" if environments and episodes % environments == 0 else "uneven rounds"
    return f"{episodes} episode(s) over {environments} environment(s) {where}, {rounds}"


def require_evaluation_shape(cfg: DictConfig, entry: SimulatorEntry | None = None) -> None:
    """Fail at the plan when the evaluation episodes cannot fill the evaluation environments.

    The evaluator scores exactly ``eval.eval_num_episodes`` episodes in rounds of
    ``eval.eval_num_envs`` (on the training workers when a shared entry has
    that many, else on evaluation workers of that size), so the count must be
    a positive multiple of the environment count. With ``entry``, the
    evaluation workers' layout (servers, batching) is checked too.
    """
    if not evaluation_enabled(cfg):
        return
    episodes = int(cfg.eval.eval_num_episodes)
    environments = int(cfg.eval.eval_num_envs)
    if episodes < environments:
        raise ValueError(
            f"eval.eval_num_episodes ({episodes}) must be at least eval.eval_num_envs ({environments})"
        )
    if episodes % environments != 0:
        lower = episodes - episodes % environments
        raise ValueError(
            f"eval.eval_num_episodes ({episodes}) must be an exact multiple of "
            f"eval.eval_num_envs ({environments}); set eval.eval_num_episodes={lower} or "
            f"{lower + environments}, or change eval.eval_num_envs"
        )
    if entry is not None and not shares_training_workers(cfg, entry):
        url = exported_server_url(entry, "eval")
        servers = environment_spec(cfg, num_envs=environments, server_url=url).num_servers
        requested_layout(entry, environments, num_servers=servers)


def require_base_model_support(cfg: DictConfig, entry: SimulatorEntry) -> None:
    """Fail before any worker starts when the entry's environments do not fit the base model.

    The mock base model stands in for any base policy (dry runs, tests), so an entry's list never excludes it.
    """
    base_model_name = selected_base_model_name(cfg)
    if base_model_name == "mock" or entry.base_models is None or base_model_name in entry.base_models:
        return
    raise ValueError(
        f"the {entry.name} entry builds environments for base model(s) {list(entry.base_models)}; "
        f"got base_model={base_model_name!r}"
    )


def resolve_simulator_entry(cfg: DictConfig) -> SimulatorEntry:
    """The simulator entry of ``env.env_name``, registering ``simulator.entry`` first when set.

    ``simulator.entry="my_package.runtime:MyRobot"`` attaches an entry that is
    not in the registry table to the environment name's simulator segment, so
    an out-of-tree simulator or robot runs without editing ``envs/providers.py``.
    """
    from isaac_rl_posttraining.envs.providers import (
        get_simulator_entry,
        register_simulator_entry,
        simulator_segment,
    )

    env_name = str(cfg.env.env_name)
    entry_path = OmegaConf.select(cfg, "simulator.entry")
    if entry_path is not None:
        register_simulator_entry(simulator_segment(env_name), str(entry_path), override=True)
    try:
        entry = get_simulator_entry(env_name)
    except KeyError as exc:
        raise ValueError(
            f"no simulator entry is registered for environment {env_name!r}; register one in "
            f"envs/providers.py or set simulator.entry=<module>:<Class>; {exc}"
        ) from exc
    # Registered in-tree entries live in simulators/<provider>/, so the simulator group and env.env_name
    # must agree; an entry attached with simulator.entry is the user's explicit choice.
    package = type(entry).__module__.split(".")
    provider = OmegaConf.select(cfg, "simulator.provider")
    if entry_path is None and provider is not None and package[0] == "simulators" and package[1] != str(provider):
        raise ValueError(
            f"env.env_name={env_name!r} runs the {package[1]} simulator, but simulator= selects a "
            f"{provider} config; set env.env_name to a {provider} environment ('<simulator>/<task>', see "
            f"simulators/{provider}/README.md) or choose simulator= to match"
        )
    return entry


def require_simulator_entry(cfg: DictConfig) -> SimulatorEntry:
    """The simulator entry the run collects and evaluates on, after checking what it supports."""
    entry = resolve_simulator_entry(cfg)
    require_base_model_support(cfg, entry)
    return entry


def selected_entry(cfg: DictConfig) -> SimulatorEntry | None:
    """The run's simulator entry; ``None`` for a plain Gymnasium id without one (the smoke path)."""
    try:
        return resolve_simulator_entry(cfg)
    except ValueError:
        return None


def steering_remove_obs_keys(env: Any) -> list[str]:
    """Raw camera keys the learner's networks drop from the observation.

    Each simulator env declares the image keys it carries only for the frozen
    policy's ``infer`` (the learner consumes their DINO embedding instead) via
    ``steering_remove_obs_keys``; the generic Gym smoke path exposes none, so
    fall back to ``["image"]``.
    """
    inner = env.envs[0] if getattr(env, "envs", None) else env
    keys = getattr(inner, "steering_remove_obs_keys", None)
    return list(keys) if keys is not None else ["image"]


def _sizing_source(
    entry: SimulatorEntry, cfg: DictConfig, phase: Phase, device: Any
) -> tuple[EnvironmentSpec, Any]:
    """The phase's environment spec and one raw environment exposing its single-env spaces."""
    from isaac_rl_posttraining.runtimes.base import (
        RolloutWorkerContext,
        partition_environments,
    )

    if phase == "train":
        num_envs = int(cfg.training.num_envs)
    elif phase == "eval":
        num_envs = int(cfg.eval.eval_num_envs)
    else:
        raise ValueError(f"Unknown sizing phase: {phase!r}")
    spec = environment_spec(cfg, num_envs=num_envs, server_url=phase_server_url(entry, phase))
    if entry.sized_from_declaration:
        return spec, declared_environment(entry, spec)
    layout = entry.worker_layout(num_envs, num_servers=spec.num_servers)
    if layout.batched:
        context = RolloutWorkerContext(
            worker_index=0, num_rollout_workers=layout.workers,
            environment_indices=partition_environments(num_envs, layout.workers)[0], device=str(device),
        )
        batch = entry.make_environment(spec, phase, context)
        try:
            if not hasattr(batch, "single_observation_space") or not hasattr(batch, "single_action_space"):
                raise TypeError(
                    f"{entry.name}.make_environment returned {type(batch).__name__}; batched model "
                    "sizing requires single_observation_space and single_action_space"
                )
            single = gym.Env()
            single.observation_space = batch.single_observation_space
            single.action_space = batch.single_action_space
        finally:
            batch.close()
    else:
        spec = environment_spec(cfg, num_envs=1, server_url=phase_server_url(entry, phase))
        context = RolloutWorkerContext(worker_index=0, num_rollout_workers=1, environment_indices=(0,), device=str(device))
        single = entry.make_environment(spec, phase, context)
    if not hasattr(single, "observation_space") or not hasattr(single, "action_space"):
        raise TypeError(
            f"{entry.name}.make_environment returned {type(single).__name__}; the launcher sizes the "
            "learner from one Gymnasium environment with observation_space and action_space"
        )
    return spec, single


class _DeclaredSpaces(gym.Env):
    """The observation and action spaces an entry declares; it is never reset or stepped."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        *,
        remove_keys: list[str],
        language_instruction: str | None,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.steering_remove_obs_keys = remove_keys
        self.language_instruction = language_instruction

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        raise RuntimeError("the declared-spaces environment only sizes the learner; it cannot be reset")

    def step(self, action: Any) -> Any:
        raise RuntimeError("the declared-spaces environment only sizes the learner; it cannot be stepped")


def _box(array: ArraySpec, *, entry: str, name: str) -> gym.spaces.Box:
    """The Box of a fixed-size array spec: unbounded floats, full-range integers and booleans."""
    if not all(isinstance(size, int) for size in array.shape):
        raise ValueError(f"{entry} declares {name} as {array}; sizing from the declaration needs fixed sizes")
    dtype = np.dtype(array.dtype)
    low: float
    high: float
    if dtype == np.bool_:
        low, high = 0, 1
    elif np.issubdtype(dtype, np.integer):
        low, high = np.iinfo(dtype).min, np.iinfo(dtype).max
    else:
        low, high = -np.inf, np.inf
    return gym.spaces.Box(low=low, high=high, shape=tuple(int(size) for size in array.shape), dtype=dtype)


def declared_environment(entry: SimulatorEntry, spec: EnvironmentSpec) -> gym.Env:
    """A space-only environment built from the entry's observation and action declarations.

    The learner reads only shapes and dtypes from it, so no simulator starts.
    It drops every declared array the steering key mapping does not name as
    the state or language entry (the cameras reach the learner as embeddings)
    and carries the entry's task instruction when it declares one.
    """
    observation = entry.observation_spec(spec)
    action = entry.action_spec(spec)
    if observation is None or action is None:
        raise ValueError(
            f"{entry.name} sets sized_from_declaration but does not declare observation_spec and action_spec"
        )
    action_shape = (action.width,) if action.horizon is None else (action.horizon, action.width)
    mapping = entry.steering_key_mapping(spec)
    kept = {mapping.get("state"), mapping.get("language")}
    try:
        instruction: str | None = entry.instruction(spec)
    except NotImplementedError:
        instruction = None
    return _DeclaredSpaces(
        gym.spaces.Dict({name: _box(array, entry=entry.name, name=name) for name, array in observation.arrays.items()}),
        _box(ArraySpec(action_shape, action.dtype), entry=entry.name, name="its action"),
        remove_keys=[name for name in observation.arrays if name not in kept],
        language_instruction=instruction,
    )


def sizing_environment(
    entry: SimulatorEntry,
    cfg: DictConfig,
    *,
    phase: Phase = "train",
    device: Any,
    dinov2_model: Any = None,
    dinov2_processor: Any = None,
    sentence_model: Any = None,
) -> Any:
    """One wrapped set of selected-phase spaces, built like the rollout workers.

    The learner's networks are sized from this environment before any worker
    starts. Scalar entries build one environment; native batches query the
    configured batch and release its client without resetting or stepping it.
    The resulting spaces carry the wrappers every training environment uses: the language
    embedding when the run uses one, the frozen image encoder's embedding, and
    the action chunking; the entry's steering keys are set on the result.
    """
    from isaac_rl_posttraining.data.spec import ObservationSpec
    from isaac_rl_posttraining.envs import VectorActionChunkingWrapper
    from isaac_rl_posttraining.envs.dino_wrapper import VectorDinoEmbeddingWrapper
    from isaac_rl_posttraining.runtimes.steering_rollouts import resolve_image_keys

    spec, single = _sizing_source(entry, cfg, phase, device)
    mapping = dict(entry.steering_key_mapping(spec))
    # A key the observation lacks would leave the raw array in the learner's view instead of an embedding.
    resolved = resolve_image_keys(cfg, entry, spec)
    mismatch = resolved.mismatch(ObservationSpec.from_space(single.observation_space).images)
    if mismatch is not None:
        raise ValueError(f"{entry.name}: {mismatch}")
    image_keys = list(resolved.keys)
    # Keys the environment itself keeps from the networks (Isaac Lab's per-group states) stay removed.
    declared = list(getattr(single, "steering_remove_obs_keys", None) or [])
    remove_keys = list(dict.fromkeys([*image_keys, *declared]))
    # The steering contract lives on the raw environment and on the outermost wrapper.
    single.steering_key_mapping = mapping
    single.steering_remove_obs_keys = remove_keys
    env: Any = gym.vector.SyncVectorEnv([lambda: single])
    if sentence_model is not None:
        from isaac_rl_posttraining.envs.language_wrappers import (
            LanguageInstructionVectorWrapper,
        )

        try:
            instruction = entry.instruction(spec)
        except NotImplementedError as exc:
            raise ValueError(
                f"model.sentence_model is set but {entry.name} declares no task instruction; "
                "implement instruction() on the entry or set model.sentence_model=null"
            ) from exc
        env = LanguageInstructionVectorWrapper(env, instruction, sentence_model)
    if dinov2_model is not None and image_keys:
        env = VectorDinoEmbeddingWrapper(
            env,
            dinov2_model,
            dinov2_processor,
            device=device,
            image_keys=image_keys,
            critic_image_size=OmegaConf.select(cfg, "env.critic_image_size", default=None),
        )
    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)
    if chunk_size is not None:
        env = VectorActionChunkingWrapper(env, chunk_size=int(chunk_size), n_action_steps=1)
    env.steering_key_mapping = mapping
    env.steering_remove_obs_keys = remove_keys
    return env
