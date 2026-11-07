# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run the off-policy families (steering and edit) from a composed config.

DSRL (Diffusion Steering RL) trains a SAC actor/critic in the latent-noise
space of a FROZEN flow/diffusion policy (GR00T): the actor emits a steering
latent per decision step, the frozen policy decodes it into an action chunk,
and the rollout workers execute the first ``dsrl.action_exec_len`` actions.

A run builds the frozen policy, one local environment that sizes the models,
the networks, the algorithm and the replay buffer, then trains on the runtime
layer: rollout workers collect through the simulator's registered entry, the
learner trains in this process, and a rollout evaluator scores the published
weights. Evaluation-only runs are expressed as
``training.num_rollouts=0 eval.eval_on_first_step=true`` (plus
``eval.eval_base=true`` for the native base-policy baseline).

Environment variables (launcher boundary only): ``ISAAC_RL_POSTTRAINING_OUTPUT_DIR`` sets
the run-output root (default ``~/isaac_rl_outputs``); everything tunable is
a config field.
"""

from __future__ import annotations

import contextlib
import functools
import os
import signal
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.discounting import chunk_discounted_gamma
from isaac_rl_posttraining.algorithms.registry import get_algorithm_config
from isaac_rl_posttraining.buffers import ReplayBuffer
from isaac_rl_posttraining.configs import (
    evaluation_enabled,
    selected_algorithm_name,
    selected_base_model_name,
)
from isaac_rl_posttraining.envs.providers import get_env_builder
from isaac_rl_posttraining.envs.simulator_entry import (
    Phase,
    SimulatorEntry,
    server_addresses,
)
from isaac_rl_posttraining.runners.off_policy_evaluation import (
    run_evaluation_only,
)
from isaac_rl_posttraining.runners.off_policy_models import (
    ParadigmModels,
    paradigm_models,
    policy_family,
    uses_reference_actions,
)
from isaac_rl_posttraining.runners.off_policy_session import (
    OffPolicySession,
    open_off_policy_session,
    rollout_worker_spec_for,
)
from isaac_rl_posttraining.runners.plan_checks import native_eval_only, native_only_error
from isaac_rl_posttraining.runners.simulator_entries import (
    require_simulator_entry,
    selected_entry,
    sizing_environment,
    steering_remove_obs_keys,
)
from isaac_rl_posttraining.runners.spec_checks import (
    check_chunk_size_at_load,
    check_policy_inputs_at_load,
)
from isaac_rl_posttraining.runtimes import (
    RolloutRuntime,
)
from isaac_rl_posttraining.runtimes.gym_worker import learning_reward_channel
from isaac_rl_posttraining.runtimes.policy_families import (
    reward_transform_for,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors
from isaac_rl_posttraining.runtimes.watchdog import on_frozen_rollout
from isaac_rl_posttraining.training import (
    FingerprintSink,
    OffPolicyTrainer,
)
from isaac_rl_posttraining.training.checkpoints import save_aborted_checkpoint
from isaac_rl_posttraining.utils.compat import add_note
from isaac_rl_posttraining.utils.env_utils import make_env
from isaac_rl_posttraining.utils.logging_utils import setup_loguru_logging
from isaac_rl_posttraining.utils.training_utils import (
    Replay,
    create_replay,
    load_checkpoint,
    seed_everything,
)

if TYPE_CHECKING:
    from isaac_rl_posttraining.algorithms.rfs.modulation import RFSActionModulator


def configure_runtime_env() -> None:
    """Quiet third-party log spam and defer CUDA module loading."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


def output_root() -> str:
    """Run-output root: ``ISAAC_RL_POSTTRAINING_OUTPUT_DIR`` or ``~/isaac_rl_outputs``.

    Checkpoints and logs live OUTSIDE the repository so the working tree
    stays small and node-local storage can be used; the env var is the
    launcher-boundary override.
    """
    return os.environ.get("ISAAC_RL_POSTTRAINING_OUTPUT_DIR") or os.path.join(os.path.expanduser("~"), "isaac_rl_outputs")


def new_run_output_dir(cfg: DictConfig) -> tuple[str, str]:
    """Create ``<output root>/<logging.wandb_name>_<timestamp>`` and return ``(exp_name, output_dir)``."""
    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{cfg.logging.wandb_name}_{string_time}"
    output_dir = os.path.join(os.path.abspath(output_root()), exp_name)
    os.makedirs(output_dir, exist_ok=True)
    return exp_name, output_dir


def build_base_model(cfg: DictConfig, device: torch.device) -> Any:
    """Load the frozen policy selected by the ``base_model`` config group.

    The returned object exposes ``infer(obs, noise)`` plus ``action_horizon``
    — everything the latent env, the SAC networks, and the collection/eval
    workers consume. The base model name (``base_model.name``) resolves through the
    policy registry (:func:`isaac_rl_posttraining.policies.get_base_model`), and
    each base model builds itself from the composed config through the uniform
    :meth:`~isaac_rl_posttraining.policies.base.BaseModelPolicy.from_dsrl_config`
    constructor. Adding a base model is therefore one registry entry plus that
    constructor, with no change here. GR00T's returns its dedicated steering
    wrapper; every other name returns its registered ``BaseModelPolicy``.

    Base models with ``supports_steering=False`` (e.g. ``dreamzero_server``) load fine
    but only support native evaluation; :func:`setup_dsrl_training` enforces
    that before any environment traffic.

    Raises:
        KeyError: For an unknown base model name (from the registry).
        ValueError: For missing required base model fields.
        ImportError: When the selected base model's optional dependencies are
            not installed (raised by the registry).
    """
    from isaac_rl_posttraining.policies import get_base_model, is_gr00t_base_model

    name = selected_base_model_name(cfg)
    base_model = get_base_model(name)
    extra: dict[str, Any] = {}
    if is_gr00t_base_model(name):
        # GR00T decodes head outputs through an embodiment adapter. Resolve it
        # here (the entry point may touch the env layer) and hand the object to
        # the embodiment-agnostic base model, so no policy module imports the env
        # layer or names a simulator.
        from isaac_rl_posttraining.envs.embodiments import get_dataset_adapter

        extra["dataset_adapter"] = get_dataset_adapter(
            OmegaConf.select(cfg, "base_model.dataset_adapter", default="g1_wbc")
        )
    return base_model.from_dsrl_config(cfg, device, **extra)


def validate_evaluation_for_paradigm(cfg: DictConfig) -> None:
    """Reject configs whose paradigm cannot run without an eval env.

    The direct-action paradigm wraps the eval env for its reference actions
    (:func:`_setup_direct_action`), so a collect-only run
    (:func:`evaluation_enabled` false) has nothing to wrap. Fail before any
    simulator boots rather than deep inside setup.
    """
    algorithm_name = str(cfg.algorithm.name).lower()
    if uses_reference_actions(algorithm_name) and not evaluation_enabled(cfg):
        raise ValueError(
            f"{algorithm_name!r} (direct-action paradigm) scores its bounded edits against the "
            "frozen reference and cannot run with evaluation disabled; set eval.eval_freq > 0 or "
            "eval.eval_on_first_step=true"
        )


def setup_dsrl_env(
    cfg: DictConfig,
    dinov2_model: Any,
    dinov2_processor: Any,
    sentence_model: Any,
    device: torch.device,
    entry: SimulatorEntry | None = None,
    *,
    phase: Phase = "train",
) -> tuple[Any, list[str]]:
    """Build the one local environment that sizes the models.

    The rollout workers build their own environments through the simulator
    entry; this environment only provides the observation and action spaces,
    the task language and the steering contract. A simulator with a registered
    training-env builder keeps it; any other entry is sized from the
    environment its workers build.

    Returns:
        Tuple ``(env, remove_obs_keys)`` where ``remove_obs_keys`` lists
        observation keys the SAC networks must drop (the raw image stays in
        the env observation for the frozen policy's ``infer``).
    """
    if entry is not None and (
        phase == "eval" or get_env_builder(str(cfg.env.env_name)) is None
    ):
        env = sizing_environment(
            entry, cfg, phase=phase, device=device, dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor, sentence_model=sentence_model,
        )
        return env, steering_remove_obs_keys(env)
    # Launcher-boundary env reads: the launchers export the per-run server
    # addresses and the sim geometry knobs (see example_workflows/g1_arena_posttraining/scripts/train.sh).
    # Several training servers share one geometry, so the first one sizes the models.
    env, _ = make_env(
        env_name=cfg.env.env_name,
        num_envs=1,
        chunk_size=cfg.training.chunk_size,
        use_full_state=cfg.env.use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model if not cfg.env.use_full_state else None,
        render_mode="rgb_array",
        terminate_on_success=True,
        max_episode_steps=cfg.env.max_episode_steps,
        seed=int(cfg.training.seed),
        server_url=next(iter(server_addresses(os.environ.get("ARENA_SERVER_URL"))), None),
        action_dim=int(OmegaConf.select(cfg, "simulator.action_dim", default=35)),
        image_resolution=int(OmegaConf.select(cfg, "simulator.image_resolution", default=0)),
        critic_image_size=OmegaConf.select(cfg, "env.critic_image_size", default=None),
        use_init_states=bool(OmegaConf.select(cfg, "env.use_init_states", default=True)),
        fixed_init_state_id=OmegaConf.select(cfg, "env.fixed_init_state_id", default=None),
        with_eval_env=False,
    )

    # The SAC actor/critic consume {state, dino_embedding}; the raw camera
    # frames stay in the env observation for the frozen policy's infer and are
    # dropped from the learner's view. Each simulator env owns its steering
    # contract -- steering_key_mapping (read by the rollout policies) and
    # steering_remove_obs_keys -- so nothing per-provider is wired here.
    remove_obs_keys = steering_remove_obs_keys(env)
    return env, remove_obs_keys


def _env_language_instruction(env: Any) -> str | None:
    """The task's language instruction, when the environment carries one.

    In-process providers (LIBERO) expose ``language_instruction`` on the
    task environment. Vector wrappers forward neither attributes nor
    ``envs``, so walk ``env`` links until the sub-env list appears, then
    unwrap the first sub-env.
    """
    node = env
    for _ in range(8):
        inner_envs = getattr(node, "envs", None)
        if inner_envs:
            node = inner_envs[0]
            break
        inner = getattr(node, "env", None)
        if inner is None:
            break
        node = inner
    instruction = getattr(getattr(node, "unwrapped", node), "language_instruction", None)
    return str(instruction) if instruction else None


def _load_frozen_policy(cfg: DictConfig, device: torch.device) -> Any:
    """Load the frozen policy and reject steering runs on a native-only base model.

    A native-only base model (BaseModelPolicy contract note 5) cannot receive
    steering latents, so anything beyond an eval-only run of the native policy
    would crash at the first steered call; fail fast with the flags to use.
    """
    base_model_name = selected_base_model_name(cfg)
    logger.info(f"Loading the frozen policy (base_model={base_model_name})")
    base_model = build_base_model(cfg, device)
    if not getattr(base_model, "supports_steering", True) and not native_eval_only(cfg):
        raise native_only_error(base_model_name)
    return base_model


def _open_run(cfg: DictConfig) -> tuple[Any, str]:
    """Create the run's output directory, experiment logger and log file.

    Returns ``(wandb_logger, save_dir)``; checkpoints and fingerprints go
    under ``save_dir``.
    """
    from isaac_rl_posttraining.loggers.wandb_logger import WandbLogger, wandb_run_dir

    exp_name, output_dir = new_run_output_dir(cfg)
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        log_dir=wandb_run_dir(cfg.logging.wandb_log_dir_base, str(cfg.logging.wandb_name)),
        prefix=cfg.logging.get("wandb_prefix", "offline"),
    )
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    setup_loguru_logging(log_level=cfg.logging.get("log_level", "INFO"), output_dir=output_dir)
    return wandb_logger, save_dir


def _load_encoders(cfg: DictConfig, device: torch.device) -> tuple[Any, Any, Any]:
    """The frozen encoders the sizing environment embeds observations with.

    Returns ``(dinov2_model, dinov2_processor, sentence_model)``, ``None`` for
    an encoder the config leaves out.
    """
    from transformers import AutoImageProcessor, AutoModel

    dinov2_model, dinov2_processor = None, None
    if cfg.model.dinov2_model:
        dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2_model).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2_model)
    sentence_model = None
    if cfg.model.sentence_model is not None:
        from sentence_transformers import SentenceTransformer

        sentence_model = SentenceTransformer(cfg.model.sentence_model)
    return dinov2_model, dinov2_processor, sentence_model


def _adopt_environment_task(base_model: Any, env: Any) -> None:
    """Give a frozen policy without a task prompt the environment's language instruction.

    Base models registered from baked checkpoint statistics have no dataset to
    read the task language from; the environment carries it (vectorized
    training envs drop the per-observation prompt). An explicit
    ``base_model.task_instruction`` override wins.
    """
    policy = getattr(base_model, "policy", None)
    if policy is None or getattr(policy, "task", None) is not None:
        return
    instruction = _env_language_instruction(env)
    if instruction:
        policy.task = instruction
        logger.info(f"Task instruction from the environment: {instruction!r}")


def algorithm_config_from(cfg: DictConfig) -> tuple[str, Any]:
    """The selected algorithm's name and its config built from the composed ``algorithm`` group.

    The registry is the single source of truth for what exists. Building the
    config before the models makes it the source of truth for the fields their
    construction reads (the reference-action key, the latent prior std).
    """
    if cfg.algorithm is None:
        raise ValueError(
            "algorithm config is required but is None. "
            "Make sure the config composes 'algorithm: <name>' in its defaults."
        )
    algorithm_name = str(cfg.algorithm.name).lower()
    config_cls = get_algorithm_config(algorithm_name)
    return algorithm_name, config_cls(**OmegaConf.to_container(cfg.algorithm, resolve=True))


def _wire_algorithm(
    cfg: DictConfig,
    algorithm_name: str,
    algorithm_config: Any,
    models: ParadigmModels,
    replay: Replay,
    wandb_logger: Any,
) -> Any:
    """Hand the algorithm config its runtime objects and create the algorithm.

    Every paradigm stores one transition per executed chunk. Direct action
    folds the receding-horizon discount into the SAC core; the steering
    paradigms fold it into gamma up front instead.
    """
    direct_action = uses_reference_actions(algorithm_name)
    algorithm_config.env = models.latent_env
    algorithm_config.actor = models.actor
    algorithm_config.critic = models.critic
    algorithm_config.buffer = replay.buffer
    algorithm_config.sampler = replay.sampler
    algorithm_config.demonstrations = replay.demonstrations
    algorithm_config.action_space = (
        models.latent_env.single_action_space if direct_action else models.latent_env.action_space
    )
    algorithm_config.logger = wandb_logger
    algorithm_config.compute_chunked_gamma = direct_action
    if not direct_action:
        algorithm_config.gamma = chunk_discounted_gamma(algorithm_config.gamma, cfg.dsrl.action_exec_len)
    logger.info(f"Effective gamma: {algorithm_config.gamma}")
    return algorithm_config.create()


def _warm_start(cfg: DictConfig, algorithm: Any, save_dir: str) -> None:
    """Warm-start (not resume) the algorithm from ``training.load_dir``; record the loaded actor's digest.

    The digest goes through the same sink as the evaluation and checkpoint
    events, so a warm start, its first evaluation and the checkpoint it came
    from can be matched by one string.
    """
    load_dir = cfg.training.load_dir
    if load_dir is None:
        return
    load_checkpoint(algorithm, load_dir)
    logger.info(f"Warm start from {load_dir}: learner state loaded; env steps and replay start empty")
    FingerprintSink(save_dir).record(
        "load",
        step=0,
        state=policy_state_from_tensors(algorithm.actor.state_dict(), version=0),
        load_dir=str(load_dir),
    )


@dataclass
class OffPolicyComponents:
    """What an off-policy run builds before its rollout workers.

    ``env`` is the controller's local environment; it only sizes the models
    and is released (set to ``None``) once the workers own their own
    environments. ``latent_env`` is the paradigm's environment stand-in the
    rollout policies sample from.
    """

    device: torch.device
    base_model: Any
    env: Any | None
    latent_env: Any
    action_modulator: RFSActionModulator | None
    algorithm_name: str
    actor: torch.nn.Module
    v_net: torch.nn.Module | None
    algorithm: Any
    buffer: ReplayBuffer
    wandb_logger: Any
    save_dir: str
    save_interval: int


def setup_dsrl_training(cfg: DictConfig) -> OffPolicyComponents:
    """Create every component of an off-policy run except its rollout workers.

    Loads the frozen policy, opens the run's outputs, builds one local
    environment to size the models (a remote simulator serves one client, so
    it closes before the workers connect), then the paradigm's networks, the
    replay buffer and the algorithm.
    """
    selected_algorithm_name(cfg)
    # Config-only checks fail before anything loads.
    algorithm_name, algorithm_config = algorithm_config_from(cfg)
    validate_evaluation_for_paradigm(cfg)
    seed_everything(int(OmegaConf.select(cfg, "training.seed", default=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    base_model = _load_frozen_policy(cfg, device)
    check_chunk_size_at_load(cfg, base_model)
    wandb_logger, save_dir = _open_run(cfg)
    sizing_phase: Phase = (
        "train" if int(cfg.training.num_rollouts) > 0 else "eval"
    )
    env, remove_obs_keys = setup_dsrl_env(
        cfg,
        *_load_encoders(cfg, device),
        device,
        selected_entry(cfg),
        phase=sizing_phase,
    )
    _adopt_environment_task(base_model, env)
    # The checkpoint's state layout is known only now; a wider state would be sliced silently.
    logger.info(check_policy_inputs_at_load(base_model, env))
    models = paradigm_models(
        cfg,
        algorithm_name,
        base_model=base_model,
        env=env,
        remove_obs_keys=remove_obs_keys,
        algorithm_config=algorithm_config,
        device=device,
    )
    logger.info(f"Actor: {models.actor.__class__.__name__} | Critic: {models.critic.__class__.__name__}")
    # The learner trains on the reward the training workers record.
    train_transform = reward_transform_for(policy_family(algorithm_name), cfg, phase="train")
    replay = create_replay(
        capacity=int(cfg.buffer.capacity),
        drop_components=models.remove_obs_keys or (),
        reward_channel=learning_reward_channel(train_transform),
        reward_aggregation=str(cfg.buffer.reward_aggregation),
        success_fraction=cfg.buffer.success_fraction,
        demonstrations=cfg.buffer.demonstrations,
        demonstration_fraction=cfg.buffer.demonstration_fraction,
    )
    algorithm = _wire_algorithm(cfg, algorithm_name, algorithm_config, models, replay, wandb_logger)
    _warm_start(cfg, algorithm, save_dir)
    wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
    return OffPolicyComponents(
        device=device,
        base_model=base_model,
        env=models.env,
        latent_env=models.latent_env,
        action_modulator=models.action_modulator,
        algorithm_name=algorithm_name,
        actor=models.actor,
        v_net=models.v_net,
        algorithm=algorithm,
        buffer=replay.buffer,
        wandb_logger=wandb_logger,
        save_dir=save_dir,
        save_interval=cfg.training.save_interval,
    )


def run(cfg: DictConfig) -> str:
    """Train one off-policy run: build its components, then collect, learn and evaluate; returns its run directory."""
    entry = require_simulator_entry(cfg)
    components = setup_dsrl_training(cfg)
    runtimes: list[RolloutRuntime] = []
    try:
        _train(cfg, components, entry, runtimes)
    finally:
        close_run_resources(components, runtimes)
    return os.path.dirname(components.save_dir)


def open_session(cfg: DictConfig) -> OffPolicySession:
    """Build the run's components and open its session for step-wise control; closing it releases everything."""
    entry = require_simulator_entry(cfg)
    components = setup_dsrl_training(cfg)
    runtimes: list[RolloutRuntime] = []
    try:
        session = open_off_policy_session(cfg, components, entry, runtimes)
    except BaseException:
        close_run_resources(components, runtimes)
        raise
    session.release = functools.partial(close_run_resources, components, runtimes)
    return session


def close_run_resources(components: OffPolicyComponents, runtimes: list[RolloutRuntime]) -> None:
    """Finish the logger, then close runtimes and the local environment, preserving an in-flight error."""
    active_error = sys.exc_info()[1]
    cleanup_errors: list[Exception] = []
    # Ending W&B first sends its last rows before the slower runtime shutdown.
    try:
        components.wandb_logger.finish()
    except Exception as exc:  # noqa: BLE001
        add_note(exc, "while closing experiment logger")
        cleanup_errors.append(exc)
    resources = [("rollout runtime", runtime) for runtime in reversed(runtimes)]
    if components.env is not None:
        resources.append(("local environment", components.env))
    for resource_name, resource in resources:
        try:
            resource.close()
        except Exception as exc:  # noqa: BLE001
            add_note(exc, f"while closing {resource_name}")
            cleanup_errors.append(exc)
    if cleanup_errors:
        if active_error is not None:
            add_note(active_error, 
                f"{len(cleanup_errors)} training resource(s) also failed to close"
            )
        else:
            raise RuntimeError(
                f"{len(cleanup_errors)} training resource(s) failed to close"
            ) from cleanup_errors[0]


@contextlib.contextmanager
def sigterm_as_interrupt() -> Iterator[None]:
    """Raise ``KeyboardInterrupt`` on SIGTERM, so a stopped job saves and closes like Ctrl-C."""
    if threading.current_thread() is not threading.main_thread():
        yield  # signal handlers can only be set from the main thread
        return

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _train(
    cfg: DictConfig,
    components: OffPolicyComponents,
    entry: SimulatorEntry,
    runtimes: list[RolloutRuntime],
) -> None:
    """Train on the runtime layer: rollout groups, a local learner group, a rollout evaluator."""
    buffer = components.buffer
    if int(cfg.training.num_rollouts) == 0:
        if not evaluation_enabled(cfg):
            raise ValueError(
                "training.num_rollouts=0 requires evaluation; set "
                "eval.eval_on_first_step=true or eval.eval_freq>0"
            )
        eval_num_envs = int(cfg.eval.eval_num_envs)
        eval_spec = rollout_worker_spec_for(
            cfg,
            components,
            entry,
            phase="eval",
            num_envs=eval_num_envs,
            configured_workers=None,
        )
        components.env.close()
        components.env = None
        run_evaluation_only(
            cfg,
            entry,
            eval_spec,
            eval_num_envs=eval_num_envs,
            runtimes=runtimes,
            actor=components.actor,
            optimizer_updates=int(components.algorithm.step_counter),
            experiment_logger=components.wandb_logger,
            output_dir=os.path.dirname(components.save_dir),
            save_dir=components.save_dir,
        )
        return
    session = open_off_policy_session(cfg, components, entry, runtimes)
    trainer = session.trainer
    logger.info(f"Starting training for {cfg.training.num_rollouts} env steps")
    try:
        with sigterm_as_interrupt(), on_frozen_rollout(functools.partial(save_aborted_checkpoint, trainer)):
            trainer.run()
        logger.info(f"Training completed. Final buffer size: {len(buffer)}")
    except KeyboardInterrupt:
        logger.warning(f"Training interrupted. Buffer size at interruption: {len(buffer)}")
        trainer.save_checkpoint("interrupted")
        trainer.save_buffer_on_signal()
        raise
    except Exception as exc:
        save_aborted_checkpoint(trainer, exc)
        raise
    finally:
        session.close()
