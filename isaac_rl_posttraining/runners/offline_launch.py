# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run the offline loop (IQL, SFT) from a composed config.

Offline methods update a trainable base model (its flow-SDE adapter) from a
demonstration dataset. There are no collection workers. Periodic evaluation,
when enabled, scores the same live policy with a deterministic ``infer``
decode on evaluation workers in this process.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import get_algorithm_config
from isaac_rl_posttraining.configs import (
    evaluation_enabled,
    selected_algorithm_name,
    selected_base_model_name,
)
from isaac_rl_posttraining.envs.embodiments import resolve_adapter_option
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry, requested_layout
from isaac_rl_posttraining.policies import is_gr00t_base_model
from isaac_rl_posttraining.policies.gr00t.flow_sde_policy import (
    FlowSDEActionChunkPolicy,
)
from isaac_rl_posttraining.policies.trainable import (
    FlowSDEConfig,
    TrainablePolicySpec,
    get_trainable_policy,
    resolve_trainable_policy_name,
)
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runners.off_policy_evaluation import open_evaluator
from isaac_rl_posttraining.runners.off_policy_launch import (
    new_run_output_dir,
    sigterm_as_interrupt,
)
from isaac_rl_posttraining.runners.ppo_launch import log_namespaced_metrics
from isaac_rl_posttraining.runners.run_placement import (
    native_runs_processes,
    open_runtime,
)
from isaac_rl_posttraining.runners.simulator_entries import (
    environment_spec,
    phase_server_url,
    require_base_model_support,
    require_evaluation_shape,
    resolve_simulator_entry,
)
from isaac_rl_posttraining.runtimes.base import PolicyContext
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures
from isaac_rl_posttraining.runtimes.policy_snapshot import (
    load_policy_snapshot,
    policy_state_from_tensors,
)
from isaac_rl_posttraining.runtimes.rollout_records import SparseOutcomeReward
from isaac_rl_posttraining.training.checkpoints import CheckpointStore, FingerprintSink
from isaac_rl_posttraining.training.offline import SOURCE_BUILDERS, open_offline_source
from isaac_rl_posttraining.utils.logging_utils import setup_loguru_logging
from isaac_rl_posttraining.utils.training_utils import seed_everything

__all__ = ["describe", "require_trainable_base_model", "run"]


def require_trainable_base_model(cfg: DictConfig) -> str:
    """The trainable adapter ``base_model`` selects, refused when it has no demonstration source."""
    algorithm = selected_algorithm_name(cfg)
    name = selected_base_model_name(cfg)
    try:
        adapter = resolve_trainable_policy_name(name, str(OmegaConf.select(cfg, "base_model.ckpt") or ""))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"algorithm={algorithm} trains the base model; base_model={name} cannot be trained "
            f"(set base_model to one of {sorted(SOURCE_BUILDERS)})"
        ) from exc
    if adapter not in SOURCE_BUILDERS:
        raise ValueError(
            f"algorithm={algorithm} has no demonstration reader for base_model={adapter}; "
            f"set base_model to one of {sorted(SOURCE_BUILDERS)}"
        )
    return adapter


def _eval_freq(cfg: DictConfig) -> int:
    """``eval.eval_freq`` in updates; unset means one evaluation per 1% of the run, as the off-policy loop does."""
    if not evaluation_enabled(cfg):
        return 0
    freq = cfg.eval.eval_freq
    if freq is None:
        return max(1, int(cfg.training.num_rollouts) // 100)
    return int(freq)


def describe(cfg: DictConfig) -> tuple[SimulatorEntry, str, str, str]:
    """Plan fields of the offline loop: entry, layout, evaluation mode, budget."""
    adapter = require_trainable_base_model(cfg)
    algorithm = selected_algorithm_name(cfg)
    if int(cfg.placement.learner.ranks) != 1:
        raise ValueError(
            f"{algorithm} trains in this process; set placement.learner.ranks=1 "
            f"(got {cfg.placement.learner.ranks})"
        )
    # Evaluation workers drive the live policy object, so they must share this process.
    if str(cfg.runtime.name) != "native" or native_runs_processes(cfg):
        raise ValueError(
            f"{algorithm} evaluates the live policy in this process; select runtime=native "
            "with runtime.native.processes=single"
        )
    entry = resolve_simulator_entry(cfg)
    require_base_model_support(cfg, entry)
    evaluation_mode = "disabled"
    if evaluation_enabled(cfg):
        require_evaluation_shape(cfg, entry)
        if requested_layout(entry, int(cfg.eval.eval_num_envs)).batched:
            raise ValueError(
                f"{algorithm} evaluates on one-environment rollout workers; the {entry.name} entry "
                "runs batched simulators. Set eval.eval_freq=0 and eval.eval_on_first_step=false"
            )
        evaluation_mode = "infer"
    budget = (
        f"{cfg.training.num_rollouts} {algorithm} updates on {adapter}, "
        f"eval_freq={_eval_freq(cfg)} ({cfg.eval.eval_num_episodes} episodes, {evaluation_mode}), "
        f"save_interval={cfg.training.save_interval}, retention={cfg.training.checkpoint_retention}"
    )
    return entry, "offline (no collection workers)", evaluation_mode, budget


def run(cfg: DictConfig) -> str:
    """Load the trainable base model, open its demonstration source, and run offline updates; returns the run directory."""
    adapter = require_trainable_base_model(cfg)
    algorithm = selected_algorithm_name(cfg)
    entry = resolve_simulator_entry(cfg)
    checkpoint = str(OmegaConf.select(cfg, "base_model.ckpt") or "")
    if not checkpoint:
        raise ValueError(f"{algorithm} needs base_model.ckpt (the checkpoint it trains)")
    updates = int(cfg.training.num_rollouts)
    denoising = OmegaConf.select(cfg, "base_model.denoising_steps")
    seed_everything(int(cfg.training.seed))
    policy = TrainablePolicySpec(
        base_model=adapter,
        checkpoint=checkpoint,
        flow_sde=FlowSDEConfig(denoising_steps=4 if denoising is None else int(denoising), keep_dropout=True),
        value_head_seed=int(cfg.training.seed),
        options=resolve_adapter_option(get_trainable_policy(adapter).spec_options(cfg.base_model)),
    ).build("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.eval()
    # Offline methods train the action head, not the PPO value head attached at load.
    for parameter in policy.value_head.parameters():
        parameter.requires_grad_(False)
    instruction = OmegaConf.select(cfg, "base_model.task_instruction")
    if instruction:
        policy.instruction = str(instruction)

    from isaac_rl_posttraining.loggers import WandbLogger

    exp_name, output_dir = new_run_output_dir(cfg)
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=bool(cfg.logging.wandb_offline),
        project=str(cfg.logging.wandb_project),
        entity=cfg.logging.wandb_entity,
        log_dir=f"{cfg.logging.wandb_log_dir_base}/{cfg.logging.wandb_name}",
        prefix=str(cfg.logging.wandb_prefix),
    )
    save_dir = os.path.join(output_dir, "checkpoints")
    setup_loguru_logging(log_level=str(cfg.logging.get("log_level", "INFO")), output_dir=output_dir)
    store = CheckpointStore(
        save_dir, retention=str(cfg.training.checkpoint_retention), algorithm=selected_algorithm_name(cfg)
    )
    evaluations: list[dict[str, Any]] = []

    def evaluate(step: int) -> None:
        evaluations.append(_evaluate(cfg, entry, policy, wandb_logger, output_dir, step=step))
        store.write_evaluation_summary(evaluations)
        FingerprintSink(save_dir).record(
            "eval", step=step, state=_policy_state(policy, step), success_rate=evaluations[-1]["success_rate"]
        )

    def save(tag: int | str, step: int) -> None:
        store.save(tag, learner_group=None, policy_state=_policy_state(policy, step), metadata={"updates": step})
        store.apply_retention({record["step"]: record["success_rate"] for record in evaluations})

    load_dir = OmegaConf.select(cfg, "training.load_dir")
    if load_dir:
        # The evaluation wrapper checks the snapshot's keys against the live policy as it loads them.
        _InferActionChunkPolicy(policy, instruction="").set_policy_state(load_policy_snapshot(str(load_dir)))
        logger.info(f"loaded the offline policy snapshot from {load_dir}")
    if updates == 0:
        if not evaluation_enabled(cfg):
            raise ValueError(
                "training.num_rollouts=0 only evaluates; set eval.eval_on_first_step=true or eval.eval_freq>0"
            )
        evaluate(0)
        return output_dir

    dataset_adapter = None
    if is_gr00t_base_model(adapter):
        from isaac_rl_posttraining.envs.embodiments import get_dataset_adapter

        dataset_adapter = get_dataset_adapter(str(OmegaConf.select(cfg, "base_model.dataset_adapter", default="none")))
    dataset_dir = str(OmegaConf.select(cfg, "base_model.dataset_dir") or "")
    source = open_offline_source(adapter, dataset_dir, policy=policy, dataset_adapter=dataset_adapter)
    config_cls = get_algorithm_config(algorithm)
    payload = OmegaConf.to_container(cfg.algorithm, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("algorithm config must compose to a mapping")
    algorithm_config = config_cls(**{key: value for key, value in payload.items() if key in config_cls.__dataclass_fields__})
    algorithm_config.policy = policy
    algorithm_config.offline_source = source
    algorithm_config.total_updates = updates * int(algorithm_config.num_updates_per_train_step)
    learner = algorithm_config.create()

    logger.info(
        f"{algorithm} on {adapter}: {updates} updates from {dataset_dir}, "
        f"batch_size={algorithm_config.batch_size}, eval_freq={_eval_freq(cfg) or 'disabled'}"
    )
    if evaluation_enabled(cfg) and bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=False)):
        evaluate(0)
    step = 0
    try:
        with sigterm_as_interrupt():
            for step in _update_loop(learner, updates, wandb_logger, prefix=algorithm):
                if _eval_freq(cfg) > 0 and step % _eval_freq(cfg) == 0:
                    evaluate(step)
                if int(cfg.training.save_interval) > 0 and step % int(cfg.training.save_interval) == 0 and step < updates:
                    save(step, step)
    except KeyboardInterrupt:
        logger.warning(f"{algorithm} interrupted after {step} updates; saving the interrupted checkpoint")
        save("interrupted", step)
        raise
    save("final", updates)
    logger.info(f"{algorithm} finished {updates} updates; checkpoints in {save_dir}")
    return output_dir


def _update_loop(learner: Any, updates: int, experiment_logger: Any, *, prefix: str) -> Any:
    """Yield each update step after running it and logging its metrics at that step."""
    for step in range(1, updates + 1):
        metrics = learner.train_step(rollout_step=step)
        if metrics:
            experiment_logger.log(metrics, step=step, prefix=prefix)
            if step % max(updates // 20, 1) == 0:
                logger.info(f"{prefix} update {step}/{updates}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
        yield step


def _policy_state(policy: Any, step: int) -> Any:
    """The trainable weights (everything outside the frozen backbone) as a policy state."""
    prefixes = tuple(f"{path}." for path in policy.frozen_submodules)
    return policy_state_from_tensors(policy.state_dict(), version=max(step, 1), exclude_prefixes=prefixes)


def _evaluate(
    cfg: DictConfig,
    entry: SimulatorEntry,
    policy: Any,
    experiment_logger: Any,
    output_dir: str,
    *,
    step: int,
) -> dict[str, Any]:
    """Score the live policy with ``infer`` on evaluation workers; returns the evaluation record."""
    eval_num_envs = int(cfg.eval.eval_num_envs)
    instruction = _task_instruction(cfg, entry)
    reward = None
    if bool(cfg.env.sparse_success_reward):
        reward = SparseOutcomeReward(
            success_reward=float(cfg.env.success_reward),
            failure_reward=float(cfg.env.failure_reward),
            step_reward=float(cfg.env.step_reward),
        )
    worker_spec = GymWorkerSpec(
        entry=entry,
        environment=environment_spec(cfg, num_envs=eval_num_envs, server_url=phase_server_url(entry, "eval")),
        phase="eval",
        policy=_LiveInferPolicySpec(policy, instruction),
        reward_transform=reward,
        chunked_step=bool(entry.supports_chunked_step),
    )
    runtime = open_runtime(cfg)
    evaluator = open_evaluator(
        cfg,
        entry,
        runtime,
        None,
        worker_spec,
        eval_num_envs=eval_num_envs,
        experiment_logger=experiment_logger,
        reward_channel=worker_spec.reward_channel,
        reward_aggregation=str(cfg.buffer.reward_aggregation),
        output_dir=output_dir,
    )
    try:
        metrics = dict(evaluator.evaluate(_policy_state(policy, step), env_steps=step))
    finally:
        evaluator.close()
        runtime.close()
    log_namespaced_metrics(experiment_logger, {f"eval/{name}": value for name, value in metrics.items()}, step)
    logger.info(
        f"offline eval at update {step}: success_rate={metrics['success_rate']:.1%} "
        f"over {int(metrics['num_eval_episodes'])} episodes"
    )
    return {"step": step, **metrics}


def _task_instruction(cfg: DictConfig, entry: SimulatorEntry) -> str:
    configured = OmegaConf.select(cfg, "base_model.task_instruction")
    if configured:
        return str(configured)
    if type(entry).instruction is not SimulatorEntry.instruction:
        return entry.instruction(environment_spec(cfg, num_envs=1, server_url=None))
    return "task"


class _LiveInferPolicySpec:
    """Hand evaluation workers in this process the live policy, so no second copy loads."""

    def __init__(self, policy: Any, instruction: str) -> None:
        self.policy = policy
        self.instruction = instruction

    def __call__(self, context: PolicyContext) -> _InferActionChunkPolicy:
        del context
        return _InferActionChunkPolicy(self.policy, instruction=self.instruction)


class _InferActionChunkPolicy(FlowSDEActionChunkPolicy):
    """One action chunk per gym observation, decoded with :meth:`FlowSDEActorCritic.infer`."""

    # Evaluation decisions carry no behaviour values (the flow-SDE policy's PPO decisions do).
    features = PolicyFeatures()

    def __init__(self, policy: Any, *, instruction: str) -> None:
        super().__init__(policy, instruction=instruction, rng=np.random.default_rng(0), cache_features=False)

    def action_chunks(
        self,
        observations: Any,
        *,
        mode: Any,
        session_ids: Any,
    ) -> tuple[ActionChunk, ...]:
        del mode, session_ids
        decoded = self.actor_critic.infer(_stack_observations(observations, instruction=self.instruction))
        if decoded.ndim != 3 or decoded.shape[0] != len(observations):
            raise ValueError(f"infer must return actions [{len(observations)}, H, A], got {decoded.shape}")
        return tuple(
            ActionChunk.from_arrays(
                learner_action=None,
                environment_actions=decoded[index : index + 1],
                execution_length=int(decoded.shape[1]),
                source="policy",
                policy_version=self._policy_version,
            )
            for index in range(len(observations))
        )


def _stack_observations(observations: Any, *, instruction: str) -> dict[str, Any]:
    """Stack a batch of mapping observations for ``infer``; the batch shares one instruction."""
    first = observations[0]
    stacked: dict[str, Any] = {
        key: np.stack([np.asarray(observation[key]) for observation in observations])
        for key in first
        if not isinstance(first[key], str)
    }
    stacked["instruction"] = next(
        (first[name] for name in ("instruction", "language", "prompt", "task") if first.get(name)), instruction
    )
    return stacked

