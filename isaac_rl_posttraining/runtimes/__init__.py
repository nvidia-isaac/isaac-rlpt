# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable execution runtimes with lazy optional-provider imports."""

from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any

from isaac_rl_posttraining.runtimes.base import (
    DECISION_MODES,
    CollectedDecision,
    DecisionMode,
    PolicyContext,
    PolicyState,
    RolloutGroup,
    RolloutInfrastructureError,
    RolloutRequest,
    RolloutResult,
    RolloutRuntime,
    RolloutWorker,
    RolloutWorkerContext,
    RolloutWorkerSpec,
    RuntimeFeatures,
    WeightsRef,
    check_runtime,
    partition_environments,
)
from isaac_rl_posttraining.runtimes.batched_worker import (
    BatchedEnvironment,
    BatchedEpisodeRolloutWorker,
    BatchedWorkerSpec,
)
from isaac_rl_posttraining.runtimes.concurrency import (
    AsyncCall,
    RolloutGroupBase,
    ThreadCall,
    train_step_async,
)
from isaac_rl_posttraining.runtimes.edit_policy import (
    EditActionChunkPolicy,
    EditPolicySpec,
)
from isaac_rl_posttraining.runtimes.gym_worker import (
    ActionChunkPolicy,
    GymEnvironment,
    GymRolloutWorker,
    GymWorkerSpec,
    PolicySpec,
    RewardTransform,
    SparseOutcomeReward,
    TransformedReward,
    resolve_num_rollout_workers,
)
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerContext,
    LearnerGroup,
    LearnerModuleManager,
    LearnerStepResult,
    LearnerWorker,
    LearnerWorkerSpec,
    LocalLearnerGroup,
    open_distributed_learner_group,
    open_learner_group,
)
from isaac_rl_posttraining.runtimes.policy import (
    BatchedActionChunkPolicy,
    BootstrapValuePolicy,
    PolicyFeatures,
    RolloutPolicy,
    SessionPolicy,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import (
    load_policy_snapshot,
    policy_state_from_tensors,
    policy_state_tensors,
    save_policy_snapshot,
)
from isaac_rl_posttraining.runtimes.steering_policy import (
    FrozenBaseModelSpec,
    SteeringActionChunkPolicy,
    SteeringPolicySpec,
)

__all__ = [
    "DECISION_MODES",
    "ActionChunkPolicy",
    "AsyncCall",
    "BatchedActionChunkPolicy",
    "BatchedEnvironment",
    "BatchedEpisodeRolloutWorker",
    "BatchedWorkerSpec",
    "BootstrapValuePolicy",
    "CollectedDecision",
    "DecisionMode",
    "EditActionChunkPolicy",
    "EditPolicySpec",
    "FrozenBaseModelSpec",
    "GymEnvironment",
    "GymRolloutWorker",
    "GymWorkerSpec",
    "LearnerBatch",
    "LearnerContext",
    "LearnerGroup",
    "LearnerModuleManager",
    "LearnerStepResult",
    "LearnerWorker",
    "LearnerWorkerSpec",
    "LocalLearnerGroup",
    "NativeRuntime",
    "NativeRuntimeConfig",
    "PolicyContext",
    "PolicyFeatures",
    "PolicySpec",
    "PolicyState",
    "RLInfRuntime",
    "RLInfRuntimeConfig",
    "RewardTransform",
    "RolloutGroup",
    "RolloutGroupBase",
    "RolloutInfrastructureError",
    "RolloutPolicy",
    "RolloutRequest",
    "RolloutResult",
    "RolloutRuntime",
    "RolloutWorker",
    "RolloutWorkerContext",
    "RolloutWorkerSpec",
    "RuntimeFeatures",
    "SessionPolicy",
    "SparseOutcomeReward",
    "SteeringActionChunkPolicy",
    "SteeringPolicySpec",
    "ThreadCall",
    "TransformedReward",
    "WeightsRef",
    "available_runtimes",
    "create_runtime",
    "create_runtime_from_config",
    "load_policy_snapshot",
    "open_distributed_learner_group",
    "open_learner_group",
    "partition_environments",
    "policy_serving_mode",
    "policy_state_from_tensors",
    "policy_state_tensors",
    "resolve_num_rollout_workers",
    "save_policy_snapshot",
    "train_step_async",
]

_RUNTIMES: dict[str, tuple[str, str, str | None]] = {
    "native": (
        "isaac_rl_posttraining.runtimes.native",
        "NativeRuntime",
        "NativeRuntimeConfig",
    ),
    "rlinf": (
        "isaac_rl_posttraining.runtimes.rlinf",
        "RLInfRuntime",
        "RLInfRuntimeConfig",
    ),
}

_LAZY_ATTRIBUTES = {
    attribute: (module_name, attribute)
    for module_name, runtime_class, config_class in _RUNTIMES.values()
    for attribute in (runtime_class, config_class)
    if attribute is not None
}


def available_runtimes() -> tuple[str, ...]:
    """Return stable configuration names for registered adapters."""
    return tuple(_RUNTIMES)


def create_runtime(name: str, **settings: Any) -> RolloutRuntime:
    """Construct a runtime without importing unselected provider packages."""
    try:
        module_name, class_name, config_name = _RUNTIMES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown runtime {name!r}; available: {', '.join(available_runtimes())}"
        ) from exc
    runtime_class = getattr(importlib.import_module(module_name), class_name)
    if config_name is None:
        if settings:
            raise ValueError(
                f"runtime {name!r} accepts no settings, got {sorted(settings)}"
            )
        return check_runtime(runtime_class())
    config_class = getattr(importlib.import_module(module_name), config_name)
    return check_runtime(runtime_class(config_class(**settings)))


def policy_serving_mode(placement: Any) -> tuple[str, Any]:
    """How ``placement.policy`` and ``placement.parts`` serve the policy, with the settings that apply.

    ``auto``, ``local`` or ``served`` (the whole policy, ``placement.policy``
    or every part served together), or ``base`` (``placement.parts.base``
    served while the actor stays in the rollout workers).
    """
    policy = placement.policy
    # A part left on auto follows placement.policy.
    parts = {
        name: part
        for name, part in dict(placement.parts or {}).items()
        if part.mode != "auto"
    }
    base, actor = parts.get("base", policy), parts.get("actor", policy)
    if base.mode == "served" and actor.mode != "served":
        return "base", base
    if actor.mode == "served" and base.mode != "served":
        raise ValueError(
            "only the base model can be served while other parts stay local "
            "(placement.parts.base.mode=served); serve the whole policy with "
            "placement.policy.mode=served"
        )
    served = parts.get("policy", parts.get("base", policy))
    if served.mode == "served":
        return "served", served
    return policy.mode, policy


def _policy_serving_settings(name: str, placement: Any) -> dict[str, Any]:
    """``placement.policy`` and ``placement.parts`` lowered onto runtime ``name``'s policy-serving settings."""
    if placement is None:
        return {}
    mode, served = policy_serving_mode(placement)
    serving = (
        {"policy_mode": "local", "base_mode": "served"}
        if mode == "base"
        else {"policy_mode": mode}
    )
    if name == "native":
        # Several replicas run as Ray actors from the placement plan (create_runtime_from_config's plan).
        settings: dict[str, Any] = dict(serving)
        if served.max_batch is not None:
            settings["policy_max_batch"] = int(served.max_batch)
        if served.max_wait_ms is not None:
            settings["policy_max_wait_ms"] = float(served.max_wait_ms)
        return settings
    if name == "rlinf":
        settings = {**serving, "num_policy_servers": served.replicas}
        if served.max_batch is not None:
            settings["policy_batch_size"] = int(served.max_batch)
        if served.max_wait_ms is not None:
            settings["policy_batch_wait_seconds"] = float(served.max_wait_ms) / 1000.0
        return settings
    return {}


def create_runtime_from_config(
    config: Any, placement: Any = None, *, plan: Any = None
) -> RolloutRuntime:
    """Construct the selected runtime from the structured ``runtime`` and ``placement`` sections.

    ``placement`` (a ``PlacementConfig``) decides policy serving: its
    ``policy`` mode, replicas and batching become the runtime's settings.
    Without it the runtime section's own settings apply. ``plan`` (a
    ``PlacementPlan``) runs ``native`` as Ray actors placed by it.
    """
    from isaac_rl_posttraining.configs.schema import RuntimeConfig

    if not isinstance(config, RuntimeConfig):
        raise TypeError(f"expected RuntimeConfig, got {type(config).__name__}")
    serving = _policy_serving_settings(config.name, placement)
    if config.name == "native":
        settings = asdict(config.native)
        settings["policy_serve_eval_unbatched"] = settings.pop(
            "serve_evaluation_unbatched"
        )
        settings["rollout_timeout_seconds"] = config.rollout_timeout_seconds
        from isaac_rl_posttraining.runtimes.native import (
            NativeRuntime,
            NativeRuntimeConfig,
        )

        native = NativeRuntimeConfig(**{**settings, **serving})
        replicas = (
            None if placement is None else policy_serving_mode(placement)[1].replicas
        )
        if plan is None and (native.processes == "ray" or (replicas or 1) > 1):
            raise ValueError(
                "several processes (runtime.native.processes=ray, or placement.policy.replicas > 1) "
                "run as Ray actors placed by the run's placement plan; pass plan= "
                "(runners.run_placement.open_runtime does)"
            )
        return check_runtime(NativeRuntime(native, plan=plan))
    if config.name == "rlinf":
        settings = asdict(config.rlinf)
        # Either name turning unbatched evaluation off turns it off.
        settings["deterministic_evaluation"] = bool(
            settings.pop("serve_evaluation_unbatched")
            and settings["deterministic_evaluation"]
        )
        if settings["node_ranks"] is not None:
            settings["node_ranks"] = tuple(settings["node_ranks"])
        settings["rollout_timeout_seconds"] = config.rollout_timeout_seconds
        # A deprecated alias; normalizing moved it onto placement.policy.mode.
        del settings["policy_serving"]
        return create_runtime("rlinf", **{**settings, **serving})
    raise KeyError(
        f"unknown runtime {config.name!r}; available: {', '.join(available_runtimes())}"
    )


def __getattr__(name: str):
    target = _LAZY_ATTRIBUTES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = target
    value = getattr(importlib.import_module(module_name), class_name)
    globals()[name] = value
    return value
