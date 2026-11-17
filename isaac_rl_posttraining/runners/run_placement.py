# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a run's ``placement`` section into a :class:`PlacementPlan` for the dry run.

The config's ``placement.*`` keys become a :class:`Placement`; the runtime's
:class:`RuntimeFeatures` and the GPUs it can place onto become the planner's
inputs. ``native`` runs every role in this process unless the placement asks
for more (:func:`native_runs_processes`); then its plan places Ray actors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import AlgorithmSpec, algorithm_spec
from isaac_rl_posttraining.configs.schema import PlacementConfig, RuntimeConfig
from isaac_rl_posttraining.runtimes import (
    RolloutRuntime,
    create_runtime_from_config,
    policy_serving_mode,
)
from isaac_rl_posttraining.runtimes.base import RuntimeFeatures
from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Local,
    Placement,
    PlacementPlan,
    PolicyPlacement,
    Resources,
    RolePlacement,
    Served,
    plan_placement,
)
from isaac_rl_posttraining.runtimes.serving.batcher import DEFAULT_MAX_WAIT_MS

__all__ = [
    "RolloutShape",
    "native_runs_processes",
    "open_runtime",
    "placement_config_of",
    "placement_from_config",
    "placement_overrides",
    "plan_resources",
    "plan_run_placement",
    "resolved_placement",
    "runtime_config_of",
    "runtime_features",
    "serves_policy_by_default",
]

#: What a policy server batches when placement leaves it open, per runtime (rlinf: today's settings).
#: Native: no request cap (a 3x16-scene layout fits one batch; a cap of 32 would make the third
#: worker wait a whole decode) and RLInf's 5 ms window, so concurrent workers' requests coalesce.
_SERVER_DEFAULTS = {"rlinf": (32, 5.0)}
_DEFAULT_SERVER = (None, DEFAULT_MAX_WAIT_MS)


@dataclass(frozen=True)
class RolloutShape:
    """The training rollout workers as the run builds them, known before they start.

    ``workers``: how many; ``batched``: each drives several environments in
    one simulator; ``servable``: a policy server may act for them (their
    spec's ``supports_batched_policy`` / ``supports_served_policy``);
    ``gpu_simulator``: each simulates on a GPU (``SimulatorEntry.simulates_on_gpu``).
    """

    workers: int
    batched: bool = False
    servable: bool = True
    gpu_simulator: bool = False

    @classmethod
    def of_spec(cls, spec: Any, workers: int) -> RolloutShape:
        """The shape of ``workers`` rollout workers built from ``spec``."""
        from isaac_rl_posttraining.runtimes.batched_worker import BatchedWorkerSpec

        batched = isinstance(spec, BatchedWorkerSpec)
        servable = (
            spec.supports_served_policy
            if batched
            else bool(getattr(spec, "supports_batched_policy", False))
        )
        return cls(
            workers=int(workers),
            batched=batched,
            servable=bool(servable),
            gpu_simulator=spec.entry.simulates_on_gpu,
        )


def _section(cfg: DictConfig, key: str) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg[key], resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"{key} config must resolve to a mapping")
    return raw


def runtime_config_of(cfg: DictConfig) -> RuntimeConfig:
    """The structured ``runtime`` section of a run config."""
    return RuntimeConfig(**_section(cfg, "runtime"))


def placement_config_of(cfg: DictConfig) -> PlacementConfig:
    """The structured ``placement`` section of a run config."""
    return PlacementConfig(**_section(cfg, "placement"))


def native_runs_processes(cfg: DictConfig) -> bool:
    """Whether ``runtime=native`` runs the roles of this run as Ray actors (``runtime.native.processes``)."""
    if str(cfg.runtime.name) != "native":
        return False
    mode = str(OmegaConf.select(cfg, "runtime.native.processes", default="auto"))
    if mode != "auto":
        return mode == "ray"
    if OmegaConf.select(cfg, "placement") is None:
        return False  # a partial config (a test's) asks for nothing beyond this process
    from isaac_rl_posttraining.runtimes.native import wants_processes

    num_nodes = int(OmegaConf.select(cfg, "runtime.native.num_nodes", default=1))
    return wants_processes(placement_from_config(cfg), num_nodes=num_nodes)


def serves_policy_by_default(cfg: DictConfig, shape: RolloutShape | None) -> bool:
    """Whether the runtime serves the whole policy when ``placement.policy.mode=auto``.

    ``native`` in one process shares one in-process policy server among
    unbatched workers whose policy may be shared; ``rlinf`` on GPUs serves
    such workers from policy servers. Otherwise every worker keeps its own.
    """
    if shape is None or not shape.servable or shape.batched:
        return False
    if policy_serving_mode(placement_config_of(cfg))[0] != "auto":
        return False
    runtime = str(cfg.runtime.name)
    if runtime == "native":
        return not native_runs_processes(cfg)
    if runtime == "rlinf":
        return str(cfg.runtime.rlinf.device) == "gpu"
    return False


def open_runtime(cfg: DictConfig, shape: RolloutShape | None = None) -> RolloutRuntime:
    """Start the selected runtime with the policy serving the placement asks for.

    A native run that asks for several processes receives its placement plan
    (the dry run's, for training rollout workers of ``shape``) and runs the
    roles as Ray actors placed by it.
    """
    plan = resolved_placement(cfg, shape) if native_runs_processes(cfg) else None
    return create_runtime_from_config(
        runtime_config_of(cfg), placement_config_of(cfg), plan=plan
    )


def runtime_features(cfg: DictConfig) -> RuntimeFeatures:
    """What the selected runtime can do, without starting it."""
    name = str(cfg.runtime.name)
    if name == "native":
        from isaac_rl_posttraining.runtimes.native import NativeRuntime

        return NativeRuntime.features
    if name == "rlinf":
        from isaac_rl_posttraining.runtimes.rlinf.runtime import RLInfRuntime

        return RLInfRuntime.features
    raise ValueError(f"unknown runtime {name!r}")


def _gpus(value: Any) -> Any:
    if value is None or isinstance(value, int):
        return value
    return tuple(str(device) for device in value)


def _policy(section: Any, runtime: str) -> PolicyPlacement:
    if section.mode != "served":
        return Local()
    max_batch, max_wait_ms = _SERVER_DEFAULTS.get(runtime, _DEFAULT_SERVER)
    return Served(
        replicas=section.replicas,
        gpus=_gpus(section.gpus),
        gpus_per_replica=int(section.gpus_per_replica),
        max_batch=max_batch if section.max_batch is None else int(section.max_batch),
        max_wait_ms=max_wait_ms
        if section.max_wait_ms is None
        else float(section.max_wait_ms),
        engine=str(section.engine),
        routing=section.routing,
    )


def placement_from_config(
    cfg: DictConfig, shape: RolloutShape | None = None
) -> Placement:
    """The :class:`Placement` the ``placement`` section asks for.

    Without ``shape``, ``placement.policy.mode=auto`` plans as local; with it,
    as the runtime resolves it (:func:`serves_policy_by_default`), and the
    rollout workers default to ``shape.workers``.
    """
    section = cfg.placement
    runtime = str(cfg.runtime.name)
    ranks = int(section.learner.ranks)
    learner_gpus = _gpus(section.learner.gpus)
    policy = _policy(section.policy, runtime)
    if serves_policy_by_default(cfg, shape):
        max_batch, max_wait_ms = _SERVER_DEFAULTS.get(runtime, _DEFAULT_SERVER)
        policy = Served(max_batch=max_batch, max_wait_ms=max_wait_ms)
    workers = section.envs.workers
    if workers is None and shape is not None:
        workers = shape.workers
    return Placement(
        envs=Envs(workers=workers, gpus=_gpus(section.envs.gpus)),
        policy=policy,
        # A part left on auto follows placement.policy.
        parts={
            name: _policy(part, runtime)
            for name, part in (section.parts or {}).items()
            if part.mode != "auto"
        },
        # Zero ranks evaluate without a learner, so the learner holds no GPU.
        learner=Learner(
            ranks=ranks or None,
            gpus=0 if ranks == 0 and learner_gpus is None else learner_gpus,
        ),
        nodes=None if section.nodes is None else list(section.nodes),
    )


def _visible_gpus() -> int:
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def node_gpus(cfg: DictConfig) -> int:
    """GPUs of one node: ``placement.gpus_per_node``, else the ones visible here."""
    gpus = cfg.placement.gpus_per_node
    return _visible_gpus() if gpus is None else int(gpus)


def plan_resources(cfg: DictConfig, features: RuntimeFeatures) -> Resources:
    """The nodes and GPUs the plan places onto.

    ``placement.gpus_per_node`` overrides the GPUs visible here; a CPU runtime
    has none, and a runtime that cannot split roles across processes plans
    onto one GPU.
    """
    name = str(cfg.runtime.name)
    gpus = node_gpus(cfg)
    num_nodes = 1
    if name == "rlinf":
        num_nodes = int(cfg.runtime.rlinf.num_nodes)
        if str(cfg.runtime.rlinf.device) == "cpu":
            gpus = 0
    elif name == "native":
        num_nodes = int(OmegaConf.select(cfg, "runtime.native.num_nodes", default=1))
        if str(cfg.runtime.native.device) == "cpu":
            gpus = 0
    if not (features.policy_service or features.distributed_learner):
        gpus = min(gpus, 1)
    return Resources.uniform(num_nodes, gpus)


def _native_collects_concurrently(cfg: DictConfig, workers: int) -> bool:
    """``runtime.native.concurrent_collection`` as the in-process runtime resolves it."""
    setting = OmegaConf.select(
        cfg, "runtime.native.concurrent_collection", default="auto"
    )
    if setting != "auto":
        return bool(setting)
    return (
        policy_serving_mode(placement_config_of(cfg))[0] in ("served", "base")
        and workers > 1
    )


def _single_process_learner(placement: Placement) -> Learner:
    """The learner of a run whose roles all stay in this process: on its one device, whatever the count."""
    learner = placement.learner
    if learner.gpus not in (None, 0, 1, ("0",), ("0:0",), ("node0:0",)):
        raise ValueError(
            "a single learner rank runs in this process on GPU 0; leave placement.learner.gpus "
            f"unset (got {learner.gpus}) or ask for several ranks"
        )
    return learner


def _controller_learner_gpu(placement: Placement, resources: Resources) -> Placement:
    """A native run's single learner rank trains in the controller process, on its first GPU."""
    learner = placement.learner
    if (learner.ranks or 1) != 1 or resources.gpus_per_node[0] == 0:
        return placement
    if learner.gpus in (None, 1):
        return dataclasses.replace(
            placement, learner=Learner(ranks=learner.ranks, gpus=("0",))
        )
    if learner.gpus not in (0, ("0",), ("0:0",), (f"{resources.nodes[0]}:0",)):
        raise ValueError(
            "a single learner rank runs in the controller process on GPU 0 of the first node; "
            f"leave placement.learner.gpus unset (got {learner.gpus}) or ask for several ranks"
        )
    return placement


def _thin_rollout_workers(
    placement: Placement, spec: AlgorithmSpec, shape: RolloutShape | None
) -> Placement:
    """Rollout workers that hold no policy part and do not simulate on a GPU need none (as on RLInf)."""
    if placement.envs.gpus is not None or shape is None or shape.gpu_simulator:
        return placement
    parts = [placement.parts.get(part, placement.policy) for part in spec.policy_parts]
    if not all(isinstance(part, Served) for part in parts):
        return placement
    return dataclasses.replace(
        placement, envs=Envs(workers=placement.envs.workers, gpus=0)
    )


def _without_gpus(part: PolicyPlacement) -> PolicyPlacement:
    return dataclasses.replace(part, gpus=None) if isinstance(part, Served) else part


def _check_semantics(
    placement: Placement,
    resources: Resources,
    spec: AlgorithmSpec,
    features: RuntimeFeatures,
) -> PlacementPlan:
    """What the runtime and algorithm allow (served parts, learner ranks), on no GPUs."""
    without_gpus = dataclasses.replace(
        placement,
        envs=Envs(workers=placement.envs.workers),
        policy=_without_gpus(placement.policy),
        parts={name: _without_gpus(part) for name, part in placement.parts.items()},
        learner=Learner(ranks=placement.learner.ranks),
    )
    return plan_placement(
        without_gpus,
        Resources(resources.nodes, (0,) * len(resources.nodes)),
        policy_parts=spec.policy_parts,
        served_supported=features.policy_service,
        distributed_learner=features.distributed_learner,
        served_parts=features.served_parts,
        runtime=f"runtime={features.name}",
    )


def _in_process_plan(
    cfg: DictConfig,
    placement: Placement,
    resources: Resources,
    spec: AlgorithmSpec,
    features: RuntimeFeatures,
) -> PlacementPlan:
    """Every role in this process on its one device: the rollout workers, an in-process policy server, the learner."""
    learner = _single_process_learner(placement)
    checked = _check_semantics(placement, resources, spec, features)
    gpus = min(resources.gpus_per_node[0], 1)
    device = (("node0", (0,)),) if gpus else ()
    workers = checked.envs.processes
    policy = []
    for role in checked.policy:
        if role.served is None:
            policy.append(dataclasses.replace(role, processes=workers))
            continue
        # Workers collecting one after another are served at once; concurrent ones are waited for.
        wait = (
            role.served.max_wait_ms
            if _native_collects_concurrently(cfg, workers)
            else 0.0
        )
        served = dataclasses.replace(
            role.served, replicas=1, gpus=gpus, max_wait_ms=wait
        )
        policy.append(
            dataclasses.replace(role, gpus=device, processes=1, served=served)
        )
    ranks = 0 if learner.gpus == 0 and learner.ranks is None else learner.ranks or 1
    return PlacementPlan(
        resources=Resources.uniform(1, gpus),
        envs=RolePlacement(role="envs", gpus=device, processes=workers),
        policy=tuple(policy),
        learner=RolePlacement(
            role="learner", gpus=device if ranks else (), processes=ranks
        ),
        shared=bool(gpus),
        in_process=True,
    )


def plan_run_placement(
    cfg: DictConfig,
    spec: AlgorithmSpec,
    features: RuntimeFeatures,
    shape: RolloutShape | None = None,
) -> PlacementPlan:
    """Resolve the run's placement as the selected runtime will run it, raising on what it cannot do.

    ``shape`` is what the run knows about its training rollout workers (how
    many, batched or not, whether a policy server may act for them); with it
    the plan shows the worker count and the policy serving the runtime picks
    for ``placement.policy.mode=auto``.
    """
    placement = placement_from_config(cfg, shape)
    resources = plan_resources(cfg, features)
    runtime = str(cfg.runtime.name)
    if runtime == "rlinf":
        from isaac_rl_posttraining.runners.rlinf_placement import (
            reject_unused_server_settings,
            rlinf_plan,
        )

        reject_unused_server_settings(placement)
        checked = _check_semantics(placement, resources, spec, features)
        return rlinf_plan(
            cfg,
            checked,
            resources,
            gpu_simulator=shape is not None and shape.gpu_simulator,
        )
    if runtime == "native" and not native_runs_processes(cfg):
        return _in_process_plan(cfg, placement, resources, spec, features)
    if runtime == "native":
        placement = _thin_rollout_workers(
            _controller_learner_gpu(placement, resources), spec, shape
        )
    return plan_placement(
        placement,
        resources,
        policy_parts=spec.policy_parts,
        served_supported=features.policy_service,
        distributed_learner=features.distributed_learner,
        served_parts=features.served_parts,
        runtime=f"runtime={features.name}",
        evaluation_workers=_evaluation_workers(cfg, spec),
    )


def _evaluation_workers(cfg: DictConfig, spec: AlgorithmSpec) -> int:
    """Workers of an off-policy evaluation group of its own (0 when evaluation runs on the training workers)."""
    from isaac_rl_posttraining.runners.plan_checks import evaluation_group_workers
    from isaac_rl_posttraining.runners.simulator_entries import resolve_simulator_entry

    if spec.loop != "off_policy":
        return 0
    return evaluation_group_workers(cfg, resolve_simulator_entry(cfg), spec.loop) or 0


def resolved_placement(
    cfg: DictConfig, shape: RolloutShape | None = None
) -> PlacementPlan:
    """The placement plan of a normalized run config (what the dry run prints and ``placement.json`` holds)."""
    features = runtime_features(cfg)
    return plan_run_placement(
        cfg, algorithm_spec(str(cfg.algorithm.name)), features, shape
    )


def placement_overrides(placement: Placement) -> list[str]:
    """Hydra overrides that set ``placement.*`` to ``placement``."""
    overrides = [
        f"placement.envs.workers={_value(placement.envs.workers)}",
        f"placement.envs.gpus={_value(placement.envs.gpus)}",
        *_policy_overrides("placement.policy", placement.policy),
        f"placement.learner.gpus={_value(placement.learner.gpus)}",
    ]
    if placement.learner.ranks is not None:
        overrides.append(f"placement.learner.ranks={placement.learner.ranks}")
    for name, part in placement.parts.items():
        overrides += [
            f"++{item}" for item in _policy_overrides(f"placement.parts.{name}", part)
        ]
    if placement.nodes is not None:
        overrides.append(f"placement.nodes={_value(list(placement.nodes))}")
    return overrides


def _policy_overrides(prefix: str, part: PolicyPlacement) -> list[str]:
    if isinstance(part, Local):
        return [f"{prefix}.mode=local"]
    return [
        f"{prefix}.mode=served",
        f"{prefix}.replicas={_value(part.replicas)}",
        f"{prefix}.gpus={_value(part.gpus)}",
        f"{prefix}.gpus_per_replica={part.gpus_per_replica}",
        f"{prefix}.max_batch={_value(part.max_batch)}",
        f"{prefix}.max_wait_ms={part.max_wait_ms}",
        f"{prefix}.engine={part.engine}",
        f"{prefix}.routing={part.routing}",
    ]


def _value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list | tuple):
        return "[" + ",".join(f"'{item}'" for item in value) + "]"
    return str(value)
