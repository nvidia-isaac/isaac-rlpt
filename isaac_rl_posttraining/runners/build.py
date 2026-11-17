# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one training run from a composed ``train`` config.

The algorithm registry (``algorithms/registry.py``) names each algorithm's
loop, action space and launcher; this module validates a config, asks the
launcher for its plan and its run, and adds what every run shares: the
data-spec check, the placement plan and the schedule. The launch plan
summarizes what a run will do from the config alone, before any policy,
simulator or learner loads; the dry run prints it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields

from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import (
    AlgorithmSpec,
    algorithm_spec,
    load_target,
)
from isaac_rl_posttraining.configs import TrainConfig, selected_algorithm_name
from isaac_rl_posttraining.configs.normalize import normalize_train_config
from isaac_rl_posttraining.runners import off_policy_launch
from isaac_rl_posttraining.runners.launchers import Launcher, LoopPlan, StepSession
from isaac_rl_posttraining.runners.plan_checks import (
    PlanChecks,
    check_base_model_servable,
    check_base_model_steerable,
    check_learner_budget,
    check_rollout_workers,
    check_runtime_installed,
    check_served_policy,
    check_server_replicas,
)
from isaac_rl_posttraining.runners.rlinf_placement import check_rlinf_accelerators
from isaac_rl_posttraining.runners.run_placement import (
    RolloutShape,
    native_runs_processes,
    placement_config_of,
    plan_run_placement,
    runtime_features,
)
from isaac_rl_posttraining.runners.run_schedule import schedule_spec
from isaac_rl_posttraining.runners.simulator_entries import (
    describe_evaluation,
    describe_launch_inputs,
    require_evaluation_shape,
    require_server_addresses,
    resolve_simulator_entry,
    selected_entry,
)
from isaac_rl_posttraining.runners.spec_checks import (
    SpecReport,
    check_data_specs,
    environment_spec_for_plan,
)
from isaac_rl_posttraining.runtimes import policy_serving_mode
from isaac_rl_posttraining.runtimes.base import RuntimeFeatures
from isaac_rl_posttraining.runtimes.placement import PlacementPlan
from isaac_rl_posttraining.runtimes.steering_rollouts import resolve_image_keys
from isaac_rl_posttraining.training.schedule import ScheduleSpec
from isaac_rl_posttraining.utils.config_utils import convert_hydra_to_dataclass

__all__ = [
    "PlanReport",
    "TrainPlan",
    "algorithm_family",
    "build_run",
    "open_session",
    "plan_report",
    "run_config",
    "selected_algorithm",
    "train_plan",
]


def selected_algorithm(cfg: DictConfig) -> AlgorithmSpec:
    """The registry entry of the algorithm a composed config selects."""
    return algorithm_spec(selected_algorithm_name(cfg))


def algorithm_family(cfg: DictConfig) -> str:
    """Today's ``algorithm.family`` of a composed config (``steering``, ``edit`` or ``on_policy``), from the registry."""
    return selected_algorithm(cfg).family


def _check_schedule(
    cfg: DictConfig, spec: AlgorithmSpec, features: RuntimeFeatures | None = None
) -> ScheduleSpec:
    schedule = schedule_spec(cfg, on_policy=spec.loop == "on_policy")
    if schedule.kind not in spec.schedules:
        raise ValueError(
            f"schedule={schedule.kind} is not implemented by the {spec.loop} loop of {spec.name}; "
            f"it runs {', '.join(spec.schedules)}"
        )
    # Overlapped publishes while a round is in flight, which only an async collector bounds by the rollout timeout.
    if (
        schedule.kind in ("async", "overlapped")
        and features is not None
        and not features.async_collect
    ):
        raise ValueError(
            f"schedule={schedule.kind} needs a runtime that collects asynchronously; {features.name} does not"
        )
    return schedule


def run_config(cfg: DictConfig) -> DictConfig:
    """The validated, resolved view of a composed config that the runs read.

    The config is first normalized (registry family, ``algorithm.*`` keys,
    renamed keys, episode cap), then round-tripped through :class:`TrainConfig`,
    which checks every section against the schema, fills the defaults of keys a
    yaml leaves out and resolves the interpolations. The result stays
    unstructured because the runs read their optional keys with defaults.
    """
    normalized = normalize_train_config(cfg)
    resolved = OmegaConf.create(
        asdict(convert_hydra_to_dataclass(normalized, TrainConfig))
    )
    OmegaConf.set_struct(resolved, False)
    return resolved


_UNRESOLVED = "unresolved (see plan.checks)"
#: TrainPlan fields that are objects for Python callers rather than ``plan.<field>=`` lines.
_OBJECT_FIELDS = ("placement_plan", "spec_report", "check_report")


@dataclass(frozen=True)
class TrainPlan:
    """What a run will do, resolved from the config alone."""

    simulator: str
    entry: str
    image_keys: str
    layout: str
    evaluation_mode: str
    evaluation: str
    algorithm: str
    family: str
    loop: str
    action_space: str
    policy_family: str
    runtime: str
    placement: str
    schedule: str
    schedule_kind: str
    checkpoint_dir: str
    launch_inputs: str
    data_specs: str
    checks: str
    placement_plan: PlacementPlan = field(repr=False, compare=False)
    spec_report: SpecReport = field(repr=False, compare=False)
    check_report: PlanChecks = field(repr=False, compare=False)

    @classmethod
    def line_fields(cls) -> tuple[str, ...]:
        """The fields printed as ``plan.<field>=<value>`` lines (``plan.placement.*`` lines join them)."""
        return tuple(
            item.name for item in fields(cls) if item.name not in _OBJECT_FIELDS
        )

    def lines(self) -> list[str]:
        """The plan as ``plan.<field>=<value>`` lines, the dry run's output; ``plan.checks`` comes last."""
        return _plan_lines(vars(self), self.placement_plan)


def _plan_lines(
    values: Mapping[str, object], placement_plan: PlacementPlan | None
) -> list[str]:
    """``plan.<field>=`` lines in field order, then the placement, then ``plan.checks``."""
    lines = [
        f"plan.{name}={values.get(name, _UNRESOLVED)}"
        for name in TrainPlan.line_fields()
        if name != "checks"
    ]
    if placement_plan is not None:
        lines += placement_plan.lines()
    return lines + [f"plan.checks={values['checks']}"]


@dataclass(frozen=True)
class PlanReport:
    """Everything the plan resolved and every check it ran, even when some failed.

    ``plan`` is ``None`` when a check failed; ``lines`` then shows what was
    resolved (``unresolved`` for the rest) and ends with ``plan.checks``.
    """

    lines: list[str]
    checks: PlanChecks
    plan: TrainPlan | None

    def raise_errors(self) -> TrainPlan:
        """The plan, or every failed check as one error."""
        self.checks.raise_errors()
        assert self.plan is not None
        return self.plan


def _launcher(spec: AlgorithmSpec) -> Launcher:
    return load_target(spec.launcher)


def plan_report(cfg: DictConfig) -> PlanReport:
    """Resolve the launch plan of a composed config, running every check instead of stopping at the first.

    The schema check of :func:`run_config` still raises at once (nothing
    else can be read from a config that does not parse). Missing simulator
    addresses are launch inputs, listed in ``plan.launch_inputs``; the
    launch requires them.
    """
    cfg = run_config(cfg)
    spec = selected_algorithm(cfg)
    checks = PlanChecks()
    values: dict[str, str] = {
        "algorithm": spec.name,
        "family": spec.family,
        "loop": spec.loop,
        "action_space": spec.action_space,
        "policy_family": spec.policy_family,
        "runtime": str(cfg.runtime.name),
        "checkpoint_dir": (
            f"{off_policy_launch.output_root()}/{cfg.logging.wandb_name}_<timestamp>/checkpoints"
        ),
    }
    loop = checks.run("loop config", lambda: _launcher(spec).plan(cfg))
    entry = selected_entry(cfg) if loop is None else loop.entry
    shape = None if loop is None else _rollout_shape(loop)
    if loop is not None:
        values.update(
            simulator=loop.entry.name,
            layout=loop.layout,
            evaluation_mode=loop.evaluation_mode,
            schedule=loop.budget,
            placement=_placement(cfg, loop.learner_ranks, spec),
        )
    if entry is not None:
        values.update(
            entry=repr(entry), launch_inputs=describe_launch_inputs(cfg, entry)
        )
        if spec.loop != "on_policy":
            checks.run(
                "evaluation layout", lambda: require_evaluation_shape(cfg, entry)
            )
            values["evaluation"] = describe_evaluation(cfg, entry)
        else:
            episodes = OmegaConf.select(cfg, "ppo.eval_episodes_per_environment")
            values["evaluation"] = (
                f"{episodes} episode(s) per environment (ppo.eval_episodes_per_environment)"
            )
    # What the simulator emits and accepts must match what the policy reads and emits before anything loads.
    data_specs = checks.run(
        "simulator specs", lambda: check_data_specs(cfg, algorithm=spec.name)
    )
    if data_specs is not None:
        checks.add(
            data_specs.comparison,
            f"data specs of simulator {data_specs.simulator} and the policy disagree",
        )
        values["data_specs"] = data_specs.summary()
    if entry is not None:
        image_keys = checks.run(
            "image keys",
            lambda: resolve_image_keys(cfg, entry, environment_spec_for_plan(cfg)),
        )
        if image_keys is not None:
            values["image_keys"] = (
                f"{', '.join(image_keys.keys) or 'none'} from {image_keys.source}"
            )
    features = checks.run("runtime", lambda: runtime_features(cfg))
    note = checks.run("runtime installed", lambda: check_runtime_installed(cfg))
    if note:
        checks.checked.remove("runtime installed")
        checks.unverified.append(note)
    schedule = checks.run("schedule", lambda: _check_schedule(cfg, spec, features))
    if schedule is not None:
        lag = schedule.policy_lag_bound
        values["schedule_kind"] = (
            f"{schedule.kind}, max policy lag {'unbounded' if lag is None else lag}"
        )
        if schedule.kind == "async":
            every = schedule.updates_per_publish
            values["schedule_kind"] += (
                ", publish once per training burst"
                if every is None
                else f", publish every {every} update(s)"
            )
    placement_plan = None
    if features is not None:
        placement_plan = checks.run(
            "placement", lambda: plan_run_placement(cfg, spec, features, shape)
        )
    if placement_plan is not None and values["runtime"] == "rlinf":
        checks.run(
            "RLInf accelerators",
            lambda: check_rlinf_accelerators(cfg, placement_plan),
        )
    if placement_plan is not None and entry is not None:
        checks.run(
            "policy-server replicas",
            lambda: check_server_replicas(cfg, placement_plan, entry),
        )
    if placement_plan is not None and loop is not None and features is not None:
        checks.run(
            "served policy",
            lambda: check_served_policy(spec.name, loop, placement_plan, features),
        )
    if placement_plan is not None:
        note = checks.run(
            "servable base_model",
            lambda: check_base_model_servable(cfg, placement_plan),
        )
        if note:
            checks.checked.remove("servable base_model")
            checks.unverified.append(note)
    if spec.loop == "off_policy":
        note = checks.run(
            "steerable base_model", lambda: check_base_model_steerable(cfg)
        )
        if note:
            checks.checked.remove("steerable base_model")
            checks.unverified.append(note)
    checks.run("rollout workers", lambda: check_rollout_workers(cfg))
    if spec.loop == "off_policy" and (budget_warning := check_learner_budget(cfg)):
        checks.warnings.append(budget_warning)
    values["checks"] = checks.summary()
    if checks.errors:
        return PlanReport(_plan_lines(values, placement_plan), checks, None)
    assert placement_plan is not None and data_specs is not None
    plan = TrainPlan(
        **values,
        placement_plan=placement_plan,
        spec_report=data_specs,
        check_report=checks,
    )
    return PlanReport(plan.lines(), checks, plan)


def train_plan(cfg: DictConfig) -> TrainPlan:
    """Resolve the launch plan of a composed config, raising every failed check (one error, or all together)."""
    return plan_report(cfg).raise_errors()


def build_run(cfg: DictConfig) -> Callable[[], str | None]:
    """The run of a composed config as a no-argument callable, built by the algorithm's launcher.

    Building validates the config against the loop's requirements; calling
    the result loads the frozen policy, the environments and the learner,
    trains, and releases them.
    """
    cfg = run_config(cfg)
    spec = selected_algorithm(cfg)
    _check_schedule(cfg, spec, runtime_features(cfg))
    require_server_addresses(cfg, resolve_simulator_entry(cfg))
    if missing := check_runtime_installed(cfg):
        raise ValueError(missing)
    return _launcher(spec).build(cfg)


def open_session(cfg: DictConfig) -> StepSession:
    """The run of a composed config opened for step-wise control, by the algorithm's launcher.

    The same checks as :func:`build_run` come first. Raises
    ``NotImplementedError`` when the algorithm's launcher offers no session.
    """
    cfg = run_config(cfg)
    spec = selected_algorithm(cfg)
    _check_schedule(cfg, spec, runtime_features(cfg))
    require_server_addresses(cfg, resolve_simulator_entry(cfg))
    open_launcher_session = getattr(_launcher(spec), "open_session", None)
    if open_launcher_session is None:
        raise NotImplementedError(
            f"algorithm={spec.name} has no step-wise control: its launcher ({spec.launcher}) "
            "defines no open_session(cfg). Use Run.loop()."
        )
    return open_launcher_session(cfg)


def _rollout_shape(loop: LoopPlan) -> RolloutShape | None:
    """The training rollout workers a launcher's plan describes (``None`` when it does not count them)."""
    if loop.rollout_workers is None:
        return None
    return RolloutShape(
        workers=loop.rollout_workers,
        batched=loop.batched_workers,
        servable=loop.served_policy,
        gpu_simulator=loop.entry.simulates_on_gpu,
    )


def _placement(cfg: DictConfig, learner_ranks: int, spec: AlgorithmSpec) -> str:
    runtime = str(cfg.runtime.name)
    if runtime == "native":
        if not native_runs_processes(cfg):
            return f"{learner_ranks} learner rank(s) ({cfg.learner.device}) and the rollout workers in this process"
        learners = (
            f"{learner_ranks} learner rank actor(s) in one process group"
            if learner_ranks > 1
            else f"{learner_ranks} learner rank(s) in this process"
        )
        nodes = int(cfg.runtime.native.num_nodes)
        return (
            f"Ray actors on {nodes} node(s): {learners}, one actor per rollout worker"
            + (
                ", policy-server actors"
                if policy_serving_mode(placement_config_of(cfg))[0]
                in ("served", "base")
                else ""
            )
            + "; the GPUs of plan.placement.*"
        )
    if runtime == "rlinf":
        settings = cfg.runtime.rlinf
        if learner_ranks == 1:
            learners = "1 learner rank in this process"
        else:
            first = int(settings.learner_first_accelerator)
            last = first + max(learner_ranks, 1) - 1
            learners = (
                f"{learner_ranks} learner rank(s) on accelerator(s) {first}-{last}"
            )
        return (
            f"{learners}; policy servers and rollout workers from accelerator {settings.first_accelerator} "
            f"on {settings.num_nodes} node(s); the GPUs of plan.placement.*"
        )
    raise ValueError(f"unknown runtime {runtime!r}")
