# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dry run's placement lines describe what the selected runtime runs.

``plan.layout`` counts the rollout workers from the simulator entry; the
``plan.placement.*`` lines must count the same workers, put them where the
runtime puts them (the native runtime in one process keeps every role on its
one device; RLInf serves the policy from ``first_accelerator`` and runs the
workers next to it on the CPU) and show the policy serving the runtime picks
for ``placement.policy.mode=auto``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pytest

from isaac_rl_posttraining.data.spec import ActionSpec, ArraySpec, ObservationSpec
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
from isaac_rl_posttraining.policies import (
    MockBaseModelPolicy,
    base_model_servable,
    base_model_steerable,
    register_base_model,
)
from isaac_rl_posttraining.runners import run_placement
from isaac_rl_posttraining.runners.build import plan_report, train_plan
from isaac_rl_posttraining.runners.run_placement import RolloutShape, resolved_placement
from isaac_rl_posttraining.train import compose_train_config


class PlanRobot(SimulatorEntry):
    """One environment per rollout worker, evaluated on the training workers."""

    name = "planmatch_robot"

    def make_environment(self, spec: Any, phase: Any, context: Any) -> Any:
        raise AssertionError("the plan must not build an environment")

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        return ObservationSpec(
            {
                "state": ArraySpec((7,), "float32"),
                "image": ArraySpec((16, 16, 3), "uint8"),
            }
        )

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        return ActionSpec(width=7)


class DedicatedEvalRobot(PlanRobot):
    """The same robot, evaluated by rollout workers of its own."""

    name = "planmatch_dedicated"
    evaluation: ClassVar[Any] = "dedicated"


def _robot(entry: type[PlanRobot], num_envs: int) -> tuple[str, ...]:
    return (
        "algorithm=dsrl_sac",
        "base_model=mock",
        f"env.env_name={entry.name}/reach",
        f"+simulator.entry={__name__}:{entry.__name__}",
        "base_model.action_dim=7",
        "dsrl.noise_dim=7",
        "env.max_episode_steps=8",
        f"training.num_envs={num_envs}",
        f"eval.eval_num_envs={num_envs}",
        f"eval.eval_num_episodes={12 * num_envs}",
    )


def _lines(*overrides: str) -> dict[str, str]:
    report = plan_report(compose_train_config(list(overrides)))
    return dict(line.split("=", 1) for line in report.lines)


def _workers(layout: str) -> int:
    match = re.match(r"(\d+) rollout worker", layout)
    assert match, layout
    return int(match.group(1))


def test_native_in_one_process_keeps_every_role_on_its_one_gpu() -> None:
    """The V2D 3-GPU case: two workers and a learner, all in this process on GPU 0."""
    lines = _lines(*_robot(PlanRobot, 2), "placement.gpus_per_node=3")
    assert lines["plan.layout"].startswith("2 rollout worker(s)")
    assert lines["plan.placement.resources"] == (
        "node0 (1 gpus); roles share gpus; every role in this process"
    )
    assert (
        lines["plan.placement.envs"] == "2 worker(s) in this process on node0 gpus [0]"
    )
    assert (
        lines["plan.placement.learner"] == "1 rank(s) in this process on node0 gpus [0]"
    )
    # Workers collecting one after another are served without waiting.
    assert lines["plan.placement.policy.base+actor"] == (
        "served by one in-process policy server on node0 gpus [0], max_batch=unbounded, max_wait_ms=0.0"
    )


def test_a_served_policy_in_one_process_waits_for_concurrent_workers() -> None:
    lines = _lines(*_robot(PlanRobot, 3), "placement.policy.mode=served")
    assert lines["plan.placement.policy.base+actor"].endswith("max_wait_ms=5.0")
    sequential = _lines(
        *_robot(PlanRobot, 3),
        "placement.policy.mode=served",
        "runtime.native.concurrent_collection=false",
    )
    assert sequential["plan.placement.policy.base+actor"].endswith("max_wait_ms=0.0")


def test_batched_workers_keep_their_policy_under_auto() -> None:
    """Isaac Lab's scenes in one batched worker: auto keeps the policy in the worker on both runtimes."""
    stack = (
        "algorithm=dsrl_sac",
        "simulator=isaaclab_franka_stack",
        "env.env_name=isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "base_model=mock",
        "base_model.action_dim=7",
        "dsrl.noise_dim=7",
        "training.num_envs=8",
        "eval.eval_num_envs=8",
    )
    lines = _lines(*stack)
    assert _workers(lines["plan.layout"]) == 1
    assert lines["plan.placement.envs"].startswith("1 worker(s)")
    assert (
        lines["plan.placement.policy.base+actor"]
        == "local (a copy in every rollout worker)"
    )


def test_rlinf_serves_unbatched_workers_from_the_first_accelerator() -> None:
    """The LIBERO RLInf case: four CPU rollout workers and one policy server on GPU 1."""
    lines = _lines(*_robot(PlanRobot, 4), "runtime=rlinf", "placement.gpus_per_node=2")
    assert lines["plan.layout"].startswith("4 rollout worker(s)")
    assert lines["plan.placement.envs"] == "4 worker(s) on the cpu"
    assert lines["plan.placement.policy.base+actor"].startswith(
        "served by 1 replica(s) x 1 gpu on node0 gpus [1]"
    )
    assert lines["plan.placement.learner"] == "1 rank(s) on node0 gpus [0]"
    assert (
        "policy servers and rollout workers from accelerator 1"
        in lines["plan.placement"]
    )
    assert "0 error(s)" in lines["plan.checks"]


@pytest.mark.parametrize("visible", [0, 1])
def test_rlinf_plans_onto_the_runs_gpus_on_a_machine_with_too_few(
    monkeypatch, visible: int
) -> None:
    """A dry run on a GPU-free or one-GPU machine plans the LIBERO RLInf case as on its own GPUs."""
    monkeypatch.setattr(run_placement, "_visible_gpus", lambda: visible)
    lines = _lines(*_robot(PlanRobot, 4), "runtime=rlinf")
    assert "0 error(s)" in lines["plan.checks"]
    assert (
        "policy servers and rollout workers from accelerator 1"
        in lines["plan.placement"]
    )


def test_rlinf_checks_a_stated_gpu_count_against_the_plan(monkeypatch) -> None:
    monkeypatch.setattr(run_placement, "_visible_gpus", lambda: 8)
    lines = _lines(*_robot(PlanRobot, 4), "runtime=rlinf", "placement.gpus_per_node=1")
    assert "1 error(s)" in lines["plan.checks"]


def test_rlinf_local_workers_take_one_accelerator_each() -> None:
    lines = _lines(
        *_robot(PlanRobot, 2),
        "runtime=rlinf",
        "placement.gpus_per_node=4",
        "placement.policy.mode=local",
    )
    assert (
        lines["plan.placement.policy.base+actor"]
        == "local (a copy in every rollout worker)"
    )
    assert lines["plan.placement.envs"] == "2 worker(s) on node0 gpus [1, 2]"


def test_rlinf_local_workers_share_accelerators_when_there_are_fewer_than_workers() -> (
    None
):
    """Four local-policy workers on one rollout card, and an evaluation group next to them, plan by default."""
    overrides = (
        *_robot(DedicatedEvalRobot, 4),
        "runtime=rlinf",
        "placement.gpus_per_node=2",
        "placement.policy.mode=local",
    )
    lines = _lines(*overrides)
    assert lines["plan.placement.envs"] == "4 worker(s) on node0 gpus [1]"
    assert "roles share gpus" in lines["plan.placement.resources"]
    assert "0 error(s)" in lines["plan.checks"]


def test_rlinf_base_served_workers_run_their_actor_on_a_gpu() -> None:
    """The actor stays in the workers next to a served base model, so they get the rollout cards."""
    lines = _lines(
        *_robot(PlanRobot, 4),
        "runtime=rlinf",
        "placement.gpus_per_node=8",
        "placement.parts.base.mode=served",
    )
    assert lines["plan.placement.policy.base"].startswith(
        "served by 1 replica(s) x 1 gpu on node0 gpus [1]"
    )
    assert lines["plan.placement.envs"] == "4 worker(s) on node0 gpus [2, 3, 4, 5]"


def test_rlinf_learner_ranks_may_not_share_with_rollout_workers() -> None:
    overrides = (
        *_robot(PlanRobot, 2),
        "algorithm=dsrl_ppo",
        "runtime=rlinf",
        "placement.gpus_per_node=4",
        "placement.policy.mode=local",
        "placement.learner.ranks=2",
        "runtime.rlinf.first_accelerator=1",
    )
    report = plan_report(compose_train_config(list(overrides)))
    (error,) = report.checks.errors
    assert "the learner ranks and the rollout workers on accelerator(s) [1]" in str(
        error
    )
    train_plan(compose_train_config([*overrides[:-1]]))


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("placement.gpus_per_node=3",),
        ("placement.policy.mode=served",),
        ("placement.policy.mode=local", "placement.gpus_per_node=2"),
        ("runtime=rlinf", "placement.gpus_per_node=3"),
        ("runtime=rlinf", "placement.policy.mode=local", "placement.gpus_per_node=8"),
        ("placement.gpus_per_node=8", "runtime.native.processes=ray"),
    ],
)
def test_the_placement_counts_the_layouts_workers(extra: tuple[str, ...]) -> None:
    cfg = compose_train_config([*_robot(PlanRobot, 3), *extra])
    plan = train_plan(cfg)
    assert plan.placement_plan.envs.processes == _workers(plan.layout) == 3
    # The run resolves the same plan once it knows its workers.
    from isaac_rl_posttraining.runners.build import run_config

    resolved = resolved_placement(run_config(cfg), RolloutShape(workers=3))
    assert resolved == plan.placement_plan


class StatefulBaseModel(MockBaseModelPolicy):
    """A mock base model that keeps one episode's state per instance."""

    servable = False


def test_a_base_model_that_cannot_be_served_is_refused_by_the_dry_run() -> None:
    """DreamZero's served-model client keeps one session; the plan refuses serving it before anything loads."""
    assert base_model_servable("dreamzero_server") is False
    assert base_model_servable("dreamzero_local") is True
    assert base_model_servable("mock") is True
    register_base_model(
        "planmatch_stateful", f"{__name__}:StatefulBaseModel", override=True
    )
    stateful = (*_robot(PlanRobot, 3), "base_model.name=planmatch_stateful")
    for placement in (
        "placement.policy.mode=served",
        "placement.parts.base.mode=served",
    ):
        report = plan_report(compose_train_config([*stateful, placement]))
        assert report.plan is None
        section = placement.rsplit(".mode", 1)[0]
        assert any(
            "planmatch_stateful keeps one episode's state per instance" in str(error)
            and f"set {section}.mode=local" in str(error)
            for error in report.checks.errors
        ), report.checks.errors
    train_plan(compose_train_config([*stateful, "placement.policy.mode=local"]))
    # auto shares an in-process server, which a one-environment evaluation of such a base model uses.
    train_plan(
        compose_train_config(
            list(_robot(PlanRobot, 1)) + ["base_model.name=planmatch_stateful"]
        )
    )


class NativeOnlyBaseModel(MockBaseModelPolicy):
    """A mock base model that cannot take steering latents."""

    supports_steering = False


def test_a_base_model_that_cannot_be_steered_is_refused_by_the_dry_run() -> None:
    """A training run on a native-only base model fails in the dry run, not after the policy loads."""
    assert base_model_steerable("dreamzero_server") is False
    assert base_model_steerable("dreamzero_local") is True
    assert base_model_steerable("mock") is True
    register_base_model(
        "planmatch_native_only", f"{__name__}:NativeOnlyBaseModel", override=True
    )
    native_only = (*_robot(PlanRobot, 1), "base_model.name=planmatch_native_only")
    report = plan_report(compose_train_config(list(native_only)))
    assert report.plan is None
    assert any(
        "planmatch_native_only is native-only" in str(error)
        and "training.num_rollouts=0 eval.eval_base=true" in str(error)
        for error in report.checks.errors
    ), report.checks.errors
    # Evaluating the frozen policy alone stays allowed.
    train_plan(
        compose_train_config(
            [
                *native_only,
                "training.num_rollouts=0",
                "eval.eval_base=true",
                "eval.eval_on_first_step=true",
            ]
        )
    )


def test_native_ray_runs_the_placement_the_dry_run_printed(monkeypatch) -> None:
    """The runtime's plan sizes the roles by the same rollout workers the dry run counted."""
    from isaac_rl_posttraining.runners import run_placement

    overrides = (
        *_robot(PlanRobot, 1),
        "algorithm=dsrl_ppo",
        "placement.gpus_per_node=4",
        "runtime.native.processes=ray",
        "placement.learner.ranks=2",
    )
    cfg = compose_train_config(list(overrides))
    report = plan_report(cfg)
    assert "plan.placement.envs=1 worker(s) on node0 gpus [0]" in report.lines
    opened = {}
    monkeypatch.setattr(
        run_placement,
        "create_runtime_from_config",
        lambda runtime, placement, plan: opened.setdefault("plan", plan),
    )
    run_placement.open_runtime(cfg, RolloutShape(workers=1))
    assert opened["plan"] == report.plan.placement_plan


def test_an_edit_algorithm_with_a_chunked_actor_is_refused_by_the_dry_run() -> None:
    """SAC-edit's single MLP covers the whole decoded chunk; the plan says so before the policy loads."""
    edit = (*_robot(PlanRobot, 1), "algorithm=sac_edit")
    report = plan_report(compose_train_config([*edit, "training.chunk_size=4"]))
    assert report.plan is None
    assert any(
        "sac_edit uses a single MLP over the full action chunk; set training.chunk_size=null"
        in str(error)
        for error in report.checks.errors
    ), report.checks.errors
    train_plan(compose_train_config([*edit, "training.chunk_size=null"]))
