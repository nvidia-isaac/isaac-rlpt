# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When runtime=native runs its roles as Ray actors, and what a single-process run keeps.

No test here starts Ray: the multi-process runs themselves are in
``tests/integration/test_native_ray_*.py`` (the ``native_ray`` marker).
"""

from __future__ import annotations

import sys
import textwrap
import types

import pytest

from isaac_rl_posttraining import train as facade
from isaac_rl_posttraining.configs.schema import (
    LearnerPlacementConfig,
    PlacementConfig,
    PolicyPlacementConfig,
    RuntimeConfig,
)
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.envs.simulator_entry import (
    SimulatorEntry,
    WorkerLayout,
    requested_layout,
)
from isaac_rl_posttraining.runners.run_placement import (
    native_runs_processes,
    resolved_placement,
)
from isaac_rl_posttraining.runtimes import create_runtime_from_config
from isaac_rl_posttraining.runtimes.native import NativeRuntime, wants_processes
from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Placement,
    Resources,
    Served,
    plan_placement,
)


@pytest.mark.parametrize(
    ("placement", "num_nodes", "expected"),
    [
        (Placement(), 1, False),
        (Placement(envs=Envs(workers=8)), 1, False),
        (
            Placement(policy=Served()),
            1,
            False,
        ),  # one in-process policy server, as today
        (Placement(policy=Served(replicas=2)), 1, True),
        (Placement(policy=Served(gpus=1)), 1, True),
        (Placement(parts={"base": Served(replicas=2)}), 1, True),
        (Placement(learner=Learner(ranks=2)), 1, True),
        (Placement(learner=Learner(ranks=1)), 1, False),
        (Placement(envs=Envs(gpus=2)), 1, True),
        (Placement(envs=Envs(gpus=0)), 1, False),
        (Placement(), 2, True),
        (Placement(nodes=("0", "1")), 1, True),
    ],
)
def test_the_placement_decides_when_native_runs_several_processes(
    placement, num_nodes, expected
):
    assert wants_processes(placement, num_nodes=num_nodes) is expected


def test_without_a_plan_native_is_todays_single_process_runtime():
    runtime = create_runtime_from_config(RuntimeConfig(name="native"))
    assert isinstance(runtime, NativeRuntime) and not runtime.multi_process
    assert not runtime.features.distributed_learner
    with pytest.raises(RuntimeError, match="one learner rank in this process"):
        runtime.open_learner_group(lambda *args: None, num_learner_ranks=2, name="x")
    runtime.close()


def test_a_plan_turns_native_into_a_multi_process_runtime_without_starting_ray():
    plan = plan_placement(
        Placement(learner=Learner(ranks=2)),
        Resources.uniform(1, 0),
        served_supported=True,
        distributed_learner=True,
    )
    runtime = create_runtime_from_config(
        RuntimeConfig(name="native"),
        PlacementConfig(learner=LearnerPlacementConfig(ranks=2)),
        plan=plan,
    )
    assert runtime.multi_process and runtime.features.distributed_learner
    runtime.close()  # Ray starts at the first group; nothing to stop yet
    with pytest.raises(ValueError, match="placement plan"):
        create_runtime_from_config(
            RuntimeConfig(name="native", native={"processes": "ray"})
        )
    with pytest.raises(ValueError, match="forbids"):
        create_runtime_from_config(
            RuntimeConfig(name="native", native={"processes": "single"}),
            PlacementConfig(policy=PolicyPlacementConfig(mode="served")),
            plan=plan,
        )


class _OldEntry(SimulatorEntry):
    """An entry written before ``worker_layout`` took a worker count."""

    name = "old_entry"

    def make_environment(self, spec, phase, context):
        raise NotImplementedError

    def worker_layout(self, num_envs: int, num_servers: int = 1) -> WorkerLayout:
        return WorkerLayout(workers=1, batched=True)


def test_an_entry_without_the_workers_argument_keeps_its_own_layout():
    assert requested_layout(_OldEntry(), 8, workers=4) == WorkerLayout(1, batched=True)


ROBOT = textwrap.dedent(
    """
    import gymnasium as gym
    import numpy as np

    from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry


    class RobotEnv(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
                "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)


    class Robot(SimulatorEntry):
        name = "plan_robot"

        def make_environment(self, spec, phase, context):
            return RobotEnv()
    """
)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    package = tmp_path / "plan_robot"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "runtime.py").write_text(ROBOT)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    providers._SIMULATOR_ENTRIES.pop("plan_robot", None)


def _compose(*overrides: str):
    return facade.compose_train_config(
        [
            "algorithm=dsrl_sac",
            "base_model=mock",
            "env.env_name=plan_robot/reach",
            "+simulator.entry=plan_robot.runtime:Robot",
            "base_model.action_dim=7",
            "dsrl.noise_dim=7",
            "model.dinov2_model=null",
            "training.num_envs=4",
            "env.max_episode_steps=8",
            "training.chunk_size=16",
            "base_model.action_horizon=16",
            "dsrl.action_exec_len=8",
            "placement.gpus_per_node=8",
            *overrides,
        ]
    )


def test_the_dry_run_shows_ray_actors_and_their_gpus(robot):
    from isaac_rl_posttraining.runners.build import run_config, train_plan

    cfg = _compose(
        "placement.policy.mode=served",
        "placement.policy.replicas=2",
        "placement.learner.ranks=2",
    )
    assert native_runs_processes(run_config(cfg))
    lines = "\n".join(train_plan(cfg).lines())
    assert (
        "plan.placement=Ray actors on 1 node(s): 2 learner rank actor(s) in one process group"
        in lines
    )
    assert "plan.placement.policy.base+actor=served by 2 replica(s)" in lines
    assert "plan.placement.learner=2 rank(s) on node0 gpus" in lines


def test_served_policy_workers_without_a_gpu_simulator_take_no_gpu(robot):
    """Thin clients of a served policy leave the GPUs to the servers; a local actor keeps its worker's GPU."""
    from isaac_rl_posttraining.runners.build import train_plan

    served = train_plan(
        _compose("placement.policy.mode=served", "runtime.native.processes=ray")
    )
    assert "plan.placement.envs=4 worker(s) on the cpu" in served.lines()
    assert served.placement_plan.part("base").gpus
    actor_local = train_plan(
        _compose("placement.parts.base.mode=served", "runtime.native.processes=ray")
    )
    assert actor_local.placement_plan.envs.gpus


def test_a_single_learner_rank_is_planned_on_the_controllers_gpu(robot):
    from isaac_rl_posttraining.runners.build import run_config

    cfg = run_config(
        _compose("placement.policy.mode=served", "placement.policy.replicas=2")
    )
    plan = resolved_placement(cfg)
    assert plan.learner.by_node() == {"node0": (0,)} and plan.learner.processes == 1
    assert 0 not in plan.envs.by_node()["node0"]
    # The same config in one process keeps today's plan.
    single = run_config(
        _compose(
            "placement.policy.mode=served",
            "placement.policy.replicas=2",
            "runtime.native.processes=single",
        )
    )
    assert not native_runs_processes(single)


def test_the_dry_run_shows_the_base_model_served_and_the_actor_local(robot):
    from isaac_rl_posttraining.runners.build import run_config, train_plan
    from isaac_rl_posttraining.runners.run_placement import (
        placement_config_of,
        runtime_config_of,
    )

    cfg = run_config(_compose("placement.parts.base.mode=served"))
    assert not native_runs_processes(cfg)  # one in-process base-model server
    lines = train_plan(cfg).lines()
    assert "plan.placement.policy.actor=local (a copy in every rollout worker)" in lines
    assert any(
        line.startswith(
            "plan.placement.policy.base=served by one in-process policy server"
        )
        for line in lines
    )
    runtime = create_runtime_from_config(
        runtime_config_of(cfg), placement_config_of(cfg)
    )
    assert (runtime.config.policy_mode, runtime.config.base_mode) == ("local", "served")
    several = run_config(
        _compose("placement.parts.base.mode=served", "placement.parts.base.replicas=2")
    )
    assert native_runs_processes(several)
    plan = resolved_placement(several)
    assert plan.base_served_alone and plan.part("base").processes == 2
    assert plan.part("actor").served is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (("placement.parts.actor.mode=served",), "only the base model"),
        (
            ("algorithm=ppo", "placement.parts.base.mode=served"),
            "placement.parts.base names a policy part .* trains as a whole.*placement.policy.mode=served",
        ),
    ],
)
def test_part_placements_a_runtime_or_algorithm_cannot_run_fail_in_the_dry_run(
    robot, overrides, match
):
    from isaac_rl_posttraining.runners.build import run_config

    with pytest.raises(ValueError, match=match):
        resolved_placement(run_config(_compose(*overrides)))


@pytest.mark.parametrize(
    ("policy", "parts", "expected"),
    [
        ("auto", {}, "auto"),
        ("served", {}, "served"),
        ("local", {"base": "served"}, "base"),
        ("auto", {"base": "served", "actor": "local"}, "base"),
        ("auto", {"base": "served", "actor": "served"}, "served"),
        ("served", {"base": "auto", "actor": "auto"}, "served"),
        ("local", {"policy": "served"}, "served"),
    ],
)
def test_policy_and_parts_lower_to_one_serving_mode(policy, parts, expected):
    from isaac_rl_posttraining.runtimes import policy_serving_mode

    placement = PlacementConfig(
        policy=PolicyPlacementConfig(mode=policy),
        parts={name: PolicyPlacementConfig(mode=mode) for name, mode in parts.items()},
    )
    assert policy_serving_mode(placement)[0] == expected


def test_an_actor_served_without_its_base_model_is_refused():
    from isaac_rl_posttraining.runtimes import policy_serving_mode

    for placement in (
        PlacementConfig(parts={"actor": PolicyPlacementConfig(mode="served")}),
        PlacementConfig(
            policy=PolicyPlacementConfig(mode="served"),
            parts={"base": PolicyPlacementConfig(mode="local")},
        ),
    ):
        with pytest.raises(ValueError, match="only the base model"):
            policy_serving_mode(placement)


def test_a_python_placement_with_parts_composes_into_the_config(robot):
    from isaac_rl_posttraining.runners.build import run_config
    from isaac_rl_posttraining.runners.run_placement import (
        placement_from_config,
        placement_overrides,
    )

    wanted = Placement(parts={"base": Served(replicas=2, max_batch=8)})
    cfg = run_config(_compose(*placement_overrides(wanted)))
    assert placement_from_config(cfg).parts == {"base": Served(replicas=2, max_batch=8)}


def _fake_ray(monkeypatch, started):
    constants = types.SimpleNamespace(RAY_ENABLE_UV_RUN_RUNTIME_ENV=True)
    monkeypatch.setitem(sys.modules, "ray._private", types.ModuleType("ray._private"))
    monkeypatch.setitem(sys.modules, "ray._private.ray_constants", constants)

    def init(address=None, **kwargs):
        started.append(constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV)
        if address == "auto":
            raise ConnectionError("no cluster")

    ray = types.SimpleNamespace(init=init, is_initialized=lambda: False)
    from isaac_rl_posttraining.runtimes.native_ray import cluster

    monkeypatch.setattr(cluster, "import_ray", lambda: ray)
    return constants, ray


def test_ray_actors_run_the_drivers_interpreter_not_a_uv_run_rebuild(monkeypatch):
    from isaac_rl_posttraining.runtimes.native_ray import cluster

    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)
    seen = []
    constants, _ = _fake_ray(monkeypatch, seen)
    assert cluster.connect_ray(None) is True
    assert seen == [False, False] and constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False


def test_an_explicit_uv_run_setting_is_left_alone(monkeypatch):
    from isaac_rl_posttraining.runtimes.native_ray import cluster

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "1")
    seen = []
    _fake_ray(monkeypatch, seen)
    cluster.connect_ray(None)
    assert seen == [True, True]


@pytest.mark.parametrize(
    ("gpus", "visible", "expected"),
    [
        ((0, 1), None, ("0", "1")),
        ((0, 2), ("1", "2", "3"), ("1", "3")),
        ((1,), ("GPU-a", "GPU-b"), ("GPU-b",)),
    ],
)
def test_actor_gpus_are_the_drivers_visible_devices(gpus, visible, expected):
    from isaac_rl_posttraining.runtimes.native_ray.cluster import device_names

    assert device_names(gpus, visible) == expected
