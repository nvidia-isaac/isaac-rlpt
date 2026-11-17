# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python API: registry routing, algorithm constructors, config aliases, placement plan, Run handle."""

from __future__ import annotations

import dataclasses
import functools
import re
from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from isaac_rl_posttraining import launch
from isaac_rl_posttraining import train as facade
from isaac_rl_posttraining.algorithms import (
    DSRL,
    DSRLPPO,
    PPO,
    RFS,
    Algorithm,
    BaseModel,
    SACEdit,
    registry,
)
from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.configs.normalize import (
    CONFIG_ALIASES,
    normalize_train_config,
    task_episode_cap,
)
from isaac_rl_posttraining.run import Run
from isaac_rl_posttraining.runners import build, launchers
from isaac_rl_posttraining.runners.off_policy_models import (
    policy_family,
    uses_reference_actions,
)
from isaac_rl_posttraining.runners.run_schedule import schedule_spec
from isaac_rl_posttraining.runtimes.placement import Learner, Placement, Served
from isaac_rl_posttraining.training import PersistSpec
from isaac_rl_posttraining.training.schedule import ScheduleSpec
from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

ARENA = ["base_model=mock"]


@pytest.fixture(autouse=True)
def _arena_servers(monkeypatch):
    """The arena entry needs its two server addresses to plan."""
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:7300")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:7301")


def _compose(*overrides: str):
    return facade.compose_train_config(list(overrides))


def _resolved(cfg) -> dict:
    return OmegaConf.to_container(build.run_config(cfg), resolve=True)


# -- registry ------------------------------------------------------------------------------------

EXPECTED_ROUTES = {
    "dsrl_sac": ("off_policy", "latent", "steering", "steering", launchers.off_policy),
    "rfs": (
        "off_policy",
        "latent_residual",
        "steering",
        "steering",
        launchers.off_policy,
    ),
    "sac_edit": ("off_policy", "edit", "edit", "edit", launchers.off_policy),
    "ppo": ("on_policy", "full_chunk", "on_policy", "flow", launchers.flow_sde_ppo),
    "dsrl_ppo": ("on_policy", "latent", "on_policy", "steering", launchers.dsrl_ppo),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_ROUTES))
def test_registry_routes_every_algorithm(name):
    loop, action_space, family, acting, launcher = EXPECTED_ROUTES[name]
    spec = registry.algorithm_spec(name)
    assert (spec.loop, spec.action_space, spec.family, spec.policy_family) == (
        loop,
        action_space,
        family,
        acting,
    )
    assert registry.load_target(spec.launcher) is launcher
    cfg = _compose(f"algorithm={name}", *ARENA)
    assert build.selected_algorithm(cfg) == spec
    assert cfg.algorithm.family == family


def test_off_policy_models_route_by_action_space():
    assert [
        policy_family(name) for name in ("dsrl_sac", "rfs", "sac_edit", "dsrl_ppo")
    ] == [
        "steering",
        "steering",
        "edit",
        "steering",
    ]
    assert uses_reference_actions("sac_edit") and not uses_reference_actions("rfs")
    with pytest.raises(ValueError, match="full_chunk"):
        policy_family("ppo")


class _ToyLauncher:
    """Plans like the off-policy loop; its run records that it was called."""

    calls: ClassVar[list[str]] = []

    def plan(self, cfg):
        return launchers.off_policy.plan(cfg)

    def build(self, cfg):
        return functools.partial(self.calls.append, str(cfg.algorithm.name))


TOY_LAUNCHER = _ToyLauncher()


@pytest.fixture
def toy_algorithm():
    spec = registry.register_algorithm(
        "toy_steer",
        "isaac_rl_posttraining.algorithms.sac:SACConfig",
        loop="off_policy",
        action_space="latent_residual",
        launcher=f"{__name__}:TOY_LAUNCHER",
        sections=("dsrl",),
    )
    register_configs()
    yield spec
    registry._ALGORITHMS.pop("toy_steer")


def test_a_registered_algorithm_routes_without_core_edits(toy_algorithm):
    # The registry stores each schema as a ``base_<name>`` config entry.
    cfg = _compose(
        "algorithm=" + f"base_{toy_algorithm.name}", "algorithm.name=toy_steer", *ARENA
    )
    plan = build.train_plan(cfg)
    assert (plan.algorithm, plan.family, plan.action_space, plan.policy_family) == (
        "toy_steer",
        "steering",
        "latent_residual",
        "steering",
    )
    _ToyLauncher.calls.clear()
    build.build_run(cfg)()
    assert _ToyLauncher.calls == ["toy_steer"]
    assert policy_family("toy_steer") == "steering"
    # The constructor routes keywords through the same registry entry.
    toy = Algorithm.named(
        "toy_steer", base="mock", batch_size=8, noise_action_bound=2.0
    )
    assert toy.overrides()[-2:] == [
        "algorithm.batch_size=8",
        "dsrl.noise_action_bound=2.0",
    ]


def test_a_duplicate_registration_needs_override():
    with pytest.raises(ValueError, match="already registered"):
        registry.register_algorithm(
            "dsrl_sac", "isaac_rl_posttraining.algorithms.sac:SACConfig"
        )


# -- constructors --------------------------------------------------------------------------------

CONSTRUCTOR_CASES = [
    (
        DSRL(
            base="mock",
            batch_size=128,
            noise_action_bound=2.0,
            warmup_exploration="normal",
        ),
        [
            "algorithm=dsrl_sac",
            "base_model=mock",
            "algorithm.batch_size=128",
            "dsrl.noise_action_bound=2.0",
            "dsrl.warmup_exploration=normal",
        ],
    ),
    (
        RFS(
            base=BaseModel("mock", action_dim=35),
            residual_action_bound=0.02,
            gamma=0.99,
        ),
        [
            "algorithm=rfs",
            "base_model=mock",
            "base_model.action_dim=35",
            "rfs.residual_action_bound=0.02",
            "algorithm.gamma=0.99",
        ],
    ),
    (
        SACEdit(base="mock", actor_edit_scale=0.05, tau=0.1),
        [
            "algorithm=sac_edit",
            "base_model=mock",
            "direct_action.actor_edit_scale=0.05",
            "algorithm.tau=0.1",
        ],
    ),
    (
        PPO(base_ckpt="/ckpt/stack", learning_rate=2e-5, update_epochs=2),
        [
            "algorithm=ppo",
            "base_model.ckpt=/ckpt/stack",
            "ppo.learning_rate=2e-5",
            "ppo.update_epochs=2",
        ],
    ),
    (
        DSRLPPO(
            base="mock", clip_ratio=0.1, action_exec_len=4, deterministic_kernels=False
        ),
        [
            "algorithm=dsrl_ppo",
            "base_model=mock",
            "ppo.clip_ratio=0.1",
            "dsrl.action_exec_len=4",
            "ppo.deterministic_kernels=false",
        ],
    ),
]


@pytest.mark.parametrize(
    ("algorithm", "cli"), CONSTRUCTOR_CASES, ids=lambda case: getattr(case, "name", "")
)
def test_constructors_compose_the_cli_config(algorithm, cli):
    env = (
        "isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
        if algorithm.name == "ppo"
        else "arena/drill_lift"
    )
    runtime = "rlinf" if algorithm.name == "ppo" else "native"
    from_python = launch(
        algorithm, env=env, runtime=runtime, overrides=["training.seed=3"], dry_run=True
    )
    from_cli = _compose(
        *cli, f"runtime={runtime}", f"env.env_name={env}", "training.seed=3"
    )
    assert OmegaConf.to_container(from_python.config, resolve=True) == _resolved(
        from_cli
    )


@pytest.mark.parametrize(
    ("key", "old", "new"),
    [
        ("training.schedule", "serial", "schedule=sync"),
        ("training.schedule", "lookahead", "schedule=overlapped"),
        ("ppo.overlap_collection", "true", "schedule=overlapped"),
        ("schedule", "serial", "schedule=sync"),
        ("schedule", "lookahead", "schedule=overlapped"),
        ("schedule.kind", "lookahead", "schedule.kind=overlapped"),
    ],
)
def test_old_schedule_names_are_refused_naming_the_new_one(key, old, new):
    with pytest.raises(RemovedConfigKeyError, match=re.escape(new)):
        _compose(*ARENA, f"{key}={old}")


def test_constructors_refuse_unknown_and_deprecated_settings():
    with pytest.raises(TypeError, match=r"unknown settings \['bogus'\]"):
        DSRL(base="mock", bogus=1)
    with pytest.raises(TypeError, match="overlap_collection"):
        DSRLPPO(base="mock", overlap_collection=True)
    with pytest.raises(TypeError, match="either"):
        DSRL(base=BaseModel("mock", ckpt="/a"), base_ckpt="/b")


def test_launch_writes_placement_and_schedule_overrides():
    overrides = facade.launch_overrides(
        DSRL(base="mock"),
        "arena/drill_lift",
        runtime="rlinf",
        placement=Placement(policy=Served(replicas=2), learner=Learner(ranks=1)),
        schedule="overlapped",
        max_policy_lag=1,
    )
    assert (
        "placement.policy.mode=served" in overrides
        and "placement.policy.replicas=2" in overrides
    )
    assert overrides[-2:] == ["schedule.kind=overlapped", "schedule.max_policy_lag=1"]


# -- aliases -------------------------------------------------------------------------------------

ALIAS_CASES = [
    ("eval.eval_deterministic=true", "eval.action_selection", "mean"),
    ("runtime.num_rollout_workers=2", "placement.envs.workers", 2),
    ("runtime.rlinf.policy_serving=shared", "placement.policy.mode", "served"),
    ("runtime.rlinf.num_policy_servers=3", "placement.policy.replicas", 3),
    ("runtime.rlinf.policy_batch_size=8", "placement.policy.max_batch", 8),
    (
        "runtime.rlinf.policy_batch_wait_seconds=0.01",
        "placement.policy.max_wait_ms",
        10.0,
    ),
    (
        "runtime.rlinf.deterministic_evaluation=false",
        "runtime.rlinf.serve_evaluation_unbatched",
        False,
    ),
]


@pytest.mark.parametrize(("old", "new", "value"), ALIAS_CASES)
def test_old_keys_map_onto_their_new_names(old, new, value):
    cfg = _compose(*ARENA, "runtime=rlinf", old)
    assert OmegaConf.select(cfg, new) == value
    key, _, raw = old.partition("=")
    # The old key is left as it was (nothing reads it), and normalizing again changes nothing.
    assert str(OmegaConf.select(cfg, key)).lower() == raw.lower()
    assert normalize_train_config(cfg) == cfg


def test_on_policy_aliases_reach_the_readers():
    cfg = _compose(
        "algorithm=dsrl_ppo",
        *ARENA,
        "schedule=overlapped",
        "ppo.deterministic_evaluation=false",
    )
    assert (cfg.schedule.kind, cfg.ppo.deterministic_kernels) == ("overlapped", False)
    assert schedule_spec(cfg, on_policy=True) == ScheduleSpec("overlapped")
    cfg = _compose(
        "algorithm=dsrl_ppo", *ARENA, "schedule=async", "eval.action_selection=mean"
    )
    assert schedule_spec(cfg, on_policy=True) == ScheduleSpec("async", max_policy_lag=1)
    assert cfg.eval.action_selection == "mean"


def test_every_alias_targets_a_declared_key():
    cfg = _compose("algorithm=dsrl_ppo", *ARENA, "runtime=rlinf")
    for alias in CONFIG_ALIASES:
        assert OmegaConf.select(cfg, alias.new, default="<missing>") != "<missing>", (
            alias.new
        )
        assert OmegaConf.select(cfg, alias.old, default="<missing>") != "<missing>", (
            alias.old
        )


def test_a_set_old_key_or_synonym_overrides_the_recipe():
    cfg = _compose(
        "algorithm=ppo", "base_model.ckpt=/x", "runtime.num_rollout_workers=1"
    )
    assert cfg.placement.envs.workers == 1


def test_learner_ranks_have_one_key_and_the_placement_api_sets_it():
    from isaac_rl_posttraining.runners.run_placement import (
        placement_from_config,
        placement_overrides,
    )

    cfg = _compose("algorithm=ppo", "base_model.ckpt=/x", "placement.learner.ranks=2")
    assert cfg.placement.learner.ranks == 2 and "num_ranks" not in cfg.learner
    assert placement_from_config(cfg).learner.ranks == 2
    overrides = placement_overrides(Placement(learner=Learner(ranks=2)))
    assert "placement.learner.ranks=2" in overrides
    with pytest.raises(
        RemovedConfigKeyError, match=r"learner\.num_ranks: .*placement\.learner\.ranks"
    ):
        _compose(*ARENA, "learner.num_ranks=2")


def test_rlinf_gpu_lists_become_first_accelerators():
    cfg = _compose(
        *ARENA,
        "runtime=rlinf",
        "placement.learner.gpus=[2]",
        "placement.envs.gpus=['0:3','0:4']",
    )
    assert (
        cfg.runtime.rlinf.learner_first_accelerator,
        cfg.runtime.rlinf.first_accelerator,
    ) == (2, 3)
    with pytest.raises(ValueError, match="contiguous"):
        _compose(*ARENA, "runtime=rlinf", "placement.envs.gpus=[1,3]")


# -- refused keys and checks before launch -------------------------------------------------------


@pytest.mark.parametrize(
    "override", ["env.use_gt_rewards=true", "+env.extra_keys_to_drop=[a]", "debug=true"]
)
def test_dead_keys_are_refused_with_their_replacement(override):
    with pytest.raises(RemovedConfigKeyError, match="nothing read it"):
        _compose(*ARENA, override)


def test_unknown_algorithm_keys_fail_the_dry_run():
    with pytest.raises(
        RemovedConfigKeyError,
        match="algorithm.bogus_key: not settings of algorithm 'dsrl_sac'",
    ):
        build.train_plan(_compose(*ARENA, "+algorithm.bogus_key=1"))
    with pytest.raises(RemovedConfigKeyError, match="settings live in ppo"):
        build.train_plan(
            _compose(
                "algorithm=ppo", "base_model.ckpt=/x", "+algorithm.learning_rate=1"
            )
        )


def test_a_disagreeing_yaml_family_is_refused():
    with pytest.raises(ValueError, match="registry"):
        _compose(*ARENA, "algorithm.family=edit")


@pytest.mark.parametrize(
    "schedule",
    [
        ("schedule=async", "schedule.max_policy_lag=2"),
        ("schedule.kind=async", "schedule.max_policy_lag=2"),
        ("schedule=overlapped",),
        ("schedule.kind=overlapped",),
    ],
)
def test_every_schedule_spelling_composes_with_its_settings(schedule):
    expected = (
        "overlapped, max policy lag 1"
        if "overlapped" in schedule[0]
        else "async, max policy lag 2, publish once per training burst"
    )
    assert build.train_plan(_compose(*ARENA, *schedule)).schedule_kind == expected


def test_schedules_the_loop_does_not_run_are_refused(toy_algorithm):
    assert build.train_plan(_compose(*ARENA, "schedule=async")).schedule_kind == (
        "async, max policy lag unbounded, publish once per training burst"
    )
    assert build.train_plan(
        _compose(*ARENA, "schedule.kind=async", "schedule.publish_every=500")
    ).schedule_kind.endswith("publish every 500 update(s)")
    with pytest.raises(ValueError, match="schedule=async is not implemented"):
        build.train_plan(
            _compose(
                "algorithm=" + f"base_{toy_algorithm.name}",
                "algorithm.name=toy_steer",
                *ARENA,
                "schedule.kind=async",
            )
        )
    with pytest.raises(ValueError, match="max_policy_lag must be 0"):
        build.train_plan(_compose(*ARENA, "schedule.max_policy_lag=2"))


def test_the_episode_cap_follows_the_simulator_group():
    assert _compose(*ARENA).env.max_episode_steps == 450  # drill_lift is LMDrillLift
    assert _compose(*ARENA, "simulator=libero_10").env.max_episode_steps == 600
    assert (
        _compose(*ARENA, "env.env_name=arena/LMPushButton").env.max_episode_steps == 350
    )
    assert _compose(*ARENA, "env.max_episode_steps=12").env.max_episode_steps == 12
    assert task_episode_cap({"Only-v0": 7}, "isaaclab/whatever") == 7
    with pytest.raises(ValueError, match="names none of the simulator's tasks"):
        task_episode_cap({"A": 1, "B": 2}, "sim/c")


# -- the plan ------------------------------------------------------------------------------------


def test_the_dry_run_prints_the_placement_plan():
    plan = build.train_plan(
        _compose("algorithm=ppo", "base_model.ckpt=/x", "placement.gpus_per_node=8")
    )
    lines = plan.lines()
    assert lines[-1].startswith("plan.checks=") and lines[-5:-1] == [
        "plan.placement.resources=node0 (8 gpus)",
        "plan.placement.envs=4 worker(s) on node0 gpus [4, 5, 6, 7]",
        "plan.placement.policy=local (a copy in every rollout worker)",
        "plan.placement.learner=4 rank(s) on node0 gpus [0, 1, 2, 3]",
    ]
    assert {
        line.split("=", 1)[0].removeprefix("plan.").split(".")[0] for line in lines
    } == set(build.TrainPlan.line_fields())


def test_served_policies_follow_the_runtime_features():
    cfg = _compose(
        *ARENA,
        "runtime=rlinf",
        "runtime.rlinf.device=gpu",
        "placement.gpus_per_node=8",
        "placement.policy.mode=served",
        "placement.policy.replicas=2",
    )
    lines = build.train_plan(cfg).lines()
    assert "plan.placement.policy.base+actor=served by 2 replica(s)" in "\n".join(lines)


# -- the run handle ------------------------------------------------------------------------------


def test_a_dry_run_plans_and_refuses_to_start():
    run = launch(DSRL(base="mock"), env="arena/drill_lift", dry_run=True)
    assert run.plan.algorithm == "dsrl_sac" and run.placement.learner.processes == 1
    assert run.data_specs.mismatches == ()
    with pytest.raises(RuntimeError, match="dry run"):
        run.loop()
    with pytest.raises(RuntimeError, match="dry run"):
        run.collect()


def test_loop_runs_the_configured_run_once(monkeypatch):
    import isaac_rl_posttraining.run as run_module

    ran = []
    monkeypatch.setattr(
        run_module, "build_run", lambda cfg: lambda: ran.append(cfg.algorithm.name)
    )
    with launch("rfs", env="arena/drill_lift", overrides=ARENA) as run:
        run.loop()
        with pytest.raises(RuntimeError, match="fresh run"):
            run.loop()
    assert ran == ["rfs"]


def test_train_is_launch_then_loop_and_the_old_form_warns(monkeypatch):
    import isaac_rl_posttraining.run as run_module

    ran = []
    monkeypatch.setattr(run_module, "build_run", lambda cfg: lambda: ran.append("new"))
    monkeypatch.setattr(facade, "build_run", lambda cfg: lambda: ran.append("old"))
    facade.train(DSRL(base="mock"), env="arena/drill_lift")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        facade.train("arena/drill_lift", "mock", "dsrl_sac")
    assert ran == ["new", "old"]


def test_step_wise_control_needs_a_launcher_session(toy_algorithm):
    """Every built-in loop opens a step-wise session; a launcher without one is named in plain words."""
    for launcher in (launchers.off_policy, launchers.flow_sde_ppo, launchers.dsrl_ppo):
        assert callable(launcher.open_session)
    run = Run(
        _compose(
            "algorithm=" + f"base_{toy_algorithm.name}",
            "algorithm.name=toy_steer",
            *ARENA,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="toy_steer has no step-wise control.*Run.loop",
    ):
        run.train(steps=1)
    run.close()


def test_launch_persist_sets_the_data_persist_keys():
    spec = PersistSpec("collected", every_episodes=5, keep_frames=True)
    run = launch(DSRL(base="mock"), env="arena/drill_lift", persist=spec, dry_run=True)
    assert OmegaConf.to_container(run.config.data.persist) == {
        "path": "collected",
        "every_episodes": 5,
        "keep_frames": True,
    }
    run = launch(
        DSRLPPO(base="mock"),
        env="arena/drill_lift",
        persist="/data/episodes",
        dry_run=True,
    )
    assert run.config.data.persist.path == "/data/episodes"
    assert (
        launch(
            DSRL(base="mock"), env="arena/drill_lift", dry_run=True
        ).config.data.persist.path
        is None
    )


def test_run_methods_are_documented():
    public = [name for name in vars(Run) if not name.startswith("_")]
    assert {
        "loop",
        "collect",
        "train",
        "evaluate",
        "publish",
        "save",
        "load",
        "close",
        "plan",
    } - set(public) <= {"plan"}
    assert all(
        getattr(Run, name).__doc__ for name in public if callable(getattr(Run, name))
    )
    assert dataclasses.is_dataclass(build.TrainPlan)
