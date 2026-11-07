# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dry run as a type check: one row per check, each with the outcome the plan must report.

Every case composes a config and resolves its plan on the CPU without
loading a model or starting a simulator: the mock base model, a GR00T
checkpoint made of its JSON metadata only, the in-tree G1 Factory, LIBERO and
Isaac Lab entries, and robots brought from outside the repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from isaac_rl_posttraining.data.spec import (
    ActionSpec,
    ArraySpec,
    ObservationSpec,
    SpecMismatch,
)
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
from isaac_rl_posttraining.runners.build import plan_report, train_plan
from isaac_rl_posttraining.runners.plan_checks import PlanCheckError
from isaac_rl_posttraining.train import compose_train_config

G1 = ("algorithm=dsrl_sac",)
LIBERO = (
    "algorithm=dsrl_sac",
    "simulator=libero_10",
    "env.env_name=libero/libero_10/0",
)
STACK_CUBE = (
    "algorithm=dsrl_sac",
    "simulator=isaaclab_franka_stack",
    "env.env_name=isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    "base_model=mock",
    "base_model.action_dim=7",
    "dsrl.noise_dim=7",
)
#: Ray-placed roles on an assumed 8-GPU node, so placement is checked without GPUs here.
EIGHT_GPUS = ("placement.gpus_per_node=8", "runtime.native.processes=ray")


class UndeclaredRobot(SimulatorEntry):
    """A robot that declares nothing beyond how to build its environment."""

    name = "plancheck_undeclared"

    def make_environment(self, spec: Any, phase: Any, context: Any) -> gym.Env:
        raise AssertionError("the plan must not build an environment")


class DeclaredRobot(UndeclaredRobot):
    """The same robot, declaring a 7-joint state, one 16x16 camera and a 7-wide action."""

    name = "plancheck_declared"

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        return ObservationSpec(
            {
                "state": ArraySpec((7,), "float32"),
                "image": ArraySpec((16, 16, 3), "uint8"),
            }
        )

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        return ActionSpec(width=7)


UNDECLARED = (
    "algorithm=dsrl_sac",
    "base_model=mock",
    "env.env_name=plancheck_undeclared/reach",
    f"+simulator.entry={__name__}:UndeclaredRobot",
    "base_model.action_dim=7",
    "dsrl.noise_dim=7",
    "env.max_episode_steps=8",
)
DECLARED = (
    "algorithm=dsrl_sac",
    "base_model=mock",
    "env.env_name=plancheck_declared/reach",
    f"+simulator.entry={__name__}:DeclaredRobot",
    "base_model.action_dim=7",
    "dsrl.noise_dim=7",
    "env.max_episode_steps=8",
)


def _gr00t_checkpoint(
    root: Path, *, state: dict[str, int], action: dict[str, int]
) -> str:
    """A GR00T N1.7 checkpoint directory holding only the JSON metadata the plan reads."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {"model_type": "Gr00tN1d7", "action_horizon": 40, "max_action_dim": 132}
        )
    )
    processor = root / "processor"
    processor.mkdir()
    baked = {
        "state": {"delta_indices": [0], "modality_keys": list(state)},
        "action": {"delta_indices": list(range(16)), "modality_keys": list(action)},
    }
    (processor / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {"modality_configs": {"new_embodiment": baked}},
            }
        )
    )
    statistics = {
        modality: {
            key: {"min": [0.0] * width, "max": [1.0] * width}
            for key, width in groups.items()
        }
        for modality, groups in (("state", state), ("action", action))
    }
    (processor / "statistics.json").write_text(
        json.dumps({"new_embodiment": statistics})
    )
    return str(root)


G1_STATE = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}
G1_ACTION = {**G1_STATE, "base_height_command": 1, "navigate_command": 3}


@pytest.fixture(autouse=True)
def _no_server_addresses(monkeypatch: pytest.MonkeyPatch):
    for variable in ("ARENA_SERVER_URL", "ARENA_EVAL_SERVER_URL"):
        monkeypatch.delenv(variable, raising=False)
    yield
    for segment in ("plancheck_undeclared", "plancheck_declared"):
        providers._SIMULATOR_ENTRIES.pop(segment, None)


@pytest.fixture
def checkpoints(tmp_path: Path) -> dict[str, str]:
    return {
        "g1": _gr00t_checkpoint(tmp_path / "g1", state=G1_STATE, action=G1_ACTION),
        "legs": _gr00t_checkpoint(
            tmp_path / "legs", state={**G1_STATE, "legs": 12}, action=G1_ACTION
        ),
    }


def _report(*overrides: str, checkpoints: dict[str, str] | None = None):
    resolved = [override.format(**(checkpoints or {})) for override in overrides]
    return plan_report(compose_train_config(resolved))


# (case id, overrides, the errors the plan reports, unverified substrings, substrings of the plan lines)
CASES: list[
    tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
] = [
    # -- data specs: G1 Factory ----------------------------------------------------
    (
        "g1-mock-binds-action-not-state",
        (*G1, "base_model=mock"),
        (),
        (
            "the policy's 'state' width binds at load: base model mock reads a state of any width",
        ),
        (
            "plan.checks=",
            "policy reads {state: float32[state_dim], ego_view_image: uint8[480, 640, 3]}",
        ),
    ),
    (
        "g1-gr00t-checkpoint-binds-everything",
        (*G1, "base_model.ckpt={g1}"),
        (),
        (),
        ("policy reads {state: float32[31]", "0 unverified, 0 error(s)"),
    ),
    (
        "g1-gr00t-checkpoint-with-legs-state",
        (*G1, "base_model.ckpt={legs}"),
        (r"observation 'state': shape float32\[31\] vs float32\[43\]",),
        (),
        (),
    ),
    (
        "g1-gr00t-model-id-stays-unverified",
        (*G1, "base_model.ckpt=nvidia/GR00T-N1.7-3B"),
        (),
        (
            "bind at load: base_model.ckpt=nvidia/GR00T-N1.7-3B is not a local directory",
        ),
        (),
    ),
    (
        "g1-action-width-vs-simulator",
        (*G1, "base_model=mock", "base_model.action_dim=43"),
        (r"action: emitted float32\[43\] x 40 steps vs accepted float32\[35\]",),
        (),
        (),
    ),
    (
        "g1-action-width-vs-checkpoint",
        (*G1, "base_model.ckpt={g1}", "base_model.action_dim=36"),
        (
            r"base_model.action_dim=36 but checkpoint .* decodes 35-wide actions; set base_model.action_dim=35 or null",
        ),
        (),
        (),
    ),
    (
        "steering-latent-wider-than-flow-head",
        (*G1, "base_model.ckpt={g1}", "dsrl.noise_dim=140"),
        (
            r"dsrl.noise_dim=140 exceeds the 132-wide flow head .*; set dsrl.noise_dim<=132",
        ),
        (),
        (),
    ),
    (
        "steering-window-longer-than-flow-horizon",
        (*G1, "base_model.ckpt={g1}", "training.chunk_size=48"),
        (r"training.chunk_size=48 exceeds the checkpoint's action_horizon .*=40",),
        (),
        (),
    ),
    (
        "executed-steps-longer-than-decode",
        (*G1, "base_model.ckpt={g1}", "dsrl.action_exec_len=20"),
        (
            r"dsrl.action_exec_len=20 exceeds the 16 steps one decode of .* emits; set dsrl.action_exec_len<=16",
        ),
        (),
        (),
    ),
    (
        "camera-key-the-simulator-lacks",
        (*G1, "base_model=mock", "env.image_keys=[wrist]"),
        (r"image keys \['wrist'\] from env.image_keys are not camera entries",),
        (),
        (),
    ),
    # -- data specs: LIBERO and Isaac Lab declare their observation and action ---
    (
        "libero-gr00t-convention",
        (*LIBERO, "base_model=mock", "base_model.action_dim=7", "dsrl.noise_dim=7"),
        (),
        (),
        (
            (
                "simulator libero emits {video.image: uint8[256, 256, 3], video.wrist_image: uint8[256, 256, 3], "
                "state: float32[8]} and accepts float32[7]"
            ),
        ),
    ),
    (
        "libero-action-width",
        (*LIBERO, "base_model=mock", "base_model.action_dim=35", "dsrl.noise_dim=7"),
        (r"action: emitted float32\[35\] x 40 steps vs accepted float32\[7\]",),
        (),
        (),
    ),
    (
        "libero-pi0-padded-latent",
        (
            "algorithm=dsrl_sac",
            "simulator=libero_10",
            "env.env_name=libero_pi0/libero_10/0",
            "base_model=pi0",
            "dsrl.noise_dim=32",
        ),
        (),
        ("base model pi0 states its sizes once loaded",),
        ("simulator libero_pi0 emits {observation/image: uint8[224, 224, 3]",),
    ),
    (
        "isaaclab-stack-cube",
        STACK_CUBE,
        (),
        (),
        (
            "simulator isaaclab emits {video.image: uint8[256, 256, 3], video.wrist_image: uint8[256, 256, 3]",
            "state: float32[8]} and accepts float32[7]",
        ),
    ),
    (
        "isaaclab-unknown-task",
        (*STACK_CUBE, "env.env_name=isaaclab/Isaac-Nope-v0"),
        ("no GR00T mapping registered for IsaacLab task 'Isaac-Nope-v0'",),
        (),
        (),
    ),
    # -- robots from outside the repository --------------------------------------
    (
        "robot-without-declarations",
        UNDECLARED,
        (),
        (
            "plancheck_undeclared declares no observation spec",
            "plancheck_undeclared declares no action spec",
        ),
        (),
    ),
    (
        "robot-with-declarations",
        DECLARED,
        (),
        (),
        ("simulator plancheck_declared emits",),
    ),
    (
        "robot-action-width",
        (*DECLARED, "base_model.action_dim=35"),
        (r"action: emitted float32\[35\] x 40 steps vs accepted float32\[7\]",),
        (),
        (),
    ),
    # -- evaluation layout -------------------------------------------------------
    (
        "isaaclab-evaluation-on-its-own-scenes",
        (
            *STACK_CUBE,
            "training.num_envs=16",
            "eval.eval_num_envs=8",
            "eval.eval_num_episodes=24",
        ),
        (),
        (),
        (
            "plan.evaluation=24 episode(s) over 8 environment(s) on workers of their own, 3 round(s)",
        ),
    ),
    (
        "isaaclab-evaluation-on-the-training-scenes",
        (
            *STACK_CUBE,
            "training.num_envs=16",
            "eval.eval_num_envs=16",
            "eval.eval_num_episodes=32",
        ),
        (),
        (),
        (
            "plan.evaluation=32 episode(s) over 16 environment(s) on the training workers, 2 round(s)",
        ),
    ),
    (
        "evaluation-episodes-not-a-multiple",
        (
            *STACK_CUBE,
            "training.num_envs=16",
            "eval.eval_num_envs=16",
            "eval.eval_num_episodes=24",
        ),
        (
            (
                r"eval.eval_num_episodes \(24\) must be an exact multiple of eval.eval_num_envs \(16\); "
                r"set eval.eval_num_episodes=16 or 32"
            ),
        ),
        (),
        (),
    ),
    (
        "g1-evaluation-servers",
        (*G1, "base_model=mock", "eval.eval_num_envs=4", "eval.eval_num_episodes=8"),
        (),
        (),
        ("plan.evaluation=8 episode(s) over 4 environment(s) on workers of their own",),
    ),
    # -- workers and placement ---------------------------------------------------
    (
        "isaaclab-scenes-over-workers",
        (
            *STACK_CUBE,
            "training.num_envs=16",
            "eval.eval_num_envs=16",
            "eval.eval_num_episodes=16",
            "placement.envs.workers=3",
        ),
        (
            r"16 scenes \(training.num_envs\) do not split evenly over 3 rollout workers \(placement.envs.workers\)",
        ),
        (),
        (),
    ),
    (
        "sac-edit-several-environments-and-workers",
        (
            *STACK_CUBE,
            "algorithm=sac_edit",
            "training.chunk_size=null",
            "training.num_envs=4",
            "placement.envs.workers=2",
            "placement.gpus_per_node=8",
        ),
        (),
        (),
        ("plan.layout=2 rollout worker(s) x 2 environment(s), batched",),
    ),
    (
        "more-workers-than-environments",
        (*DECLARED, "training.num_envs=2", "placement.envs.workers=3"),
        (r"placement.envs.workers=3 exceeds training.num_envs=2",),
        (),
        (),
    ),
    (
        "overlapping-gpu-lists",
        (
            *DECLARED,
            *EIGHT_GPUS,
            "placement.learner.ranks=6",
            "placement.learner.gpus=['0','1','2','3','4','5']",
            "placement.envs.gpus=['4','5','6']",
        ),
        (
            (
                r"envs \(placement.envs.gpus\) and learner \(placement.learner.gpus\) both ask for GPU\(s\) node0:4, "
                r"node0:5; requested 9 \(envs 3, learner 6\), available 8"
            ),
        ),
        (),
        (),
    ),
    (
        "gpu-the-node-lacks",
        (*DECLARED, *EIGHT_GPUS, "placement.envs.gpus=['9']"),
        (
            r"envs \(placement.envs.gpus\) asks for GPU 9, which does not exist: node0 has 8 GPU\(s\) \(0-7\)",
        ),
        (),
        (),
    ),
    (
        "more-server-replicas-than-environments",
        (
            *DECLARED,
            *EIGHT_GPUS,
            "training.num_envs=2",
            "placement.policy.mode=served",
            "placement.policy.replicas=4",
        ),
        (
            (
                r"placement.policy.replicas=4 policy servers of base\+actor for 3 environment\(s\) \(training and "
                r"evaluation\) leave servers idle; set placement.policy.replicas<=3"
            ),
        ),
        (),
        (),
    ),
    # -- schedule ----------------------------------------------------------------
    (
        "schedule-the-loop-does-not-run",
        (*DECLARED, "schedule.kind=bogus"),
        ("schedule",),
        (),
        (),
    ),
    # -- launch inputs -----------------------------------------------------------
    (
        "missing-server-address-is-a-launch-input",
        (*G1, "base_model=mock"),
        (),
        (),
        (
            (
                "plan.launch_inputs=ARENA_SERVER_URL (train) needs to be exported at launch; "
                "ARENA_EVAL_SERVER_URL (eval) needs to be exported at launch"
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    ("overrides", "errors", "unverified", "lines"),
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_the_dry_run_reports_each_check(
    checkpoints, overrides, errors, unverified, lines
) -> None:
    report = _report(*overrides, checkpoints=checkpoints)
    messages = [str(error) for error in report.checks.errors]
    assert len(messages) == len(errors), messages
    for pattern in errors:
        assert any(re.search(pattern, message) for message in messages), (
            pattern,
            messages,
        )
    for fragment in unverified:
        assert any(fragment in item for item in report.checks.unverified), (
            fragment,
            report.checks.unverified,
        )
    text = "\n".join(report.lines)
    assert report.lines[-1].startswith("plan.checks=")
    for fragment in lines:
        assert fragment in text, (fragment, text)
    if not errors:
        assert report.plan is not None and report.plan.lines() == report.lines


def test_every_problem_is_reported_at_once(checkpoints) -> None:
    overrides = (
        *G1,
        "base_model.ckpt={g1}",
        "dsrl.noise_dim=140",
        "dsrl.action_exec_len=20",
        "eval.eval_num_envs=4",
        "eval.eval_num_episodes=6",
    )
    report = _report(*overrides, checkpoints=checkpoints)
    assert report.plan is None
    summary = report.lines[-1]
    assert (
        "3 error(s)" not in summary
    )  # the two data-spec mismatches are one error naming both
    assert "dsrl.noise_dim=140" in summary and "dsrl.action_exec_len=20" in summary
    assert "eval.eval_num_episodes (6)" in summary
    assert any(
        line == "plan.data_specs=" + line.split("=", 1)[1] for line in report.lines
    )
    with pytest.raises(
        PlanCheckError, match=r"the launch plan found 2 problems:\n  1\. .*\n  2\. "
    ) as raised:
        train_plan(compose_train_config([o.format(**checkpoints) for o in overrides]))
    assert {type(error) for error in raised.value.errors} == {SpecMismatch, ValueError}


def test_a_single_problem_keeps_its_own_type() -> None:
    with pytest.raises(SpecMismatch, match="action: emitted float32"):
        train_plan(
            compose_train_config([*G1, "base_model=mock", "base_model.action_dim=43"])
        )


@pytest.mark.parametrize(
    ("overrides", "warning"),
    [
        (
            ("training.num_rollouts=1000",),
            "ends before algorithm.learning_starts=1200, so the learner never updates",
        ),
        (
            (
                "training.num_rollouts=3000",
                "training.num_envs=8",
                "training.rollout_count_by=episode",
            ),
            "may end before the first learner update, which can come as late as 4800 env steps",
        ),
        (
            (
                "training.num_rollouts=10000",
                "training.num_envs=8",
                "training.rollout_count_by=episode",
            ),
            None,
        ),
        (
            (
                "training.num_rollouts=1000",
                "training.train_after_episode=false",
                "algorithm.learning_starts=500",
            ),
            None,
        ),
    ],
)
def test_a_budget_that_cannot_reach_a_learner_update_is_warned(
    overrides, warning
) -> None:
    """The plan warns, without failing, when the budget can end before the first update."""
    checks = train_plan(
        compose_train_config([*G1, "base_model=mock", *overrides])
    ).checks
    if warning is None:
        assert "warnings" not in checks
    else:
        assert warning in checks.split(" | warnings: ")[1]


def test_an_exported_server_address_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:5555")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:5556")
    plan = train_plan(compose_train_config([*G1, "base_model=mock"]))
    assert plan.launch_inputs == (
        "ARENA_SERVER_URL (train) = tcp://127.0.0.1:5555; ARENA_EVAL_SERVER_URL (eval) = tcp://127.0.0.1:5556"
    )


@pytest.mark.parametrize(
    ("entry_name", "image_size", "cameras", "state_key"),
    [
        ("Libero", 256, ("video.image", "video.wrist_image"), "state"),
        (
            "LiberoPi0",
            224,
            ("observation/image", "observation/wrist_image"),
            "observation/state",
        ),
        ("LiberoDreamZero", 256, ("main_image", "wrist_image"), "state"),
    ],
)
def test_libero_declarations_match_the_adapters(
    entry_name, image_size, cameras, state_key
) -> None:
    """Each LIBERO entry declares what its adapter builds over a LIBERO env (stood in for here)."""
    from simulators.libero import runtime as libero_runtime
    from simulators.libero.policies.dreamzero import LiberoDreamZeroAdapter
    from simulators.libero.policies.gr00t import LiberoGr00tAdapter
    from simulators.libero.policies.pi0 import LiberoPi0Adapter

    adapters = {
        "Libero": LiberoGr00tAdapter,
        "LiberoPi0": LiberoPi0Adapter,
        "LiberoDreamZero": LiberoDreamZeroAdapter,
    }
    inner = gym.Env()
    frame = gym.spaces.Box(0, 255, (256, 256, 3), np.uint8)
    inner.observation_space = gym.spaces.Dict(
        {
            "agentview_image": frame,
            "robot0_eye_in_hand_image": frame,
            "state": gym.spaces.Box(
                -np.inf, np.inf, (libero_runtime.EEF_STATE_DIM,), np.float32
            ),
        }
    )
    inner.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
    adapter = adapters[entry_name](inner)
    entry = getattr(libero_runtime, entry_name)()
    spec = EnvironmentSpec(
        env_name=f"{entry.name}/libero_10/0", num_envs=1, max_episode_steps=8, seed=0
    )
    declared = entry.observation_spec(spec)
    assert declared == ObservationSpec.from_space(adapter.observation_space)
    assert set(declared.images) == set(cameras) and declared.arrays[
        state_key
    ].shape == (8,)
    assert all(
        declared.arrays[key].shape == (image_size, image_size, 3) for key in cameras
    )
    assert entry.action_spec(spec) == ActionSpec(width=adapter.action_space.shape[0])


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "Isaac-Assemble-Trocar-G129-Dex3-v0",
    ],
)
def test_isaaclab_declaration_matches_the_scene_spaces(task) -> None:
    """The Isaac Lab entry declares the spaces its scenes build from the same task mapping."""
    from simulators.isaaclab.env import _observation_space, mapping_for_task
    from simulators.isaaclab.runtime import IsaacLab

    spec = EnvironmentSpec(
        env_name=f"isaaclab/{task}",
        num_envs=1,
        max_episode_steps=8,
        seed=0,
        action_dim=7,
        image_resolution=64,
    )
    declared = IsaacLab().observation_spec(spec)
    assert declared == ObservationSpec.from_space(
        _observation_space(mapping_for_task(task), 64)
    )
    assert IsaacLab().action_spec(spec) == ActionSpec(width=7)


@pytest.mark.parametrize(("training_envs", "shares"), [(16, False), (8, True)])
def test_a_shared_entry_evaluates_on_the_training_workers_only_at_the_evaluation_size(
    monkeypatch: pytest.MonkeyPatch, training_envs: int, shares: bool
) -> None:
    """Isaac Lab trains 16 scenes and evaluates 8: the evaluation runs on 8 scenes of its own, not on the 16."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from isaac_rl_posttraining.runners import off_policy_evaluation

    opened: list[int] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        off_policy_evaluation, "resolve_num_rollout_workers", lambda *a, **k: 1
    )
    monkeypatch.setattr(off_policy_evaluation, "environment_spec", lambda *a, **k: None)
    monkeypatch.setattr(
        off_policy_evaluation,
        "resolve_image_keys",
        lambda *a, **k: SimpleNamespace(keys=["image"]),
    )
    monkeypatch.setattr(
        off_policy_evaluation,
        "RolloutEvaluator",
        lambda **kwargs: captured.update(kwargs),
    )
    cfg = OmegaConf.create(
        {
            "eval": {
                "eval_num_episodes": 24,
                "eval_max_steps": 10,
                "reset_seed": None,
                "eval_record_video": False,
                "video_fps": 20,
            }
        }
    )
    train_group = SimpleNamespace(num_envs=training_envs)
    runtime = SimpleNamespace(
        open_rollout_group=lambda *a, num_envs, **k: (
            opened.append(num_envs) or "eval group"
        )
    )
    off_policy_evaluation.open_evaluator(
        cfg,
        SimpleNamespace(evaluation="shared"),
        runtime,
        train_group,
        SimpleNamespace(),
        eval_num_envs=8,
        experiment_logger=SimpleNamespace(log_dir="unused"),
        reward_channel="sparse_outcome",
        reward_aggregation="last",
        output_dir="unused",
    )
    if shares:
        assert (
            captured["group"] is train_group
            and captured["shared_group"]
            and opened == []
        )
    else:
        assert (
            captured["group"] == "eval group"
            and "shared_group" not in captured
            and opened == [8]
        )


@pytest.mark.parametrize(
    "runtime,note",
    [
        (("runtime=rlinf",), "install the [rlinf] extra"),
    ],
)
def test_a_runtime_this_machine_lacks_is_unverified_and_refused_at_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime, note
) -> None:
    from isaac_rl_posttraining.compat import rlinf
    from isaac_rl_posttraining.runners.build import build_run

    monkeypatch.setattr(rlinf, "is_available", lambda: False)
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:5555")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:5556")
    overrides = [*G1, "base_model=mock"] + [
        override.format(missing=tmp_path) for override in runtime
    ]
    report = _report(*overrides)
    assert report.plan is not None, report.lines[-1]
    assert any(note in item for item in report.checks.unverified), report.lines[-1]
    with pytest.raises(ValueError, match=re.escape(note)):
        build_run(compose_train_config(overrides))


def test_only_an_off_policy_evaluation_group_next_to_training_counts_as_extra_workers():
    from isaac_rl_posttraining.runners.plan_checks import evaluation_group_workers
    from isaac_rl_posttraining.runners.simulator_entries import resolve_simulator_entry

    cfg = compose_train_config([*LIBERO, "eval.eval_num_envs=3"])
    entry = resolve_simulator_entry(cfg)
    assert evaluation_group_workers(cfg, entry, "off_policy") == 3
    # PPO closes its training workers before evaluation workers replace them.
    ppo = compose_train_config(
        ["algorithm=dsrl_ppo", *LIBERO[1:], "ppo.eval_episodes_per_environment=1"]
    )
    assert evaluation_group_workers(ppo, entry, "on_policy") is None
    evaluation_only = compose_train_config(
        [*LIBERO, "eval.eval_num_envs=3", "training.num_rollouts=0"]
    )
    assert evaluation_group_workers(evaluation_only, entry, "off_policy") is None


@pytest.mark.parametrize(
    "setting", ["routing=least_loaded", "gpus_per_replica=2", "engine=other"]
)
def test_rlinf_refuses_served_settings_it_would_ignore(setting):
    cfg = compose_train_config(
        [
            *LIBERO,
            "runtime=rlinf",
            "placement.policy.mode=served",
            f"placement.policy.{setting}",
        ]
    )
    with pytest.raises(
        ValueError, match=rf"set placement\.policy\.{setting.split('=')[0]}="
    ):
        train_plan(cfg)


def test_the_builtin_engine_refuses_replicas_on_several_gpus():
    from types import SimpleNamespace

    from isaac_rl_posttraining.runners.plan_checks import check_served_policy
    from isaac_rl_posttraining.runtimes.placement import (
        Placement,
        Resources,
        Served,
        plan_placement,
    )

    def plan(gpus_per_replica):
        return plan_placement(
            Placement(policy=Served(gpus_per_replica=gpus_per_replica)),
            Resources.uniform(1, 4),
            served_supported=True,
            distributed_learner=True,
        )

    loop = SimpleNamespace(served_policy=True, batched_workers=False)
    with pytest.raises(ValueError, match="builtin engine runs a replica on 1 GPU"):
        check_served_policy("dsrl_sac", loop, plan(2), None)
    check_served_policy("dsrl_sac", loop, plan(1), None)
