# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python facade composes the same config as scripts/train.py and hands it to build_run."""

from __future__ import annotations

import pytest

from isaac_rl_posttraining import train as facade
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.envs.providers import find_simulator_entry
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy


class _FacadeSimulator(SimulatorEntry):
    name = "facade_sim"
    evaluation = "fresh_workers"

    def make_environment(self, spec, phase, context):
        del spec, phase, context
        return object()

    def steering_key_mapping(self, env_name):
        del env_name
        return {"state": "state", "language": None, "image": []}


@pytest.fixture(autouse=True)
def _restore_simulator_registry():
    """The facade registers entries globally; do not leak facade_sim into other tests."""
    saved = dict(providers._SIMULATOR_ENTRIES)
    yield
    providers._SIMULATOR_ENTRIES.clear()
    providers._SIMULATOR_ENTRIES.update(saved)


@pytest.fixture
def captured_run(monkeypatch):
    """Replace build_run with a capture so the facade composes without a simulator or GPU."""
    # The launch plan requires a served simulator's addresses; the arena entry names two.
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:7300")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:7301")
    captured = {}

    def fake_build_run(cfg):
        captured["cfg"] = cfg
        return lambda: captured.setdefault("ran", True)

    monkeypatch.setattr(facade, "build_run", fake_build_run)
    return captured


def test_names_select_the_groups_and_run(captured_run):
    facade.train(
        "arena/drill_lift",
        "mock",
        "rfs",
        overrides=["training.num_rollouts=0", "training.seed=7"],
    )
    cfg = captured_run["cfg"]
    assert (cfg.algorithm.name, cfg.algorithm.family) == ("rfs", "steering")
    assert (cfg.base_model.name, cfg.runtime.name, cfg.env.env_name) == (
        "mock",
        "native",
        "arena/drill_lift",
    )
    assert (cfg.training.num_rollouts, cfg.training.seed) == (0, 7)
    assert captured_run["ran"] is True


def test_objects_register_and_compose(captured_run):
    entry = _FacadeSimulator()
    facade.train(
        entry,
        MockBaseModelPolicy,
        "dsrl_sac",
        runtime="rlinf",
        gpus=2,
        overrides=["env.max_episode_steps=100"],
    )
    cfg = captured_run["cfg"]
    assert find_simulator_entry("facade_sim/task") is entry
    assert cfg.env.env_name == "facade_sim"
    assert (
        cfg.base_model.name == "isaac_rl_posttraining.policies.base.MockBaseModelPolicy"
    )
    # The off-policy learner keeps its own device; only the RLInf placement follows gpus.
    assert (cfg.learner.device, cfg.runtime.rlinf.first_accelerator) == ("auto", 2)


def test_the_launch_plan_checks_run_before_the_run(captured_run):
    """A policy whose action width disagrees with the simulator fails in the plan, as it does on the CLI."""
    from isaac_rl_posttraining.data.spec import SpecMismatch

    with pytest.raises(
        SpecMismatch, match=r"emitted float32\[43\] .* vs accepted float32\[35\]"
    ):
        facade.train(
            "arena/drill_lift",
            "mock",
            "dsrl_sac",
            overrides=["base_model.action_dim=43"],
        )
    assert "ran" not in captured_run


def test_an_entry_class_registers_under_its_name(captured_run):
    facade.train(
        _FacadeSimulator, "mock", "dsrl_sac", overrides=["env.max_episode_steps=100"]
    )
    assert isinstance(find_simulator_entry("facade_sim/task"), _FacadeSimulator)
    assert captured_run["cfg"].env.env_name == "facade_sim"


def test_gpus_place_the_on_policy_learner(captured_run):
    # The plan needs the GR00T generation; the generic gr00t base model reads it from a local
    # checkpoint, so name the generation the PPO recipe pins (gr00t_n15).
    facade.train(
        "isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "gr00t_n15",
        "ppo",
        runtime="rlinf",
        gpus=4,
        overrides=["base_model.ckpt=/ckpt/x"],
    )
    cfg = captured_run["cfg"]
    assert (
        cfg.learner.device,
        cfg.placement.learner.ranks,
        cfg.runtime.rlinf.first_accelerator,
    ) == ("gpu", 4, 4)
    facade.train(
        "isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "gr00t_n15",
        "ppo",
        gpus=0,
        overrides=[
            "base_model.ckpt=/ckpt/x",
            "placement.learner.ranks=1",
            "runtime.num_rollout_workers=1",
            "training.num_envs=16",
        ],
    )
    assert captured_run["cfg"].learner.device == "cpu"


def test_negative_gpu_budget_is_rejected(captured_run):
    with pytest.raises(ValueError, match="gpus"):
        facade.train("arena/drill_lift", "mock", "dsrl_sac", gpus=-1)
