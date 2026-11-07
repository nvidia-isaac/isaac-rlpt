# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""By-value pickling of user-defined specs for the Ray-based runtimes, however deeply the entry is nested."""

from __future__ import annotations

import functools
import importlib
import shutil
import sys
import textwrap

import numpy as np
import pytest

from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec
from isaac_rl_posttraining.runtimes._serialization import (
    _spec_modules,
    serialize_specs_by_value,
)
from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec

ray_cloudpickle = pytest.importorskip(
    "ray.cloudpickle", reason="the by-value registry is Ray's cloudpickle"
)


def _throwaway_entry_module(tmp_path, monkeypatch):
    """A simulator entry class defined in a module outside the package, as a user's script would."""
    source = textwrap.dedent(
        """
        from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

        class ThrowawayRobot(SimulatorEntry):
            name = "throwaway_robot"

            def make_environment(self, spec, phase, context):
                raise NotImplementedError
        """
    )
    (tmp_path / "throwaway_robot_entry.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = __import__("throwaway_robot_entry")
    return module


def _served_like_spec(entry):
    """A worker spec nested one level down, the shape shared policy serving hands to Ray."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Wrapper:
        worker: GymWorkerSpec
        actors: tuple = ()

    worker = GymWorkerSpec(
        entry=entry,
        environment=EnvironmentSpec(
            env_name="throwaway/task", num_envs=1, max_episode_steps=4, seed=0
        ),
        phase="train",
        policy=lambda context: None,
    )
    return Wrapper(worker=worker)


def test_nested_entry_module_is_registered_and_round_trips_without_its_module(
    tmp_path, monkeypatch
):
    module = _throwaway_entry_module(tmp_path, monkeypatch)
    spec = _served_like_spec(module.ThrowawayRobot())
    assert module in _spec_modules(spec)
    with serialize_specs_by_value(spec, enabled=True):
        payload = ray_cloudpickle.dumps(spec)
    # The receiving process has neither the module on its path nor in sys.modules.
    monkeypatch.delitem(sys.modules, "throwaway_robot_entry")
    (tmp_path / "throwaway_robot_entry.py").unlink()
    restored = ray_cloudpickle.loads(payload)
    assert (
        type(restored.worker.entry).__name__ == "ThrowawayRobot"
        and restored.worker.entry.name == "throwaway_robot"
    )


def test_an_entry_importing_a_sibling_module_round_trips_without_its_package(
    tmp_path, monkeypatch
):
    """The template layout: ``runtime.py`` defines the entry and imports its env from ``env.py``."""
    package = tmp_path / "throwaway_robot_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "env.py").write_text("STEPS = 7\n")
    (package / "runtime.py").write_text(
        textwrap.dedent(
            """
            from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

            from .env import STEPS


            class ThrowawayRobot(SimulatorEntry):
                name = "throwaway_robot"

                def make_environment(self, spec, phase, context):
                    return STEPS
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    runtime = importlib.import_module("throwaway_robot_pkg.runtime")
    spec = _served_like_spec(runtime.ThrowawayRobot())
    names = [module.__name__ for module in _spec_modules(spec)]
    assert "throwaway_robot_pkg" in names and "throwaway_robot_pkg.runtime" not in names
    with serialize_specs_by_value(spec, enabled=True):
        payload = ray_cloudpickle.dumps(spec)
    for name in (
        "throwaway_robot_pkg",
        "throwaway_robot_pkg.env",
        "throwaway_robot_pkg.runtime",
    ):
        monkeypatch.delitem(sys.modules, name)
    shutil.rmtree(package)
    restored = ray_cloudpickle.loads(payload)
    assert restored.worker.entry.make_environment(None, "train", None) == 7


def test_scan_stops_at_cycles_and_plain_values(tmp_path, monkeypatch):
    module = _throwaway_entry_module(tmp_path, monkeypatch)
    entry = module.ThrowawayRobot()
    entry.loop = entry  # a self-reference must not recurse forever
    modules = _spec_modules({"entries": [entry], "count": 3, "name": "x"})
    assert module in modules


def _build_nothing():
    return None


def test_library_modules_are_never_registered_but_the_entry_module_still_is(
    tmp_path, monkeypatch
):
    module = _throwaway_entry_module(tmp_path, monkeypatch)
    spec = {
        "learner": functools.partial(module.ThrowawayRobot, name="x"),
        "reduce": functools.partial(np.mean, axis=0),
        "sort": sorted,
        "array": np.zeros(2),
    }
    names = {m.__name__ for m in _spec_modules(spec)}
    assert module.__name__ in names
    assert not names & {"functools", "numpy", "builtins"}
    assert not any(name.startswith("numpy") for name in names)


def test_in_tree_modules_import_by_name_on_workers():
    """The package's own specs are not pickled by value: every worker has the package installed."""
    from isaac_rl_posttraining.runtimes.gym_worker import GymWorkerSpec

    assert _spec_modules(GymWorkerSpec) == ()
    # A test module inside the checkout is not installed anywhere: its package takes the by-value path.
    assert [module.__name__ for module in _spec_modules({"build": _build_nothing})] == [
        __name__.partition(".")[0]
    ]


def test_dsrl_ppo_learner_spec_pickles_by_value():
    """The learner spec is a functools.partial; registering its modules by value must not drag in the stdlib."""
    from isaac_rl_posttraining.algorithms.ppo.ppo import PPO

    spec = PPO().learner(_build_nothing, frozen_submodules=("model.backbone",))
    # functools, torch and the package import by name; only the test's build_model is by value.
    assert [module.__name__ for module in _spec_modules(spec)] == [
        __name__.partition(".")[0]
    ]
    with serialize_specs_by_value(spec, enabled=True):
        payload = ray_cloudpickle.dumps(spec)
    restored = ray_cloudpickle.loads(payload)
    assert restored.func.__name__ == "PPOLearner"
    assert restored.keywords["frozen_submodules"] == ("model.backbone",)
    assert restored.keywords["build_model"]() is None
