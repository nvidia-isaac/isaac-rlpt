# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-integration structure tests (see simulators/README.md).

Pins the seams every simulator integration plugs into:

* the environment-provider registry (register/get semantics, lazy imports,
  optional-dependency reporting),
* the ``simulator`` Hydra config group (entries compose, task enumerations
  are sane, every ``provider`` field names a registered provider),
* launcher/config agreement: ``simulators/g1_arena/configs/simulator/g1_arena.yaml``
  is the single source of truth for the per-task episode caps; the launchers
  must resolve exactly the values it declares,
* import hygiene: the envs package (including the provider registry and the
  LiberoEnv module) imports without the LIBERO stack installed, and
  requesting the provider then fails with ``ImportError``, not a crash.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from conftest import (
    EVAL_LAUNCHER,
    PACKAGE_NAME,
    REPO_ROOT,
    TASK_MAX_STEPS,
    dryrun_text,
    run_launcher_dryrun,
)
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(REPO_ROOT) / PACKAGE_NAME / "configs"
# The ``simulator`` config groups are co-located with each simulator
# (simulators/<sim>/configs/simulator/) and reach the launch config through
# dsrl.yaml's Hydra search path; discover them by scanning that layout rather
# than a central directory.
SIMULATORS_ROOT = Path(REPO_ROOT) / "simulators"

LIBERO_CONFIG_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _simulator_config_files() -> list[Path]:
    return sorted(SIMULATORS_ROOT.glob("*/configs/simulator/*.yaml"))


def _load_simulator_yaml(name: str) -> dict:
    matches = [path for path in _simulator_config_files() if path.stem == name]
    assert len(matches) == 1, (
        f"expected one simulator config named {name!r}, found {matches}"
    )
    with open(matches[0], encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@pytest.fixture()
def providers_module():
    """The registry module, with its state restored after each test."""
    from isaac_rl_posttraining.envs import providers

    saved_providers = dict(providers._PROVIDERS)
    saved_requirements = dict(providers._OPTIONAL_REQUIREMENTS)
    yield providers
    providers._PROVIDERS.clear()
    providers._PROVIDERS.update(saved_providers)
    providers._OPTIONAL_REQUIREMENTS.clear()
    providers._OPTIONAL_REQUIREMENTS.update(saved_requirements)


def test_provider_registry_resolves_the_shipped_simulators(providers_module):
    """g1_arena resolves to the Arena client, unknown names list the registered ones, and availability
    follows the installed stacks (the libero entries and builders are pinned in test_libero_runtime)."""
    from simulators.g1_arena.env import ArenaEnv

    assert providers_module.get_provider("g1_arena") is ArenaEnv
    with pytest.raises(KeyError, match="g1_arena"):
        providers_module.get_provider("no_such_simulator")
    available = providers_module.available_providers()
    assert "g1_arena" in available, "the transport-based Arena client needs no extras"
    libero_installed = _module_installed("libero") and _module_installed("robosuite")
    assert ("libero" in available) == libero_installed


def test_provider_registry_round_trip(providers_module):
    """register/get round-trips, duplicates need override=True, and a missing requirement reports as unavailable."""
    from simulators.g1_arena.env import ArenaEnv
    from simulators.libero.env import LiberoEnv

    providers_module.register_provider(
        "registry_test", "simulators.g1_arena.env", "ArenaEnv"
    )
    assert providers_module.get_provider("registry_test") is ArenaEnv
    assert "registry_test" in providers_module.available_providers()
    with pytest.raises(ValueError, match="already registered"):
        providers_module.register_provider(
            "g1_arena", "simulators.g1_arena.env", "ArenaEnv"
        )
    providers_module.register_provider(
        "g1_arena", "simulators.libero.env", "LiberoEnv", override=True
    )
    assert providers_module.get_provider("g1_arena") is LiberoEnv
    providers_module.register_provider(
        "needs_ghost_dep",
        "simulators.g1_arena.env",
        "ArenaEnv",
        requires=("ghost_dependency_that_does_not_exist",),
    )
    assert "needs_ghost_dep" not in providers_module.available_providers()
    with pytest.raises(ImportError, match="ghost_dependency_that_does_not_exist"):
        providers_module.get_provider("needs_ghost_dep")


@pytest.mark.parametrize(
    ("env_name", "entry_name"),
    [
        ("arena/drill_lift", "g1_arena"),
        ("isaac/drill_lift", "g1_arena"),
        ("isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", "isaaclab"),
    ],
)
def test_simulator_entries_resolve_lazily_on_the_exact_segment(
    providers_module, env_name, entry_name
):
    entry = providers_module.get_simulator_entry(env_name)

    assert entry.name == entry_name
    assert providers_module.find_simulator_entry(env_name) is entry


def test_environments_without_an_entry_are_reported(providers_module):
    assert providers_module.find_simulator_entry("liberox/libero_10/0") is None
    assert providers_module.find_simulator_entry("isaaclabx/Task-v0") is None
    with pytest.raises(KeyError, match="registered simulators"):
        providers_module.get_simulator_entry("CartPole-v1")


def test_builders_match_the_simulator_segment_exactly(providers_module):
    # Isaac Lab is sized from its entry's declaration, so the isaac builder must not match it.
    assert providers_module.get_env_builder("isaaclab/Task-v0") is None
    assert (
        providers_module.get_env_builder("isaac/x").__module__
        == "simulators.g1_arena.env"
    )
    assert (
        providers_module.get_env_builder("libero_pi0/libero_10/0").__module__
        == "simulators.libero.policies.pi0"
    )
    assert (
        providers_module.get_env_builder("libero/libero_10/0").__module__
        == "simulators.libero.policies.gr00t"
    )
    assert providers_module.get_env_builder("liberox/libero_10/0") is None
    assert providers_module.get_env_builder("ZeroSuccessTask") is None
    with pytest.raises(ValueError, match="already registered"):
        providers_module.register_env_builder(
            "libero", "simulators.libero.policies.gr00t:build_dsrl_env"
        )


# ---------------------------------------------------------------------------
# Simulator config group
# ---------------------------------------------------------------------------


def test_canonical_config_composes_g1_arena_simulator():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train")
    assert str(cfg.simulator.provider) == "g1_arena"
    assert list(cfg.simulator.tasks) == list(TASK_MAX_STEPS)
    assert int(cfg.env.max_episode_steps) == TASK_MAX_STEPS["LMDrillLift"], (
        "the canonical env cap must come from the simulator group's map"
    )


def test_every_simulator_config_names_a_registered_provider():
    from isaac_rl_posttraining.envs import providers

    yaml_files = _simulator_config_files()
    assert yaml_files, (
        f"no simulator configs under {SIMULATORS_ROOT}/*/configs/simulator/"
    )
    for path in yaml_files:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        provider = doc.get("provider")
        assert provider, f"{path.name}: missing the provider field"
        try:
            providers.get_provider(provider)
        except KeyError:
            pytest.fail(f"{path.name}: provider {provider!r} is not registered")
        except ImportError:
            pass  # registered; optional dependencies absent in this venv


@pytest.mark.parametrize("name", LIBERO_CONFIG_NAMES)
def test_libero_simulator_configs_compose_and_enumerate_their_tasks(name):
    doc = _load_simulator_yaml(name)
    tasks = doc["tasks"]
    assert isinstance(tasks, list) and len(tasks) == 10, (
        f"every LIBERO suite has exactly 10 tasks; {name} lists {len(tasks or [])}"
    )
    assert len(set(tasks)) == 10, f"{name}: duplicate task names"
    assert all(isinstance(t, str) and t for t in tasks)
    assert int(doc["max_episode_steps"]) == 600, (
        "LIBERO's own evaluation harness caps every suite at 600 steps"
    )
    assert doc["suite"] == name
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"simulator={name}"])
    assert (str(cfg.simulator.provider), str(cfg.simulator.suite)) == ("libero", name)
    assert list(cfg.simulator.tasks) == tasks


# ---------------------------------------------------------------------------
# Launcher/config agreement (the caps map has one source of truth)
# ---------------------------------------------------------------------------


def test_g1_arena_config_map_is_the_pinned_map():
    """The config's per-task caps equal the map the launcher dry-runs pin.

    simulators/g1_arena/tests/test_launchers.py drives the train launcher for
    all nine tasks against conftest.TASK_MAX_STEPS; this equality closes the
    triangle config == conftest == launcher.
    """
    doc = _load_simulator_yaml("g1_arena")
    assert doc["provider"] == "g1_arena"
    assert doc["max_episode_steps"] == TASK_MAX_STEPS
    assert list(doc["tasks"]) == list(TASK_MAX_STEPS), (
        "the task list and the caps map must enumerate the same nine tasks"
    )


@pytest.mark.parametrize("task", ["LMPushButton", "LMPushShelfForward"])
def test_eval_launcher_resolves_caps_from_config(launcher_sandbox, task):
    """The eval launcher's resolved cap equals the value read from the yaml."""
    expected = _load_simulator_yaml("g1_arena")["max_episode_steps"][task]
    result = run_launcher_dryrun(EVAL_LAUNCHER, [task, "base", 5], launcher_sandbox)
    assert result.returncode == 0, dryrun_text(result)
    text = dryrun_text(result)
    assert re.search(rf"max_episode_steps\s*=\s*{expected}\b", text), (
        f"eval plan must resolve max_episode_steps={expected} for {task}:\n{text}"
    )
    assert "plan.simulator=g1_arena" in text


@pytest.mark.parametrize("mode", ["base", "policy"])
def test_eval_launcher_uses_the_checkpoint_modality(launcher_sandbox, mode):
    """Both eval modes evaluate the checkpoint's baked modality: no dataset, quoted prompt.

    Passing the LeRobot dataset registers the dataset's state modality instead
    of the checkpoint's, and the GR00T state normalizer crashes on the first
    evaluation (this is how `policy` mode used to fail).
    """
    args = ["LMBoxLiftFloor", mode, 5]  # this task's prompt contains a comma
    if mode == "policy":
        args.append(launcher_sandbox["ckpt"])
    result = run_launcher_dryrun(EVAL_LAUNCHER, args, launcher_sandbox)
    assert result.returncode == 0, dryrun_text(result)
    text = dryrun_text(result)
    dataset_lines = [
        line
        for line in text.splitlines()
        if "launch.override=base_model.dataset_dir=" in line
    ]
    assert dataset_lines == ["launch.override=base_model.dataset_dir=null"], text
    assert (
        "launch.override=base_model.task_instruction='Walk to the box and pick it up from the floor'"
        in text
    ), text
    if mode == "policy":
        assert (
            f"launch.override=training.load_dir={launcher_sandbox['ckpt']}" in text
        ), text
        assert "launch.override=eval.eval_base=true" not in text, text
    else:
        assert "launch.override=eval.eval_base=true" in text, text
        assert "launch.override=training.load_dir=" not in text, text


def test_launchers_reject_unsupported_simulator(launcher_sandbox):
    """The DSRL launchers drive g1_arena; another SIMULATOR is a loud error."""
    result = run_launcher_dryrun(
        EVAL_LAUNCHER,
        ["DrillLift", "base", 5],
        launcher_sandbox,
        extra_env={"SIMULATOR": "libero_spatial"},
    )
    assert result.returncode != 0, "SIMULATOR=libero_spatial must not dry-run as arena"
    assert "SIMULATOR" in dryrun_text(result)


# ---------------------------------------------------------------------------
# Import hygiene without the LIBERO stack
# ---------------------------------------------------------------------------


def test_envs_package_imports_without_libero_stack():
    """`import isaac_rl_posttraining.envs` (incl. the LiberoEnv module) must
    succeed with the LIBERO stack blocked, and requesting the provider must
    raise ImportError -- not crash at import time."""
    blocked = ("libero", "robosuite", "mujoco", "gym")
    code = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            blocked = {blocked!r}

            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in self.blocked):
                    raise ImportError(f"optional dependency {{name!r}} blocked for hygiene test")
                return None

        sys.meta_path.insert(0, _Blocker())
        for mod in list(sys.modules):
            if any(mod == b or mod.startswith(b + ".") for b in _Blocker.blocked):
                del sys.modules[mod]

        # Must import w/o the stack.
        from simulators.libero.env import LiberoEnv  # noqa: F401
        from {PACKAGE_NAME}.envs import available_providers, get_provider
        assert "libero" not in available_providers()
        try:
            get_provider("libero")
        except ImportError:
            pass
        else:
            raise AssertionError("get_provider('libero') must raise ImportError without the stack")
        print("IMPORT_OK")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0 and "IMPORT_OK" in result.stdout, (
        f"envs package must import without the LIBERO stack.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
