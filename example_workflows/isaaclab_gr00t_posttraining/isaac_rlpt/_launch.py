# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared launch logic for the ``isaac_rlpt`` train/play modules.

Order of operations, mirroring the other Isaac Lab library modules:

1. parse the CLI, 2. resolve the output root, 3. compose the library's
``isaac_rl_posttraining`` Hydra ``train`` config with the launch overrides,
4. hand it to the library's entry script, which prints and checks the launch
plan before anything loads. This process starts no Isaac Sim: every scene runs
in a simulator child process of a rollout worker, and the learner is sized
from the task's declared observation and action layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType


def import_isaac_rl_posttraining_entry(script: str) -> ModuleType:
    """Import an entry script (path relative to the checkout) from the installed (editable) library."""
    import isaac_rl_posttraining

    repo_root = Path(isaac_rl_posttraining.__file__).resolve().parents[1]
    entry = repo_root / script
    if not entry.is_file():
        raise SystemExit(
            f"isaac_rl_posttraining must be an editable install of its git checkout (missing {entry}); "
            "run `pip install -e /path/to/isaac_rl_posttraining --no-deps`."
        )
    spec = importlib.util.spec_from_file_location(
        "isaac_rl_posttraining_" + entry.stem, entry
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"{entry} cannot be loaded as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_run_name(args: argparse.Namespace, mode: str) -> str:
    """``<mode>_<algorithm>_<task>_<timestamp>`` unless ``--run_name`` is given."""
    if args.run_name:
        return args.run_name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{mode}_{args.algorithm}_{args.task}_{stamp}"


def output_root() -> Path:
    """Run-output root: ``ISAAC_RL_POSTTRAINING_OUTPUT_DIR`` if set, else ``logs/isaac_rlpt`` under the working directory.

    Isaac Lab's other libraries write to ``logs/<library>/`` next to where
    ``isaaclab train`` runs; the library creates ``<root>/<run_name>_<timestamp>``
    there. The module directory is a symlink into the library checkout, so it
    is never used as a log location.
    """
    configured = os.environ.get("ISAAC_RL_POSTTRAINING_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return Path("logs", "isaac_rlpt").absolute()


#: The library entry every algorithm runs through: its Hydra config and its script.
LIBRARY_CONFIG = "train"
LIBRARY_SCRIPT = "scripts/train.py"


def launch(args: argparse.Namespace, overrides: list[str], run_name: str) -> None:
    """Run the library's training entry with ``overrides``."""
    root = output_root()
    # Launcher boundary: the library reads its output root from this variable.
    os.environ["ISAAC_RL_POSTTRAINING_OUTPUT_DIR"] = str(root)
    # The launcher's own facts as launch.* lines; the library prints its plan.* lines after them.
    print(f"launch.output_root={root}")
    print(f"launch.run_dir={root / run_name}_<timestamp>")
    print(
        "\n".join(f"launch.override={override}" for override in overrides), flush=True
    )

    if args.dry_run:
        os.environ["ISAAC_RL_POSTTRAINING_DRYRUN"] = "1"
    module, cfg = _compose(overrides)
    module.run_from_hydra_cfg(cfg)


def _compose(overrides: list[str]) -> tuple[ModuleType, object]:
    """The library entry module and its config composed with ``overrides``."""
    module = import_isaac_rl_posttraining_entry(LIBRARY_SCRIPT)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import isaac_rl_posttraining
    from isaac_rl_posttraining.configs import register_configs

    module.configure_runtime_env()
    register_configs()
    config_dir = Path(isaac_rl_posttraining.__file__).resolve().parent / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=LIBRARY_CONFIG, overrides=overrides)
    return module, cfg
