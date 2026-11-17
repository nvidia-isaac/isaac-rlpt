# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export the repo root ``tests/conftest.py`` scaffolding for every workflow's tests.

Workflow tests are imported as packages (``example_workflows.<workflow>.tests``), so they reach the
shared helpers through ``from example_workflows.conftest import ...``; pytest registers this module as
their conftest, which also gives them the hardware/sim marker gate and the autouse fixtures the core
suite has.
"""

import importlib.util as _u
import os
import pathlib as _p
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_root = _p.Path(__file__).resolve().parents[1] / "tests" / "conftest.py"
_spec = _u.spec_from_file_location("_root_conftest_shared", _root)
assert _spec is not None and _spec.loader is not None
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("__")})
# Named explicitly so the workflow tests' imports type-check.
launcher_sandbox = _mod.launcher_sandbox
run_launcher_dryrun = _mod.run_launcher_dryrun
dryrun_text = _mod.dryrun_text
TRAIN_LAUNCHER = _mod.TRAIN_LAUNCHER
EVAL_LAUNCHER = _mod.EVAL_LAUNCHER
TASK_MAX_STEPS = _mod.TASK_MAX_STEPS
