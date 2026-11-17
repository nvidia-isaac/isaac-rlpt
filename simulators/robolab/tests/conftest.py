# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export the repo root ``tests/conftest.py`` scaffolding.

The co-located simulator tests import shared helpers/fixtures via
``from conftest import ...``. Pytest puts this directory on ``sys.path`` (its
default prepend import mode), so this module answers that name and re-exports
the root conftest (constants, launcher/sandbox helpers, and the arena/libero
pytest fixtures) for tests that moved out of ``tests/``.
"""

import importlib.util as _u
import pathlib as _p

_root = _p.Path(__file__).resolve().parents[3] / "tests" / "conftest.py"
_spec = _u.spec_from_file_location("_root_conftest_shared", _root)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export every public name (constants, helpers, and @pytest.fixture objects)
# so both ``from conftest import X`` and pytest fixture injection resolve here.
globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("__")})
