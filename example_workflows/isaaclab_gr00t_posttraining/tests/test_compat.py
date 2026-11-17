# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launcher helpers resolve on both Isaac Lab layouts; release/3.0.0 has no ``common.py`` and no ``--headless``."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPAT = Path(__file__).resolve().parent.parent / "isaac_rlpt" / "_compat.py"


def _load(monkeypatch: pytest.MonkeyPatch, common: types.ModuleType | None):
    """Load ``_compat`` with ``common`` (Isaac Lab beta2's script module) present or, for ``None``, missing."""
    app = types.ModuleType("isaaclab.app")
    app.add_launcher_args = lambda parser: parser.add_argument("--device")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    monkeypatch.setitem(
        sys.modules, "common", common
    )  # None makes `import common` raise ImportError
    spec = importlib.util.spec_from_file_location("isaac_rlpt_compat_test", COMPAT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compat(monkeypatch: pytest.MonkeyPatch):
    return _load(monkeypatch, None)


def test_falls_back_to_isaaclab_app_and_loads_modules_by_path(monkeypatch) -> None:
    compat = _load(monkeypatch, None)
    parser = argparse.ArgumentParser()
    compat.add_isaaclab_launcher_args(parser)
    assert parser.parse_known_args(["--device", "cuda:0"])[0].device == "cuda:0"
    loaded = compat.import_local_module("compat_probe", COMPAT.parent / "__init__.py")
    assert sys.modules["compat_probe"] is loaded


def test_uses_the_common_helpers_when_the_checkout_has_them(monkeypatch) -> None:
    common = types.ModuleType("common")
    common.add_isaaclab_launcher_args = object()  # type: ignore[attr-defined]
    common.import_local_module = object()  # type: ignore[attr-defined]
    compat = _load(monkeypatch, common)
    assert compat.add_isaaclab_launcher_args is common.add_isaaclab_launcher_args
    assert compat.import_local_module is common.import_local_module


def test_headless_is_accepted_and_not_a_hydra_override(compat) -> None:
    parser = argparse.ArgumentParser()
    compat.add_isaaclab_launcher_args(parser)
    args, extra = parser.parse_known_args(["--headless", "--device", "cuda:0", "x=1"])
    assert args.device == "cuda:0" and extra == ["x=1"]


def test_a_launcher_that_defines_headless_does_not_conflict(compat) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    compat.add_isaaclab_launcher_args(parser)
