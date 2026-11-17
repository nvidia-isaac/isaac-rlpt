# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/install.py`` links the module into Isaac Lab's RL scripts directory and registers it, reversibly."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

WORKFLOW = Path(__file__).resolve().parent.parent
MODULE_DIR = WORKFLOW / "isaac_rlpt"
#: The shape of Isaac Lab's dispatch table in ``scripts/reinforcement_learning/{train,play}.py``.
ENTRYPOINTS_TABLE = (
    'LIBRARY_ENTRYPOINTS = {\n    "rsl_rl": SCRIPT_DIR / "rsl_rl" / "train.py",\n}\n'
)


def _install() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "isaaclab_gr00t_posttraining_install", WORKFLOW / "scripts" / "install.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rl_dir(tmp_path: Path) -> Path:
    """A stand-in for Isaac Lab's ``scripts/reinforcement_learning`` directory."""
    directory = tmp_path / "IsaacLab" / "scripts" / "reinforcement_learning"
    directory.mkdir(parents=True)
    for name in ("train.py", "play.py"):
        (directory / name).write_text(ENTRYPOINTS_TABLE, encoding="utf-8")
    (directory / "common.py").write_text("", encoding="utf-8")
    return directory


def _entries(rl_dir: Path, name: str) -> int:
    return (rl_dir / name).read_text(encoding="utf-8").count('"isaac_rlpt":')


def test_install_links_registers_idempotently_and_uninstalls(
    rl_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install = _install()
    install.main(["--rl_dir", str(rl_dir), "--check"])
    assert "missing" in capsys.readouterr().out

    install.main(["--rl_dir", str(rl_dir)])
    target = rl_dir / "isaac_rlpt"
    assert target.is_symlink() and target.resolve() == MODULE_DIR
    for name in ("train.py", "play.py"):
        text = (rl_dir / name).read_text(encoding="utf-8")
        entry = f'"isaac_rlpt": SCRIPT_DIR / "isaac_rlpt" / "{name[:-3]}_isaac_rlpt.py"'
        assert entry in text and _entries(rl_dir, name) == 1

    install.main(["--rl_dir", str(rl_dir)])
    assert _entries(rl_dir, "train.py") == 1 and _entries(rl_dir, "play.py") == 1

    install.main(["--rl_dir", str(rl_dir), "--uninstall"])
    assert not target.exists() and not target.is_symlink()
    for name in ("train.py", "play.py"):
        assert (rl_dir / name).read_text(encoding="utf-8") == ENTRYPOINTS_TABLE


def test_install_from_the_checkout_root_and_in_copy_mode(rl_dir: Path) -> None:
    install = _install()
    checkout = rl_dir.parent.parent
    install.main(["--isaaclab", str(checkout), "--copy"])
    target = rl_dir / "isaac_rlpt"
    assert target.is_dir() and not target.is_symlink()
    for name in (
        "cli_args.py",
        "_launch.py",
        "train_isaac_rlpt.py",
        "play_isaac_rlpt.py",
    ):
        assert (target / name).is_file()
    assert not (target / "__pycache__").exists()
    assert _entries(rl_dir, "train.py") == 1 and _entries(rl_dir, "play.py") == 1


def test_install_rejects_a_directory_that_is_not_isaac_labs(tmp_path: Path) -> None:
    install = _install()
    with pytest.raises(
        SystemExit, match="not Isaac Lab's scripts/reinforcement_learning"
    ):
        install.main(["--rl_dir", str(tmp_path)])
    without_table = tmp_path / "rl"
    without_table.mkdir()
    for name in ("train.py", "play.py", "common.py"):
        (without_table / name).write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="LIBRARY_ENTRYPOINTS table not found"):
        install.main(["--rl_dir", str(without_table)])


def test_install_into_the_isaac_lab_3_0_0_dispatch_table(tmp_path: Path) -> None:
    install = _install()
    entrypoints = tmp_path / "source" / "isaaclab_rl" / "isaaclab_rl" / "entrypoints"
    (entrypoints / "backends").mkdir(parents=True)
    table = '_BACKEND_MODULES = {\n    "train": {\n        "sb3": "a",\n    },\n    "play": {\n        "sb3": "b",\n    },\n}\n'
    (entrypoints / "dispatch.py").write_text(table, encoding="utf-8")

    install.main(["--isaaclab", str(tmp_path)])
    install.main(["--isaaclab", str(tmp_path)])
    text = (entrypoints / "dispatch.py").read_text(encoding="utf-8")
    assert text.count('"isaac_rlpt":') == 2
    for action in ("train", "play"):
        entry = f'"isaac_rlpt": "isaaclab_rl.entrypoints.backends.isaac_rlpt.{action}_isaac_rlpt"'
        assert text.count(entry) == 1
    assert (entrypoints / "backends" / "isaac_rlpt").resolve() == MODULE_DIR

    install.main(["--isaaclab", str(tmp_path), "--uninstall"])
    assert (entrypoints / "dispatch.py").read_text(encoding="utf-8") == table
    assert not (entrypoints / "backends" / "isaac_rlpt").exists()
